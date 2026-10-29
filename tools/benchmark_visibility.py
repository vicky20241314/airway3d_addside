"""Compare exact reference and accelerated ray masks on saved real meshes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from core.multiview_data import load_session
from core.multiview_geometry import RigidCamera,ray_visible
from core.pipeline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    out=a.output/('visibility_benchmark_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    ext=json.loads((a.run/'extrinsics.json').read_text(encoding='utf-8'));rows=[]
    for obs in load_session(a.run/'prepared/session.json'):
        if obs.view_id not in ext:continue
        T=RigidCamera(ext[obs.view_id]['R'],ext[obs.view_id]['t'])
        mesh=trimesh.load(a.run/obs.state/f'mesh_{obs.state}_stage2.obj',process=False,force='mesh')
        for level in ('stage2','subdivided'):
            v,f=(mesh.vertices,mesh.faces) if level=='stage2' else trimesh.remesh.subdivide(mesh.vertices,mesh.faces)
            v=T.to_view(v);samples=np.vstack((v,v[f].mean(1)))
            timings={};masks={}
            for backend in ('reference','projected'):
                t=time.perf_counter();masks[backend]=ray_visible(v,f,samples,backend=backend);timings[backend]=time.perf_counter()-t
            mismatch=int(np.count_nonzero(masks['reference']!=masks['projected']))
            row=dict(view_id=obs.view_id,level=level,rays=len(samples),mismatches=mismatch,seconds=timings,
                speedup=timings['reference']/max(timings['projected'],1e-12))
            rows.append(row);print(json.dumps(row),flush=True)
            write_json(out/'benchmark.json',dict(source=str(a.run.resolve()),rows=rows,all_masks_equal=all(r['mismatches']==0 for r in rows)))
            if mismatch:raise AssertionError('Visibility masks differ; inspect before enabling acceleration')
    print(out,flush=True)


if __name__=='__main__':main()
