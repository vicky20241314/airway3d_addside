"""Controlled reference/projected comparison from identical saved stage2 input."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from core.multiview_data import load_session
from core.multiview_geometry import RigidCamera
from core.pipeline import finish_multiview_refinement,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    out=a.output/('refinement_backends_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    observations=load_session(a.run/'prepared/session.json')
    ext=json.loads((a.run/'extrinsics.json').read_text(encoding='utf-8'))
    assert all(x['direction']=='state_front_to_view' for x in ext.values())
    transforms={k:RigidCamera(x['R'],x['t']) for k,x in ext.items()}
    stages=[trimesh.load(a.run/s/f'mesh_{s}_stage2.obj',process=False,force='mesh') for s in ('closed','open')]
    record=dict(status='started',source=str(a.run.resolve()),seconds={},comparison=[])
    write_json(out/'comparison.json',record);meshes={}
    previous=os.environ.get('AIRWAY3D_RAY_BACKEND')
    try:
        for backend in ('reference','projected'):
            os.environ['AIRWAY3D_RAY_BACKEND']=backend;folder=out/backend;folder.mkdir()
            (folder/'metrics.json').write_bytes((a.run/'metrics.json').read_bytes())
            for state in ('closed','open'):
                (folder/state).mkdir();name=f'mesh_{state}_stage2.obj'
                (folder/state/name).write_bytes((a.run/state/name).read_bytes())
            start=time.perf_counter()
            meshes[backend],info=finish_multiview_refinement(stages,observations,transforms,folder,
                lambda s:print(backend,s,flush=True))
            elapsed=time.perf_counter()-start;record['seconds'][backend]=elapsed
            write_json(folder/'report.json',dict(status='complete',source=str(a.run.resolve()),refinement=info,
                stage3_enabled=True,metric_geometry_stage=2,visibility_backend=backend,elapsed_s=elapsed))
            write_json(out/'comparison.json',record)
        for i,state in enumerate(('closed','open')):
            b,c=meshes['reference'][i],meshes['projected'][i]
            np.testing.assert_array_equal(b.faces,c.faces)
            delta=np.linalg.norm(b.vertices-c.vertices,axis=1)
            record['comparison'].append(dict(state=state,max_delta_mm=float(delta.max()*1000),
                mean_delta_mm=float(delta.mean()*1000),passed=bool(delta.max()<=.00002)))
        record.update(status='complete',passed=all(r['passed'] for r in record['comparison']),
            speedup=record['seconds']['reference']/record['seconds']['projected'])
        write_json(out/'comparison.json',record);print(json.dumps(record),flush=True);print(out,flush=True)
        if not record['passed']:raise AssertionError('Controlled backend refinement differs by more than 0.02 mm')
    finally:
        if previous is None:os.environ.pop('AIRWAY3D_RAY_BACKEND',None)
        else:os.environ['AIRWAY3D_RAY_BACKEND']=previous


if __name__=='__main__':main()
