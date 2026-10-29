"""Replay joint refinement from saved stage2; preserve metrics and source run."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from core.multiview_data import load_session
from core.multiview_geometry import RigidCamera
from core.pipeline import finish_multiview_refinement,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    observations=load_session(a.run/'prepared/session.json')
    ext=json.loads((a.run/'extrinsics.json').read_text(encoding='utf-8'))
    if any(row['direction']!='state_front_to_view' for row in ext.values()):raise ValueError('Extrinsic direction mismatch')
    transforms={k:RigidCamera(row['R'],row['t']) for k,row in ext.items()}
    folder=a.output/('mv_refine_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(parents=True,exist_ok=False)
    snapshots=[]
    for state in ('closed','open'):
        path=a.run/state/f'mesh_{state}_stage2.obj'
        mesh=trimesh.load(path,process=False,force='mesh');snapshots.append(mesh)
        (folder/state).mkdir();mesh.export(folder/state/path.name)
    metrics=(a.run/'metrics.json').read_bytes();(folder/'metrics.json').write_bytes(metrics)
    report=dict(status='started',source=str(a.run.resolve()),metric_geometry_stage=2,stage3_enabled=True,real_side_effect_validated=False,
        version=(ROOT/'VERSION').read_text().strip())
    write_json(folder/'report.json',report)
    try:
        display,info=finish_multiview_refinement(snapshots,observations,transforms,folder,lambda s:print(s,flush=True))
        for state,mesh in zip(('closed','open'),display):
            offsets=np.load(folder/state/'vertex_offsets.npz')
            np.testing.assert_allclose(mesh.vertices,offsets['reference_vertices']+offsets['offsets'],rtol=0,atol=1e-12)
            saved=trimesh.load(folder/state/f'mesh_{state}.obj',process=False,force='mesh')
            np.testing.assert_allclose(saved.vertices,mesh.vertices,rtol=0,atol=1e-8)
            assert np.max(np.linalg.norm(offsets['offsets'],axis=1))<=.002+1e-10
        assert (folder/'metrics.json').read_bytes()==metrics
        report.update(status='complete',refinement=info,offsets_applied_once=True,metrics_byte_identical=True)
        write_json(folder/'report.json',report);print(folder,flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc));write_json(folder/'report.json',report);raise


if __name__=='__main__':main()
