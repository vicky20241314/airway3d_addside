"""Replay joint refinement from saved stage2; preserve metrics and source run."""
import argparse
import json
import sys
import shutil
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
    p.add_argument('--geometry',choices=('stage2','shape_corrected'),default='stage2')
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    source_report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    if source_report['status']!='complete':raise ValueError('Refinement requires a completed source')
    observations=load_session(a.run/'prepared/session.json')
    policies=source_report.get('fitting',{}).get('constraint_policies',{})
    for o in observations:o.quality.update(policies.get(o.view_id,{}))
    ext=json.loads((a.run/'extrinsics.json').read_text(encoding='utf-8'))
    if any(row['direction']!='state_front_to_view' for row in ext.values()):raise ValueError('Extrinsic direction mismatch')
    transforms={k:RigidCamera(row['R'],row['t']) for k,row in ext.items()}
    folder=a.output/('mv_refine_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(parents=True,exist_ok=False)
    snapshots=[];stage2=[]
    for state in ('closed','open'):
        path=a.run/state/f'mesh_{state}_stage2.obj'
        mesh=trimesh.load(path,process=False,force='mesh');stage2.append(mesh)
        (folder/state).mkdir();shutil.copyfile(path,folder/state/path.name)
        if a.geometry=='shape_corrected':
            path=a.run/state/f'mesh_{state}_shape_corrected.obj'
            mesh=trimesh.load(path,process=False,force='mesh');shutil.copyfile(path,folder/state/path.name)
        snapshots.append(mesh)
    metrics=(a.run/'metrics.json').read_bytes();(folder/'metrics.json').write_bytes(metrics)
    report=dict(status='started',source=str(a.run.resolve()),metric_geometry_stage=2,stage3_enabled=True,real_side_effect_validated=False,
        version=(ROOT/'VERSION').read_text().strip(),refinement_source=a.geometry,fit_configuration=source_report.get('fit_configuration',{}))
    write_json(folder/'report.json',report)
    try:
        display,info=finish_multiview_refinement(snapshots,observations,transforms,folder,lambda s:print(s,flush=True))
        for state,original,mesh,row in zip(('closed','open'),stage2,display,info):
            raw_reference,_=trimesh.remesh.subdivide(original.vertices,original.faces)
            row.update(input_geometry=a.geometry,cumulative_change_from_stage2_max_m=float(np.linalg.norm(mesh.vertices-raw_reference,axis=1).max()))
            write_json(folder/state/'stage3_report.json',row)
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
