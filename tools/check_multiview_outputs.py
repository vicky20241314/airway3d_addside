"""Read-only validation of completed multi-view outputs; write a separate report."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from core.conservative_refine import RefineOptions,ShapeGuard
from core.multiview_geometry import RigidCamera
from core.multiview_data import load_session
from core.pipeline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and report['metric_geometry_stage']==2
    metrics=json.loads((a.run/'metrics.json').read_text(encoding='utf-8'));assert metrics['geometry_stage']==2
    source=Path(report['source']) if 'source' in report else a.run
    ext=json.loads((source/'extrinsics.json').read_text(encoding='utf-8'))
    fronts={o.view_id for o in load_session(source/'prepared/session.json') if o.view=='front'}
    for key,row in ext.items():
        assert row['direction']=='state_front_to_view'
        T=RigidCamera(row['R'],row['t'])
        if key in fronts:
            np.testing.assert_allclose(T.R,np.eye(3),atol=1e-6);np.testing.assert_allclose(T.t,0,atol=1e-6)
    with np.load(source/'flame_parameters.npz') as parameters:
        assert parameters['shared_shape'].ndim==1 and 0<len(parameters['shared_shape'])<=300
        for state in ('closed','open'):
            ids=np.flatnonzero(parameters['states']==state)
            for name in ('expression_by_view','pose_by_view'):
                for i in ids:np.testing.assert_array_equal(parameters[name][i],parameters[name][ids[0]])
    details=[]
    for state in ('closed','open'):
        path=a.run/state
        stage2=trimesh.load(path/f'mesh_{state}_stage2.obj',process=False,force='mesh')
        if report['stage3_enabled']:
            with np.load(path/'vertex_offsets.npz') as offsets:
                reference=offsets['reference_vertices'];faces=offsets['faces'];D=offsets['offsets']
            mesh=trimesh.load(path/f'mesh_{state}.obj',process=False,force='mesh')
            np.testing.assert_allclose(mesh.vertices,reference+D,rtol=0,atol=1e-8)
            v,f=trimesh.remesh.subdivide(stage2.vertices,stage2.faces)
            np.testing.assert_allclose(v,reference,rtol=0,atol=1e-8);np.testing.assert_array_equal(f,faces)
            assert np.linalg.norm(D,axis=1).max()<=.002+1e-12
            bad,quality=ShapeGuard(reference,faces,RefineOptions()).inspect(reference+D);assert not len(bad)
            info=json.loads((path/'stage3_report.json').read_text(encoding='utf-8'))
            assert info['shared_displacement'] and info['state']==state
            assert all(row['step_m']<=.00025+1e-12 for row in info['history'])
            fair=info.get('displacement_fairing',{})
            assert fair.get('correction_max_mm',0)<=.1+1e-10
            details.append(dict(state=state,max_offset_mm=float(np.linalg.norm(D,axis=1).max()*1000),
                stage2_vertices=len(stage2.vertices),stage3_vertices=len(reference),quality=quality))
    out=a.output/('mv_output_checks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    write_json(out/'checks.json',dict(source=str(a.run.resolve()),passed=True,states=details,
        parameter_sharing=True,metric_geometry_stage=2,geometry_checks_performed=bool(report['stage3_enabled']),
        offsets_applied_once=bool(report['stage3_enabled']),real_side_effect_validated=False))
    print(json.dumps(dict(passed=True,states=details,report=str(out/'checks.json'))),flush=True)


if __name__=='__main__':main()
