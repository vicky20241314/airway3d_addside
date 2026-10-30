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
            info=json.loads((path/'stage3_report.json').read_text(encoding='utf-8'))
            refinement_base=stage2
            if info.get('input_geometry','stage2')=='shape_corrected':
                refinement_base=trimesh.load(path/f'mesh_{state}_shape_corrected.obj',process=False,force='mesh')
            v,f=trimesh.remesh.subdivide(refinement_base.vertices,refinement_base.faces)
            np.testing.assert_allclose(v,reference,rtol=0,atol=1e-8);np.testing.assert_array_equal(f,faces)
            assert np.linalg.norm(D,axis=1).max()<=.002+1e-12
            bad,quality=ShapeGuard(reference,faces,RefineOptions()).inspect(reference+D);assert not len(bad)
            assert info['shared_displacement'] and info['state']==state
            assert all(row['step_m']<=.00025+1e-12 for row in info['history'])
            fair=info.get('displacement_fairing',{})
            assert fair.get('correction_max_mm',0)<=.1+1e-10
            details.append(dict(state=state,max_offset_mm=float(np.linalg.norm(D,axis=1).max()*1000),
                stage2_vertices=len(stage2.vertices),stage3_vertices=len(reference),quality=quality))
    if report.get('fit_configuration',{}).get('canonical_shape_enabled'):
        shape_report=json.loads((source/'canonical_shape_report.json').read_text(encoding='utf-8'))
        with np.load(source/'canonical_shape.npz') as shape_arrays:
            np.testing.assert_allclose(np.einsum('vck,k->vc',shape_arrays['basis'],shape_arrays['alpha']),shape_arrays['canonical_offset'],atol=1e-12)
            assert np.linalg.norm(shape_arrays['canonical_offset'],axis=1).max()<=.003+1e-12
            assert not np.any(shape_arrays['canonical_offset'][~shape_arrays['active']])
        assert shape_report['shared_alpha'] and shape_report['max_state_change_m']<=.003+1e-8
        assert shape_report['exact_lbs_reconstruction_error_m']<=1e-6
        if 'joint_moment_max_m' in shape_report:
            assert shape_report['joint_moment_max_m']<=1e-8
            assert shape_report['protected_posed_change_m']<=1e-6
    out=a.output/('mv_output_checks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    write_json(out/'checks.json',dict(source=str(a.run.resolve()),passed=True,states=details,
        parameter_sharing=True,metric_geometry_stage=2,geometry_checks_performed=bool(report['stage3_enabled']),
        offsets_applied_once=bool(report['stage3_enabled']),real_side_effect_validated=False))
    print(json.dumps(dict(passed=True,states=details,report=str(out/'checks.json'))),flush=True)


if __name__=='__main__':main()
