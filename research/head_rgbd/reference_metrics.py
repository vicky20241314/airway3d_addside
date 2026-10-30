"""Bind coarse/refined landmarks and geometric mouth gaps to exact reference frames."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from audit import dump,sha
from fit_b import normals
from core.fit_options import FitOptions
from core.integrated_fitting import make_flame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    report=json.loads((a.fit/'report.json').read_text(encoding='utf-8'))
    sequence=json.loads(a.sequence.read_text(encoding='utf-8'))
    with np.load(a.fit/'parameters.npz',allow_pickle=False) as d:params={k:d[k] for k in d.files}
    n=len(report['frames']);flame=make_flame(FitOptions(shape_components=50,expression_components=10),'cpu',batch=n)
    with torch.no_grad():
        pose=torch.cat((torch.zeros(n,3),torch.tensor(params['jaw'])),1)
        full=torch.cat((flame.head_pose,pose,flame.eye_pose),1)
        fi,bc=flame._find_dynamic_lmk_idx_and_bcoords(torch.tensor(params['raw_vertices']),full,
            flame.dynamic_lmk_faces_idx,flame.dynamic_lmk_bary_coords,flame.neck_kin_chain,
            view_rotation=torch.tensor(params['R']))
        fi=torch.cat((fi,flame.lmk_faces_idx[None].expand(n,-1)),1).numpy()
        bc=torch.cat((bc,flame.lmk_bary_coords[None].expand(n,-1,-1)),1).numpy()
    output=dict(schema='head_rgbd.reference_geometry.v1',mesh_unit='m',landmark_numbering='zero-based FLAME 68',
        report_sha256=sha(a.fit/'report.json'),parameters_sha256=sha(a.fit/'parameters.npz'),
        sequence_sha256=sha(a.sequence),references={},clinical_validation=False)
    for state,reference in report['references'].items():
        i=next(i for i,r in enumerate(report['frames']) if r['id']==reference['frame_id'])
        source=next(r for r in sequence['frames'] if r['id']==reference['frame_id'])
        coarse=params['posed_vertices'][i]
        refined=coarse+normals(coarse,params['faces'])*params['deviation_m'][:,None]
        triangles=params['faces'][fi[i]]
        lm_coarse=(coarse[triangles]*bc[i,:,:,None]).sum(1)
        lm_refined=(refined[triangles]*bc[i,:,:,None]).sum(1)
        np.testing.assert_allclose(lm_coarse,params['landmarks'][i],atol=2e-6)
        with np.load(a.sequence.parent/source['file'],allow_pickle=False) as d:
            observed=d['xyz'];valid=d['valid']
        def gap(lm):return float(np.linalg.norm(lm[62]-lm[66])*1000)
        output['references'][state]=dict(frame_id=reference['frame_id'],timestamp_s=reference['timestamp_s'],
            source_frame=reference['source_frame'],source=reference['source'],
            coordinates='X_right_Y_up_Z_backward; metres; exact reference camera',
            coarse_mesh=state+'_coarse_m.obj',refined_mesh=state+'_refined_m.obj',
            coarse_landmarks_m=lm_coarse,refined_landmarks_m=lm_refined,observed_landmarks_m=np.nan_to_num(observed),
            observed_valid=valid,inner_lip_pair=[62,66],
            coarse_inner_lip_gap_mm=gap(lm_coarse),refined_inner_lip_gap_mm=gap(lm_refined),
            observed_inner_lip_gap_mm=gap(observed) if valid[[62,66]].all() else None,
            metric_scope='fitted 3D inner-lip Euclidean gap; not a validated airway/clinical metric')
    dump(a.output,output)
    print(json.dumps({s:{k:v for k,v in row.items() if k.endswith('gap_mm') or k=='frame_id'} for s,row in output['references'].items()},indent=2))


if __name__=='__main__':main()
