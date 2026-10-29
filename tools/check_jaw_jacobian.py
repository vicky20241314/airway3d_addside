"""Finite-difference test of added contour residuals using the actual FLAME model."""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from core.fit_options import FitOptions
from core.integrated_fitting import make_flame
from core.joint_lm import JointProblem
from core.camera import Camera

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    prep=np.load(a.run/'prepared.npz');cap=np.load(a.run/'capture_pair.npz');saved=np.load(a.run/'flame_parameters.npz')
    options=FitOptions(jaw_contour_weight=.5,jaw_region_balance=True)
    model=make_flame(options,'cpu',dtype=torch.float64)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
    names=('closed','open');camera=Camera(*cap['intrinsics'])
    points=[tensor(prep['points_'+s][::max(1,len(prep['points_'+s])//300)]) for s in names]
    problem=JointProblem(model,camera,points,[tensor(prep['lmk3d_'+s]) for s in names],
        [tensor(prep['lmk2d_'+s]) for s in names],[torch.as_tensor(prep['mask_'+s],dtype=torch.bool) for s in names],
        [prep['parsing_'+s] for s in names],options)
    theta=problem.theta.clone();layout=problem.layout
    theta[layout.shape]=tensor(saved['shared_shape'])
    problem.rotations=tensor(saved['rotation'])
    for b in range(2):
        theta[layout.expr[b]]=tensor(saved['expression'][b]);theta[layout.pose[b]]=tensor(saved['pose'][b,list(options.pose_indices)])
        theta[layout.rot[b]]=0;theta[layout.trans[b]]=tensor(saved['translation'][b])
    problem.refresh(theta,2);r,j=problem.residual(theta,2)
    row=0;rows=[]
    for b in range(2):
        row+=4*len(points[b])+3*len(problem.semantic_ids[b])
        count=len(problem.jaw[b]['ids']);rows.extend(range(row,row+count));row+=count+len(problem.mouth_pairs[2][b])
    assert len(rows)>=12,'Too few gated jaw landmarks'
    columns=[0,8,50,layout.expr[0].start,layout.pose[0].start,layout.rot[0].start+1,layout.trans[0].start]
    errors={}
    for col in columns:
        eps=1e-5;plus=theta.clone();minus=theta.clone();plus[col]+=eps;minus[col]-=eps
        numeric=(problem.residual(plus,2,False)[0][rows]-problem.residual(minus,2,False)[0][rows])/(2*eps)
        error=float((numeric-j[rows,col]).abs().max());errors[str(col)]=error
        assert error<2e-5,(col,error)
    for data in problem.jaw:
        assert torch.isfinite(data['weights']).all() and abs(float(data['weights'].sum())-1)<1e-10
    from core.jaw_constraints import prepare_jaw
    for bad_parsing in (None,np.full_like(prep['parsing_closed'],17),np.full_like(prep['parsing_closed'],14)):
        gated=prepare_jaw(bad_parsing,prep['lmk2d_closed'],prep['lmk3d_closed'],prep['mask_closed'],camera,points[0].numpy())
        assert len(gated['ids'])==0 and np.isfinite(gated['weights']).all()
    accepted_ids=[d['ids'].tolist() for d in problem.jaw]
    angle=np.deg2rad(40.)
    problem.rotations=tensor([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]).repeat(2,1,1)
    problem.jaw_initialized=False;problem.refresh(theta,2)
    assert all(len(d['ids'])==0 and d['reason']=='oblique_pose_fallback' for d in problem.jaw)
    assert all(torch.allclose(d['weights'],torch.full_like(d['weights'],1/len(d['weights']))) for d in problem.jaw)
    assert torch.isfinite(problem.residual(theta,2,False)[0]).all()
    report=dict(jaw_rows=len(rows),max_absolute_derivative_errors=errors,
        accepted_ids=accepted_ids,finite_difference_passed=True,oblique_pose_fallback=True,missing_or_bad_parsing_fallback=True)
    out=ROOT/'diagnostics/jaw_candidate';out.mkdir(parents=True,exist_ok=True)
    (out/'jacobian.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
