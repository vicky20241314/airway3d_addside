"""FaceCap-inspired offline FLAME experiment, NOT a full FaceCap reproduction."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import trimesh
from smplx.lbs import batch_rodrigues

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from audit import dump,sha
from prepare import rigid,backproject
from core.integrated_fitting import make_flame
from core.fit_options import FitOptions


def normals(v,f):
    tri=v[f]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    out=np.zeros_like(v)
    for j in range(3): np.add.at(out,f[:,j],n)
    return out/np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)


def visible(v,f,depth,intr,face_mask,tolerance=.015):
    """Backface, image validity, measured occlusion and coarse self z-buffer."""
    z=-v[:,2]; fx,fy,cx,cy=intr
    uv=np.column_stack((fx*v[:,0]/np.maximum(z,1e-6)+cx,-fy*v[:,1]/np.maximum(z,1e-6)+cy))
    pix=np.rint(uv).astype(int);h,w=depth.shape
    keep=(z>.05)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    n=normals(v,f);keep&=(n*(-v)).sum(1)>0
    ids=np.flatnonzero(keep)
    measured=np.zeros(len(v));measured[ids]=depth[pix[ids,1],pix[ids,0]]*.001
    semantic=np.zeros(len(v),bool);semantic[ids]=face_mask[pix[ids,1],pix[ids,0]]
    keep&=semantic&(measured>.1)&(np.abs(measured-z)<tolerance)
    # Vertex z buffer is conservative at low image resolution; not a full rasterizer.
    cell=(np.clip(pix[:,1]//4,0,(h-1)//4)*((w+3)//4)+np.clip(pix[:,0]//4,0,(w-1)//4))
    buf=np.full(((h+3)//4)*((w+3)//4),np.inf)
    np.minimum.at(buf,cell[ids],z[ids])
    keep&=z<=buf[cell]+.008
    return keep,uv,n,measured


def choose(rows,initial_r,max_per_segment):
    """Quality ranks within each segment, then yaw novelty and temporal spacing."""
    selected=[]
    for seg in dict.fromkeys(r['segment'] for r in rows):
        ids=[i for i,r in enumerate(rows) if r['segment']==seg and r['status']!='excluded']
        if not ids:continue
        ranked=sorted(ids,key=lambda i:rows[i]['sharpness'],reverse=True)
        chosen=[ranked[0]]
        while len(chosen)<min(max_per_segment,len(ids)):
            candidates=[i for i in ids if i not in chosen]
            def score(i):
                yaw=Rotation.from_matrix(initial_r[i]).as_euler('xyz',degrees=True)[1]
                novelty=min(abs(yaw-Rotation.from_matrix(initial_r[j]).as_euler('xyz',degrees=True)[1]) for j in chosen)
                spacing=min(abs(rows[i]['timestamp_s']-rows[j]['timestamp_s']) for j in chosen)
                return novelty+spacing*10+.01*rows[i]['sharpness']
            chosen.append(max(candidates,key=score))
        selected.extend(chosen)
    return sorted(selected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--keyframes-per-segment',type=int,default=3)
    p.add_argument('--front-only',action='store_true')
    p.add_argument('--points',type=int,default=1200)
    a=p.parse_args()
    if a.steps<2 or a.points<100 or a.keyframes_per_segment<1: p.error('invalid experiment budget')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(1234);rng=np.random.default_rng(1234)
    start=time.perf_counter();info=json.loads(a.sequence.read_text(encoding='utf-8'))
    if info['schema']!='head_rgbd.sequence.v1' or info['depth_unit']!='mm' or info['mesh_unit']!='m' or info['coordinate_convention']!='X_right_Y_up_Z_backward':
        raise ValueError('Unsupported sequence units/coordinates')
    rows=[r.copy() for r in info['frames'] if r['status']=='candidate' and (not a.front_only or r['view']=='front')]
    data=[]
    for r in rows:
        with np.load(a.sequence.parent/r['file'],allow_pickle=False) as d:data.append({k:d[k] for k in d.files})
    options=FitOptions(shape_components=50,expression_components=10)
    flame=make_flame(options,a.device,batch=len(rows))
    dev=torch.device(a.device);tensor=lambda x:torch.as_tensor(x,device=dev,dtype=torch.float32)
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    with torch.no_grad():
        _,lm=flame(shape_params=torch.zeros(len(rows),50,device=dev),expression_params=torch.zeros(len(rows),10,device=dev),
                   pose_params=torch.zeros(len(rows),6,device=dev),transl=torch.zeros(len(rows),3,device=dev))
    initial=[];translations=[];good=[]
    for i,(r,d) in enumerate(zip(rows,data)):
        valid=d['valid'].copy();valid[:27]=False;valid[48:]=False
        try:
            R,t=rigid(lm[i].cpu().numpy()[valid],d['xyz'][valid])
            overlap=r.get('overlap',{})
            if i>0 and r['segment']==rows[i-1]['segment'] and overlap.get('status')=='tracked':
                rel=np.array(overlap['previous_to_current_R']);shift=np.array(overlap['previous_to_current_t'])
                tracked_r=rel@initial[i-1];tracked_t=rel@translations[i-1]+shift
                src=lm[i].cpu().numpy()[valid];target=d['xyz'][valid]
                direct_error=np.median(np.linalg.norm(src@R.T+t-target,axis=1))
                track_error=np.median(np.linalg.norm(src@tracked_r.T+tracked_t-target,axis=1))
                if track_error<=direct_error+.002:
                    R,t=tracked_r,tracked_t;r['pose_initialization']='adjacent_rgbd_overlap'
                else:r['pose_initialization']='landmark_relocalization'
            else:r['pose_initialization']='landmark_segment_initialization'
            initial.append(R);translations.append(t);good.append(i)
        except (ValueError,np.linalg.LinAlgError):
            r.update(status='excluded',reason='initialization_failed')
            initial.append(np.eye(3));translations.append(np.array([0,0,-.5]))
    if not good:raise ValueError('No frame can be initialized')
    initial=np.array(initial);selected=choose(rows,initial,a.keyframes_per_segment)
    # Quality gate once per frame, preserve other views if an individual frame is poor.
    for i,r in enumerate(rows):
        r['initial_yaw_deg']=float(Rotation.from_matrix(initial[i]).as_euler('xyz',degrees=True)[1])
        r['selected']=i in selected;r['selection_reason']='quality_angle_time_keyframe' if i in selected else 'redundant_within_segment'
    n=len(rows);faces=flame.faces_tensor.cpu().numpy()
    shape=torch.nn.Parameter(torch.zeros(1,50,device=dev))
    expr=torch.nn.Parameter(torch.zeros(n,10,device=dev))
    jaw=torch.nn.Parameter(torch.zeros(n,3,device=dev))
    rotation=torch.nn.Parameter(torch.zeros(n,3,device=dev))
    trans=torch.nn.Parameter(tensor(np.array(translations)))
    base=tensor(initial)
    opt=torch.optim.Adam([{'params':[shape,expr],'lr':.025},{'params':[jaw,rotation],'lr':.002}, {'params':[trans],'lr':.0005}])
    points=[]
    for d in data:
        ys,xs=np.where(d['face_mask']&(d['depth']>100)&(d['depth']<1500))
        take=rng.choice(len(xs),min(a.points,len(xs)),replace=False)
        pts,_=backproject(np.column_stack((xs[take],ys[take])),d['depth'],d['intrinsics']);points.append(pts)
    edges=[(i-1,i) for i in good if i-1 in good and rows[i]['segment']==rows[i-1]['segment'] and 0<rows[i]['timestamp_s']-rows[i-1]['timestamp_s']<.2]
    target3=[tensor(np.nan_to_num(d['xyz'])) for d in data]
    target2=[tensor(np.nan_to_num(d['uv'])) for d in data]
    masks=[torch.as_tensor(d['valid'],device=dev) for d in data]
    correspond={};history=[]
    def forward():
        R=batch_rodrigues(rotation)@base
        pose=torch.cat((torch.zeros(n,3,device=dev),jaw),1)
        identity=torch.stack([shape[0] if i in selected else shape[0].detach() for i in range(n)])
        v,l=flame(shape_params=identity,expression_params=expr,pose_params=pose,
            transl=torch.zeros_like(trans),view_rotation=R)
        return v,torch.einsum('bij,bvj->bvi',R,v)+trans[:,None],torch.einsum('bij,bkj->bki',R,l)+trans[:,None],R
    for step in range(a.steps):
        opt.zero_grad();raw,v,l,R=forward()
        coarse=step<a.steps//3
        loss=v.sum()*0
        if not coarse and step%5==0:
            with torch.no_grad():
                for i in selected:
                    vv=v[i].cpu().numpy()
                    keep,_,_,_=visible(vv,faces,data[i]['depth'],data[i]['intrinsics'],data[i]['face_mask'],.035)
                    ids=np.flatnonzero(keep)
                    if len(ids)<30:correspond[i]=None;continue
                    distance,index=cKDTree(vv[ids]).query(points[i])
                    valid=distance<.025
                    if valid.sum()<30:correspond[i]=None;continue
                    correspond[i]=(torch.as_tensor(ids[index[valid]],device=dev),tensor(points[i][valid]))
        for i in good:
            valid=masks[i].clone()
            # The 2D contour is view-dependent; stable inner landmarks drive rigid fitting.
            valid[:17]=False
            if rows[i]['view']!='front':valid[48:]=False
            delta=l[i][valid]-target3[i][valid]
            loss+=torch.nn.functional.smooth_l1_loss(delta/.01,torch.zeros_like(delta),beta=.3)/len(good)
            fx,fy,cx,cy=data[i]['intrinsics'];z=(-l[i,:,2]).clamp(min=.05)
            uv=torch.stack((fx*l[i,:,0]/z+cx,-fy*l[i,:,1]/z+cy),1)
            loss+=.1*torch.nn.functional.smooth_l1_loss((uv[valid]-target2[i][valid])/10,torch.zeros_like(uv[valid]),beta=.3)/len(good)
            if not coarse and correspond.get(i) is not None:
                ids,target=correspond[i];diff=(v[i,ids]-target)/.01
                loss+=.7*torch.nn.functional.smooth_l1_loss(diff,torch.zeros_like(diff),beta=.3)/len(selected)
        loss+=.002*shape.square().mean()+.002*expr[selected].square().mean()+.01*jaw[selected].square().mean()
        for i,j in edges:
            dt=rows[j]['timestamp_s']-rows[i]['timestamp_s'];factor=(.033/max(dt,.01))**2
            loss+=factor*(.005*(expr[j]-expr[i]).square().mean()+.02*(jaw[j]-jaw[i]).square().mean()+.01*(rotation[j]-rotation[i]).square().mean())/max(len(edges),1)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite objective; no invalid mesh exported')
        loss.backward()
        if coarse:shape.grad.zero_();expr.grad.zero_()
        torch.nn.utils.clip_grad_norm_([shape,expr,jaw,rotation,trans],10.)
        opt.step()
        with torch.no_grad():jaw.clamp_(-.7,.7);shape.clamp_(-4,4);expr.clamp_(-4,4)
        if step%20==0 or step==a.steps-1:
            record=dict(step=step,phase='rigid_jaw' if coarse else 'shared_identity',objective=float(loss.detach()))
            history.append(record);print(record,flush=True)
    with torch.no_grad():
        raw,v,l,R=forward();raw=raw.cpu().numpy();v=v.cpu().numpy();R=R.cpu().numpy();l=l.cpu().numpy()
    fit_end=time.perf_counter()
    # Shared normal displacement, after fitting each frame's pose/expression.
    # Each scalar observation is in its current animated normal direction, never raw-depth averaging.
    samples=np.full((len(selected),raw.shape[1]),np.nan)
    for k,i in enumerate(selected):
        keep,uv,norm,measured=visible(v[i],faces,data[i]['depth'],data[i]['intrinsics'],data[i]['face_mask'],.008)
        target,_=backproject(uv,data[i]['depth'],data[i]['intrinsics'])
        offset=((target-v[i])*norm).sum(1)
        keep&=np.abs(offset)<.005
        samples[k,keep]=offset[keep]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);deviation=np.nanmedian(samples,axis=0)
    support=np.isfinite(samples).sum(0)
    supported=(support>=2)&np.isfinite(deviation)
    deviation=np.where(supported,np.clip(np.nan_to_num(deviation),-.003,.003),0)
    # Leave unobserved template unchanged, label it explicitly as prior completion.
    references={}
    for state in ('closed','open'):
        candidates=[i for i in selected if rows[i]['state']==state and rows[i]['view']=='front']
        if not candidates:continue
        idx=max(candidates,key=lambda i:rows[i]['sharpness'])
        refined=v[idx]+normals(v[idx],faces)*deviation[:,None]
        trimesh.Trimesh(v[idx],faces,process=False).export(a.output/(state+'_coarse_m.obj'))
        trimesh.Trimesh(refined,faces,process=False).export(a.output/(state+'_refined_m.obj'))
        references[state]=dict(frame_id=rows[idx]['id'],timestamp_s=rows[idx]['timestamp_s'],
            source=rows[idx]['source'],source_frame=rows[idx]['source_frame'],
            coordinate_system='reference frame camera: X right, Y up, Z backward; metres',
            mesh=state+'_refined_m.obj',landmark_indices='FLAME dynamic contour + static 68',
            landmarks_3d_m=l[idx].tolist(),expression=expr[idx].detach().cpu().tolist(),jaw_axis_angle_rad=jaw[idx].detach().cpu().tolist())
    # Neck KLT/RGB-D motion is estimated separately in preparation; never feeds face shape.
    neck_reports=[]
    for i in selected:
        ys,xs=np.where(data[i]['neck_mask']&(data[i]['depth']>100)&(data[i]['depth']<1500))
        if len(xs)<30:neck_reports.append(dict(frame_id=rows[i]['id'],status='unavailable'));continue
        take=rng.choice(len(xs),min(1000,len(xs)),replace=False)
        cloud,_=backproject(np.column_stack((xs[take],ys[take])),data[i]['depth'],data[i]['intrinsics'])
        trimesh.points.PointCloud(cloud).export(a.output/(rows[i]['id']+'_neck_observed_m.ply'))
        motion=rows[i].get('neck_overlap',dict(status='unavailable',reason='preparation_without_neck_tracking'))
        head=rows[i].get('overlap',{})
        disagreement=None
        if motion.get('status')=='tracked' and head.get('status')=='tracked':
            relative=np.array(motion['previous_to_current_R'])@np.array(head['previous_to_current_R']).T
            disagreement=float(np.degrees(Rotation.from_matrix(relative).magnitude()))
        neck_reports.append(dict(frame_id=rows[i]['id'],status='observations_exported_separately',
            movement_model='separate neck RGB-D feature tracking; no face feedback',motion=motion,
            head_neck_rotation_difference_deg=disagreement,points=len(cloud),face_identity_constraint=False))
    np.savez_compressed(a.output/'parameters.npz',shape=shape.detach().cpu().numpy(),
        expression=expr.detach().cpu().numpy(),jaw=jaw.detach().cpu().numpy(),
        R=R,t=trans.detach().cpu().numpy(),raw_vertices=raw,posed_vertices=v,faces=faces,
        landmarks=l,deviation_m=deviation,support_frames=support,selected=np.array(selected))
    if a.device=='cuda':torch.cuda.synchronize()
    report=dict(algorithm='B: FaceCap-inspired shared FLAME + per-frame pose/expression/jaw + normal median',
        full_paper_reproduction=False,status='pipeline_complete_effect_unverified',device=a.device,
        elapsed_s=time.perf_counter()-start,fitting_s=fit_end-start,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else 0,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if a.device=='cuda' else 0,
        steps=a.steps,points_per_frame=a.points,frames=rows,references=references,neck=neck_reports,
        sequence_sha256=sha(a.sequence),source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        model_sha256=sha(ROOT/'assets/flame2023.pkl'),history=history,
        observed_refinement_vertices=int(supported.sum()),total_vertices=len(supported),
        validation=dict(real_side_improvement='unverified',independent_extrinsics=False,human_silhouette=False),
        limitations=['offline experiment; not real time', 'vertex-nearest coarse correspondences, conservative vertex z-buffer',
            'normal deviation limited to 3mm, no UV upsampling', 'neck tracking may degrade if texture/depth is insufficient; no neck mesh identity fit',
            'no clinically validated metrics', 'unobserved head geometry is FLAME prior completion',
            'all valid frames fit motion; only keyframes update shared identity'])
    dump(a.output/'report.json',report)
    print(json.dumps(dict(output=str(a.output),elapsed_s=report['elapsed_s'],selected=[rows[i]['id'] for i in selected],references=list(references),cuda_peak_allocated_mib=report['cuda_peak_allocated_mib']),indent=2),flush=True)


if __name__=='__main__': main()
