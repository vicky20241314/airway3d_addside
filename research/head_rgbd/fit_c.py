"""C: alternating identity/expression FLAME fit with measured-ray surface ICP.

Independent research entry; retains B and the production pipeline unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from audit import dump,sha
from fit_b import choose,normals
from prepare import rigid
from geometry_c import sample_depth,ray_surface,landmark_validity,refine_shared,neck_surfaces,front_protected_vertices,safe_refinement
from core.integrated_fitting import make_flame
from core.fit_options import FitOptions


def landmark_embedding(flame, raw, jaw, R):
    n=len(raw)
    pose=torch.cat((torch.zeros(n,3,device=raw.device),jaw),1)
    full=torch.cat((flame.head_pose,pose,flame.eye_pose),1)
    fi,bc=flame._find_dynamic_lmk_idx_and_bcoords(raw,full,flame.dynamic_lmk_faces_idx,
        flame.dynamic_lmk_bary_coords,flame.neck_kin_chain,view_rotation=R)
    return torch.cat((fi,flame.lmk_faces_idx[None].expand(n,-1)),1),torch.cat((bc,flame.lmk_bary_coords[None].expand(n,-1,-1)),1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--front-only',action='store_true');p.add_argument('--points',type=int,default=1200)
    p.add_argument('--steps',type=int,default=240);p.add_argument('--keyframes-per-segment',type=int,default=3)
    p.add_argument('--expression-components',type=int,default=30)
    p.add_argument('--front-anchor',type=Path,help='conservative side-detail mode; preserve this real front-only identity/motion')
    a=p.parse_args()
    if a.front_anchor and a.front_only:p.error('anchor mode requires real oblique input')
    if a.steps<40 or a.points<100 or not 10<=a.expression_components<=50:p.error('invalid fitting budget/components')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(1234);rng=np.random.default_rng(1234)
    start=time.perf_counter();info=json.loads(a.sequence.read_text(encoding='utf-8'))
    if (info['schema'],info['depth_unit'],info['mesh_unit'],info['coordinate_convention']) != (
        'head_rgbd.sequence.v1','mm','m','X_right_Y_up_Z_backward'):raise ValueError('unsupported sequence contract')
    rows=[r.copy() for r in info['frames'] if r['status']=='candidate' and (not a.front_only or r['view']=='front')]
    data=[]
    for r in rows:
        with np.load(a.sequence.parent/r['file'],allow_pickle=False) as d:data.append({k:d[k] for k in d.files})
    n=len(rows);dev=torch.device(a.device);T=lambda x:torch.as_tensor(x,device=dev,dtype=torch.float32)
    anchor_report=None;anchor=None;anchored_ids=[]
    if a.front_anchor:
        anchor_report=json.loads((a.front_anchor/'report.json').read_text(encoding='utf-8'))
        if anchor_report['sequence_sha256']!=sha(a.sequence):raise ValueError('anchor sequence mismatch')
        if any(r['view']!='front' for r in anchor_report['frames']):raise ValueError('anchor must be real front-only control')
        with np.load(a.front_anchor/'parameters.npz',allow_pickle=False) as z:anchor={k:z[k] for k in z.files}
        anchor_index={r['id']:i for i,r in enumerate(anchor_report['frames'])}
        if anchor['expression'].shape[1]!=a.expression_components:raise ValueError('anchor expression dimensions mismatch')
    ne=a.expression_components
    flame=make_flame(FitOptions(shape_components=50,expression_components=ne),a.device,batch=n)
    if a.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _,template_lm=flame(shape_params=T(np.zeros((n,50))),expression_params=T(np.zeros((n,ne))),
            pose_params=T(np.zeros((n,6))),transl=T(np.zeros((n,3))))
    initial=[];translations=[];good=[]
    for i,(r,d) in enumerate(zip(rows,data)):
        valid=d['valid'].copy();valid[:27]=False;valid[48:]=False
        try:
            R,t=rigid(template_lm[i].cpu().numpy()[valid],d['xyz'][valid])
            motion=r.get('overlap',{});r['pose_initialization']='landmark_relocalization'
            if i and r['segment']==rows[i-1]['segment'] and 0<r['timestamp_s']-rows[i-1]['timestamp_s']<.2 and motion.get('status')=='tracked':
                rr=np.asarray(motion['previous_to_current_R'])@initial[i-1]
                tt=np.asarray(motion['previous_to_current_R'])@translations[i-1]+np.asarray(motion['previous_to_current_t'])
                src=template_lm[i].cpu().numpy()[valid];target=d['xyz'][valid]
                if np.median(np.linalg.norm(src@rr.T+tt-target,axis=1))<=np.median(np.linalg.norm(src@R.T+t-target,axis=1))+.002:
                    R,t=rr,tt;r['pose_initialization']='adjacent_rgbd_overlap'
            initial.append(R);translations.append(t);good.append(i)
        except (ValueError,np.linalg.LinAlgError):
            initial.append(np.eye(3));translations.append(np.array([0.,0.,-.5]));r.update(status='excluded',reason='initialization_failed')
    if not good:raise ValueError('no usable frame')
    initial=np.asarray(initial);selected=choose(rows,initial,a.keyframes_per_segment)
    if anchor is not None:
        for i,r in enumerate(rows):
            if r['view']=='front' and r['id'] in anchor_index:
                j=anchor_index[r['id']];initial[i]=anchor['R'][j];translations[i]=anchor['t'][j];anchored_ids.append(i)
    identity_ids=[i for i in selected if rows[i]['state']=='closed']
    if not identity_ids:raise ValueError('no closed identity keyframe')
    for i,r in enumerate(rows):
        r.update(initial_yaw_deg=float(Rotation.from_matrix(initial[i]).as_euler('xyz',degrees=True)[1]),
            selected=i in selected,identity_keyframe=i in identity_ids,
            role='identity_and_motion' if i in identity_ids else 'motion_expression_only',
            selection_reason='closed_quality_angle_time_keyframe' if i in identity_ids else (
                'open_state_excluded_from_shared_identity' if r['state']=='open' else 'redundant_within_segment'))
    faces=flame.faces_tensor.cpu().numpy();ft=torch.as_tensor(faces,device=dev)
    shape=torch.nn.Parameter(T(np.zeros((1,50))));expr=torch.nn.Parameter(T(np.zeros((n,ne))))
    jaw=torch.nn.Parameter(T(np.zeros((n,3))));rotation=torch.nn.Parameter(T(np.zeros((n,3))))
    trans=torch.nn.Parameter(T(np.asarray(translations)));base=T(initial)
    if anchor is not None:
        with torch.no_grad():
            shape.copy_(T(anchor['shape']))
            for i in anchored_ids:
                j=anchor_index[rows[i]['id']];expr[i]=T(anchor['expression'][j]);jaw[i]=T(anchor['jaw'][j])
        for i,r in enumerate(rows):
            if i in anchored_ids:r['role']='front_identity_anchor' if i in identity_ids else 'front_motion_anchor'
            elif i in identity_ids:r['role']='side_identity_detail_and_motion'
    opt=torch.optim.Adam([{'params':[shape,expr],'lr':.025},{'params':[jaw,rotation],'lr':.002},{'params':[trans],'lr':.0005}])
    target3=[T(np.nan_to_num(d['xyz'])) for d in data];target2=[T(np.nan_to_num(d['uv'])) for d in data]
    masks=[];masks2=[];points=[];identity_points=[]
    for i,d in enumerate(data):
        valid,bad=landmark_validity(d);masks.append(torch.as_tensor(valid,device=dev))
        valid2=np.isfinite(d['uv']).all(1);masks2.append(torch.as_tensor(valid2,device=dev))
        rows[i]['landmark_depth_invalid']=np.flatnonzero(~d['valid']).tolist()
        rows[i]['landmark_depth_outlier_rejected']=bad
        points.append(sample_depth(d,a.points,rng)[0]);identity_points.append(sample_depth(d,a.points,rng,True)[0])
    edges=[(i-1,i) for i in good if i-1 in good and rows[i]['segment']==rows[i-1]['segment'] and 0<rows[i]['timestamp_s']-rows[i-1]['timestamp_s']<.2]
    def forward():
        R=batch_rodrigues(rotation)@base
        v,l=flame(shape_params=shape.expand(n,-1),expression_params=expr,
            pose_params=torch.cat((torch.zeros(n,3,device=dev),jaw),1),transl=torch.zeros_like(trans),view_rotation=R)
        return v,torch.einsum('bij,bvj->bvi',R,v)+trans[:,None],torch.einsum('bij,bkj->bki',R,l)+trans[:,None],R
    def robust(x):
        return torch.nn.functional.smooth_l1_loss(x,torch.zeros_like(x),beta=.3)
    # Total budget equals B by default. Each stage has distinct active variables.
    cuts=np.rint(np.array([0,.167,.458,.625,.833,1])*a.steps).astype(int)
    phase_names=['motion_initial','identity_large','expression_reestimate','identity_surface','motion_final']
    correspond={};history=[];correspondence_log=[];landmark_visibility={}
    for step in range(a.steps):
        stage=min(np.searchsorted(cuts[1:],step,side='right'),4);identity_phase=stage in (1,3)
        phase=phase_names[stage];active=identity_ids if identity_phase else good
        if anchor is not None:
            phase=['side_motion_initial','side_rigid_alignment','side_expression','side_surface_alignment','side_motion_final'][stage]
            active=[i for i in active if i not in anchored_ids]
        if not active:raise ValueError('no unanchored real frames')
        opt.zero_grad();raw,v,l,R=forward();loss=v.sum()*0
        view_weights={i:3. if identity_phase and rows[i]['view']=='front' else 1. for i in active}
        denominator=sum(view_weights.values())
        # First phase uses landmarks. Later phases use true triangle surfaces.
        if stage>0 and (step==cuts[stage] or (step-cuts[stage])%10==0):
            correspond={}
            with torch.no_grad():
                for i in active:
                    corr,stats=ray_surface(v[i].cpu().numpy(),faces,identity_points[i] if identity_phase else points[i])
                    if anchor is not None:
                        lc,ls=ray_surface(v[i].cpu().numpy(),faces,l[i].cpu().numpy(),max_depth=.002)
                        visible_lm=np.zeros(68,bool)
                        if lc is not None:visible_lm[lc['ray']]=True
                        discrepancy=np.linalg.norm(l[i].cpu().numpy()-np.nan_to_num(data[i]['xyz']),axis=1)
                        reliable=visible_lm&(discrepancy<.012)
                        landmark_visibility[i]=torch.as_tensor(reliable,device=dev)
                        rows[i]['motion_landmark_visibility_or_outlier_rejected']=np.flatnonzero(~reliable).tolist()
                        stats['visible_depth_consistent_landmarks']=int(reliable[17:48].sum())
                    if corr is not None and stats['accepted']>=30:
                        correspond[i]=(ft[torch.as_tensor(corr['face'],device=dev)],T(corr['bary']),T(corr['normal']),T(corr['target']))
                    correspondence_log.append(dict(step=step,phase=phase,frame_id=rows[i]['id'],**stats))
            print(json.dumps(dict(step=step,phase=phase,correspondence_frames=len(correspond))),flush=True)
        for i in active:
            frame_weight=view_weights[i]/denominator
            valid=masks[i].clone();valid[:17]=False
            if i in landmark_visibility:valid &= landmark_visibility[i]
            if identity_phase or rows[i]['view']!='front':valid[48:]=False
            # Inner lip XYZ is a noisy cavity-edge measurement, downweight it.
            stable=valid.clone();stable[60:]=False
            landmark_weight=.2 if anchor is not None and stage>0 else 1.
            if stable.any():loss+=frame_weight*landmark_weight*robust((l[i][stable]-target3[i][stable])/.01)
            inner=valid.clone();inner[:60]=False
            if inner.any():loss+=frame_weight*.25*robust((l[i][inner]-target3[i][inner])/.01)
            fx,fy,cx,cy=data[i]['intrinsics'];z=(-l[i,:,2]).clamp(min=.05)
            uv=torch.stack((fx*l[i,:,0]/z+cx,-fy*l[i,:,1]/z+cy),1)
            valid2=masks2[i].clone();valid2[:17]=False
            if i in landmark_visibility:valid2 &= landmark_visibility[i]
            if identity_phase or rows[i]['view']!='front':valid2[48:]=False
            if valid2.any():loss+=frame_weight*landmark_weight*.15*robust((uv[valid2]-target2[i][valid2])/10)
            if not identity_phase and rows[i]['view']=='front':
                # Valid 2D detections remain useful when depth is missing.
                mouth=masks2[i].clone();mouth[:48]=False
                if mouth.any():loss+=frame_weight*.35*robust((uv[mouth]-target2[i][mouth])/10)
            if not identity_phase and rows[i]['state']=='closed':
                # State is known even when oblique 2D lip detections are poor.
                gaps=torch.linalg.norm(l[i,[61,62,63]]-l[i,[67,66,65]],dim=1)
                loss+=frame_weight*.15*(torch.relu(gaps-.002)/.003).square().mean()
            if i in correspond and stage>0:
                tri,bc,normal,target=correspond[i]
                surface=(v[i,tri]*bc[:,:,None]).sum(1)
                residual=((surface-target)*normal).sum(1)
                loss+=frame_weight*1.4*robust(residual/.01)
        # Preserve B's per-coefficient expression prior when adding components.
        loss+=.002*shape.square().mean()+.002*expr.square().sum()/(n*10)+.01*jaw.square().mean()
        if not identity_phase:
            for i,j in edges:
                dt=rows[j]['timestamp_s']-rows[i]['timestamp_s'];factor=(.033/max(dt,.01))**2
                loss+=factor*(.005*(expr[j]-expr[i]).square().sum()/10+.02*(jaw[j]-jaw[i]).square().mean()+.01*(rotation[j]-rotation[i]).square().mean())/max(len(edges),1)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
        loss.backward()
        if identity_phase:expr.grad=None;jaw.grad=None
        if not identity_phase or anchor is not None:shape.grad=None
        torch.nn.utils.clip_grad_norm_([shape,expr,jaw,rotation,trans],10.)
        # Adam otherwise moves inactive rows using momentum from the last stage.
        inactive=[i for i in good if i not in active]
        frozen_r=rotation.detach()[inactive].clone();frozen_t=trans.detach()[inactive].clone()
        frozen_e=expr.detach()[anchored_ids].clone();frozen_j=jaw.detach()[anchored_ids].clone()
        opt.step()
        with torch.no_grad():
            rotation[inactive]=frozen_r;trans[inactive]=frozen_t
            expr[anchored_ids]=frozen_e;jaw[anchored_ids]=frozen_j
        with torch.no_grad():jaw.clamp_(-.7,.7);shape.clamp_(-4,4);expr.clamp_(-4,4)
        if step%20==0 or step==a.steps-1:
            record=dict(step=step,phase=phase,objective=float(loss.detach()));history.append(record);print(record,flush=True)
    with torch.no_grad():
        raw,v,l,R=forward();fi,bc=landmark_embedding(flame,raw,jaw,R)
        raw=raw.cpu().numpy();v=v.cpu().numpy();l=l.cpu().numpy();R=R.cpu().numpy();fi=fi.cpu().numpy();bc=bc.cpu().numpy()
    fit_end=time.perf_counter()
    for i,r in enumerate(rows):r['fitted_yaw_deg']=float(Rotation.from_matrix(R[i]).as_euler('xyz',degrees=True)[1])
    # Define a generous canonical mouth exclusion around the closed front lips.
    ri=next(i for i in identity_ids if rows[i]['view']=='front')
    mouth=raw[ri][faces[fi[ri,48:]]];lmraw=(mouth*bc[ri,48:,:,None]).sum(1)
    rv=raw[ri];mouth_vertex=(rv[:,0]>lmraw[:,0].min()-.008)&(rv[:,0]<lmraw[:,0].max()+.008)&(rv[:,1]>lmraw[:,1].min()-.008)&(rv[:,1]<lmraw[:,1].max()+.008)&(rv[:,2]>lmraw[:,2].min()-.02)
    protected=None;fixed_deviation=None
    if anchor is not None:
        for i in anchored_ids:np.testing.assert_allclose(v[i],anchor['posed_vertices'][anchor_index[rows[i]['id']]],atol=1e-7)
        fixed_deviation=anchor['deviation_m']
        anchor_refined=np.array([vv+normals(vv,faces)*fixed_deviation[:,None] for vv in v])
        for i in anchored_ids:anchor_refined[i]=anchor['refined_vertices'][anchor_index[rows[i]['id']]]
        protected=front_protected_vertices(anchor_refined,faces,data,anchored_ids)
    deviation,support,attempted,refinement_log=refine_shared(v,faces,data,identity_ids,mouth_vertex,rng,rows,protected,fixed_deviation)
    before_refined=np.array([vv+normals(vv,faces)*(fixed_deviation if fixed_deviation is not None else np.zeros(len(vv)))[:,None] for vv in v])
    locked_effective=None
    if anchor is not None:
        locked_effective={i:anchor['effective_deviation_m'][anchor_index[rows[i]['id']]] for i in anchored_ids}
    refined,effective_deviation,safety_log=safe_refinement(v,faces,deviation,protected,locked_effective)
    before_refined,_,_=safe_refinement(v,faces,fixed_deviation if fixed_deviation is not None else np.zeros(len(deviation)))
    # Evidence is counted separately from refinement participation, including
    # open-state observations without allowing them to change identity.
    observed=np.zeros((n,len(deviation)),bool);seen=np.zeros(len(deviation),bool)
    for i in good:
        pts,_,_=sample_depth(data[i],12000,rng,stride=4)
        corr,stats=ray_surface(refined[i],faces,pts,.008,.005)
        if corr is None:continue
        ids=faces[corr['face']];observed[i,np.unique(ids[corr['bary']>.05])]=True
        seen[np.unique(ids)]=True
    observed_count=observed.sum(0)
    # 0=unobserved completion, 1=one-frame/model-prior support, 2=repeat observed.
    labels=np.where(observed_count>=2,2,np.where(seen,1,0)).astype(np.uint8)
    confidence=np.minimum(observed_count/3,1.)
    colors=np.array([[110,110,120,255],[230,175,55,255],[55,175,215,255]],np.uint8)[labels]
    references={};reference_geometry={}
    for state in ('closed','open'):
        candidates=[i for i in selected if rows[i]['state']==state and rows[i]['view']=='front']
        if not candidates:continue
        i=max(candidates,key=lambda i:rows[i]['sharpness']);triangles=faces[fi[i]]
        lc=(v[i,triangles]*bc[i,:,:,None]).sum(1);lr=(refined[i,triangles]*bc[i,:,:,None]).sum(1)
        np.testing.assert_allclose(lc,l[i],atol=2e-6)
        for stage,vv in [('coarse',v[i]),('refined',refined[i])]:
            if not np.isfinite(vv).all():raise ValueError('nonfinite mesh')
            trimesh.Trimesh(vv,faces,process=False).export(a.output/f'{state}_{stage}_m.obj')
        trimesh.Trimesh(refined[i],faces,vertex_colors=colors,process=False).export(a.output/f'{state}_support_m.ply')
        references[state]=dict(frame_id=rows[i]['id'],timestamp_s=rows[i]['timestamp_s'],source=rows[i]['source'],source_frame=rows[i]['source_frame'],
            coordinate_system='reference camera X_right_Y_up_Z_backward; metres',mesh=f'{state}_refined_m.obj',
            expression=expr[i].detach().cpu().numpy(),jaw_axis_angle_rad=jaw[i].detach().cpu().numpy(),landmarks_3d_m=lr,
            landmark_stage='refined',landmark_indices='FLAME dynamic contour + static 68; zero based')
        observed_lm=[data[i]['xyz'][j].tolist() if data[i]['valid'][j] else None for j in range(68)]
        gap=lambda lm:float(np.linalg.norm(lm[62]-lm[66])*1000)
        reference_geometry[state]=dict(frame_id=rows[i]['id'],coordinates=references[state]['coordinate_system'],
            coarse_mesh=f'{state}_coarse_m.obj',refined_mesh=f'{state}_refined_m.obj',
            coarse_landmarks_m=lc,refined_landmarks_m=lr,observed_landmarks_m=observed_lm,
            observed_valid=data[i]['valid'],used_depth_valid=masks[i].cpu().numpy(),
            landmark_triangle_indices=fi[i],landmark_barycentric=bc[i],
            coarse_inner_lip_gap_mm=gap(lc),refined_inner_lip_gap_mm=gap(lr),
            observed_inner_lip_gap_mm=gap(data[i]['xyz']) if data[i]['valid'][[62,66]].all() else None,
            closure_soft_interval_mm=[0,2],observation_is_manual_truth=False)
    neck=neck_surfaces(rows,data,a.output)
    np.savez_compressed(a.output/'parameters.npz',shape=shape.detach().cpu().numpy(),expression=expr.detach().cpu().numpy(),jaw=jaw.detach().cpu().numpy(),
        R=R,t=trans.detach().cpu().numpy(),raw_vertices=raw,posed_vertices=v,refined_vertices=refined,faces=faces,landmarks=l,
        deviation_m=deviation,support_frames=support.sum(0),selected=np.asarray(selected),identity_selected=np.asarray(identity_ids),
        landmark_faces=fi,landmark_barycentric=bc,mouth_identity_excluded=mouth_vertex,
        before_side_detail_vertices=before_refined,front_protected_vertices=protected if protected is not None else np.zeros(len(deviation),bool),
        effective_deviation_m=effective_deviation)
    np.savez_compressed(a.output/'support.npz',observed_frame_mask=observed,observed_frames=observed_count,refinement_frame_mask=support,
        refinement_frame_indices=np.asarray(identity_ids),refinement_frames=support.sum(0),confidence=confidence,region_label=labels,
        label_names=np.array(['unobserved_completion','insufficient_observation_model_prior','repeat_observation_supported']),deviation_m=deviation)
    dump(a.output/'reference_geometry.json',dict(schema='head_rgbd.reference_geometry.v2',mesh_unit='m',references=reference_geometry,
        metric_scope='same-frame fitted geometry diagnosis; not independent or clinical accuracy'))
    if a.device=='cuda':torch.cuda.synchronize()
    report=dict(algorithm='C: alternating FLAME + exact first-hit triangle point-to-plane + bounded closed-only refinement',
        full_paper_reproduction=False,status='pipeline_complete_effect_unverified',device=a.device,steps=a.steps,points_per_frame=a.points,
        elapsed_s=time.perf_counter()-start,fitting_s=fit_end-start,expression_components=ne,
        identity_front_observation_weight=3.,identity_other_view_weight=1.,expression_prior_per_coefficient=.002/(n*10),
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else 0,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if a.device=='cuda' else 0,
        timing_scope='wall time including CPU ray correspondences, refinement, evidence queries and exports; CUDA synchronized; no prepare/environment time',
        memory_scope='this process torch allocator peak; not total device VRAM',
        frames=rows,references=references,neck=neck,history=history,temporal_edges=[[rows[i]['id'],rows[j]['id']] for i,j in edges],
        correspondence_log=correspondence_log,refinement_log=refinement_log,safety_log=safety_log,
        observed_refinement_vertices=int((support.sum(0)>=2).sum()),total_vertices=len(deviation),support_label_counts=np.bincount(labels,minlength=3),
        sequence_sha256=sha(a.sequence),model_sha256=sha(ROOT/'assets/flame2023.pkl'),source_hashes={q.name:sha(q) for q in Path(__file__).parent.glob('*.py')},
        acceptance=dict(independent_surface_error='unverified',human_contour_error='unverified',independent_coverage='unverified',real_side_accuracy='unverified'),
        limitations=['four disconnected short bursts, not a continuous head turn','no open side observations or 90-degree measured profiles',
            'local normal displacement, limited to 3mm; mouth identity refinement excluded',
            'neck open patches have independent motion; no global neck identity model','training camera poses and 2D detections are not independent truth',
            'head back, occluded ears and unseen profile remain model completion','offline batch optimisation; long-term tracking and live camera capture unverified'])
    if anchor is not None:
        report['algorithm']='C guarded: real frontal identity anchor + real oblique motion + bounded new-visibility identity detail'
        report['front_anchor']=dict(path=str(a.front_anchor),report_sha256=sha(a.front_anchor/'report.json'),
            parameters_sha256=sha(a.front_anchor/'parameters.npz'),protected_vertices=int(protected.sum()),
            changed_side_detail_vertices=int((np.abs(deviation-fixed_deviation)>1e-7).sum()),
            maximum_side_detail_change_mm=float(np.abs(deviation-fixed_deviation).max()*1000),
            anchor_elapsed_s=anchor_report['elapsed_s'],cold_pipeline_elapsed_s=anchor_report['elapsed_s']+report['elapsed_s'],
            policy='front base identity, pose, expression and covered triangle vertices fixed; only genuinely new side-supported vertices may refine',
            limitation='conservative side detail; overlapping frontal face does not receive global side shape updates')
        report['limitations'].append(report['front_anchor']['limitation'])
        report['identity_coarse_fit']='inherited shared identity from front-only anchor; no new global shape update'
        report['identity_detail_frames']=[rows[i]['id'] for i in identity_ids if i not in anchored_ids]
        report['side_motion_landmark_policy']='first-hit self-visibility within 2mm + 12mm 3D outlier gate; landmark weight 0.2 after surface ICP starts'
    dump(a.output/'report.json',report)
    print(json.dumps(dict(output=str(a.output),elapsed_s=report['elapsed_s'],cuda_peak_allocated_mib=report['cuda_peak_allocated_mib'],
        identity_frames=[rows[i]['id'] for i in identity_ids],mouth_gaps_mm={s:r['refined_inner_lip_gap_mm'] for s,r in reference_geometry.items()})),flush=True)


if __name__=='__main__':main()
