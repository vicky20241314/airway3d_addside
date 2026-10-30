"""Shared-state multi-camera FLAME LM using the existing progressive solver."""
from dataclasses import dataclass
from pathlib import Path
import math
import time
import numpy as np
import torch
import trimesh
from smplx.lbs import batch_rodrigues
from core.fit_options import FitOptions
from core.joint_lm import JointProblem, solve_stage, solve_progressive, project, vertex_normals
from core.initial_surface import closest_surface, matched_points
from core.multiview_geometry import RigidCamera, visible_faces, visible_landmarks


@dataclass
class MultiLayout:
    shape_n: int
    expr_n: int
    pose_indices: tuple
    states: tuple

    def __post_init__(self):
        self.frames=len(self.states);self.shape=slice(0,self.shape_n)
        self.expr=[];self.pose=[];self.rot=[];self.trans=[]
        self.state_slots={};cursor=self.shape_n
        for state in dict.fromkeys(self.states):
            self.state_slots[state]=(slice(cursor,cursor+self.expr_n),
                slice(cursor+self.expr_n,cursor+self.expr_n+len(self.pose_indices)))
            cursor+=self.expr_n+len(self.pose_indices)
        for state in self.states:
            e,p=self.state_slots[state];self.expr.append(e);self.pose.append(p)
            self.rot.append(slice(cursor,cursor+3));self.trans.append(slice(cursor+3,cursor+6));cursor+=6
        self.size=cursor

    def active(self,stage):
        if stage==2:
            omitted={i for b in getattr(self,'fixed_views',()) for s in (self.rot[b],self.trans[b]) for i in range(s.start,s.stop)}
            return [i for i in range(self.size) if i not in omitted]
        # First camera-only pass: shape, expression AND joint pose are fixed.
        return [i for a,b in zip(self.rot,self.trans) for s in (a,b) for i in range(s.start,s.stop)]


class MultiProblem(JointProblem):
    def __init__(self,flame,observations,initial,transforms,options,raw_profiles=None):
        self.flame=flame;self.options=options;self.observations=observations
        self.faces=flame.faces_tensor;self.device=flame.v_template.device;self.dtype=flame.v_template.dtype
        self.layout=MultiLayout(options.shape_components,options.expression_components,options.pose_indices,
                                tuple(o.state for o in observations))
        self.theta=flame.v_template.new_zeros(self.layout.size)
        self.rotations=torch.stack([torch.as_tensor(T.R,device=self.device,dtype=self.dtype) for T in transforms])
        tensor=lambda a:torch.as_tensor(a,device=self.device,dtype=self.dtype)
        self.theta[self.layout.shape]=tensor(initial['closed']['shape'])
        for b,o in enumerate(observations):
            self.theta[self.layout.expr[b]]=tensor(initial[o.state]['expression'])
            self.theta[self.layout.pose[b]]=tensor(initial[o.state]['pose'])[list(options.pose_indices)]
            self.theta[self.layout.trans[b]]=tensor(transforms[b].t)
        self.points=[];self.lmk3d=[];self.lmk2d=[];self.masks=[]
        rng=np.random.default_rng(options.seed)
        for o in observations:
            p=np.asarray(o.points);p=p[np.isfinite(p).all(1)&(p[:,2]<0)]
            if len(p)>options.max_points:p=p[rng.choice(len(p),options.max_points,replace=False)]
            if len(p)<20:raise ValueError(f'{o.view_id}: insufficient face/neck points')
            self.points.append(tensor(p));self.lmk3d.append(tensor(o.xyz));self.lmk2d.append(tensor(o.uv))
            self.masks.append(torch.as_tensor(o.valid,device=self.device,dtype=torch.bool))
        self.quality=np.array([o.quality.get('weight',1.) for o in observations],float)
        self.quality/=self.quality.sum()
        self.context=[];self.embedding=None;self.jaw=[];self.view_diagnostics=[]
        self.front_indices={s:next(i for i,o in enumerate(observations) if o.state==s and o.view=='front') for s in self.layout.state_slots}
        from core.multiview_constraints import relative_transforms
        self.initial_relative=relative_transforms(self.rotations.cpu().numpy(),
            np.stack([T.t for T in transforms]),self.front_indices,observations)
        self.fixed_relative={}
        self.profiles=[];self.silhouette_topology=None
        if options.side_contour_weight>0:
            from core.side_silhouette import observed_profile,silhouette_topology
            self.profiles=[raw_profiles[o.view_id] if raw_profiles and o.view_id in raw_profiles else
                observed_profile(o) if o.view!='front' else None for o in observations]
            self.silhouette_topology=silhouette_topology(self.faces.cpu().numpy())
        from core.jaw_constraints import prepare_jaw
        for b,o in enumerate(observations):
            yaw=math.degrees(math.atan2(transforms[b].R[0,2],transforms[b].R[2,2]))
            jaw=prepare_jaw(o.parsing,o.uv,o.xyz,o.valid,o.camera,self.points[b].cpu().numpy())
            if o.view!='front' or abs(yaw)>options.jaw_max_yaw_deg:
                jaw['ids']=np.empty(0,int);jaw['normals']=np.empty((0,2));jaw['weights']=np.full(len(self.points[b]),1/len(self.points[b]))
                jaw['reason']='side_depth_only' if o.view!='front' else 'oblique_pose_fallback'
            jaw['initial_yaw_deg']=yaw
            self.jaw.append(jaw)

    def parameters(self,theta):
        shape,expr,pose,rot,trans=super().parameters(theta)
        if self.fixed_relative:
            rot=rot.clone();trans=trans.clone()
            for b,T in self.fixed_relative.items():
                i=self.front_indices[self.observations[b].state]
                r=theta.new_tensor(T.R);t=theta.new_tensor(T.t)
                rot[b]=r@rot[i];trans[b]=r@trans[i]+t
        return shape,expr,pose,rot,trans

    def begin_shape_fit(self,theta):
        if self.options.pose_strategy=='fixed':
            # Freeze the trustworthy registration, not drift accumulated by
            # independently aligning a model whose identity is still frontal.
            self.fixed_relative=dict(self.initial_relative)
            self.layout.fixed_views=tuple(self.fixed_relative)

    def pose_prior(self,theta):
        from core.multiview_constraints import pose_prior_values
        *_,rot,trans=self.parameters(theta)
        return pose_prior_values(rot.cpu().numpy(),trans.cpu().numpy(),self.front_indices,
            self.observations,self.initial_relative,self.options)

    def iteration_diagnostics(self,theta,stage):
        from core.multiview_constraints import pose_drift
        *_,rot,trans=self.parameters(theta)
        return dict(relative_pose_from_registration=pose_drift(rot.cpu().numpy(),trans.cpu().numpy(),
            self.front_indices,self.observations,self.initial_relative),views=self.view_diagnostics,
            residual_terms=getattr(self,'residual_terms',[]),shared_regularizers_once=True)

    def _embedding(self,theta,vertices):
        pose=self.parameters(theta)[2];rot=self.parameters(theta)[3]
        full=torch.cat((self.flame.head_pose,pose,self.flame.eye_pose),1)
        fi,bc=self.flame._find_dynamic_lmk_idx_and_bcoords(vertices,full,
            self.flame.dynamic_lmk_faces_idx,self.flame.dynamic_lmk_bary_coords,
            self.flame.neck_kin_chain,dtype=self.dtype,view_rotation=rot)
        fi=torch.cat((fi,self.flame.lmk_faces_idx[None].expand(self.layout.frames,-1)),1)
        bc=torch.cat((bc,self.flame.lmk_bary_coords[None].expand(self.layout.frames,-1,-1)),1)
        bc=bc/bc.sum(-1,keepdim=True)
        return self.faces[fi],bc

    @torch.no_grad()
    def forward(self,theta):
        shape,expr,pose,rot,trans=self.parameters(theta)
        raw,_=self.flame(shape_params=shape,expression_params=expr,pose_params=pose,transl=torch.zeros_like(trans),view_rotation=rot)
        vertices=torch.einsum('bij,bvj->bvi',rot,raw)+trans[:,None]
        tri,bary=self.embedding if self.embedding is not None else self._embedding(theta,vertices)
        lm=torch.stack([(vertices[b][tri[b]]*bary[b,:,:,None]).sum(1) for b in range(self.layout.frames)])
        return vertices,lm

    def landmark_embedding(self,theta,vertices):
        return self.embedding if self.embedding is not None else self._embedding(theta,vertices)

    @torch.no_grad()
    def vertex_jacobians(self,theta,stage):
        if stage==2:
            fixed=getattr(self.layout,'fixed_views',())
            if not fixed:return super().vertex_jacobians(theta,stage)
            self.layout.fixed_views=()
            try:full=super().vertex_jacobians(theta,stage)
            finally:self.layout.fixed_views=fixed
            # A reference-camera perturbation now moves every same-state view.
            eps=1e-4 if theta.dtype==torch.float32 else 1e-5
            for b in self.front_indices.values():
                for s in (self.layout.rot[b],self.layout.trans[b]):
                    for i in range(s.start,s.stop):
                        plus=theta.clone();minus=theta.clone();plus[i]+=eps;minus[i]-=eps
                        full[:,:,:,i]=(self.forward(plus)[0]-self.forward(minus)[0])/(2*eps)
            return full[:,:,:,self.layout.active(stage)]
        active=self.layout.active(stage);col={i:j for j,i in enumerate(active)}
        jac=theta.new_zeros((self.layout.frames,len(self.flame.v_template),3,len(active)))
        eps=1e-4 if theta.dtype==torch.float32 else 1e-5
        for b in range(self.layout.frames):
            for i in range(self.layout.rot[b].start,self.layout.rot[b].stop):
                plus=theta.clone();minus=theta.clone();plus[i]+=eps;minus[i]-=eps
                jac[b,:,:,col[i]]=(self.forward(plus)[0][b]-self.forward(minus)[0][b])/(2*eps)
            for k,i in enumerate(range(self.layout.trans[b].start,self.layout.trans[b].stop)):jac[b,:,k,col[i]]=1
        return jac

    @torch.no_grad()
    def refresh(self,theta,stage):
        self.embedding=None
        verts,lm=self.forward(theta);self.embedding=self._embedding(theta,verts)
        self.context=[];self.view_diagnostics=[]
        for b,o in enumerate(self.observations):
            v=verts[b];vnp=v.cpu().numpy();fnp=self.faces.cpu().numpy()
            visible=visible_faces(vnp,fnp,o.camera,o.depth,depth_tolerance=.030 if stage==1 else .020)
            allowed=self.faces[torch.as_tensor(visible,device=self.device)]
            if len(allowed)<10:raise ValueError(f'{o.view_id}: 可见表面不足，检查配准或重新采集')
            q,d,face_ids,bary=closest_surface(v,allowed,self.points[b])
            tri=allowed[face_ids]
            keep=d<(.030 if stage==1 else .020)
            if int(keep.sum())<20:raise ValueError(f'{o.view_id}: 有效对应不足，检查点云叠加并重采')
            tri=tri[keep];bary=bary[keep];distance=d[keep]
            normal=matched_points(vertex_normals(v,self.faces),(tri,bary),normals=True)
            base_weights=np.asarray(self.jaw[b]['weights'])[keep.cpu().numpy()] if stage==2 and self.options.jaw_region_balance else np.ones(int(keep.sum()))
            weights=torch.as_tensor(base_weights,device=self.device,dtype=self.dtype)
            weights*=torch.clamp(.003/distance.clamp_min(1e-12),max=1)
            weights=(weights/weights.sum().clamp_min(1e-12)*self.quality[b]).sqrt()
            valid=visible_landmarks(vnp,fnp,lm[b].cpu().numpy(),o.uv,o.valid,o.camera,o.depth)
            ldiag={};lw=None
            if o.view!='front' and self.options.side_landmark_mode=='soft':
                from core.multiview_constraints import soft_side_landmarks
                ids,lw,valid,ldiag=soft_side_landmarks(o,vnp,fnp,lm[b].cpu().numpy(),
                    self.options.side_landmark_huber_m,self.options.side_landmark_gross_m)
            elif o.view!='front':
                # Never trust occluded-side detector predictions. Static internal
                # landmarks must also agree in 3-D with the current surface.
                valid &= np.linalg.norm(np.nan_to_num(o.xyz)-lm[b].cpu().numpy(),axis=1)<.006
            # Even frontal chin 8 is a view contour, not a fixed 3-D material
            # landmark. Keep it for observed metrics, use only 2-D jaw normals.
            if lw is None:
                ids=np.array([21,22,33,50,51,52,56,57,58]);ids=ids[valid[ids]];lw=np.ones(len(ids))
            jaw=self.jaw[b];jids=jaw['ids']
            if len(jids):
                r=((project(lm[b][jids],o.camera)-self.lmk2d[b][jids])*torch.as_tensor(jaw['normals'],device=self.device,dtype=self.dtype)).sum(1)*jaw['pixel_to_m']
                irls=torch.clamp(self.options.jaw_contour_huber_m/r.abs().clamp_min(1e-12),max=1)
            else:irls=theta.new_empty(0)
            self.context.append(dict(points=self.points[b][keep],tri=tri,bary=bary,normals=normal,weights=weights,ids=ids,valid=valid,jaw_irls=irls,
                landmark_weights=theta.new_tensor(lw).sqrt()))
            profile_diag={}
            if self.profiles and self.profiles[b] is not None and stage==2:
                from core.side_silhouette import freeze_profile
                frozen=freeze_profile(vnp,fnp,o,self.profiles[b],self.silhouette_topology)
                if len(frozen['edges']):
                    mid=v[torch.as_tensor(frozen['edges'],device=self.device)].mean(1)
                    residual=((project(mid,o.camera)-theta.new_tensor(frozen['target']))*theta.new_tensor(frozen['normal'])).sum(1)
                    frozen['irls']=torch.clamp(self.options.side_contour_huber_px/residual.abs().clamp_min(1e-12),max=1)
                self.context[-1]['profile']=frozen
                profile_diag=dict(profile_status=self.profiles[b]['status'],profile_observed_points=len(self.profiles[b]['uv']),
                    profile_equations=len(frozen['edges']),profile_target_uv=frozen['target'].tolist(),profile_filter_counts=self.profiles[b].get('filter_counts',{}))
            self.view_diagnostics.append(dict(view_id=o.view_id,valid_points=int(keep.sum()),visible_faces=int(visible.sum()),
                observed_points=len(o.points),sampled_points=len(self.points[b]),distance_rejected=int((~keep).sum()),
                correspondence_keep_ratio=float(keep.float().mean()),captured=True,registration_accepted=True,stage2_used=True,
                landmark_ids=ids.tolist(),quality=float(self.quality[b]),jaw_reason=jaw['reason'],initial_yaw_deg=jaw['initial_yaw_deg'],**ldiag,**profile_diag))

    @torch.no_grad()
    def residual(self,theta,stage,need_jac=True):
        verts,lm=self.forward(theta)
        dj=self.vertex_jacobians(theta,stage) if need_jac else None
        tri_lm,bary_lm=self.embedding
        active=self.layout.active(stage);mapping={i:k for k,i in enumerate(active)}
        rs=[];js=[];terms=[];view='shared'
        def add(r,j,scale=1.,name='other'):
            rs.append((r*scale).reshape(-1))
            if need_jac:js.append((j*scale).reshape(r.numel(),len(active)))
            terms.append(dict(view_id=view,term=name,equations=r.numel(),weighted_square_sum=float((r*scale).square().sum())))
        for b,(o,ctx) in enumerate(zip(self.observations,self.context)):
            view=o.view_id
            tri=ctx['tri'];bary=ctx['bary'];w=ctx['weights']
            diff=(verts[b][tri]*bary[:,:,None]).sum(1)-ctx['points']
            j=(dj[b][tri]*bary[:,:,None,None]).sum(1) if need_jac else None
            add(diff*w[:,None],j*w[:,None,None] if need_jac else None,math.sqrt(.1),name='depth_point_m')
            terms[-1]['raw_rmse_m']=float(diff.square().sum(1).mean().sqrt())
            add((diff*ctx['normals']).sum(1)*w,
                torch.einsum('ni,nip->np',ctx['normals'],j)*w[:,None] if need_jac else None,math.sqrt(.9),name='depth_plane_m')
            terms[-1]['raw_rmse_m']=float((diff*ctx['normals']).sum(1).square().mean().sqrt())
            jl=(dj[b][tri_lm[b]]*bary_lm[b,:,:,None,None]).sum(1) if need_jac else None
            uv,pj=project(lm[b],o.camera,True);juv=pj@jl if need_jac else None
            if stage==2:
                profile=ctx.get('profile')
                if profile and len(profile['edges']):
                    edges=torch.as_tensor(profile['edges'],device=self.device,dtype=torch.long);mid=verts[b][edges].mean(1)
                    puv,pjac=project(mid,o.camera,True)
                    normals=theta.new_tensor(profile['normal']);target=theta.new_tensor(profile['target'])
                    raw=((puv-target)*normals).sum(1)
                    derivative=torch.einsum('ni,nip->np',normals,pjac@dj[b][edges].mean(1)) if need_jac else None
                    # Pixels are whitened by their own noise, then assigned a
                    # 1 mm residual budget; depth/regularizer units stay metres.
                    weight=(profile['irls']*self.quality[b]*self.options.side_contour_weight/len(edges)).sqrt()*.001/self.options.side_contour_noise_px
                    add(raw*weight,derivative*weight[:,None] if need_jac else None,name='side_profile_px_whitened')
                    terms[-1]['raw_rmse_px']=float(raw.square().mean().sqrt())
                ids=ctx['ids']
                if len(ids):
                    wl=ctx.get('landmark_weights',theta.new_ones(len(ids)))
                    add((lm[b][ids]-self.lmk3d[b][ids])*wl[:,None],jl[ids]*wl[:,None,None] if need_jac else None,
                        math.sqrt(2*self.quality[b]/(3*len(ids))),name='landmark_m')
                    terms[-1]['raw_rmse_m']=float((lm[b][ids]-self.lmk3d[b][ids]).square().sum(1).mean().sqrt())
                jaw=self.jaw[b];ids=jaw['ids']
                if len(ids) and self.options.jaw_contour_weight>0:
                    normals=theta.new_tensor(jaw['normals']);unit=jaw['pixel_to_m']
                    wj=(ctx['jaw_irls']*self.options.jaw_contour_weight*self.quality[b]/len(ids)).sqrt()
                    add(((uv[ids]-self.lmk2d[b][ids])*normals).sum(1)*unit*wj,
                        torch.einsum('ni,nip->np',normals,juv[ids])*unit*wj[:,None] if need_jac else None)
            if o.view=='front':
                pairs=((50,58),(51,57),(52,56)) if stage==1 else ((61,67),(62,66),(63,65))
                pairs=[(i,k) for i,k in pairs if ctx['valid'][i] and ctx['valid'][k]]
                for i,k in pairs:
                    value=lm[b] if stage==1 else uv;target=self.lmk3d[b] if stage==1 else self.lmk2d[b]
                    derivative=jl if stage==1 else juv
                    delta=value[i]-value[k];distance=delta.norm().clamp_min(1e-12)
                    add(distance-(target[i]-target[k]).norm(),(delta/distance)@(derivative[i]-derivative[k]) if need_jac else None,
                        math.sqrt((10 if stage==1 else 1e-5)*self.quality[b]/max(len(pairs),1)))
        if stage==2:
            view='shared'
            if self.options.pose_strategy=='prior':
                value=self.pose_prior(theta);j=theta.new_zeros((len(value),len(active))) if need_jac else None
                if need_jac and len(value):
                    eps=1e-4 if theta.dtype==torch.float32 else 1e-6
                    columns={i for s in (*self.layout.rot,*self.layout.trans) for i in range(s.start,s.stop)}
                    for i in columns:
                        plus=theta.clone();minus=theta.clone();plus[i]+=eps;minus[i]-=eps
                        j[:,mapping[i]]=theta.new_tensor((self.pose_prior(plus)-self.pose_prior(minus))/(2*eps))
                if len(value):add(theta.new_tensor(value),j,math.sqrt(self.options.pose_prior_weight),name='relative_pose_whitened')
            # One identity prior and ONE expression prior per state, not per camera.
            regs=[(self.layout.shape,self.options.shape_reg)]+[(s[0],self.options.expression_reg) for s in self.layout.state_slots.values()]
            for s,weight in regs:
                j=theta.new_zeros((s.stop-s.start,len(active))) if need_jac else None
                if need_jac:
                    for row,i in enumerate(range(s.start,s.stop)):j[row,mapping[i]]=1
                add(theta[s],j,math.sqrt(weight),name='shared_regularizer')
        self.residual_terms=terms
        return torch.cat(rs),torch.cat(js) if need_jac else None


def fit_multiview(observations,folder,*,options=None,device=None,progress=lambda _:None):
    from core.integrated_fitting import train,make_flame
    from core.multiview_registration import select_registered_views
    from core.pipeline import write_json
    options=options or FitOptions();options.schedule();folder=Path(folder)
    timing={};started=time.perf_counter()
    observations,side_to_front,registration,excluded=select_registered_views(observations,folder,progress,options=options)
    # Extract silhouette evidence BEFORE constraint_observation rewrites the
    # outside semantic labels to 255 for the head-only depth/transition path.
    raw_profiles={}
    if options.side_contour_weight>0:
        from core.side_silhouette import observed_profile
        raw_profiles={o.view_id:observed_profile(o) for o in observations if o.view!='front'}
    from core.side_constraints import constraint_observation
    observations=[constraint_observation(o) for o in observations]
    timing['registration_s']=time.perf_counter()-started
    fronts={o.state:o for o in observations if o.view=='front'}
    c=fronts['closed'];o=fronts['open']
    # Real legacy front fit provides metric-scale identity and state initialization.
    progress('使用正面闭口/张口拟合初始化多视角 FLAME...')
    started=time.perf_counter()
    from core.multiview_initialization import initialization_signature,load_initialization,save_initialization
    signature=initialization_signature(fronts,options)
    cache=Path(options.front_initialization_cache) if options.front_initialization_cache else None
    if cache and cache.exists():
        initial=load_initialization(cache,signature);progress('复用输入与配置哈希一致的正面初始化')
    else:
        initial_meshes=train(c.xyz,o.xyz,c.uv,o.uv,c.valid,o.valid,c.points,o.points,c.camera,
            parsing_closed=c.parsing,parsing_open=o.parsing,options=options,device=device,
            progress=lambda s:progress('正面初始化｜'+s),camera_open=o.camera)
        initial={s:m.metadata['flame_parameters'] for s,m in zip(('closed','open'),initial_meshes)}
        if cache:save_initialization(cache,signature,initial)
    save_initialization(folder/'front_initialization.npz',signature,initial)
    timing['front_initialization_s']=time.perf_counter()-started
    transforms=[]
    for obs in observations:
        p=initial[obs.state];front_T=RigidCamera(p['rotation'],p['translation'])
        if obs.view=='front':transforms.append(front_T);continue
        transforms.append(side_to_front[obs.view_id].inverse().compose(front_T))
    write_json(folder/'registration.json',registration)
    device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    flame=make_flame(options,device,batch=len(observations))
    problem=MultiProblem(flame,observations,initial,transforms,options,raw_profiles=raw_profiles)
    started=time.perf_counter()
    theta,stage1=solve_stage(problem,problem.theta,1,lambda s:progress('多视角相机对齐｜'+s))
    problem.begin_shape_fit(theta)
    timing['camera_fit_s']=time.perf_counter()-started
    started=time.perf_counter()
    theta,stage2=solve_progressive(problem,theta,lambda s:progress('多视角联合拟合｜'+s))
    timing['joint_fit_s']=time.perf_counter()-started
    problem.refresh(theta,2);problem.residual(theta,2,False)
    from core.multiview_diagnostics import save_fit_diagnostics
    save_fit_diagnostics(problem,theta,folder)
    problem.embedding=None;vertices,landmarks=problem.forward(theta)
    shape,expr,pose,rotation,translation=problem.parameters(theta)
    meshes={};view_transforms={};front_indices={s:next(i for i,v in enumerate(observations) if v.state==s and v.view=='front') for s in fronts}
    for s,i in front_indices.items():
        mesh=trimesh.Trimesh(vertices[i].cpu().numpy(),flame.faces_tensor.cpu().numpy(),process=False)
        mesh.metadata.update(geometry_stage=2,shared_shape=True,flame_landmarks_3d=landmarks[i].cpu().numpy())
        meshes[s]=mesh
    for b,obs in enumerate(observations):
        i=front_indices[obs.state]
        if b==i:
            view_transforms[obs.view_id]=RigidCamera(np.eye(3),np.zeros(3))
            continue
        T=RigidCamera(rotation[b].cpu().numpy(),translation[b].cpu().numpy()).compose(
            RigidCamera(rotation[i].cpu().numpy(),translation[i].cpu().numpy()).inverse())
        view_transforms[obs.view_id]=T
    np.savez_compressed(folder/'flame_parameters.npz',shared_shape=shape[0].cpu().numpy(),
        expression_by_view=expr.cpu().numpy(),pose_by_view=pose.cpu().numpy(),rotation=rotation.cpu().numpy(),translation=translation.cpu().numpy(),
        view_ids=np.array([o.view_id for o in observations]),states=np.array([o.state for o in observations]))
    canonical_report=dict(enabled=False,geometry_stage='stage2')
    if options.canonical_shape_enabled:
        from core.canonical_shape_refine import correct_shape
        progress('实验：固定状态/相机，联合求解共享静止空间低频形状修正…')
        started=time.perf_counter();_,canonical_report=correct_shape(problem,theta,folder)
        timing['canonical_shape_s']=time.perf_counter()-started
    report=dict(stage1=stage1,stage2=stage2,views=problem.view_diagnostics,registration=registration,
        pose_strategy=options.pose_strategy,final_diagnostics=problem.iteration_diagnostics(theta,2),
        front_initialization_signature=signature,
        canonical_shape=canonical_report,
        constraint_policies={v.view_id:dict(upper_face_only=bool(v.quality.get('upper_face_only')),expression_status=v.quality.get('expression_status')) for v in observations},
        degraded_to_front_only=not any(v.view!='front' for v in observations),
        timing=timing,side_constraint_region='head_only',side_neck_used=False,
        used_view_ids=[o.view_id for o in observations],excluded_views=excluded,
        sharing='one shape; one expression/neck/jaw per state; SE3 per view; no scale',
        front_initialization_open_intrinsics_equal=bool(np.allclose([c.camera.fx,c.camera.fy,c.camera.cx,c.camera.cy],
            [o.camera.fx,o.camera.fy,o.camera.cx,o.camera.cy])),parameter_count=problem.layout.size)
    write_json(folder/'fitting_report.json',report)
    return meshes,view_transforms,report
