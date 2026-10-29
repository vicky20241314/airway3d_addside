"""Shared-state multi-camera FLAME LM using the existing progressive solver."""
from dataclasses import dataclass
from pathlib import Path
import math
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
        if stage==2:return list(range(self.size))
        # First camera-only pass: shape, expression AND joint pose are fixed.
        return [i for a,b in zip(self.rot,self.trans) for s in (a,b) for i in range(s.start,s.stop)]


class MultiProblem(JointProblem):
    def __init__(self,flame,observations,initial,transforms,options):
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
        from core.jaw_constraints import prepare_jaw
        for b,o in enumerate(observations):
            yaw=math.degrees(math.atan2(transforms[b].R[0,2],transforms[b].R[2,2]))
            jaw=prepare_jaw(o.parsing,o.uv,o.xyz,o.valid,o.camera,self.points[b].cpu().numpy())
            if o.view!='front' or abs(yaw)>options.jaw_max_yaw_deg:
                jaw['ids']=np.empty(0,int);jaw['normals']=np.empty((0,2));jaw['weights']=np.full(len(self.points[b]),1/len(self.points[b]))
                jaw['reason']='side_depth_only' if o.view!='front' else 'oblique_pose_fallback'
            jaw['initial_yaw_deg']=yaw
            self.jaw.append(jaw)

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
        if stage==2:return super().vertex_jacobians(theta,stage)
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
            if o.view!='front':
                # Never trust occluded-side detector predictions. Static internal
                # landmarks must also agree in 3-D with the current surface.
                valid &= np.linalg.norm(np.nan_to_num(o.xyz)-lm[b].cpu().numpy(),axis=1)<.006
            # Even frontal chin 8 is a view contour, not a fixed 3-D material
            # landmark. Keep it for observed metrics, use only 2-D jaw normals.
            ids=np.array([21,22,33,50,51,52,56,57,58])
            ids=ids[valid[ids]]
            jaw=self.jaw[b];jids=jaw['ids']
            if len(jids):
                r=((project(lm[b][jids],o.camera)-self.lmk2d[b][jids])*torch.as_tensor(jaw['normals'],device=self.device,dtype=self.dtype)).sum(1)*jaw['pixel_to_m']
                irls=torch.clamp(self.options.jaw_contour_huber_m/r.abs().clamp_min(1e-12),max=1)
            else:irls=theta.new_empty(0)
            self.context.append(dict(points=self.points[b][keep],tri=tri,bary=bary,normals=normal,weights=weights,ids=ids,valid=valid,jaw_irls=irls))
            self.view_diagnostics.append(dict(view_id=o.view_id,valid_points=int(keep.sum()),visible_faces=int(visible.sum()),
                landmark_ids=ids.tolist(),quality=float(self.quality[b]),jaw_reason=jaw['reason'],initial_yaw_deg=jaw['initial_yaw_deg']))

    @torch.no_grad()
    def residual(self,theta,stage,need_jac=True):
        verts,lm=self.forward(theta)
        dj=self.vertex_jacobians(theta,stage) if need_jac else None
        tri_lm,bary_lm=self.embedding
        active=self.layout.active(stage);mapping={i:k for k,i in enumerate(active)}
        rs=[];js=[]
        def add(r,j,scale=1.):
            rs.append((r*scale).reshape(-1))
            if need_jac:js.append((j*scale).reshape(r.numel(),len(active)))
        for b,(o,ctx) in enumerate(zip(self.observations,self.context)):
            tri=ctx['tri'];bary=ctx['bary'];w=ctx['weights']
            diff=(verts[b][tri]*bary[:,:,None]).sum(1)-ctx['points']
            j=(dj[b][tri]*bary[:,:,None,None]).sum(1) if need_jac else None
            add(diff*w[:,None],j*w[:,None,None] if need_jac else None,math.sqrt(.1))
            add((diff*ctx['normals']).sum(1)*w,
                torch.einsum('ni,nip->np',ctx['normals'],j)*w[:,None] if need_jac else None,math.sqrt(.9))
            jl=(dj[b][tri_lm[b]]*bary_lm[b,:,:,None,None]).sum(1) if need_jac else None
            uv,pj=project(lm[b],o.camera,True);juv=pj@jl if need_jac else None
            if stage==2:
                ids=ctx['ids']
                if len(ids):add(lm[b][ids]-self.lmk3d[b][ids],jl[ids] if need_jac else None,math.sqrt(2*self.quality[b]/(3*len(ids))))
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
            # One identity prior and ONE expression prior per state, not per camera.
            regs=[(self.layout.shape,self.options.shape_reg)]+[(s[0],self.options.expression_reg) for s in self.layout.state_slots.values()]
            for s,weight in regs:
                j=theta.new_zeros((s.stop-s.start,len(active))) if need_jac else None
                if need_jac:
                    for row,i in enumerate(range(s.start,s.stop)):j[row,mapping[i]]=1
                add(theta[s],j,math.sqrt(weight))
        return torch.cat(rs),torch.cat(js) if need_jac else None


def fit_multiview(observations,folder,*,options=None,device=None,progress=lambda _:None):
    from core.integrated_fitting import train,make_flame
    from core.multiview_registration import register_to_front
    from core.pipeline import write_json
    options=options or FitOptions();options.schedule();folder=Path(folder)
    fronts={o.state:o for o in observations if o.view=='front'}
    c=fronts['closed'];o=fronts['open']
    # Real legacy front fit provides metric-scale identity and state initialization.
    progress('使用正面闭口/张口拟合初始化多视角 FLAME...')
    initial_meshes=train(c.xyz,o.xyz,c.uv,o.uv,c.valid,o.valid,c.points,o.points,c.camera,
        parsing_closed=c.parsing,parsing_open=o.parsing,options=options,device=device,progress=progress,camera_open=o.camera)
    initial={s:m.metadata['flame_parameters'] for s,m in zip(('closed','open'),initial_meshes)}
    transforms=[];registration=[]
    for obs in observations:
        p=initial[obs.state];front_T=RigidCamera(p['rotation'],p['translation'])
        if obs.view=='front':transforms.append(front_T);continue
        progress(f'配准 {obs.view_id}：稳定面部重叠区域...')
        try:T,diag,source,target=register_to_front(fronts[obs.state],obs)
        except Exception as exc:
            write_json(folder/'registration_failure.json',dict(view_id=obs.view_id,error=str(exc)));raise
        transforms.append(T.inverse().compose(front_T));registration.append(dict(view_id=obs.view_id,**diag))
        # Colored overlay in STATE FRONT camera coordinates; explicit direction.
        overlay=trimesh.points.PointCloud(np.vstack((target,T.to_view(source))),
            colors=np.vstack((np.tile([60,160,255,255],(len(target),1)),np.tile([255,110,60,255],(len(source),1)))))
        overlay.export(folder/f'registration_{obs.view_id}_front.ply')
    write_json(folder/'registration.json',registration)
    device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    flame=make_flame(options,device,batch=len(observations))
    problem=MultiProblem(flame,observations,initial,transforms,options)
    theta,stage1=solve_stage(problem,problem.theta,1,progress)
    theta,stage2=solve_progressive(problem,theta,progress)
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
    report=dict(stage1=stage1,stage2=stage2,views=problem.view_diagnostics,registration=registration,
        sharing='one shape; one expression/neck/jaw per state; SE3 per view; no scale',
        front_initialization_open_intrinsics_equal=bool(np.allclose([c.camera.fx,c.camera.fy,c.camera.cx,c.camera.cy],
            [o.camera.fx,o.camera.fy,o.camera.cx,o.camera.cy])),parameter_count=problem.layout.size)
    write_json(folder/'fitting_report.json',report)
    return meshes,view_transforms,report
