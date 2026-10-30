"""Experimental shared rest-space low-frequency identity correction (opt-in).

Only alpha is optimized. Identity/state/cameras remain fixed. Corrected rest
joints are regressed by original FLAME LBS; no camera-space displacement is
shared between states. Stage2 geometry and business metrics are immutable.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
import trimesh
from scipy import sparse
from scipy.sparse.linalg import eigsh
from core.multiview_geometry import project,visible_faces
from core.initial_surface import closest_surface
from core.conservative_refine import ShapeGuard,RefineOptions


def low_frequency_basis(vertices,faces,active,modes,joint_regressor=None):
    """Dirichlet graph modes; six infinitesimal rigid fields projected out."""
    n=len(vertices);edges=trimesh.Trimesh(vertices,faces,process=False).edges_unique
    A=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.asarray(A.sum(1)).ravel();L=sparse.diags(degree)-A
    active=np.asarray(active,bool);ids=np.flatnonzero(active)
    if len(ids)<max(50,modes+2):raise ValueError('Insufficient reliably observed canonical support')
    values,scalar=eigsh(L[ids][:,ids].astype(float),k=min(modes,len(ids)-2),sigma=-1e-6,which='LM',v0=np.ones(len(ids)))
    scalar=scalar[:,np.argsort(values)];B=np.zeros((n,3,scalar.shape[1]*3))
    for axis in range(3):B[ids,axis,axis::3]=scalar
    centered=vertices-vertices[ids].mean(0);rigid=np.zeros((n,3,6))
    rigid[ids,:,:3]=np.eye(3)
    for axis in range(3):rigid[ids,:,axis+3]=np.cross(np.eye(3)[axis],centered[ids])
    constraints=rigid.reshape(-1,6)
    if joint_regressor is not None:
        # Zero canonical offsets alone do not freeze posed neck vertices:
        # changing regressed joints can move them through the LBS pivot terms.
        # Preserve all joint moments, in addition to removing rigid modes.
        reg=np.asarray(joint_regressor);moments=np.zeros((n,3,3*len(reg)))
        for axis in range(3):moments[ids,axis,axis::3]=reg[:,ids].T
        constraints=np.column_stack((constraints,moments.reshape(3*n,-1)))
    norm=np.linalg.norm(constraints,axis=0);constraints=constraints[:,norm>1e-12]/norm[norm>1e-12]
    Q,s,_=np.linalg.svd(constraints,full_matrices=False);Q=Q[:,s>1e-10]
    flat=B.reshape(-1,B.shape[-1]);flat-=Q@(Q.T@flat)
    # Remove any modes made redundant by the rigid projection.
    u,s,_=np.linalg.svd(flat,full_matrices=False);flat=u[:,s>1e-8]
    B=flat.reshape(n,3,-1);B/=np.maximum(np.linalg.norm(B,axis=1).max(0),1e-12)
    B[~active]=0
    return B,L


def posed_vertices(problem,theta,offset):
    shape,expr,pose,rotation,translation=problem.parameters(theta)
    with torch.no_grad():
        raw,_=problem.flame(shape_params=shape,expression_params=expr,pose_params=pose,
            transl=torch.zeros_like(translation),view_rotation=rotation,canonical_offset=theta.new_tensor(offset))
        return (torch.einsum('bij,bvj->bvi',rotation,raw)+translation[:,None]).cpu().numpy()


def observed_support(problem,vertices):
    active=np.zeros(vertices.shape[1],bool);protected=np.zeros_like(active);faces=problem.faces.cpu().numpy()
    for b,o in enumerate(problem.observations):
        uv,z=project(vertices[b],o.camera);xy=np.rint(uv).astype(int);h,w=o.depth.shape;x,y=xy.T
        inside=(z>0)&(x>=1)&(x<w-1)&(y>=1)&(y<h-1);ids=np.flatnonzero(inside)
        if not len(ids):continue
        labels=o.parsing[y[ids],x[ids]];protected[ids[np.isin(labels,(11,12,13,14,16))]]=True
        safe=cv2.erode(np.isin(o.parsing,(1,2,3,10)).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        low=cv2.erode(np.asarray(o.depth,float),np.ones((3,3),np.uint8));high=cv2.dilate(np.asarray(o.depth,float),np.ones((3,3),np.uint8))
        good=safe[y[ids],x[ids]]&(low[y[ids],x[ids]]>0)&(high[y[ids],x[ids]]-low[y[ids],x[ids]]<=8)
        seen=np.zeros_like(active);seen[np.unique(faces[visible_faces(vertices[b],faces,o.camera,o.depth)])]=True
        active[ids[good&seen[ids]]]=True
    # Protect the unobserved neck by canonical topology position as well.
    template=problem.flame.v_template.cpu().numpy()
    protected|=template[:,1]<np.quantile(template[:,1],.25)
    return active&~protected


def correct_shape(problem,theta,folder):
    from core.pipeline import write_json
    options=problem.options;folder=Path(folder);faces=problem.faces.cpu().numpy()
    base=posed_vertices(problem,theta,np.zeros_like(problem.flame.v_template.cpu().numpy()))
    active=observed_support(problem,base)
    B,L=low_frequency_basis(problem.flame.v_template.cpu().numpy(),faces,active,options.canonical_shape_modes,
        problem.flame.J_regressor.cpu().numpy())
    eps=.001
    derivatives=np.stack([(posed_vertices(problem,theta,B[:,:,k]*eps)-posed_vertices(problem,theta,-B[:,:,k]*eps))/(2*eps)
        for k in range(B.shape[-1])],axis=-1)
    guards=[ShapeGuard(v,faces,RefineOptions()) for v in base]
    canonical_guard=ShapeGuard(problem.flame.v_template.cpu().numpy(),faces,RefineOptions())
    alpha=np.zeros(B.shape[-1]);history=[];current=base.copy();rejected=0
    smooth=(L@B.reshape(len(B),-1)).reshape(-1,len(alpha));flat=B.reshape(-1,len(alpha))
    regular=options.canonical_shape_reg*(flat.T@flat/len(B)+smooth.T@smooth/len(B))
    for outer in range(6):
        rows=[];targets=[]
        for b,o in enumerate(problem.observations):
            vis=visible_faces(current[b],faces,o.camera,o.depth);allowed=faces[vis]
            if not len(allowed):continue
            points=problem.points[b].cpu().numpy()
            tensors=closest_surface(torch.as_tensor(current[b],device=problem.device,dtype=problem.dtype),
                torch.as_tensor(allowed,device=problem.device,dtype=torch.long),problem.points[b])
            q,d,fi,bc=(value.cpu().numpy() for value in tensors)
            tri=allowed[fi];keep=(d<.020)&active[tri].all(1)
            if keep.sum()<20:continue
            tri=tri[keep];bc=bc[keep];r=q[keep]-points[keep]
            j=(derivatives[b][tri]*bc[:,:,None,None]).sum(1)
            weight=np.sqrt(np.minimum(1.,.003/np.maximum(d[keep],1e-12))*problem.quality[b]/keep.sum())
            rows.append((j*weight[:,None,None]).reshape(-1,len(alpha)));targets.append((r*weight[:,None]).ravel())
        if not rows:break
        J=np.vstack(rows);r=np.concatenate(targets);H=J.T@J+regular+np.eye(len(alpha))*1e-10
        step=np.linalg.solve(H,-J.T@r-regular@alpha)
        cost=float(r@r+alpha@regular@alpha);accepted=False
        for scale in (1.,.5,.25,.125,.0625,.03125,.015625):
            trial=alpha+scale*step;D=np.einsum('vck,k->vc',B,trial)
            candidate=base+np.einsum('bvck,k->bvc',derivatives,trial)
            if np.linalg.norm(D,axis=1).max()>options.canonical_shape_max_m:continue
            if any(np.linalg.norm(v-v0,axis=1).max()>options.canonical_shape_max_m for v,v0 in zip(candidate,base)):continue
            if len(canonical_guard.inspect(canonical_guard.base+D)[0]) or any(len(g.inspect(v)[0]) for g,v in zip(guards,candidate)):continue
            predicted=r+J@(scale*step);new_cost=float(predicted@predicted+trial@regular@trial)
            if new_cost<cost:
                alpha=trial;current=candidate;accepted=True;break
        history.append(dict(outer=outer+1,equations=len(r),accepted=accepted,weighted_cost=cost,
            max_canonical_offset_m=float(np.linalg.norm(np.einsum('vck,k->vc',B,alpha),axis=1).max())))
        if not accepted:rejected+=1;break
        if np.linalg.norm(scale*step)<1e-6:break
    offset=np.einsum('vck,k->vc',B,alpha);exact=posed_vertices(problem,theta,offset)
    error=float(np.abs(exact-current).max())
    if error>1e-6:raise ValueError('Canonical LBS derivative/reconstruction mismatch')
    protected_change=float(np.linalg.norm((exact-base)[:,~active],axis=-1).max(initial=0))
    if protected_change>1e-6:raise ValueError('Canonical correction moved protected posed vertices')
    if any(len(g.inspect(v)[0]) for g,v in zip(guards,exact)):raise ValueError('Corrected state violates geometry guard')
    hashes=dict(basis_sha256=hashlib.sha256(B.tobytes()).hexdigest(),topology_sha256=hashlib.sha256(faces.tobytes()).hexdigest())
    np.savez_compressed(folder/'canonical_shape.npz',alpha=alpha,basis=B,active=active,canonical_offset=offset,faces=faces,**hashes)
    meshes={}
    for state,i in problem.front_indices.items():
        mesh=trimesh.Trimesh(exact[i],faces,process=False);mesh.metadata['geometry_stage']='shape_corrected'
        out=folder/state;out.mkdir(parents=True,exist_ok=True);mesh.export(out/f'mesh_{state}_shape_corrected.obj',digits=16);meshes[state]=mesh
    report=dict(enabled=True,experimental=True,independent_accuracy_validated=False,geometry_stage='shape_corrected',
        shared_alpha=True,parameters_fixed='FLAME identity/expression/pose and all cameras',rest_joint_policy='regress from corrected shaped rest vertices in original LBS',
        active_vertices=int(active.sum()),coefficients=len(alpha),max_canonical_offset_m=float(np.linalg.norm(offset,axis=1).max()),
        max_state_change_m=float(np.linalg.norm(exact-base,axis=-1).max()),limit_m=options.canonical_shape_max_m,
        protected_posed_change_m=protected_change,joint_moment_max_m=float(np.abs(problem.flame.J_regressor.cpu().numpy()@offset).max()),
        joint_policy='regress corrected rest joints; basis has zero joint moments, preventing indirect neck motion',
        exact_lbs_reconstruction_error_m=error,history=history,rejected_steps=rejected,**hashes,
        geometry_guards=[g.inspect(v)[1] for g,v in zip(guards,exact)])
    write_json(folder/'canonical_shape_report.json',report)
    return meshes,report


def replay_saved_shape(source,output_root):
    """Re-solve alpha from immutable saved stage2; no camera/state re-fitting."""
    import copy
    import json
    import shutil
    import time
    from dataclasses import replace,asdict
    from datetime import datetime
    from core.fit_options import FitOptions
    from core.integrated_fitting import make_flame
    from core.multiview_data import load_session
    from core.multiview_fitting import MultiProblem
    from core.multiview_geometry import RigidCamera
    from core.side_constraints import constraint_observation
    from core.pipeline import write_json
    source=Path(source);r=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if r['status']!='complete':raise ValueError('Canonical replay requires completed stage2')
    options=replace(FitOptions(**r['fit_configuration']),canonical_shape_enabled=True);options.schedule()
    with np.load(source/'flame_parameters.npz') as a:p={k:a[k] for k in a.files}
    by_id={o.view_id:o for o in load_session(source/'prepared/session.json')};observations=[]
    for view_id in p['view_ids']:
        o=by_id[str(view_id)];o.quality.update(r['fitting'].get('constraint_policies',{}).get(str(view_id),{}))
        observations.append(constraint_observation(o))
    initial={s:dict(shape=p['shared_shape'],expression=p['expression_by_view'][i],pose=p['pose_by_view'][i])
        for i,s in enumerate(p['states'])}
    flame=make_flame(options,'cpu',batch=len(observations),dtype=torch.float64)
    transforms=[RigidCamera(R,t) for R,t in zip(p['rotation'],p['translation'])]
    problem=MultiProblem(flame,observations,initial,transforms,options)
    folder=Path(output_root)/('mv_canonical_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(parents=True,exist_ok=False)
    for name in ('flame_parameters.npz','front_initialization.npz','extrinsics.json','metrics.json'):
        shutil.copyfile(source/name,folder/name)
    shutil.copytree(source/'prepared',folder/'prepared');shutil.copytree(source/'capture',folder/'capture')
    for file in source.glob('registration*.json'):shutil.copyfile(file,folder/file.name)
    for s in ('closed','open'):
        (folder/s).mkdir();shutil.copyfile(source/s/f'mesh_{s}_stage2.obj',folder/s/f'mesh_{s}_stage2.obj')
    report=copy.deepcopy(r);report.update(status='started',stage='canonical_shape_replay',source_stage2=str(source.resolve()),stage3_enabled=False,
        run_purpose='canonical_shape_replay',timing={},inherited_stage2=True,fit_configuration=asdict(options),
        inherited_fitting_configuration=r['fit_configuration'],source_run_elapsed_s=r['elapsed_s'])
    write_json(folder/'report.json',report);started=time.perf_counter()
    try:
        _,info=correct_shape(problem,problem.theta,folder)
        report['fitting']['canonical_shape']=info
        report.update(status='complete',stage='complete',elapsed_s=time.perf_counter()-started)
        write_json(folder/'report.json',report);write_json(folder/'fitting_report.json',report['fitting'])
        from core.multiview_diagnostics import save_geometry_changes
        save_geometry_changes(folder)
        write_json(folder/'provenance.json',dict(inherited_stage2=str(source.resolve()),
            source_provenance_sha256=hashlib.sha256((source/'provenance.json').read_bytes()).hexdigest(),
            canonical_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            stage2_byte_identical=True,metrics_byte_identical=(source/'metrics.json').read_bytes()==(folder/'metrics.json').read_bytes()))
        return folder
    except Exception as exc:
        report.update(status='failed',error=str(exc));write_json(folder/'report.json',report);raise
