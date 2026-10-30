"""One joint D per state; observations and equations have explicit SE(3)."""
import numpy as np
from scipy import sparse
from core.multiview_geometry import visible_faces,visible_landmarks
from core.initial_surface import closest_surface_numpy


def camera_equations(C,t,transform,n_vertices):
    """C_view * (R * V_common + t_camera) = t_view, row-major xyz.

    For isotropic 3D depth rows the equivalent common target is R.T(p-t),
    since R.T R=I. Semantic rays/depth use the full matrix mapping here.
    """
    rotation=sparse.kron(sparse.eye(n_vertices,format='csr'),sparse.csr_matrix(transform.R),format='csr')
    return (C@rotation).tocsr(),np.asarray(t)-C@np.tile(transform.t,n_vertices)


def normalize_view_equations(equations,qualities):
    """Mean squared rows per view, quality weighted, preserving one-view scale.

    A common mean row budget preserves the legacy regularizer balance. It
    does not affect the relative contributions of cameras with different
    numbers of reliable semantic rows.
    """
    if not equations:return []
    budget=sum(C.shape[0] for C,t in equations)/len(equations)
    qsum=sum(qualities)
    result=[]
    for (C,t),quality in zip(equations,qualities):
        scale=np.sqrt(budget*quality/qsum/max(C.shape[0],1))
        result.append((C*scale,t*scale))
    return result


class JointSemantic:
    def __init__(self,equations,n_vertices):
        self.C=sparse.vstack([a for a,b in equations],format='csr') if equations else sparse.csr_matrix((0,3*n_vertices))
        self.t=np.concatenate([b for a,b in equations]) if equations else np.empty(0)
    def linear_system(self,base,ids):
        columns=(3*ids[:,None]+np.arange(3)).ravel()
        return self.C[:,columns],self.t-self.C@base.ravel()
    def allowed_correspondences(self,triangle_ids,points):
        # Per-view lip/visibility checks have already been performed by provider.
        return np.ones(len(points),bool)


class ViewCorrespondences:
    def __init__(self,observations,transforms,groups,semantics):
        self.observations=observations;self.transforms=transforms
        self.groups=np.asarray(groups);self.semantics=semantics;self.history=[]

    def __call__(self,current,faces,points,source_indices):
        groups=self.groups[source_indices]
        closest=np.zeros_like(points);distance=np.full(len(points),np.inf)
        face_ids=np.zeros(len(points),int);bary=np.tile([1.,0,0],(len(points),1));valid=np.zeros(len(points),bool)
        report=[]
        for b,o in enumerate(self.observations):
            ids=np.flatnonzero(groups==b);T=self.transforms[o.view_id]
            view=T.to_view(current)
            visible=visible_faces(view,faces,o.camera,o.depth)
            allowed=np.flatnonzero(visible)
            if len(ids) and len(allowed):
                p_view=T.to_view(points[ids])
                q,d,local,bc=closest_surface_numpy(view,faces[allowed],p_view)
                original=allowed[local]
                keep=np.isfinite(d)&(d<.020)
                if self.semantics[b] is not None:
                    keep &= self.semantics[b].allowed_correspondences(original,p_view)
                closest[ids]=T.to_common(q);distance[ids]=d;face_ids[ids]=original;bary[ids]=bc;valid[ids]=keep
            report.append(dict(view_id=o.view_id,samples=len(ids),eligible_before_active_support=int(valid[ids].sum()),visible_faces=len(allowed)))
        self.history.append(report)
        return closest,distance,face_ids,bary,valid


def prepare_state(base,faces,J,observations,transforms,max_points=16000):
    from core.side_constraints import constraint_observation
    observations=[constraint_observation(o) for o in observations]
    from core.student.offset_refine import depth_inputs
    from core.student.student_refine import transition_mask
    from core.student.semantic_refine import SemanticConstraints
    points=[];confidence=[];groups=[];equations=[];semantic_quality=[];semantics=[]
    active=np.zeros(len(base),bool);quality={};diagnostics=[]
    rng=np.random.default_rng(42)
    for b,o in enumerate(observations):
        T=transforms[o.view_id];v=T.to_view(base)
        # Every origin-dependent operation runs in this actual camera frame:
        # projection, depth, rays, normals/facing, image labels, transition mask.
        p,w,seen=depth_inputs(v,faces,o.depth,o.camera,o.parsing,o.uv)
        mask=transition_mask(v,faces,seen,o.camera,o.parsing)
        if len(p)>max_points:
            select=np.sort(rng.choice(len(p),max_points,replace=False));p=p[select];w=w[select]
        quality[b]=o.quality.get('weight',1.)
        model_lm=np.vstack((np.full((17,3),np.nan),J@v))
        reliable=visible_landmarks(v,faces,model_lm,o.uv,o.valid,o.camera,o.depth)
        target=o.uv.copy();target[~reliable]=np.nan
        semantic=None
        if reliable[17:].any():
            semantic=SemanticConstraints(v,faces,target,o.camera,o.depth,o.parsing,landmark_operator=J)
            C,t=camera_equations(semantic.C,semantic.t,T,len(base))
            equations.append((C,t));semantic_quality.append(quality[b])
            mask=semantic.activate(mask,v,faces)
        active|=mask;semantics.append(semantic)
        points.append(T.to_common(p));confidence.append(w);groups.append(np.full(len(p),b,int))
        diagnostics.append(dict(view_id=o.view_id,reliable_points=len(p),visible_vertices=int(seen.sum()),
            active_vertices=int(mask.sum()),reliable_landmarks=int(reliable.sum()),quality=quality[b]))
    p=np.vstack(points);w=np.concatenate(confidence);g=np.concatenate(groups)
    joint=JointSemantic(normalize_view_equations(equations,semantic_quality),len(base))
    provider=ViewCorrespondences(observations,transforms,g,semantics)
    return p,w,g,quality,active,joint,provider,diagnostics


def refine_state(mesh,observations,transforms):
    from core.cross_refine import subdivide_with_anchors
    from core.conservative_refine import RefineOptions,ShapeGuard
    from core.student.offset_refine import refine_pointcloud
    from scipy.spatial import cKDTree
    import trimesh
    if len({o.state for o in observations})!=1:raise ValueError('One joint D must belong to exactly one state')
    base,faces,J=subdivide_with_anchors(mesh)
    n=len(mesh.vertices);edges=mesh.edges_unique
    distance,which=cKDTree(mesh.vertices[edges].mean(1)).query(base[n:])
    if distance.max(initial=0)>1e-10:raise ValueError('Invalid midpoint parents')
    count=len(base)-n
    C=sparse.coo_matrix((np.tile([1.,-.5,-.5],count),
        (np.repeat(np.arange(count),3),np.column_stack((np.arange(n,len(base)),edges[which])).ravel())),
        shape=(count,len(base))).tocsr()
    options=RefineOptions();options.validate();guard=ShapeGuard(base,faces,options)
    p,w,g,quality,active,semantic,provider,diagnostics=prepare_state(base,faces,J,observations,transforms)
    vertices,info=refine_pointcloud(base,faces,p,active=active,confidence=w,
        semantic_constraints=semantic,correspondence_provider=provider,point_groups=g,group_quality=quality,
        stages=(4.,1.,.25),iterations_per_stage=8,max_points=max(len(p),1),
        max_offset_m=.002,max_step_m=.00025,max_distance_m=.020,huber_m=.002,
        edge_weight=.05,anchor_weight=.01,shape_guard=guard,
        consistency_operator=C,consistency_weight=16.,edge_length_scale_m=.002,
        fairing_strength=1.,fairing_max_change_m=.0001)
    if not np.isfinite(vertices).all():raise RuntimeError('Invalid joint refinement output')
    result=trimesh.Trimesh(vertices,faces,process=False)
    result.metadata.update(geometry_stage=3,metrics_geometry_stage=2)
    info.update(algorithm='multiview_joint_guarded_D',state=observations[0].state,views=diagnostics,
        per_view_history=provider.history,coordinate_frame='state_front',
        shared_displacement=True,subdivision='one midpoint iteration',
        max_step_m=.00025,fairing_max_change_m=.0001,
        original_vertices=n,subdivided_vertices=len(base),original_faces=len(mesh.faces),subdivided_faces=len(faces))
    return result,base,info
