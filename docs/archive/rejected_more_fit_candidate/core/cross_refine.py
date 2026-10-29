"""New stage2 -> exact midpoint subdivision -> guarded student 3-D offsets."""
import numpy as np
import trimesh
from scipy import sparse
from core.student.arap_refine import landmark_matrix
from core.student.student_refine import refine_depth
from core.conservative_refine import ShapeGuard,RefineOptions
from scipy.spatial import cKDTree
from dataclasses import asdict
from core.student.early_stop import StudentEarlyStop

DEFAULT_PROFILE = 'v2'
PROFILES = {
    'v2': dict(algorithm='cross-v2-guarded', stages=(4.,1.,.25), consistency=16.,
               tangent=0., step_tol_m=1e-4, continue_after_small_step=False),
    'v3': dict(algorithm='cross-v3-coherent', stages=(8.,4.,2.), consistency=32.,
               tangent=2., step_tol_m=1e-5, continue_after_small_step=True),
}

def subdivide_with_anchors(mesh):
    v=np.asarray(mesh.vertices).copy(); f=np.asarray(mesh.faces).copy()
    J=landmark_matrix(f,len(v))
    sv,sf=trimesh.remesh.subdivide(v,f)
    if not np.array_equal(sv[:len(v)],v) or len(sf)!=4*len(f):
        raise ValueError('Subdivision changed original vertices or face count')
    # Map each material landmark to its child triangle. Padding old columns
    # only preserves initial positions, not the deformed child surface.
    edges=mesh.edges_unique
    distances,which=cKDTree(sv[len(v):]).query(v[edges].mean(1))
    if distances.max()>1e-10: raise ValueError('Invalid midpoint mapping')
    midpoints={tuple(edge):int(idx+len(v)) for edge,idx in zip(edges,which)}
    rows=[];cols=[];values=[]
    for row in range(J.shape[0]):
        original=J.getrow(row).indices
        allowed=list(original)
        for i in range(3):
            allowed.append(midpoints[tuple(sorted((original[i],original[(i+1)%3])))])
        candidates=sf[np.isin(sf,allowed).all(1)]
        point=(J.getrow(row)@v)[0]
        for tri in candidates:
            weights=np.linalg.lstsq(np.vstack((sv[tri].T,np.ones(3))),np.r_[point,1.],rcond=None)[0]
            if weights.min()>=-1e-8 and weights.max()<=1+1e-8:
                rows.extend([row]*3);cols.extend(tri);values.extend(weights);break
        else: raise ValueError('Landmark is outside subdivided parent face')
    JS=sparse.coo_matrix((values,(rows,cols)),shape=(51,len(sv))).tocsr()
    if not np.allclose(J@v,JS@sv,atol=1e-12,rtol=0):
        raise ValueError('Subdivision changed anatomical anchors')
    return sv,sf,JS

def refine_mesh(mesh,depth,camera,parsing,landmarks_2d,*,profile=DEFAULT_PROFILE):
    if profile not in PROFILES: raise ValueError('Unknown cross refinement profile')
    config=PROFILES[profile]
    base,faces,J=subdivide_with_anchors(mesh)
    n=len(mesh.vertices)
    edge_mid=mesh.vertices[mesh.edges_unique].mean(1)
    distances,edge_ids=cKDTree(edge_mid).query(base[n:])
    if distances.max()>1e-10: raise ValueError('Invalid midpoint parents')
    endpoints=mesh.edges_unique[edge_ids]
    count=len(base)-n
    C=sparse.coo_matrix((np.tile([1.,-.5,-.5],count),
        (np.repeat(np.arange(count),3),np.column_stack((np.arange(n,len(base)),endpoints)).ravel())),
        shape=(count,len(base))).tocsr()
    options=RefineOptions();options.validate()
    guard=ShapeGuard(base,faces,options)
    vertices,info=refine_depth(base,faces,depth,camera,parsing,
        landmarks_2d=landmarks_2d,landmark_operator=J,
        max_offset_m=options.max_offset_m,max_step_m=options.max_step_m,
        shape_guard=guard,consistency_operator=C,consistency_weight=config['consistency'],edge_length_scale_m=.002,
        stages=config['stages'],tangent_weight=config['tangent'],
        early_stop=StudentEarlyStop(step_tol_m=config['step_tol_m']),
        continue_after_small_step=config['continue_after_small_step'])
    if not np.isfinite(vertices).all(): raise ValueError('Invalid stage3 vertices')
    result=trimesh.Trimesh(vertices,faces,process=False)
    result.metadata.update(geometry_stage=3,metrics_geometry_stage=2)
    residual=np.linalg.norm(C@(vertices-base),axis=1)*1000
    info.update(algorithm=config['algorithm'],profile=profile,subdivision='one midpoint iteration',
        guard_options={k:v for k,v in asdict(options).items() if k.startswith(('min_edge','max_edge','min_area','max_area','max_normal','max_dihedral'))},
        midpoint_consistency_weight=config['consistency'],edge_length_scale_m=.002,
        tangent_weight=config['tangent'],continue_after_small_step=config['continue_after_small_step'],
        midpoint_residual_mm=dict(p95=float(np.percentile(residual,95)),max=float(residual.max())),
        parameters=dict(stages=list(config['stages']),iterations_per_stage=8,max_offset_m=.002,max_step_m=.00025,
            edge_weight=.05,anchor_weight=.01,max_distance_m=.020,huber_m=.002,
            semantic_weight=40.,max_points=16000),
        original_vertices=len(mesh.vertices),subdivided_vertices=len(base),
        original_faces=len(mesh.faces),subdivided_faces=len(faces),
        landmark_adapter='barycentric anchors on subdivided child triangles')
    return result,base,info
