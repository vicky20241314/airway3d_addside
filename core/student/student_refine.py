"""Adapt tools/train.py stage 3: V0+D, scan surface fit, offset and smoothness.

Reuse the tested sparse nonrigid ICP solver instead of the original Adam loop.
Smooth D (preserve input detail), not absolute V; output V0+D exactly once.
"""
import time
import numpy as np
from core.student.offset_refine import depth_inputs, refine_pointcloud, _operators
from core.student.detail_snap import _project_model_to_uv, _sample_labels, _vertex_normals


def transition_mask(vertices, faces, active, camera, parsing, rings=2):
    uv,z = _project_model_to_uv(vertices,camera)
    labels,inside = _sample_labels(uv,parsing)
    view=-vertices/np.maximum(np.linalg.norm(vertices,axis=1,keepdims=True),1e-12)
    facing=(_vertex_normals(vertices,faces)*view).sum(1)>.15
    eligible=inside & (z>0) & facing & np.isin(labels,[0,1,2,3,10,12,13])
    _,_,edges=_operators(vertices,faces)
    mask=active.copy()
    for _ in range(rings):
        neighbors=np.zeros(len(mask),bool)
        neighbors[edges[mask[edges[:,0]],1]]=True
        neighbors[edges[mask[edges[:,1]],0]]=True
        mask |= neighbors & eligible
    return mask


def refine_depth(vertices, faces, depth, camera, parsing, landmarks_2d=None, semantic_landmarks=True, landmark_operator=None, **kwargs):
    started=time.perf_counter()
    vertices=np.asarray(vertices,float)
    points,confidence,observed=depth_inputs(vertices,faces,depth,camera,parsing,landmarks_2d)
    active=transition_mask(vertices,faces,observed,camera,parsing)
    semantic = None
    if semantic_landmarks and landmarks_2d is not None:
        from core.student.semantic_refine import SemanticConstraints
        semantic=SemanticConstraints(vertices,faces,landmarks_2d,camera,depth,parsing,landmark_operator=landmark_operator)
        active=semantic.activate(active,vertices,faces)
    options=dict(semantic_constraints=semantic, stages=(4.,1.,.25), iterations_per_stage=8,
                 max_offset_m=.008, edge_weight=.05, anchor_weight=.01)
    options.update(kwargs)
    result,info=refine_pointcloud(vertices,faces,points,active=active,confidence=confidence,**options)
    info.update(method='student_stage3_semantic' if semantic is not None else 'student_stage3_adapted', semantic=semantic.info if semantic is not None else None, n_observed_vertices=int(observed.sum()),
                n_transition_vertices=int((active & ~observed).sum()),
                basis='tools/train.py stage 3; sparse robust surface ICP and displacement Laplacian',
                elapsed_s=time.perf_counter()-started)
    return result,info
