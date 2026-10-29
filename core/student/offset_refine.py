"""Non-rigid surface ICP after FLAME fitting (all geometry in metres).

Solve V = V0 + D, with dynamic scan-to-triangle barycentric constraints,
Huber IRLS, and coarse-to-fine regularisation of D, not of V. CPU only.
"""
from __future__ import annotations

import time
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve
from core.student.early_stop import ConsecutiveRelativeStop, STUDENT_EARLY_STOP, as_report
from core.student.initial_surface import closest_surface_numpy


def _operators(vertices, faces):
    edges = np.unique(np.sort(trimesh.geometry.faces_to_edges(faces), axis=1), axis=0)
    i, j = edges.T
    n = len(vertices)
    adjacency = sparse.coo_matrix((np.ones(2*len(i)),
                                  (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    L = sparse.diags((degree > 0).astype(float)) - sparse.diags(1/np.maximum(degree, 1)) @ adjacency
    E = sparse.coo_matrix((np.tile([1., -1.], len(edges)),
                          (np.repeat(np.arange(len(edges)), 2), edges.ravel())),
                         shape=(len(edges), n)).tocsr()
    return L.tocsr(), E, edges


def _cross(vertices, faces):
    tri = vertices[faces]
    return np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])


def refine_pointcloud(vertices, faces, points, *, active=None, confidence=None,
                      stages=(4., 1., .25), iterations_per_stage=8,
                      max_distance_m=.020, max_offset_m=.015, huber_m=.002,
                      edge_weight=.05, anchor_weight=.002, max_points=16000,
                      seed=42, semantic_constraints=None, early_stop=STUDENT_EARLY_STOP,
                      shape_guard=None, max_step_m=None, consistency_operator=None,
                      consistency_weight=0., edge_length_scale_m=None,
                      fairing_strength=0., fairing_max_change_m=.0001,
                      correspondence_provider=None, point_groups=None, group_quality=None):
    """Return new vertices and diagnostics without modifying the inputs.

    Requires an already aligned mesh and face-only cloud. ``active`` fixes
    unobserved/unsafe vertices exactly; RGB-D callers should use refine_depth.
    Samples and their confidence stay fixed; triangle IDs and barycentrics are
    recomputed each outer iteration. No scanner ground truth or new rigid ICP.
    """
    started = time.perf_counter()
    base = np.asarray(vertices, dtype=float).copy()
    faces = np.asarray(faces, dtype=np.int64)
    points = np.asarray(points, dtype=float)
    if (base.ndim != 2 or base.shape[1] != 3 or len(base) == 0
            or not np.isfinite(base).all() or faces.ndim != 2 or faces.shape[1] != 3
            or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(base)
            or points.ndim != 2 or points.shape[1] != 3):
        raise ValueError('Expected finite vertices Nx3, valid triangles Mx3 and points Px3')
    if (not stages or not np.isfinite(stages).all() or min(stages) < 0
            or iterations_per_stage < 1 or max_points < 1
            or not np.isfinite([max_distance_m, max_offset_m, huber_m,
                                    edge_weight, anchor_weight]).all()
            or min(max_distance_m, max_offset_m, huber_m, anchor_weight) <= 0
            or edge_weight < 0):
        raise ValueError('Invalid optimisation settings')
    active = np.ones(len(base), bool) if active is None else np.asarray(active, bool)
    if (not np.isfinite(consistency_weight) or consistency_weight < 0
            or any(x is not None and (not np.isfinite(x) or x <= 0)
                   for x in (max_step_m, edge_length_scale_m))):
        raise ValueError('Invalid guarded refinement settings')
    if consistency_operator is not None:
        if (consistency_operator.shape[1] != len(base)
                or not np.isfinite(consistency_operator.data).all()):
            raise ValueError('Invalid midpoint consistency operator')
    if (not np.isfinite([fairing_strength,fairing_max_change_m]).all()
            or fairing_strength<0 or fairing_max_change_m<=0
            or (fairing_strength>0 and shape_guard is None)):
        raise ValueError('Invalid bounded displacement fairing settings')
    weights = np.ones(len(points)) if confidence is None else np.asarray(confidence, float)
    source_indices=np.arange(len(points))
    groups=None if point_groups is None else np.asarray(point_groups)
    if groups is not None and groups.shape!=(len(points),):raise ValueError('Invalid view groups')
    if groups is not None:
        quality={} if group_quality is None else group_quality
        if any(not np.isfinite(quality.get(int(g),1.)) or not 0<quality.get(int(g),1.)<=1 for g in np.unique(groups)):
            raise ValueError('Invalid per-view quality')
    if active.shape != (len(base),) or weights.shape != (len(points),):
        raise ValueError('Mask/confidence shape mismatch')
    good = np.isfinite(points).all(1) & np.isfinite(weights) & (weights > 0)
    points, weights = points[good], np.clip(weights[good], 0, 1)
    source_indices=source_indices[good]
    if groups is not None:groups=groups[good]
    if len(points) > max_points:
        selected = np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))
        points, weights = points[selected], weights[selected]
        source_indices=source_indices[selected]
        if groups is not None:groups=groups[selected]
    ids = np.flatnonzero(active)
    semantic_C = semantic_t = None
    if semantic_constraints is not None:
        semantic_C, semantic_t = semantic_constraints.linear_system(base, ids)
    info = dict(method='offset_surface_icp', n_samples=len(points), n_active=len(ids),
                max_offset_m=max_offset_m, stages=list(stages), history=[])
    if early_stop is not None:
        info['student_early_stop'] = as_report(early_stop, iters_run=0, stopped=None)
    if len(points) == 0 or len(ids) == 0:
        info.update(status='no_observations', elapsed_s=time.perf_counter()-started)
        return base, info
    L, E, edges = _operators(base, faces)
    if edge_length_scale_m is not None:
        lengths=np.linalg.norm(base[edges[:,0]]-base[edges[:,1]],axis=1)
        E=sparse.diags(edge_length_scale_m/np.maximum(lengths,.0002))@E
    L, E = L[:, ids], E[:, ids]
    lap = L.T @ L
    edge = E.T @ E
    consistency=sparse.csr_matrix((len(ids),len(ids)))
    if consistency_operator is not None:
        C=consistency_operator[:,ids]
        consistency=consistency_weight*(C.T@C)
    cross0 = _cross(base, faces)
    area0 = np.linalg.norm(cross0, axis=1)
    valid_faces = area0 > 1e-14
    edge0 = np.linalg.norm(base[edges[:, 0]]-base[edges[:, 1]], axis=1)
    delta = np.zeros((len(ids), 3))
    current = base.copy()
    ever = np.zeros(len(base), bool)
    previous_ids = None
    stop_reason = None
    for stage, smooth in enumerate(stages):
        regularizer = smooth*lap + edge_weight*edge + anchor_weight*sparse.eye(len(ids)) + consistency
        rmse_guard = None
        if early_stop is not None:
            rmse_guard = ConsecutiveRelativeStop(
                early_stop.rmse_rel_tol, early_stop.rmse_patience, early_stop.rmse_min_iters)
        stage_done = False
        for iteration in range(iterations_per_stage):
            provider_valid=np.ones(len(points),bool)
            if correspondence_provider is None:
                closest, distance, triangle_ids, bary = closest_surface_numpy(current, faces, points)
            else:
                closest,distance,triangle_ids,bary,provider_valid=correspondence_provider(current,faces,points,source_indices)
            triangles = faces[triangle_ids]
            bary = np.clip(bary, 0, 1)
            bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
            support = (bary*active[triangles]).sum(1)
            keep = ((distance < max_distance_m) & (support > .5)
                    & np.isfinite(bary).all(1) & valid_faces[triangle_ids] & provider_valid)
            if semantic_constraints is not None:
                keep &= semantic_constraints.allowed_correspondences(triangle_ids, points)
            rows = np.flatnonzero(keep)
            if not len(rows):
                break
            B = sparse.coo_matrix((bary[keep].ravel(),
                                   (np.repeat(np.arange(len(rows)), 3), triangles[keep].ravel())),
                                  shape=(len(rows), len(base))).tocsr()
            A = B[:, ids]
            target = points[keep] - B @ base
            # Normalize by sample support per active vertex: tuning is largely
            # independent of depth resolution / point count.
            density = max(float(np.asarray(A.sum(0)).sum()) / max(np.count_nonzero(A.getnnz(0)), 1), 1e-12)
            w = weights[keep] * np.minimum(1., huber_m/np.maximum(distance[keep], 1e-12)) / density
            if groups is not None:
                # Normalize EACH view after all gates, then quality-weight.
                # Budget ~active support keeps the legacy regularizer scale.
                w=weights[keep]*np.minimum(1.,huber_m/np.maximum(distance[keep],1e-12))
                present=np.unique(groups[keep]);qsum=sum(quality.get(int(g),1.) for g in present)
                budget=max(np.count_nonzero(A.getnnz(0)),1)
                for g in present:
                    part=groups[keep]==g
                    w[part]*=budget*quality.get(int(g),1.)/qsum/max(w[part].sum(),1e-15)
            H = A.T @ sparse.diags(w) @ A + regularizer
            rhs = A.T @ (w[:, None]*target)
            if semantic_C is not None:
                H3 = sparse.kron(H, sparse.eye(3), format='csc') + semantic_C.T @ semantic_C
                rhs3 = rhs.ravel() + semantic_C.T @ semantic_t
                proposal = spsolve(H3, rhs3).reshape(-1,3)
            else:
                proposal = spsolve(H.tocsc(), rhs)
            if not np.isfinite(proposal).all():
                raise RuntimeError('Nonfinite student displacement proposal')
            proposal *= np.minimum(1., max_offset_m/np.maximum(np.linalg.norm(proposal, axis=1), 1e-12))[:, None]
            # Limit only the neighbourhood of a threatened triangle first.
            # Otherwise a tiny eyelid triangle can stall the entire face.
            direction = np.zeros_like(base)
            direction[ids] = proposal-delta
            if max_step_m is not None:
                direction *= np.minimum(1.,max_step_m/np.maximum(np.linalg.norm(direction,axis=1),1e-12))[:,None]
            for _ in range(20):
                trial = current+direction
                projection = np.einsum('ij,ij->i', _cross(trial, faces), cross0)
                lengths = np.linalg.norm(trial[edges[:, 0]]-trial[edges[:, 1]], axis=1)
                bad_faces = valid_faces & (projection <= .1*area0**2)
                bad_edges = (lengths < .25*edge0) | (lengths > 4.*edge0+1e-12)
                guarded_bad=shape_guard.inspect(trial)[0] if shape_guard is not None else np.array([],dtype=int)
                if not bad_faces.any() and not bad_edges.any() and not len(guarded_bad):
                    break
                blocked = np.unique(np.r_[faces[bad_faces].ravel(), edges[bad_edges].ravel(),guarded_bad])
                direction[blocked] *= .5
            proposal = delta+direction[ids]
            def energy(d):
                value = np.sum(w[:, None]*(A@d-target)**2) + np.sum(d*(regularizer@d))
                if semantic_C is not None:
                    value += np.sum((semantic_C @ d.ravel()-semantic_t)**2)
                return float(value)
            initial_energy = energy(delta)
            alpha = 1.
            accepted = False
            for _ in range(16):
                candidate_delta = delta + alpha*(proposal-delta)
                candidate = base.copy()
                candidate[ids] += candidate_delta
                crosses = _cross(candidate, faces)
                # Conservative orientation and collapse guards relative to the
                # input; not a global self-intersection guarantee.
                projection = np.einsum('ij,ij->i', crosses, cross0)
                lengths = np.linalg.norm(candidate[edges[:, 0]]-candidate[edges[:, 1]], axis=1)
                safe = (np.all(projection[valid_faces] > .1*area0[valid_faces]**2)
                        and np.all(lengths >= .25*edge0) and np.all(lengths <= 4.*edge0 + 1e-12))
                if shape_guard is not None:
                    safe = safe and not len(shape_guard.inspect(candidate)[0])
                if safe and np.isfinite(candidate).all() and energy(candidate_delta) <= initial_energy+1e-15:
                    accepted = True
                    break
                alpha *= .5
            step = float(np.max(np.linalg.norm(candidate_delta-delta, axis=1))) if accepted else 0.
            if accepted:
                delta, current = candidate_delta, candidate
                ever[np.unique(triangles[keep])] = True
            changed = int(np.count_nonzero(triangle_ids != previous_ids)) if previous_ids is not None else 0
            previous_ids = triangle_ids.copy()
            rmse_m = float(np.sqrt(np.mean(distance[keep]**2)))
            step_limit = early_stop.step_tol_m if early_stop is not None else 1e-6
            inner_reason = "max_inner"
            if not accepted:
                inner_reason = "rejected"
            elif step < step_limit:
                inner_reason = "step_tol"
            elif rmse_guard is not None and rmse_guard.step(rmse_m):
                inner_reason = "rmse_plateau"
            info['history'].append(dict(stage=stage, iteration=iteration, smooth_weight=smooth,
                                        correspondences=len(rows), changed_triangles=changed,
                                        rmse_m=rmse_m, step_m=step,
                                        step_scale=alpha if accepted else 0.,
                                        reason=inner_reason))
            if inner_reason == "step_tol":
                stop_reason = "step_tol"
                if early_stop is not None:
                    stage_done = True
                break
            if inner_reason in ("rejected", "rmse_plateau"):
                stop_reason = inner_reason
                break
        if stage_done:
            break
    if early_stop is not None:
        info['student_early_stop'] = as_report(
            early_stop, iters_run=len(info['history']),
            stopped=stop_reason or "max_iters")
    if fairing_strength>0:
        from core.displacement_fairing import fair_displacement
        # Freeze every vertex supporting a semantic equation, not merely the
        # original FLAME vertices: material landmarks stay exactly at student output.
        protected=np.zeros(len(ids),bool)
        if semantic_C is not None:
            protected=np.asarray(semantic_C.getnnz(axis=0)).reshape(-1,3).any(axis=1)
        current,info['displacement_fairing']=fair_displacement(
            base,current,ids,L,protected,shape_guard,strength=fairing_strength,
            max_change_m=fairing_max_change_m,max_total_m=max_offset_m)
    info.update(status='ok' if ever.any() else 'no_correspondences',
                n_constrained=int((ever & active).sum()),
                mean_move_m=float(np.linalg.norm(current-base, axis=1).mean()),
                max_move_m=float(np.linalg.norm(current-base, axis=1).max()),
                elapsed_s=time.perf_counter()-started)
    if shape_guard is not None:
        bad,info['quality']=shape_guard.inspect(current)
        if len(bad): raise RuntimeError('Student update violated reference geometry bounds')
    return current, info


def depth_inputs(vertices, faces, depth, camera, parsing, landmarks_2d=None, pixel_stride=2):
    """Build reliable camera points and a visible semantic vertex mask."""
    from core.student.detail_snap import _project_model_to_uv, _sample_labels, _vertex_normals, SNAP_LABELS
    from core.student.depth_refine import depth_observation, fragile_pixel_mask
    depth, parsing = np.asarray(depth), np.asarray(parsing)
    if depth.ndim != 2 or depth.shape != parsing.shape or min(depth.shape) < 5:
        raise ValueError('Depth and parsing must be matching images at least 5x5')
    if pixel_stride < 1 or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError('Invalid depth or sampling stride')
    if not np.isfinite([camera.fx, camera.fy, camera.cx, camera.cy]).all() or min(camera.fx, camera.fy) <= 0:
        raise ValueError('Invalid camera intrinsics')
    vertices = np.asarray(vertices, float)
    fragile = fragile_pixel_mask(parsing, depth, landmarks_2d)
    yy, xx = np.mgrid[2:depth.shape[0]-2:pixel_stride, 2:depth.shape[1]-2:pixel_stride]
    keep = np.isin(parsing[yy, xx], SNAP_LABELS) & ~fragile[yy, xx] & (depth[yy, xx] > 0)
    uv = np.column_stack((xx[keep], yy[keep]))
    z, confidence = depth_observation(uv, depth)
    points = np.column_stack(((uv[:, 0]-camera.cx)*z/camera.fx,
                              -(uv[:, 1]-camera.cy)*z/camera.fy, -z))
    good = confidence > .05
    uv0, z0 = _project_model_to_uv(vertices, camera)
    labels, inside = _sample_labels(uv0, parsing)
    normals = _vertex_normals(vertices, faces)
    view = -vertices / np.maximum(np.linalg.norm(vertices, axis=1, keepdims=True), 1e-12)
    # Fragile depth is excluded from DATA, not made into a hard ring in skin.
    # Visible skin holes inherit a regularized displacement from surrounding
    # observations. Actual eyes/mouth interior/back side stay exactly fixed.
    active = (inside & (z0 > 0) & (vertices[:, 2] < 0) & np.isin(labels, SNAP_LABELS)
              & ((normals*view).sum(1) > .25))
    ids = np.flatnonzero(active)
    if len(ids):
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        locations, ray_ids, _ = mesh.ray.intersects_location(np.zeros((len(ids), 3)), -view[ids], multiple_hits=False)
        visible = np.zeros(len(ids), bool)
        visible[ray_ids] = np.linalg.norm(locations-vertices[ids[ray_ids]], axis=1) < .001
        active[ids[~visible]] = False
    return points[good], confidence[good], active


def refine_depth(vertices, faces, depth, camera, parsing, landmarks_2d=None, **kwargs):
    started = time.perf_counter()
    points, confidence, active = depth_inputs(vertices, faces, depth, camera, parsing, landmarks_2d)
    result, info = refine_pointcloud(vertices, faces, points, active=active, confidence=confidence, **kwargs)
    info['elapsed_s'] = time.perf_counter()-started
    return result, info
