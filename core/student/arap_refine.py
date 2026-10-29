"""Single-frame RGB-D adaptation of ARAP (Sorkine & Alexa, SGP 2007).

Local rotations / global sparse positions are alternated inside a nonrigid ICP
loop. Projective point-to-plane matches and FLAME landmark reprojection are
task-specific additions, not a reproduction of an entire published system.
Geometry is in metres; camera depth is in millimetres.
"""
from pathlib import Path
import hashlib
import pickle
import time

import cv2
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve

from core.student.offset_refine import _operators, _cross, depth_inputs
from core.student.detail_snap import _project_model_to_uv, _sample_labels, _vertex_normals, SNAP_LABELS
from core.student.depth_refine import depth_observation, fragile_pixel_mask, _sample_mask


FLAME_FACE_SHA256 = 'c495737954d547081aa5134549ad04f38e5c25c55729cb6527b6a3821e097576'


def landmark_matrix(faces, n_vertices, embedding_path=None):
    """Use the project's 51 static FLAME landmarks (iBUG indices 17..67).

    Refuse a different topology rather than attaching anatomical constraints to
    arbitrary vertices. Custom topologies can call the method without landmarks.
    """
    if n_vertices != 5023 or hashlib.sha256(np.asarray(faces, dtype='<i8').tobytes()).hexdigest() != FLAME_FACE_SHA256:
        raise ValueError('Landmark constraints require the original project FLAME topology')
    path = embedding_path or Path(__file__).resolve().parents[2]/'assets/flame_static_embedding.pkl'
    with open(path, 'rb') as stream:
        embedding = pickle.load(stream, encoding='latin1')
    fi = np.asarray(embedding['lmk_face_idx'], int)
    bary = np.asarray(embedding['lmk_b_coords'], float)
    if fi.shape != (51,) or bary.shape != (51, 3) or not np.isfinite(bary).all() or not np.allclose(bary.sum(1), 1):
        raise ValueError('Invalid static FLAME embedding')
    return sparse.coo_matrix((bary.ravel(), (np.repeat(np.arange(51), 3), faces[fi].ravel())),
                             shape=(51, n_vertices)).tocsr()


def edge_weights(vertices, faces, edges):
    """Positive cotangent weights, bounded for small/obtuse scan triangles."""
    weights = {}
    tri = vertices[faces]
    for k in range(3):
        a, b = (k+1) % 3, (k+2) % 3
        u, v = tri[:, a]-tri[:, k], tri[:, b]-tri[:, k]
        cot = np.einsum('ij,ij->i', u, v)/np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
        for edge, value in zip(np.sort(faces[:, [a, b]], axis=1), .5*cot):
            key = tuple(edge)
            weights[key] = weights.get(key, 0.)+value
    result = np.clip([weights[tuple(edge)] for edge in edges], .05, 10.)
    return result/np.mean(result)


def local_rotations(base, current, edges, weights):
    """Weighted polar decomposition; reflection corrected to SO(3)."""
    i, j = edges.T
    e0, e = base[i]-base[j], current[i]-current[j]
    outer = weights[:, None, None]*e[:, :, None]*e0[:, None, :]
    covariance = np.zeros((len(base), 3, 3))
    np.add.at(covariance, i, outer)
    np.add.at(covariance, j, outer)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.ones((len(base), 3))
    correction[:, -1] = np.linalg.det(u@vt)
    return (u*correction[:, None, :])@vt


def _reprojection(J, landmarks, camera):
    """Linear camera ray constraints x + a*z = 0, y - b*z = 0."""
    row, col = J.nonzero()
    data = np.asarray(J[row, col]).ravel()
    a = (landmarks[:, 0]-camera.cx)/camera.fx
    b = (landmarks[:, 1]-camera.cy)/camera.fy
    return sparse.coo_matrix((np.r_[data, data*a[row], data, -data*b[row]],
                              (np.r_[2*row, 2*row, 2*row+1, 2*row+1],
                               np.r_[3*col, 3*col+2, 3*col+1, 3*col+2])),
                             shape=(2*J.shape[0], 3*J.shape[1])).tocsr()


def mouth_contact_matrix(J, landmarks):
    """Preserve the baseline inner-lip gap only when the image is closed-mouth.

    This constrains relative displacements, not absolute lip positions. An open
    capture gets no contact constraint. It does not manufacture unseen depth.
    """
    upper, lower = np.array([61, 62, 63]), np.array([67, 66, 65])
    width = np.linalg.norm(landmarks[45]-landmarks[36])
    gap = np.mean(np.linalg.norm(landmarks[upper]-landmarks[lower], axis=1))
    if width < 8 or gap/width >= .035:
        return sparse.csr_matrix((0, 3*J.shape[1]))
    pairs = J[upper-17]-J[lower-17]
    return sparse.kron(pairs, sparse.eye(3), format='csr')


def _normal_map(depth, camera):
    z = cv2.bilateralFilter(depth.astype(np.float32)/1000, 5, .003, 2)
    dzdu = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)/8
    dzdv = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)/8
    v, u = np.indices(z.shape)
    a, b = (u-camera.cx)/camera.fx, (v-camera.cy)/camera.fy
    du = np.stack((z/camera.fx+a*dzdu, -b*dzdu, -dzdu), axis=-1)
    dv = np.stack((a*dzdv, -z/camera.fy-b*dzdv, -dzdv), axis=-1)
    normal = -np.cross(du, dv)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
    return normal


def _sample_operator(faces, n_vertices, active):
    ids = np.flatnonzero(active)
    selected = faces[np.any(active[faces], axis=1)]
    return sparse.vstack((sparse.eye(n_vertices, format='csr')[ids],
                          sparse.coo_matrix((np.full(selected.size, 1/3),
                                             (np.repeat(np.arange(len(selected)), 3), selected.ravel())),
                                            shape=(len(selected), n_vertices)).tocsr()), format='csr')


def refine_depth(vertices, faces, depth, camera, parsing, landmarks_2d=None,
                 stiffness=(8., 3., 1.), iterations_per_stage=6,
                 landmark_weight=12., max_offset_m=.015, max_distance_m=.020,
                 normal_cos=.5, huber_m=.002, point_weight=.08):
    started = time.perf_counter()
    base = np.asarray(vertices, float).copy()
    faces = np.asarray(faces, np.int64)
    if (base.ndim != 2 or base.shape[1] != 3 or len(base) == 0 or not np.isfinite(base).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.min() < 0 or faces.max() >= len(base)):
        raise ValueError('Invalid mesh')
    if (len(stiffness) == 0 or not np.isfinite(stiffness).all() or min(stiffness) <= 0
            or iterations_per_stage < 1 or int(iterations_per_stage) != iterations_per_stage
            or not np.isfinite([landmark_weight, max_offset_m, max_distance_m, normal_cos, huber_m, point_weight]).all()
            or min(max_offset_m, max_distance_m, huber_m) <= 0 or landmark_weight < 0
            or not 0 <= normal_cos <= 1 or not 0 <= point_weight <= 1):
        raise ValueError('Invalid ARAP settings')
    # Reuse conservative input validation/visibility; this does not run offset ICP.
    cloud, _, active = depth_inputs(base, faces, depth, camera, parsing, landmarks_2d)
    info = dict(method='arap_projective', source='Sorkine & Alexa, SGP 2007; RGB-D adaptation',
                history=[], stiffness=list(stiffness), n_landmarks=0)
    if not len(cloud) or not active.any():
        info.update(status='no_observations', elapsed_s=time.perf_counter()-started)
        return base, info
    L, E, edges = _operators(base, faces)
    average = sparse.eye(len(base))-L
    # Let a short surface band (including eyelids) follow observed skin; the
    # eye itself never receives depth constraints. ARAP preserves its shape.
    for _ in range(3):
        active |= np.asarray(average@active.astype(float)).ravel() > 0
    uv0, _ = _project_model_to_uv(base, camera)
    labels0, inside0 = _sample_labels(uv0, parsing)
    active &= inside0 & (base[:, 2] < 0)
    J = None
    contact = sparse.csr_matrix((0, 3*len(base)))
    if landmarks_2d is not None:
        lmk = np.asarray(landmarks_2d, float)
        if lmk.shape != (68, 2) or not np.isfinite(lmk).all():
            raise ValueError('Expected finite 68x2 landmarks')
        Jall = landmark_matrix(faces, len(base))
        contact = mouth_contact_matrix(Jall, lmk)
        model_uv, _ = _project_model_to_uv(Jall@base, camera)
        good = np.linalg.norm(model_uv-lmk[17:], axis=1) < 40
        good &= (lmk[17:, 0] >= 0) & (lmk[17:, 0] < depth.shape[1]) & (lmk[17:, 1] >= 0) & (lmk[17:, 1] < depth.shape[0])
        J = Jall[good]
        target_uv = lmk[17:][good]
        info['n_landmarks'] = int(good.sum())
        # Enable landmark vertices so their own constraints are not fixed away.
        active[np.unique(J.indices)] = True
    ids = np.flatnonzero(active)
    dofs = (3*ids[:, None]+np.arange(3)).ravel()
    B = _sample_operator(faces, len(base), active)
    B0 = B@base
    weights = edge_weights(base, faces, edges)
    # Preserve smaller high-curvature regions more strongly than broad skin.
    fragile = fragile_pixel_mask(parsing, depth, landmarks_2d)
    feature = _sample_mask(uv0, fragile).astype(float)
    weights *= 1+3*np.maximum(feature[edges[:, 0]], feature[edges[:, 1]])
    Esub = E[:, ids]
    K = Esub.T@sparse.diags(weights)@Esub
    normal_map = _normal_map(depth, camera)
    cross0 = _cross(base, faces)
    area0 = np.linalg.norm(cross0, axis=1)
    valid_faces = area0 > 1e-14
    length0 = np.linalg.norm(base[edges[:, 0]]-base[edges[:, 1]], axis=1)
    base_edges = E@base
    C = _reprojection(J, target_uv, camera) if J is not None else sparse.csr_matrix((0, 3*len(base)))
    Csub, landmark_rhs = C[:, dofs], -C@base.ravel()
    anchor = .002*sparse.eye(3*len(ids), format='csr')
    landmark_H = landmark_weight*(Csub.T@Csub)
    contact_sub = contact[:, dofs]
    landmark_H += 150.*(contact_sub.T@contact_sub)
    info['closed_mouth_contact'] = bool(contact.shape[0])
    landmark_b = landmark_weight*(Csub.T@landmark_rhs)
    delta = np.zeros((len(ids), 3))
    current = base.copy()
    previous_pixels = None
    for stage, strength in enumerate(stiffness):
        regularizer = strength*sparse.kron(K, sparse.eye(3), format='csr')+anchor+landmark_H
        for iteration in range(iterations_per_stage):
            sampled = B@current
            uv, _ = _project_model_to_uv(sampled, camera)
            z, confidence = depth_observation(uv, depth)
            labels, inside = _sample_labels(uv, parsing)
            observed = np.column_stack(((uv[:, 0]-camera.cx)*z/camera.fx,
                                        -(uv[:, 1]-camera.cy)*z/camera.fy, -z))
            pixel = np.rint(uv).astype(int)
            x, y = np.clip(pixel[:, 0], 0, depth.shape[1]-1), np.clip(pixel[:, 1], 0, depth.shape[0]-1)
            normals = normal_map[y, x]
            model_normals = B@_vertex_normals(current, faces)
            model_normals /= np.maximum(np.linalg.norm(model_normals, axis=1, keepdims=True), 1e-12)
            compatible = np.einsum('ij,ij->i', normals, model_normals) > normal_cos
            distance = np.linalg.norm(observed-sampled, axis=1)
            keep = (inside & np.isin(labels, SNAP_LABELS) & (sampled[:, 2] < 0)
                    & ~_sample_mask(uv, fragile) & (confidence > .05)
                    & (distance < max_distance_m) & compatible)
            # Recompute visibility after every deformation (not only on FLAME).
            selected = np.flatnonzero(keep)
            if len(selected):
                directions = sampled[selected]/np.maximum(np.linalg.norm(sampled[selected], axis=1, keepdims=True), 1e-12)
                mesh = trimesh.Trimesh(current, faces, process=False)
                hit, ray_id, _ = mesh.ray.intersects_location(np.zeros_like(directions), directions, multiple_hits=False)
                visible = np.zeros(len(selected), bool)
                visible[ray_id] = np.linalg.norm(hit-sampled[selected[ray_id]], axis=1) < .0005
                keep[selected[~visible]] = False
            selected = np.flatnonzero(keep)
            if not len(selected):
                break
            A = sparse.kron(B[selected][:, ids], sparse.eye(3), format='csr')
            n = normals[selected]
            N = sparse.coo_matrix((n.ravel(), (np.repeat(np.arange(len(selected)), 3),
                                              np.arange(3*len(selected)))), shape=(len(selected), 3*len(selected))).tocsr()
            plane = N@A
            target = (observed[selected]-B0[selected]).ravel()
            residual = np.einsum('ij,ij->i', observed[selected]-sampled[selected], n)
            density = max(len(selected)/len(ids), 1.)
            w = confidence[selected]*np.minimum(1., huber_m/np.maximum(np.abs(residual), 1e-12))/density
            H = regularizer + plane.T@sparse.diags(w)@plane + point_weight*(A.T@sparse.diags(np.repeat(w, 3))@A)
            R = local_rotations(base, current, edges, weights)
            rotated_edges = np.einsum('ijk,ik->ij', .5*(R[edges[:, 0]]+R[edges[:, 1]]), base_edges)
            arap_target = rotated_edges-base_edges
            rhs = (plane.T@(w*(N@target)) + point_weight*(A.T@(np.repeat(w, 3)*target))
                   + strength*(Esub.T@(weights[:, None]*arap_target)).ravel()+landmark_b)
            proposal = spsolve(H.tocsc(), rhs).reshape(-1, 3)
            if not np.isfinite(proposal).all():
                raise RuntimeError('Nonfinite ARAP solve')
            proposal *= np.minimum(1., max_offset_m/np.maximum(np.linalg.norm(proposal, axis=1), 1e-12))[:, None]
            direction = np.zeros_like(base)
            direction[ids] = proposal-delta
            # More restrictive than a sign-only test; not a global collision test.
            def bad_vertices(v):
                cross = _cross(v, faces)
                area = np.linalg.norm(cross, axis=1)
                dot = np.einsum('ij,ij->i', cross, cross0)
                lengths = np.linalg.norm(v[edges[:, 0]]-v[edges[:, 1]], axis=1)
                bad_f = valid_faces & ((dot <= .25*area0**2) | (area > 2.5*area0))
                bad_e = (lengths < .5*length0) | (lengths > 2*length0+1e-12)
                return np.unique(np.r_[faces[bad_f].ravel(), edges[bad_e].ravel()])
            for _ in range(18):
                bad = bad_vertices(current+direction)
                if not len(bad):
                    break
                direction[bad] *= .5
            def quadratic(d):
                return .5*float(d@(H@d))-float(d@rhs)
            before = quadratic(delta.ravel())
            accepted = False
            for _ in range(16):
                candidate = current+direction
                new_delta = candidate[ids]-base[ids]
                if not len(bad_vertices(candidate)) and quadratic(new_delta.ravel()) <= before+1e-15:
                    accepted = True
                    break
                direction *= .5
            step = float(np.linalg.norm(direction, axis=1).max()) if accepted else 0.
            if accepted:
                current, delta = candidate, new_delta
            changed = int(np.any(pixel != previous_pixels, axis=1).sum()) if previous_pixels is not None else 0
            previous_pixels = pixel.copy()
            info['history'].append(dict(stage=stage, iteration=iteration, observations=len(selected),
                                        changed_projection_pixels=changed, max_step_m=step))
            if step < 1e-6:
                break
    info.update(status='ok' if info['history'] else 'no_correspondences', n_active=len(ids),
                max_move_m=float(np.linalg.norm(current-base, axis=1).max()),
                elapsed_s=time.perf_counter()-started)
    return current, info
