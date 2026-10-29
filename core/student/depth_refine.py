"""Confidence-weighted sparse depth refinement; input meters, depth millimeters."""
import numpy as np
import cv2
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve


def laplacian(vertices, faces, normals):
    edges = trimesh.geometry.faces_to_edges(faces)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    i, j = edges.T
    # Reduce coupling across sharp changes; do not smooth vertex positions directly.
    weight = np.maximum((normals[i] * normals[j]).sum(1), 0.05) ** 2
    adjacency = sparse.coo_matrix((np.r_[weight, weight],
                                  (np.r_[i, j], np.r_[j, i])),
                                 shape=(len(vertices), len(vertices))).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    return sparse.eye(len(vertices), format="csr") - sparse.diags(1 / np.maximum(degree, 1e-12)) @ adjacency


def depth_observation(uv, depth):
    """Strict support and edge confidence: never interpolate through missing pixels."""
    h, w = depth.shape
    pixel = np.rint(uv).astype(int)
    inside = (pixel[:, 0] >= 2) & (pixel[:, 0] < w-2) & (pixel[:, 1] >= 2) & (pixel[:, 1] < h-2)
    x, y = np.clip(pixel[:, 0], 2, w-3), np.clip(pixel[:, 1], 2, h-3)
    patches = np.stack([depth[y+dy, x+dx] for dy in range(-2, 3) for dx in range(-2, 3)], axis=1).astype(float)
    valid = patches > 0
    center = patches[:, 12]
    median = np.median(patches, axis=1)
    offsets = np.array([(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)])
    plane = np.column_stack((offsets, np.ones(25)))
    coefficients = patches @ np.linalg.pinv(plane).T
    fitted = coefficients @ plane.T
    detrended = patches - fitted
    # A steep but smooth nose surface is not a depth discontinuity.
    spread = np.percentile(detrended, 90, axis=1) - np.percentile(detrended, 10, axis=1)
    n_valid = valid.sum(1)
    raw_span = np.maximum(np.where(valid, patches, -np.inf).max(1)
                          - np.where(valid, patches, np.inf).min(1), 0.)
    raw_span[~np.isfinite(raw_span)] = 0.
    confidence = inside * (center > 0) * (n_valid >= 22) * (np.abs(center - median) < 5.)
    confidence = confidence * (n_valid / 25.0) * np.exp(-(spread / 4.) ** 2)
    confidence[spread > 10.] = 0
    # Nostril / canthus holes look like 2–5 cm jumps inside one patch.
    confidence[raw_span > 18.] = 0
    # Range-preserving bilateral weights within each patch, centered on the median.
    weights = np.exp(-np.sum(offsets ** 2, axis=1)[None] / 4.) * np.exp(-((patches-median[:, None])/3.)**2) * valid
    z = (weights * patches).sum(1) / np.maximum(weights.sum(1), 1e-12)
    # Preserve subpixel projection: rounding the sampling position creates a
    # staircase on steep nose surfaces even when the depth itself is smooth.
    fraction = uv - pixel
    use_subpixel = raw_span <= 10.
    z = np.where(use_subpixel, z + np.sum(coefficients[:, :2] * fraction, axis=1), z)
    z /= 1000.
    return z, confidence


CANTHUS_IDS = (36, 39, 42, 45)
NOSTRIL_IDS = (31, 32, 33, 34, 35)


def _landmark_disks(shape, landmarks_2d):
    """Disks on outer/inner canthi and nostril landmarks; radii scale with IOD."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if landmarks_2d is None:
        return mask
    lmk = np.asarray(landmarks_2d, dtype=float)
    if lmk.ndim != 2 or lmk.shape[0] < 46:
        return mask
    used = lmk[list(CANTHUS_IDS + NOSTRIL_IDS)]
    if not np.isfinite(used).all():
        return mask
    iod = float(np.linalg.norm(lmk[45] - lmk[36]))
    if not np.isfinite(iod) or iod < 8:
        iod = 80.0
    canthus_r = max(14, int(round(0.08 * iod)))
    nostril_r = max(12, int(round(0.07 * iod)))
    for i in CANTHUS_IDS:
        cv2.circle(mask, (int(round(lmk[i, 0])), int(round(lmk[i, 1]))), canthus_r, 1, -1)
    for i in NOSTRIL_IDS:
        cv2.circle(mask, (int(round(lmk[i, 0])), int(round(lmk[i, 1]))), nostril_r, 1, -1)
    return mask


def fragile_pixel_mask(parsing, depth, landmarks_2d=None):
    """Pixels around eyes and nostrils where TOF depth is not a face surface."""
    depth = np.asarray(depth)
    eyes = np.isin(parsing, (4, 5, 6)).astype(np.uint8)
    eyes = cv2.dilate(eyes, np.ones((15, 15), np.uint8))
    nose = (parsing == 10).astype(np.uint8)
    holes = (depth <= 0).astype(np.uint8)
    skin = np.isin(parsing, (1, 2, 3, 10))
    skin_depth = depth[skin & (depth > 0)]
    far = np.zeros(depth.shape, dtype=np.uint8)
    if skin_depth.size:
        far_limit = float(np.median(skin_depth)) + 20.0
        far[(nose > 0) & (depth > far_limit)] = 1
    d = depth.astype(np.float32)
    span = cv2.dilate(np.where(d > 0, d, 0), np.ones((7, 7), np.uint8))
    span = span - cv2.erode(np.where(d > 0, d, 1e6), np.ones((7, 7), np.uint8))
    noisy_nose = (nose > 0) & (span > 20)
    hole_band = cv2.dilate((holes | far | noisy_nose.astype(np.uint8)), np.ones((9, 9), np.uint8))
    disks = _landmark_disks(depth.shape, landmarks_2d)
    if disks.any():
        local = cv2.dilate(disks, np.ones((5, 5), np.uint8))
        return (eyes > 0) | (disks > 0) | ((hole_band > 0) & (local > 0))
    nose_band = cv2.dilate(nose, np.ones((11, 11), np.uint8))
    return (eyes > 0) | ((hole_band > 0) & (nose_band > 0))


def _sample_mask(uv, mask):
    h, w = mask.shape[:2]
    ui = np.rint(uv[:, 0]).astype(np.int32)
    vi = np.rint(uv[:, 1]).astype(np.int32)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros(len(uv), dtype=bool)
    out[inside] = mask[vi[inside], ui[inside]]
    return out


def _grow_vertex_mask(mask, average, rings=2):
    grown = mask.astype(np.float64)
    for _ in range(rings):
        grown = np.maximum(grown, (average @ grown) > 1e-12)
    return grown > 0


def refine_vertices(verts, faces, depth, camera, parsing, n_iters=12,
                    max_total_m=.012, max_residual_m=.020,
                    smooth_weight=.08, facing_min=.25, step_smoothing=1,
                    step_m=.004, landmarks_2d=None):
    from core.student.detail_snap import _vertex_normals, _project_model_to_uv, _sample_labels, SNAP_LABELS
    orig = np.asarray(verts, dtype=float).copy()
    faces = np.asarray(faces, dtype=np.int64)
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.shape != parsing.shape or min(depth.shape) < 5:
        raise ValueError("Depth and parsing must be matching images at least 5x5")
    if (not np.isfinite(orig).all() or n_iters < 1 or max_total_m <= 0
            or smooth_weight < 0 or step_smoothing < 0 or step_m <= 0):
        raise ValueError("Invalid refinement inputs/settings")
    normal0 = _vertex_normals(orig, faces)
    uv0, _ = _project_model_to_uv(orig, camera)
    labels0, inside0 = _sample_labels(uv0, parsing)
    fragile = fragile_pixel_mask(parsing, depth, landmarks_2d)
    view = -orig / np.maximum(np.linalg.norm(orig, axis=1, keepdims=True), 1e-12)
    facing = (view * normal0).sum(1)
    L = laplacian(orig, faces, normal0)
    average = sparse.eye(len(orig), format="csr") - L
    protect = _sample_mask(uv0, fragile) | ((labels0 == 10) & (facing < 0.55))
    seed = protect & np.isin(labels0, SNAP_LABELS)
    protect = protect | _grow_vertex_mask(seed, average, rings=1)
    active = (inside0 & (-orig[:,2] > 0) & np.isin(labels0, SNAP_LABELS)
              & (facing > facing_min) & ~protect)
    # Only vertices visible from the depth camera can receive observations.
    active_ids = np.flatnonzero(active)
    if active_ids.size:
        mesh = trimesh.Trimesh(vertices=orig, faces=faces, process=False)
        rays = -view[active_ids]
        locations, ray_ids, _ = mesh.ray.intersects_location(np.zeros_like(rays), rays, multiple_hits=False)
        visible = np.zeros(len(active_ids), dtype=bool)
        visible[ray_ids] = np.linalg.norm(locations-orig[active_ids[ray_ids]], axis=1) < .001
        active[active_ids[~visible]] = False
    active_ids = np.flatnonzero(active)
    current = orig.copy()
    z0 = np.zeros(len(orig))
    conf0 = np.zeros(len(orig))
    if active_ids.size:
        z0[active_ids], conf0[active_ids] = depth_observation(uv0[active_ids], depth)
    observed0 = np.column_stack(((uv0[:,0]-camera.cx)*z0/camera.fx,
                                 -(uv0[:,1]-camera.cy)*z0/camera.fy, -z0))
    initial_residual = ((observed0-orig)*normal0).sum(1)
    supported = active & (conf0 > .05) & (np.abs(initial_residual) < max_residual_m)
    ever = np.zeros(len(orig), dtype=bool)
    peak_conf = conf0.copy()
    for _ in range(n_iters):
        normal = _vertex_normals(current, faces)
        uv, _ = _project_model_to_uv(current, camera)
        z = np.zeros(len(orig))
        confidence = np.zeros(len(orig))
        ids = np.flatnonzero(active)
        if ids.size:
            z[ids], confidence[ids] = depth_observation(uv[ids], depth)
        lab, inside = _sample_labels(uv, parsing)
        observed = np.column_stack(((uv[:,0]-camera.cx)*z/camera.fx,
                                     -(uv[:,1]-camera.cy)*z/camera.fy, -z))
        residual = ((observed-current)*normal).sum(1)
        ok = (active & inside & np.isin(lab, SNAP_LABELS) & ~_sample_mask(uv, fragile)
              & (confidence > .01) & (np.abs(residual) < max_residual_m))
        ever |= ok
        peak_conf = np.maximum(peak_conf, np.where(ok, confidence, 0.))
        step = np.where(ok, np.clip(residual, -step_m, step_m) * (0.5 + 0.5 * confidence), 0.)
        # Sparse matrix multiplication replaces 5k Python loops per smoothing pass.
        for _ in range(step_smoothing):
            step = np.where(ok, .55*step+.45*(average@step), 0.)
        current += normal*step[:,None]
        offset = current-orig
        current = orig + offset*np.minimum(1.,max_total_m/np.maximum(np.linalg.norm(offset,axis=1),1e-12))[:,None]
    ids = np.flatnonzero(active)
    if ids.size and ever.any():
        # Smooth the accumulated VECTOR displacement, preserving baseline detail.
        sub = L[:,ids]
        # Observed verts keep their fit; holes in visible skin get a weak fill.
        data_weight = np.where(ever[ids], np.clip(peak_conf[ids], 0.2, 1.0), 0.02)
        system = sparse.diags(data_weight) + smooth_weight*(sub.T@sub)
        offset = np.zeros_like(orig)
        offset[ids] = spsolve(system.tocsc(),data_weight[:,None]*(current-orig)[ids])
        offset *= np.minimum(1.,max_total_m/np.maximum(np.linalg.norm(offset,axis=1),1e-12))[:,None]
        current = orig+offset
    # Keep canthi / nostrils on FLAME. Do not blend toward snapped neighbors.
    current[protect] = orig[protect]
    face_protect = protect & np.isin(labels0, SNAP_LABELS)
    if face_protect.any() and len(orig) > 100:
        smoothed = orig.copy()
        for _ in range(2):
            smoothed[face_protect] = 0.65 * smoothed[face_protect] + 0.35 * (average @ smoothed)[face_protect]
        current[face_protect] = smoothed[face_protect]
        seam = (~protect) & _grow_vertex_mask(protect, average, rings=1)
        current[seam] = 0.55 * current[seam] + 0.45 * orig[seam]
    rms = lambda x: float(np.sqrt(np.mean(x[supported]**2))) if supported.any() else 0.
    final_fixed = ((observed0-current)*normal0).sum(1)
    return current, {"method":"robust_sparse", "n_constrained":int(supported.sum()),
                     "n_protected":int(protect.sum()),
                     "rms_before_m":rms(initial_residual), "rms_after_m":rms(final_fixed),
                     "residual_definition":"fixed initial observations and normals; same supported set",
                     "mean_move_m":float(np.linalg.norm(current-orig,axis=1).mean())}
