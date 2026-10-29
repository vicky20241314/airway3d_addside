"""Fast FLAME-to-depth snap: move face vertices along normals toward RGB-D."""
from __future__ import annotations

import time

import numpy as np
import trimesh

from core.student.camera import backprojection_inverse, coordinate_transform


# Skin / brows / nose / outer lips. Skip eyes, ears, neck, hair.
SNAP_LABELS = (1, 2, 3, 10, 12, 13)


def _adjacency(faces, n_verts):
    neighbors = [set() for _ in range(n_verts)]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.fromiter(nb, dtype=np.int64) for nb in neighbors]


def _neighbor_mean(values, neighbors):
    out = np.empty_like(values)
    for i, nb in enumerate(neighbors):
        out[i] = values[nb].mean() if nb.size else values[i]
    return out


def _vertex_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(verts)
    np.add.at(normals, faces[:, 0], face_n)
    np.add.at(normals, faces[:, 1], face_n)
    np.add.at(normals, faces[:, 2], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(length, 1e-12, None)


def _project_model_to_uv(points, camera):
    points_cam = np.stack(
        [points[..., 0], -points[..., 1], np.clip(-points[..., 2], 1e-6, None)],
        axis=-1,
    )
    uv = backprojection_inverse(points_cam, camera)
    return np.asarray(uv, dtype=np.float64), points_cam[..., 2]


def _sample_depth_m(uv, depth):
    """Bilinear depth in meters; invalid samples are 0."""
    u = uv[:, 0]
    v = uv[:, 1]
    h, w = depth.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    in_bounds = (u0 >= 0) & (u1 < w) & (v0 >= 0) & (v1 < h)
    z = np.zeros(len(uv), dtype=np.float64)
    idx = np.flatnonzero(in_bounds)
    if idx.size == 0:
        return z, np.zeros(len(uv), dtype=bool)

    wu = u[idx] - u0[idx]
    wv = v[idx] - v0[idx]
    d00 = depth[v0[idx], u0[idx]].astype(np.float64)
    d01 = depth[v0[idx], u1[idx]].astype(np.float64)
    d10 = depth[v1[idx], u0[idx]].astype(np.float64)
    d11 = depth[v1[idx], u1[idx]].astype(np.float64)
    m00 = d00 > 0
    m01 = d01 > 0
    m10 = d10 > 0
    m11 = d11 > 0
    w00 = (1.0 - wu) * (1.0 - wv) * m00
    w01 = wu * (1.0 - wv) * m01
    w10 = (1.0 - wu) * wv * m10
    w11 = wu * wv * m11
    w_sum = w00 + w01 + w10 + w11
    good = w_sum > 0
    d_sum = w00 * d00 + w01 * d01 + w10 * d10 + w11 * d11
    z[idx[good]] = d_sum[good] / w_sum[good] / 1000.0
    return z, z > 0


def _sample_labels(uv, parsing):
    h, w = parsing.shape[:2]
    ui = np.rint(uv[:, 0]).astype(np.int32)
    vi = np.rint(uv[:, 1]).astype(np.int32)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    labels = np.zeros(len(uv), dtype=parsing.dtype)
    labels[inside] = parsing[vi[inside], ui[inside]]
    return labels, inside


def snap_vertices_to_depth_legacy(
    verts,
    faces,
    depth,
    camera,
    parsing,
    n_iters=8,
    smooth_iters=10,
    step_m=0.004,
    max_residual_m=0.02,
    max_total_m=0.012,
    facing_min=0.25,
    labels=SNAP_LABELS,
):
    """
    Displace vertices toward the depth surface along normals.

    Only vertices that project onto facial parsing labels are moved.
    Returns (new_verts, info).
    """
    verts = np.asarray(verts, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    orig = verts.copy()
    neighbors = _adjacency(faces, len(verts))
    label_set = np.asarray(labels, dtype=parsing.dtype)

    uv0, _ = _project_model_to_uv(verts, camera)
    depth0, depth_ok0 = _sample_depth_m(uv0, depth)
    labels0, inside0 = _sample_labels(uv0, parsing)
    nrm0 = _vertex_normals(verts, faces)
    p0 = coordinate_transform(np.stack([
        (uv0[:, 0] - camera.cx) * depth0 / camera.fx,
        (uv0[:, 1] - camera.cy) * depth0 / camera.fy,
        depth0,
    ], axis=1).astype(np.float64))
    view0 = -verts
    view0 /= np.clip(np.linalg.norm(view0, axis=1, keepdims=True), 1e-12, None)
    facing0 = (nrm0 * view0).sum(axis=1) > facing_min
    ok0 = inside0 & depth_ok0 & np.isin(labels0, label_set) & facing0
    residual0 = ((p0 - verts) * nrm0).sum(axis=1)
    ok0 = ok0 & (np.abs(residual0) < max_residual_m)
    rms_before = float(np.sqrt(np.mean(residual0[ok0] ** 2))) if ok0.any() else 0.0

    for _ in range(int(n_iters)):
        nrm = _vertex_normals(verts, faces)
        uv, _z_cam = _project_model_to_uv(verts, camera)
        depth_m, depth_ok = _sample_depth_m(uv, depth)
        lab, inside = _sample_labels(uv, parsing)
        view = -verts
        view /= np.clip(np.linalg.norm(view, axis=1, keepdims=True), 1e-12, None)
        facing = (nrm * view).sum(axis=1) > facing_min
        p = coordinate_transform(np.stack([
            (uv[:, 0] - camera.cx) * depth_m / camera.fx,
            (uv[:, 1] - camera.cy) * depth_m / camera.fy,
            depth_m,
        ], axis=1).astype(np.float64))
        residual = ((p - verts) * nrm).sum(axis=1)
        ok = (
            inside
            & depth_ok
            & np.isin(lab, label_set)
            & facing
            & (np.abs(residual) < max_residual_m)
        )
        disp = np.zeros(len(verts), dtype=np.float64)
        disp[ok] = np.clip(residual[ok], -step_m, step_m)
        for _s in range(int(smooth_iters)):
            avg = _neighbor_mean(disp, neighbors)
            disp = np.where(ok, 0.55 * disp + 0.45 * avg, 0.0)
        verts = verts + nrm * disp[:, None]
        offset = verts - orig
        dist = np.linalg.norm(offset, axis=1)
        scale = np.minimum(1.0, max_total_m / np.clip(dist, 1e-12, None))
        verts = orig + offset * scale[:, None]

    nrm = _vertex_normals(verts, faces)
    uv, _ = _project_model_to_uv(verts, camera)
    depth_m, depth_ok = _sample_depth_m(uv, depth)
    lab, inside = _sample_labels(uv, parsing)
    view = -verts
    view /= np.clip(np.linalg.norm(view, axis=1, keepdims=True), 1e-12, None)
    facing = (nrm * view).sum(axis=1) > facing_min
    p = coordinate_transform(np.stack([
        (uv[:, 0] - camera.cx) * depth_m / camera.fx,
        (uv[:, 1] - camera.cy) * depth_m / camera.fy,
        depth_m,
    ], axis=1).astype(np.float64))
    residual = ((p - verts) * nrm).sum(axis=1)
    ok = inside & depth_ok & np.isin(lab, label_set) & facing & (np.abs(residual) < max_residual_m)
    rms_after = float(np.sqrt(np.mean(residual[ok] ** 2))) if ok.any() else 0.0
    mean_move = float(np.linalg.norm(verts - orig, axis=1).mean())
    info = {
        "n_constrained": int(ok0.sum()),
        "rms_before_m": rms_before,
        "rms_after_m": rms_after,
        "mean_move_m": mean_move,
    }
    return verts, info


def snap_vertices_to_depth(verts, faces, depth, camera, parsing, method="offset", **kwargs):
    if method == "student":
        from core.student.student_refine import refine_depth
        return refine_depth(verts, faces, depth, camera, parsing, **kwargs)
    if method == "arap":
        from core.student.arap_refine import refine_depth
        return refine_depth(verts, faces, depth, camera, parsing, **kwargs)
    if method == "offset":
        from core.student.offset_refine import refine_depth
        return refine_depth(verts, faces, depth, camera, parsing, **kwargs)
    if method == "legacy":
        kwargs.pop("landmarks_2d", None)
        return snap_vertices_to_depth_legacy(verts, faces, depth, camera, parsing, **kwargs)
    if method != "robust":
        raise ValueError(f"Unknown depth refinement method: {method}")
    from core.student.depth_refine import refine_vertices
    return refine_vertices(verts, faces, depth, camera, parsing, **kwargs)


def snap_mesh_to_depth(mesh, depth, camera, parsing, **kwargs):
    """Return a new refined mesh; preserve the input mesh and FLAME metric metadata."""
    t0 = time.time()
    verts, info = snap_vertices_to_depth(
        mesh.vertices, mesh.faces, depth, camera, parsing, **kwargs
    )
    info["elapsed_s"] = time.time() - t0
    snapped = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
    if getattr(mesh, "metadata", None):
        snapped.metadata.update(mesh.metadata)
    if info.get('semantic') is not None and 'flame_landmarks_3d' in snapped.metadata:
        from core.student.arap_refine import landmark_matrix
        landmarks=np.asarray(snapped.metadata['flame_landmarks_3d']).copy()
        if landmarks.shape == (68,3):
            landmarks[17:]=landmark_matrix(mesh.faces,len(verts)) @ verts
            snapped.metadata['flame_landmarks_3d']=landmarks
    return snapped, info


def snap_closed_open_meshes(
    mesh_closed,
    mesh_open,
    depth_closed,
    depth_open,
    camera,
    parsing_closed,
    parsing_open=None,
    lmk2d_closed=None,
    lmk2d_open=None,
    method="offset",
):
    """Snap closed always; snap open if parsing is provided."""
    mesh_closed, info_c = snap_mesh_to_depth(
        mesh_closed, depth_closed, camera, parsing_closed,
        landmarks_2d=lmk2d_closed,
        method=method,
    )
    info_o = None
    if mesh_open is not None and parsing_open is not None and depth_open is not None:
        mesh_open, info_o = snap_mesh_to_depth(
            mesh_open, depth_open, camera, parsing_open,
            landmarks_2d=lmk2d_open,
            method=method,
        )
    return mesh_closed, mesh_open, info_c, info_o
