"""Frozen triangle correspondences for differentiable FLAME parameter fitting.

Closest-triangle via trimesh AABB is too slow for Stage2 ICP. Use k nearest
vertices plus their incident faces (same point-to-plane / surface idea as the
LM fitter and Bouaziz et al. 2013), staying on the tensor device.
"""
from __future__ import annotations

import numpy as np
import torch
import hashlib


_TABLE_CACHE = {}


def _faces_key(faces, n_verts):
    f = faces.detach()
    # View-specific visible face subsets may share size/first/middle/last
    # elements while differing elsewhere. Hash the complete topology.
    return (int(n_verts), hashlib.blake2b(f.cpu().numpy().tobytes(), digest_size=16).digest())


def vertex_triangle_table(faces, n_verts):
    """Packed [V, max_incident] triangle ids; unused slots are -1."""
    key = _faces_key(faces, n_verts)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached.to(device=faces.device)
    faces_np = np.asarray(faces.detach().cpu().numpy(), dtype=np.int64)
    counts = np.bincount(faces_np.ravel(), minlength=n_verts)
    max_k = int(max(int(counts.max()), 1))
    table = np.full((n_verts, max_k), -1, dtype=np.int64)
    slot = np.zeros(n_verts, dtype=np.int64)
    for tid, tri in enumerate(faces_np):
        for v in tri:
            table[v, slot[v]] = tid
            slot[v] += 1
    packed = torch.as_tensor(table, dtype=torch.long)
    if len(_TABLE_CACHE) > 16:
        _TABLE_CACHE.clear()
    _TABLE_CACHE[key] = packed
    return packed.to(device=faces.device)


def closest_point_on_triangles(p, a, b, c):
    """Ericson §5.1.5; p,a,b,c broadcast to [..., 3]."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1, keepdim=True)
    d2 = (ac * ap).sum(-1, keepdim=True)
    closest = a.clone()
    done = (d1 <= 0) & (d2 <= 0)

    bp = p - b
    d3 = (ab * bp).sum(-1, keepdim=True)
    d4 = (ac * bp).sum(-1, keepdim=True)
    region_b = (~done) & (d3 >= 0) & (d4 <= d3)
    closest = torch.where(region_b, b, closest)
    done = done | region_b

    cp = p - c
    d5 = (ab * cp).sum(-1, keepdim=True)
    d6 = (ac * cp).sum(-1, keepdim=True)
    region_c = (~done) & (d6 >= 0) & (d5 <= d6)
    closest = torch.where(region_c, c, closest)
    done = done | region_c

    vc = d1 * d4 - d3 * d2
    region_ab = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / (d1 - d3).clamp(min=1e-12)
    closest = torch.where(region_ab, a + t * ab, closest)
    done = done | region_ab

    vb = d5 * d2 - d1 * d6
    region_ac = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / (d2 - d6).clamp(min=1e-12)
    closest = torch.where(region_ac, a + t * ac, closest)
    done = done | region_ac

    va = d3 * d6 - d5 * d4
    region_bc = (~done) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=1e-12)
    closest = torch.where(region_bc, b + t * (c - b), closest)
    done = done | region_bc

    denom = (va + vb + vc).clamp(min=1e-12)
    v = vb / denom
    w = vc / denom
    return torch.where(~done, a + ab * v + ac * w, closest)


def _points_to_barycentric(points, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = (d00 * d11 - d01 * d01).clamp(min=1e-12)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary = torch.stack((u, v, w), dim=-1).clamp(0, 1)
    return bary / bary.sum(-1, keepdim=True).clamp(min=1e-12)


@torch.no_grad()
def closest_surface(vertices, faces, points, k_vertices=3):
    """Return closest, distance, triangle_ids, barycentric for each query point."""
    vertices = vertices.reshape(-1, 3)
    points = points.reshape(-1, 3)
    faces = faces.reshape(-1, 3).long()
    n_verts = vertices.shape[0]
    k = min(max(int(k_vertices), 1), n_verts)
    table = vertex_triangle_table(faces, n_verts)
    chunks = []
    batch = 4096
    for start in range(0, points.shape[0], batch):
        chunk = points[start:start + batch]
        dist = torch.cdist(chunk, vertices)
        nn = dist.topk(k, largest=False, dim=1).indices
        cand = table[nn].reshape(chunk.shape[0], -1)
        valid = cand >= 0
        safe = cand.clamp(min=0)
        tri = faces[safe]
        a = vertices[tri[..., 0]]
        b = vertices[tri[..., 1]]
        c = vertices[tri[..., 2]]
        q = closest_point_on_triangles(chunk[:, None, :], a, b, c)
        d2 = (q - chunk[:, None, :]).pow(2).sum(-1)
        d2 = torch.where(valid, d2, d2.new_full(d2.shape, 1e12))
        pick = d2.argmin(dim=1)
        rows = torch.arange(chunk.shape[0], device=chunk.device)
        chunks.append((q[rows, pick], d2[rows, pick].clamp(min=0).sqrt(), cand[rows, pick]))
    closest = torch.cat([c[0] for c in chunks], dim=0)
    distance = torch.cat([c[1] for c in chunks], dim=0)
    triangle_ids = torch.cat([c[2] for c in chunks], dim=0).clamp(min=0)
    bary = _points_to_barycentric(closest, vertices[faces[triangle_ids]])
    return closest, distance, triangle_ids, bary


def closest_surface_numpy(vertices, faces, points, k_vertices=3):
    device = vertices.device if torch.is_tensor(vertices) else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    v = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    f = torch.as_tensor(faces, dtype=torch.long, device=device)
    p = torch.as_tensor(points, dtype=torch.float32, device=device)
    closest, distance, triangle_ids, bary = closest_surface(v, f, p, k_vertices=k_vertices)
    return (
        closest.detach().cpu().numpy(),
        distance.detach().cpu().numpy(),
        triangle_ids.detach().cpu().numpy(),
        bary.detach().cpu().numpy(),
    )


@torch.no_grad()
def surface_correspondence(vertices, faces, points):
    _, _, triangle_ids, bary = closest_surface(vertices, faces, points)
    triangles = faces.long()[triangle_ids]
    return triangles, bary.to(dtype=vertices.dtype)


def matched_points(values, correspondence, normals=False):
    if isinstance(correspondence, tuple):
        ids, bary = correspondence
        result = (values[ids] * bary[..., None]).sum(dim=1)
        if normals:
            result = result / result.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return result
    return values[correspondence]
