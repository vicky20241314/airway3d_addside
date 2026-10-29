"""FLAME shape/expression Jacobians adapted from tools/lm/jacobians.py.

Fixed-pose LBS derivatives for point-to-point residuals. Pose uses finite
differences in core/lm_stage.py, matching the senior LM fitter.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def shape_expr_bases(flame, n_shape: int, n_expr: int) -> Tuple[Tensor, Tensor]:
    shapedirs = flame.shapedirs
    return shapedirs[:, :, :n_shape], shapedirs[:, :, 300:300 + n_expr]


def joints_jacobian(J_regressor: Tensor, basis: Tensor) -> Tensor:
    return torch.einsum("kv,vcp->kcp", J_regressor, basis)


def posed_joints_jacobian_fixed_pose(global_rot_mats: Tensor, dJ: Tensor, parents: Tensor) -> Tensor:
    K, _, P = dJ.shape
    dg = torch.zeros_like(dJ)
    dg[0] = dJ[0]
    for k in range(1, K):
        parent = int(parents[k].item())
        rel = dJ[k] - dJ[parent]
        dg[k] = global_rot_mats[parent] @ rel + dg[parent]
    return dg


def vertices_jacobian_fixed_pose(
    lbs_weights: Tensor,
    global_rot_mats: Tensor,
    basis: Tensor,
    dJ: Tensor,
    dg: Tensor,
    vertex_ids: Optional[Tensor] = None,
) -> Tensor:
    if vertex_ids is not None:
        basis_sel = basis[vertex_ids]
        W = lbs_weights[vertex_ids]
    else:
        basis_sel = basis
        W = lbs_weights
    diff = basis_sel[:, None, :, :] - dJ[None, :, :, :]
    rotated = torch.einsum("kcd,nkdp->nkcp", global_rot_mats, diff)
    term = dg[None, :, :, :] + rotated
    return torch.einsum("nk,nkcp->ncp", W, term)


def dV_shape_expr(flame, global_rot_mats: Tensor, n_shape: int, n_expr: int) -> Tuple[Tensor, Tensor]:
    """Return dV/dshape and dV/dexpr as [V, 3, P] under the given global rotations."""
    S_shape, S_expr = shape_expr_bases(flame, n_shape, n_expr)
    dJ_s = joints_jacobian(flame.J_regressor, S_shape)
    dJ_e = joints_jacobian(flame.J_regressor, S_expr)
    dg_s = posed_joints_jacobian_fixed_pose(global_rot_mats, dJ_s, flame.parents)
    dg_e = posed_joints_jacobian_fixed_pose(global_rot_mats, dJ_e, flame.parents)
    dV_s = vertices_jacobian_fixed_pose(flame.lbs_weights, global_rot_mats, S_shape, dJ_s, dg_s)
    dV_e = vertices_jacobian_fixed_pose(flame.lbs_weights, global_rot_mats, S_expr, dJ_e, dg_e)
    return dV_s, dV_e


def correspondence_vertex_ids(correspondence) -> Tensor:
    if isinstance(correspondence, tuple):
        return correspondence[0].reshape(-1).unique()
    return correspondence.unique()


def gather_matched_jacobian(dV: Tensor, correspondence) -> Tensor:
    """dV [V,3,P] -> matched [N,3,P] for vertex ids or (triangles, bary)."""
    if isinstance(correspondence, tuple):
        triangles, bary = correspondence
        return (dV[triangles] * bary[:, :, None, None]).sum(dim=1)
    return dV[correspondence]


def dV_matched_shape_expr(flame, global_rot_mats: Tensor, n_shape: int, n_expr: int, correspondence) -> Tensor:
    """Shape+expr Jacobian at matched points, [N, 3, n_shape+n_expr]."""
    S_shape, S_expr = shape_expr_bases(flame, n_shape, n_expr)
    dJ_s = joints_jacobian(flame.J_regressor, S_shape)
    dJ_e = joints_jacobian(flame.J_regressor, S_expr)
    dg_s = posed_joints_jacobian_fixed_pose(global_rot_mats, dJ_s, flame.parents)
    dg_e = posed_joints_jacobian_fixed_pose(global_rot_mats, dJ_e, flame.parents)
    vids = correspondence_vertex_ids(correspondence)
    dV_s = vertices_jacobian_fixed_pose(flame.lbs_weights, global_rot_mats, S_shape, dJ_s, dg_s, vids)
    dV_e = vertices_jacobian_fixed_pose(flame.lbs_weights, global_rot_mats, S_expr, dJ_e, dg_e, vids)
    dV = torch.cat([dV_s, dV_e], dim=2)
    lut = torch.full((int(flame.v_template.shape[0]),), -1, dtype=torch.long, device=dV.device)
    lut[vids] = torch.arange(vids.shape[0], device=dV.device, dtype=torch.long)
    if isinstance(correspondence, tuple):
        triangles, bary = correspondence
        return (dV[lut[triangles]] * bary[:, :, None, None]).sum(dim=1)
    return dV[lut[correspondence]]
