"""Resolution-independent, robust RGB feature constraints for FLAME fitting."""
import torch

# Separate groups prevent the 20 lip points from overwhelming each six-point eye.
FEATURE_GROUPS = {'brow_a': list(range(17,22)), 'brow_b': list(range(22,27)), 'eye_a': list(range(36, 42)), 'eye_b': list(range(42, 48)),
                  'nose': list(range(27, 36)), 'lips': list(range(48, 68))}
# iBUG 0 and 16 sit on the ears; they are not a cheek-width signal.
CONTOUR_GROUPS = {'jaw_left': list(range(1, 8)), 'chin': [8], 'jaw_right': list(range(9, 16))}


def _grouped_reprojection_loss(projected, targets, groups):
    """Equal group means, Huber penalty; lengths normalized by outer-eye span."""
    target = torch.stack(targets) if isinstance(targets, (list, tuple)) else targets
    scale = torch.linalg.vector_norm(target[:, 45] - target[:, 36], dim=-1)
    valid_frame = torch.isfinite(scale) & (scale >= 8)
    scale = torch.where(valid_frame, scale, torch.ones_like(scale))
    losses = []
    for indices in groups.values():
        gt = target[:, indices]
        good = torch.isfinite(gt).all(-1) & valid_frame[:, None]
        if good.any():
            residual = (projected[:, indices] - torch.nan_to_num(gt)) / scale[:, None, None]
            distance = torch.linalg.vector_norm(residual[good], dim=-1)
            delta = .02
            losses.append(torch.where(distance <= delta, .5*distance.square(),
                                      delta*(distance-.5*delta)).mean())
    return torch.stack(losses).mean() if losses else projected.sum()*0


def feature_reprojection_loss(projected, targets):
    """Eye/nose/lip image positions. RGB does not invent missing depth."""
    return _grouped_reprojection_loss(projected, targets, FEATURE_GROUPS)


def contour_reprojection_loss(projected, targets):
    """Jaw outline only. Does not pull the skull to the hairline."""
    return _grouped_reprojection_loss(projected, targets, CONTOUR_GROUPS)
