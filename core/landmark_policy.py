"""Old-version landmark selection and capture requirements (zero-based IDs)."""
import numpy as np

ALIGN_START = 17
ALIGN_STOP = 48
ALIGN_EXCLUDED = (17, 18, 19, 24, 25, 26)
ALIGN_IDS = tuple(i for i in range(ALIGN_START, ALIGN_STOP) if i not in ALIGN_EXCLUDED)
MIN_ALIGN_LANDMARKS = 8
REQUIRED_IDS = (8, 21, 22, 33, 50, 51, 52, 56, 57, 58)


def validate_landmarks(points2d, points3d, mask):
    """Apply identical quality checks to GUI capture and the fitting entry point.

    Return a mask that also excludes non-finite coordinates. Required jaw/lip
    points serve later constraints/metrics; they are not rigid alignment points.
    """
    uv, xyz = np.asarray(points2d), np.asarray(points3d)
    valid = np.asarray(mask, dtype=bool)
    if uv.shape != (68, 2) or xyz.shape != (68, 3) or valid.shape != (68,):
        raise ValueError('需要完整的 68 点二维／三维关键点和有效标记')
    valid = valid & np.isfinite(uv).all(1) & np.isfinite(xyz).all(1)
    count = int(valid[list(ALIGN_IDS)].sum())
    if count < MIN_ALIGN_LANDMARKS:
        raise ValueError(f'可用于对齐的关键点深度不足，请重新采集 ({count}/{len(ALIGN_IDS)}，至少需要 {MIN_ALIGN_LANDMARKS} 个)')
    missing = [i for i in REQUIRED_IDS if not valid[i]]
    if missing:
        raise ValueError('关键语义点深度无效，请重新采集 ('+', '.join(map(str, missing))+')')
    return valid
