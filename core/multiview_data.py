"""Versioned RGB-D observations. Coordinates: metres, X right/Y up/Z backward.

Each observation stores its OWN intrinsics. Raw depth is millimetres. No frame
from a different view/state may be passed to fuse_sequence.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from core.camera import Camera, merge_depth_frames
from core.landmark_policy import validate_landmarks


@dataclass
class Observation:
    view_id: str
    state: str
    view: str
    rgb: np.ndarray
    depth: np.ndarray
    camera: Camera
    uv: np.ndarray
    xyz: np.ndarray
    valid: np.ndarray
    quality: dict = field(default_factory=dict)
    parsing: np.ndarray | None = None
    points: np.ndarray | None = None

    def validate(self):
        if self.state not in ('closed', 'open') or self.view not in ('front', 'left', 'right'):
            raise ValueError('Unknown state/view')
        if not self.view_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in self.view_id):
            raise ValueError('Unsafe view ID')
        if self.depth.ndim != 2 or self.rgb.shape != (*self.depth.shape, 3):
            raise ValueError(f'{self.view_id}: RGB/depth must be aligned')
        if not np.isfinite(self.depth).all() or (self.depth < 0).any():
            raise ValueError('Invalid depth')
        intr = [self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy]
        if not np.isfinite(intr).all() or min(intr[:2]) <= 0:
            raise ValueError('Invalid per-view intrinsics')
        if self.uv.shape != (68, 2) or self.xyz.shape != (68, 3) or self.valid.shape != (68,):
            raise ValueError('Expected 68 landmarks and validity flags')
        self.valid = self.valid.astype(bool) & np.isfinite(self.xyz).all(1) & np.isfinite(self.uv).all(1)
        if self.view == 'front':
            self.valid = validate_landmarks(self.uv, self.xyz, self.valid)
        if self.quality.get('motion_mode', 'camera_moved') != 'camera_moved':
            raise ValueError('请保持头颈不动并移动相机重采；转头数据不能按整头颈刚性模型处理')
        weight = self.quality.get('weight', 1.)
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError('Quality weight must be in (0,1]')
        return self


def validate_session(observations, *, allow_open_sides=True):
    if not observations:
        raise ValueError('Empty RGB-D session')
    for o in observations:
        o.validate()
    keys = [(o.state, o.view) for o in observations]
    if len(set(keys)) != len(keys) or len({o.view_id for o in observations}) != len(keys):
        raise ValueError('Duplicate state/view or ID')
    if ('closed', 'front') not in keys or ('open', 'front') not in keys:
        raise ValueError('Both state front observations are required for metrics')
    if not allow_open_sides and any(o.state == 'open' and o.view != 'front' for o in observations):
        raise ValueError('This release supports open front only')
    # Fronts first makes initialization and diagnostics deterministic.
    return sorted(observations, key=lambda o: (o.view != 'front', o.state != 'closed', o.view))


def fuse_sequence(frames, camera):
    """frames = [(state, view, rgb, depth, timestamp_s, intrinsics), ...]."""
    if len(frames) < 3 or len({(f[0], f[1]) for f in frames}) != 1:
        raise ValueError('至少三帧，且只能融合同一状态、同一视角的短序列')
    intr = np.array([camera.fx, camera.fy, camera.cx, camera.cy])
    if any(not np.allclose(f[5], intr, rtol=0, atol=1e-5) for f in frames):
        raise ValueError('采集期间内参变化，请重新采集')
    if frames[-1][4] - frames[0][4] > 1. or frames[-1][4] <= frames[0][4]:
        raise ValueError('短序列时间无效或过长，请停稳后重采')
    ds = np.stack([f[3] for f in frames]).astype(float)
    common = (ds > 0).all(0)
    center = np.zeros(common.shape, bool)
    h, w = center.shape
    center[h//4:3*h//4, w//4:3*w//4] = True
    common &= center
    if common.sum() < 100:
        raise ValueError('有效深度不足，请重新采集')
    jitter = float(np.median(np.ptp(ds[:, common], axis=0)))
    rgb_motion = float(np.mean(np.abs(frames[-1][2].astype(float)-frames[0][2].astype(float))[center]))
    if jitter > 3. or rgb_motion > 12.:
        raise ValueError('相机或人脸尚未停稳，请保持姿态后重采')
    quality = dict(motion_mode='camera_moved', frames=len(frames), depth_jitter_mm=jitter,
                   rgb_temporal_change=rgb_motion, timestamps_s=[float(f[4]) for f in frames],
                   weight=float(np.clip(1/(1+jitter/3), .25, 1)))
    return frames[len(frames)//2][2].copy(), merge_depth_frames([f[3] for f in frames]), quality


def save_session(folder, observations):
    """Never overwrite a capture or an existing manifest."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'session.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    entries = []
    for i, o in enumerate(observations):
        name = f'observation_{i:02d}.npz'
        with (folder/name).open('xb') as stream:
            arrays = dict(rgb=o.rgb, depth=o.depth, uv=o.uv, xyz=o.xyz, valid=o.valid,
                          intrinsics=np.array([o.camera.fx, o.camera.fy, o.camera.cx, o.camera.cy]))
            if o.parsing is not None: arrays['parsing'] = o.parsing
            if o.points is not None: arrays['points'] = o.points
            np.savez_compressed(stream, **arrays)
        entries.append(dict(file=name, view_id=o.view_id, state=o.state, view=o.view, quality=o.quality))
    manifest.write_text(json.dumps(dict(schema='airway3d.rgbd.multiview.v1', depth_unit='mm',
        point_unit='m', coordinate_convention='X_right_Y_up_Z_backward', observations=entries),
        ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def load_session(path):
    path = Path(path).resolve()
    info = json.loads(path.read_text(encoding='utf-8'))
    if (info['schema'] != 'airway3d.rgbd.multiview.v1' or info['depth_unit'] != 'mm' or info['point_unit'] != 'm'
            or info.get('coordinate_convention') != 'X_right_Y_up_Z_backward'):
        raise ValueError('Unsupported session schema/units')
    result = []
    for row in info['observations']:
        file = (path.parent/row['file']).resolve()
        if file.parent != path.parent: raise ValueError('Observation must be inside session folder')
        with np.load(file, allow_pickle=False) as a:
            result.append(Observation(row['view_id'], row['state'], row['view'], a['rgb'], a['depth'],
                Camera(*a['intrinsics']), a['uv'], a['xyz'], a['valid'], row['quality'],
                a['parsing'] if 'parsing' in a else None, a['points'] if 'points' in a else None))
    return result


def load_legacy_pair(folder):
    """Read historical data without rewriting it. Cached preparation is optional."""
    folder = Path(folder)
    with np.load(folder/'capture_pair.npz', allow_pickle=False) as cap, np.load(folder/'prepared.npz', allow_pickle=False) as prep:
        return [Observation(f'{s}_front', s, 'front', cap['rgb_'+s], cap['depth_'+s],
            Camera(*cap['intrinsics']), prep['lmk2d_'+s], prep['lmk3d_'+s], prep['mask_'+s],
            dict(weight=1., source='legacy_pair', motion_mode='camera_moved'),
            prep['parsing_'+s], prep['points_'+s]) for s in ('closed', 'open')]
