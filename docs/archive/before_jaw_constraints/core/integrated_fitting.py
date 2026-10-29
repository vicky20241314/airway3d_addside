"""The old GUI's train() interface, backed by two-stage shared-shape LM."""
from pathlib import Path
import time
import numpy as np
import torch
import trimesh

from third_party.flame import FLAME, FlameConfig
from core.fit_options import FitOptions
from core.joint_lm import JointProblem, solve_stage, solve_progressive
from core.landmark_policy import validate_landmarks


def make_flame(options, device, batch=2, dtype=torch.float32):
    root = Path(__file__).resolve().parents[1] / 'assets'
    cfg = FlameConfig()
    cfg.flame_model_path = str(root/'flame2023.pkl')
    cfg.static_landmark_embedding_path = str(root/'flame_static_embedding.pkl')
    cfg.dynamic_landmark_embedding_path = str(root/'flame_dynamic_embedding.npy')
    cfg.batch_size = batch
    cfg.shape_params = options.shape_components
    cfg.expression_params = options.expression_components
    cfg.use_face_contour = True
    cfg.use_3D_translation = True
    flame = FLAME(cfg).to(device=device, dtype=dtype)
    flame.dtype = dtype
    flame.eval()
    return flame


def train(lmk3d_gt_closed, lmk3d_gt_open, lmk2d_gt_closed, lmk2d_gt_open,
          lmk_mask_closed, lmk_mask_open, ptcloud_closed, ptcloud_open, camera,
          rgb_closed=None, rgb_open=None, ptcloud_face_closed=None, ptcloud_face_open=None,
          device=None, debug_dir=None, parsing_closed=None, parsing_open=None,
          options=None, progress=lambda _: None):
    options = options or FitOptions()
    options.schedule()
    if (min(options.gradient_tol, options.parameter_step_tol, options.vertex_step_tol_m,
            options.loss_relative_tol, options.plateau_vertex_tol_m) <= 0
            or min(options.plateau_patience, options.min_outer, options.max_rejections) < 1):
        raise ValueError('Invalid convergence settings')
    if options.max_points < 20 or options.inner_steps < 1 or min(options.stage1_outer, options.stage2_outer) < 1:
        raise ValueError('Invalid LM iteration or point sampling settings')
    if not (1 <= options.shape_components <= 300 and 1 <= options.expression_components <= 100):
        raise ValueError('Invalid FLAME component counts')
    if not np.isfinite([camera.fx,camera.fy,camera.cx,camera.cy]).all() or min(camera.fx,camera.fy)<=0:
        raise ValueError('Invalid camera intrinsics')
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable')
    started = time.perf_counter()
    flame = make_flame(options, device)
    as_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    l3 = [as_tensor(lmk3d_gt_closed), as_tensor(lmk3d_gt_open)]
    l2 = [as_tensor(lmk2d_gt_closed), as_tensor(lmk2d_gt_open)]
    masks = [torch.as_tensor(x, dtype=torch.bool, device=device) for x in (lmk_mask_closed,lmk_mask_open)]
    for frame,(a,b,m) in enumerate(zip(l3,l2,masks)):
        if a.shape != (68,3) or b.shape != (68,2) or m.shape != (68,):
            raise ValueError('Expected 68 2D/3D landmarks and validity masks for each frame')
        try:
            valid = validate_landmarks(b.cpu().numpy(), a.cpu().numpy(), m.cpu().numpy())
        except ValueError as exc:
            raise ValueError(f'{"闭口" if frame == 0 else "张口"}帧：{exc}') from exc
        masks[frame] = torch.as_tensor(valid, dtype=torch.bool, device=device)
    rng = np.random.default_rng(options.seed)
    points = []
    # Both stages retain face+neck data. Sampling is independent per frame.
    for p in (ptcloud_closed, ptcloud_open):
        p = np.asarray(p)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('Expected point cloud Nx3')
        p = p[np.isfinite(p).all(1) & (p[:,2]<0)]
        if len(p) < 20:
            raise ValueError('有效面颈点云不足')
        if len(p) > options.max_points:
            p = p[rng.choice(len(p), options.max_points, replace=False)]
        points.append(as_tensor(p))
    problem = JointProblem(flame, camera, points, l3, l2, masks,
                           [parsing_closed,parsing_open], options)
    t0 = time.perf_counter()
    theta, stage1 = solve_stage(problem, problem.theta, 1, progress)
    stage1['elapsed_s'] = time.perf_counter()-t0
    t0 = time.perf_counter()
    theta, stage2 = solve_progressive(problem, theta, progress)
    stage2['elapsed_s'] = time.perf_counter()-t0
    vertices, landmarks = problem.forward(theta)
    shape, expr, pose, rotation, translation = problem.parameters(theta)
    meshes = []
    for b in range(2):
        mesh = trimesh.Trimesh(vertices=vertices[b].cpu().numpy(),
                               faces=flame.faces_tensor.cpu().numpy(), process=False)
        mesh.metadata.update(geometry_stage=2, shared_shape=True,
            flame_landmarks_3d=landmarks[b].cpu().numpy(),
            flame_parameters=dict(shape=shape[b].cpu().numpy().copy(),
                expression=expr[b].cpu().numpy().copy(), pose=pose[b].cpu().numpy().copy(),
                rotation=rotation[b].cpu().numpy().copy(), translation=translation[b].cpu().numpy().copy()),
            fit_report=dict(stage1=stage1, stage2=stage2,
                loss_profile='old_version', elapsed_s=time.perf_counter()-started,
                device=str(device), metric_geometry='stage2', identity_parameter_count=options.shape_components))
        meshes.append(mesh)
    if debug_dir:
        out = Path(debug_dir); out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'flame_parameters.npz', shared_shape=shape[0].cpu().numpy(),
            expression=expr.cpu().numpy(), pose=pose.cpu().numpy(),
            rotation=rotation.cpu().numpy(), translation=translation.cpu().numpy())
    return tuple(meshes)
