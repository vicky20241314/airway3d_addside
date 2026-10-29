"""Progressive FLAME fitting, stage-2 metrics, guarded subdivided student refinement."""
from datetime import datetime
from pathlib import Path
import copy
import json
import time
import numpy as np


def write_json(path, value):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def finish_meshes(stage2, depths, camera, parsing, targets, adam_3d, observed_landmarks,
                  folder, progress=lambda _: None):
    """Compute metrics before refining independent copies for display."""
    from core.metrics import metrics_calculate
    from core.cross_refine import refine_mesh
    folder = Path(folder)
    snapshots = tuple(copy.deepcopy(mesh) for mesh in stage2)
    for name, mesh in zip(('closed','open'), snapshots):
        mesh.metadata['geometry_stage'] = 2
        out = folder/name; out.mkdir(parents=True, exist_ok=True)
        mesh.export(out/f'mesh_{name}_stage2.obj')
    progress('正在计算第二阶段指标...')
    values = metrics_calculate(*snapshots, adam_3d, *observed_landmarks)
    names = ['张口度','甲颌距离','头后仰角度','下颌活动角度','下颌活动度','颈围','舌厚度','声门视野']
    units = ['cm','cm','°','°','cm','cm','cm','cm']
    # Never display NaN as a successful measurement.
    if not np.isfinite(values).all():
        raise ValueError('第二阶段指标出现无效数值；第二阶段网格已保存')
    numeric = {n:dict(value=float(v),unit=u) for n,v,u in zip(names,values,units)}
    write_json(folder/'metrics.json',dict(geometry_stage=2, formula_source='old_version', values=numeric))
    metrics = {n:f'{v:.2f}{" " if u == "cm" else ""}{u}' for n,v,u in zip(names,values,units)}
    display, diagnostics = [], []
    for name, mesh, depth, labels, target in zip(('closed','open'),snapshots,depths,parsing,targets):
        progress(f'正在执行交叉第三阶段：{"闭口" if name == "closed" else "张口"}网格...')
        result, reference, info = refine_mesh(copy.deepcopy(mesh),depth,camera,labels,target)
        result.export(folder/name/f'mesh_{name}.obj')
        np.savez_compressed(folder/name/'vertex_offsets.npz',
            reference_vertices=reference,faces=result.faces,offsets=result.vertices-reference)
        write_json(folder/name/'stage3_report.json',info)
        diagnostics.append(info)
        display.append(result)
    return metrics, snapshots, tuple(display), diagnostics


def run_pair(rgb_closed, rgb_open, depth_closed, depth_open, camera, *, landmarks=None,
             bboxes=(None,None), output_root=None, progress=lambda _: None, device=None, options=None):
    from core.aruco import detect_aruco
    root = Path(output_root) if output_root else Path(__file__).resolve().parents[1]/'output'
    folder = root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(algorithm='cross-v2-guarded', mesh_unit='m', depth_unit='mm', status='started', metric_geometry_stage=2, shared_shape=True, aruco_required=True)
    try:
        np.savez_compressed(folder/'capture_pair.npz',rgb_closed=rgb_closed,rgb_open=rgb_open,
            depth_closed=depth_closed,depth_open=depth_open,
            intrinsics=np.array([camera.fx,camera.fy,camera.cx,camera.cy]))
        from core.provenance import record_inputs
        record_inputs(folder, options)
        progress('正在检测 ArUco 标记...')
        corners, _, adam_3d = detect_aruco(rgb_closed,depth_closed,camera)
        if adam_3d is None or not np.isfinite(adam_3d).all():
            raise ValueError('未检测到 ArUco 标记或喉结深度无效，请重新采集闭口图像')
        # Load detection/fitting only after the mandatory marker check.
        from core.mask_detection import get_face_data_single
        from core.integrated_fitting import train
        if landmarks is None:
            from third_party.tufa.Test import get_lmk_single
            from core.face_test import detect_face_bbox
            from core.camera import backproject_window, coordinate_transform
            landmarks = {}
            for name,rgb,depth in zip(('closed','open'),(rgb_closed,rgb_open),(depth_closed,depth_open)):
                import cv2
                bbox = detect_face_bbox(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                if bbox is None: raise ValueError(f'{name}: 未检测到人脸')
                uv = get_lmk_single(rgb,bbox)
                xyz, valid = backproject_window(uv,depth,camera,window=2,trim=0,keep_shape=True)
                landmarks[f'lmk2d_{name}'] = uv
                landmarks[f'lmk3d_{name}'] = coordinate_transform(xyz)
                landmarks[f'mask_{name}'] = valid
        progress('正在提取面部和颈部点云...')
        data = [get_face_data_single(camera,rgb,depth,bbox_xywh=bbox) for rgb,depth,bbox in
                zip((rgb_closed,rgb_open),(depth_closed,depth_open),bboxes)]
        if not all(d['valid'] for d in data): raise ValueError('有效面部点云不足')
        l3 = [landmarks[f'lmk3d_{n}'] for n in ('closed','open')]
        l2 = [landmarks[f'lmk2d_{n}'] for n in ('closed','open')]
        masks = [landmarks[f'mask_{n}'] for n in ('closed','open')]
        saved = dict(landmarks)
        for name,d in zip(('closed','open'),data):
            saved[f'points_{name}'] = d['pts_face_neck']; saved[f'parsing_{name}'] = d['parsing']
        np.savez_compressed(folder/'prepared.npz', **saved, adam_3d=adam_3d,
                            intrinsics=np.array([camera.fx,camera.fy,camera.cx,camera.cy]))
        report['stage']='stage1_stage2'; write_json(folder/'report.json',report)
        stage2 = train(*l3,*l2,*masks,*[d['pts_face_neck'] for d in data],camera,
            parsing_closed=data[0]['parsing'],parsing_open=data[1]['parsing'],device=device,
            debug_dir=folder,options=options,progress=progress)
        np.savez_compressed(folder/'stage2_landmarks.npz',
            closed=stage2[0].metadata['flame_landmarks_3d'],open=stage2[1].metadata['flame_landmarks_3d'])
        report['fitting'] = stage2[0].metadata['fit_report']
        from core.provenance import sha
        report['input_hashes'] = {n:sha(folder/n) for n in ['capture_pair.npz','prepared.npz','flame_parameters.npz','stage2_landmarks.npz']}
        write_json(folder/'fitting_report.json',report['fitting'])
        report['stage']='subdivision_student_stage3'; write_json(folder/'report.json',report)
        metrics,snapshots,display,diagnostics = finish_meshes(stage2,[depth_closed,depth_open],camera,
            [d['parsing'] for d in data],l2,adam_3d,l3,folder,progress)
        report.update(stage='complete', status='complete',refinement=diagnostics, elapsed_s=time.perf_counter()-started)
        write_json(folder/'report.json',report)
        return dict(metrics=metrics, stage2=snapshots, meshes=display,
                    aruco_corners={'closed':corners,'open':None}, folder=folder)
    except Exception as exc:
        report.update(status='failed', error=str(exc),elapsed_s=time.perf_counter()-started)
        write_json(folder/'report.json',report)
        raise RuntimeError(f'{exc}\n已保存数据：{folder}') from exc
