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
                  folder, progress=lambda _: None, *, refine=True):
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
    if not refine:
        return metrics, snapshots, tuple(copy.deepcopy(m) for m in snapshots), []
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
    from core.cross_refine import DEFAULT_PROFILE,ALGORITHMS
    from core.fit_options import FitOptions
    root = Path(output_root) if output_root else Path(__file__).resolve().parents[1]/'output'
    folder = root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(algorithm=(options or FitOptions()).algorithm_version(ALGORITHMS[DEFAULT_PROFILE]), mesh_unit='m', depth_unit='mm', status='started', metric_geometry_stage=2, shared_shape=True, aruco_required=True)
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


def run_multiview(observations, *, output_root=None, options=None, device=None,
                  progress=lambda _:None, front_only=False, refine=True,smoke_test=False):
    """Explicit shared-state entry; accepts both states' side observations.

    Both returned state meshes are in their respective FRONT camera frames.
    No source capture or historical pair output is ever rewritten.
    """
    from core.multiview_data import save_session, validate_session
    from core.multiview_fitting import fit_multiview
    from core.aruco import detect_aruco
    from core.multiview_geometry import RigidCamera
    import hashlib
    import os
    from dataclasses import asdict
    from core.fit_options import FitOptions
    options=options or FitOptions()
    root=Path(output_root) if output_root else Path(__file__).resolve().parents[1]/'output'
    folder=root/('mv_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report=dict(status='started',algorithm='multiview_shared_state_LM',metric_geometry_stage=2,
        stage3_enabled=bool(refine),mesh_unit='m',depth_unit='mm',real_side_effect_validated=False,
        run_purpose='smoke_test' if smoke_test else 'reconstruction',fit_configuration=asdict(options),
        visibility_backend=os.environ.get('AIRWAY3D_RAY_BACKEND','projected'))
    if smoke_test:
        (folder/'QUICK_TEST_ONLY.txt').write_text('快速流程检查结果，不能用于判断相似度或重建精度。正式回放请去掉 --quick。\n',encoding='utf-8')
    try:
        # Save even failed validation attempts for diagnosis, then work on copies.
        observations=copy.deepcopy(observations)
        save_session(folder/'capture',observations)
        observations=validate_session(observations)
        if front_only:observations=[o for o in observations if o.view=='front']
        fronts={o.state:o for o in observations if o.view=='front'}
        c=fronts['closed'];o=fronts['open']
        corners,_,adam=detect_aruco(c.rgb,c.depth,c.camera)
        if adam is None or not np.isfinite(adam).all():raise ValueError('正面闭口 ArUco 或喉结深度无效，请重采')
        for obs in observations:
            if obs.parsing is None or obs.points is None:
                from core.mask_detection import get_face_data_single
                data=get_face_data_single(obs.camera,obs.rgb,obs.depth)
                if not data['valid']:
                    if obs.view!='front':
                        obs.parsing=None;obs.points=None
                        progress(f'{obs.view_id} 面部区域不足，将记录为未采用的可选观测。')
                        continue
                    raise ValueError(f'{obs.view_id}: 面颈分割失败')
                obs.parsing=data['parsing'];obs.points=data['pts_face_neck']
        save_session(folder/'prepared',observations)
        report.update(stage='fitting',view_ids=[v.view_id for v in observations],
            metric_sources=dict(closed_front=c.view_id,open_front=o.view_id,aruco=c.view_id,
                coordinates='each state front camera; never side landmarks'))
        write_json(folder/'report.json',report)
        timing=dict(preparation_s=time.perf_counter()-started)
        phase_started=time.perf_counter()
        meshes,transforms,fitting=fit_multiview(observations,folder,options=options,device=device,progress=progress)
        timing['fitting_s']=time.perf_counter()-phase_started
        used=fitting.get('used_view_ids',[v.view_id for v in observations])
        report.update(captured_view_ids=report['view_ids'],view_ids=used,excluded_views=fitting.get('excluded_views',[]))
        observations=[v for v in observations if v.view_id in used]
        write_json(folder/'extrinsics.json',{k:dict(R=v.R,t=v.t,direction='state_front_to_view') for k,v in transforms.items()})
        phase_started=time.perf_counter()
        metrics,snapshots,display,diagnostics=finish_meshes((meshes['closed'],meshes['open']),
            [c.depth,o.depth],c.camera,[c.parsing,o.parsing],[c.uv,o.uv],adam,[c.xyz,o.xyz],folder,progress,refine=False)
        timing['stage2_metrics_s']=time.perf_counter()-phase_started
        phase_started=time.perf_counter()
        if refine:
            report['stage']='joint_refinement';write_json(folder/'report.json',report)
            display,diagnostics=finish_multiview_refinement(snapshots,observations,transforms,folder,progress)
        timing['refinement_s']=time.perf_counter()-phase_started if refine else 0.
        report.update(status='complete',stage='complete',fitting=fitting,refinement=diagnostics,timing=timing,elapsed_s=time.perf_counter()-started)
        write_json(folder/'report.json',report)
        version=(Path(__file__).resolve().parents[1]/'VERSION').read_text().strip()
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
            for base in ('core','ui','third_party/flame') for p in (Path(__file__).resolve().parents[1]/base).rglob('*.py')}
        inputs={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for sub in ('capture','prepared') for p in (folder/sub).iterdir() if p.is_file()}
        write_json(folder/'provenance.json',dict(version=version,sources=sources,inputs=inputs,real_side_effect_validated=False))
        return dict(metrics=metrics,stage2=snapshots,meshes=display,aruco_corners={'closed':corners,'open':None},folder=folder,
            used_view_ids=used,excluded_views=report['excluded_views'])
    except Exception as exc:
        report.update(status='failed',error=str(exc),elapsed_s=time.perf_counter()-started)
        write_json(folder/'report.json',report)
        raise RuntimeError(f'{exc}\n采集与诊断已保存：{folder}') from exc


def finish_multiview_refinement(stage2,observations,transforms,folder,progress=lambda _:None):
    from core.cross_refine import refine_multiview_mesh
    display=[];diagnostics=[];folder=Path(folder)
    for state,mesh in zip(('closed','open'),stage2):
        views=[o for o in observations if o.state==state and o.view_id in transforms]
        if not any(o.view=='front' for o in views):raise ValueError(f'{state}: missing fitted front camera')
        progress(f'联合细化 {state}：{len(views)} 个视角，一份位移场...')
        phase_started=time.perf_counter()
        result,reference,info=refine_multiview_mesh(copy.deepcopy(mesh),views,transforms)
        info['total_elapsed_s']=time.perf_counter()-phase_started
        progress(f"{'闭口' if state=='closed' else '张口'}联合细化完成：{info['total_elapsed_s']:.1f}秒")
        out=folder/state;out.mkdir(parents=True,exist_ok=True)
        # Preserve small-triangle geometry bounds in the exported mesh too.
        result.export(out/f'mesh_{state}.obj',digits=16)
        np.savez_compressed(out/'vertex_offsets.npz',reference_vertices=reference,faces=result.faces,offsets=result.vertices-reference)
        write_json(out/'stage3_report.json',info)
        display.append(result);diagnostics.append(info)
    return tuple(display),diagnostics
