import os
import traceback
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal as Signal


def snapshot_frame_camera(camera):
    """A separate immutable-by-convention intrinsics/time snapshot per frame."""
    from core.camera import Camera
    result=Camera(camera.fx,camera.fy,camera.cx,camera.cy)
    result.captured_at_monotonic=time.monotonic()
    return result


class CameraThread(QThread):
    """Background thread that continuously reads camera frames and emits them via signal."""

    frame_ready = Signal(np.ndarray, np.ndarray, object)  # rgb, depth, camera
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False

    def run(self):
        try:
            self._run_camera()
        except Exception as exc:
            self.error.emit('请检查 Orbbec 相机、USB 3 连接及驱动。\n'+str(exc))

    def _run_camera(self):
        from core.capture import OrbbecCapture

        # 相机线程职责：
        # 持续读 Orbbec 帧并把 RGB/Depth 发给采集页预览。
        cap = OrbbecCapture(timeout_ms=100)
        cap.start()
        self._running = True

        try:
            while self._running:
                out = cap.read()
                if out is None:
                    continue
                camera, rgb, depth = out
                self.frame_ready.emit(rgb.copy(), depth.copy(), snapshot_frame_camera(camera))
        finally:
            cap.stop()

    def stop(self):
        self._running = False
        self.wait()


class PipelineWorker(QThread):
    """Background thread that runs the full pipeline: landmarks -> point cloud -> fitting -> metrics -> render."""

    progress = Signal(str)                              # 当前步骤描述
    finished = Signal(dict, object, object, object)    # metrics, proj_closed, proj_open, aruco_corners
    error    = Signal(str)                              # 错误 traceback

    def __init__(self, rgb_closed, rgb_open, depth_closed, depth_open, camera,
                 bbox_closed_xywh=None, bbox_open_xywh=None, landmarks=None, parent=None):
        super().__init__(parent)
        self.rgb_closed       = rgb_closed
        self.rgb_open         = rgb_open
        self.depth_closed     = depth_closed
        self.depth_open       = depth_open
        self.camera           = camera
        self.bbox_closed_xywh = bbox_closed_xywh
        self.bbox_open_xywh   = bbox_open_xywh
        self.landmarks        = landmarks

    @staticmethod
    def _xywh_to_xyxy(bbox):
        """[x, y, w, h] → [x1, y1, x2, y2]，供 infer_landmark_rgbd 使用。"""
        if bbox is None:
            return None
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h], dtype=np.int32)

    def run(self):
        try:
            from core.pipeline import run_pair
            from core.render import auto_render_mesh
            if self.landmarks is None:
                raise RuntimeError("未获取到关键点结果，请重新采集")
            result = run_pair(
                self.rgb_closed, self.rgb_open, self.depth_closed, self.depth_open,
                self.camera, landmarks=self.landmarks,
                bboxes=(self.bbox_closed_xywh,self.bbox_open_xywh),
                progress=self.progress.emit,
            )
            self.progress.emit("正在渲染拟合结果...")
            closed, opened = result["meshes"]
            proj_closed = auto_render_mesh(closed.vertices, closed.faces, 800, 800)
            proj_open = auto_render_mesh(opened.vertices, opened.faces, 800, 800)
            self.finished.emit(result["metrics"], proj_closed, proj_open, result["aruco_corners"])
        except Exception:
            self.error.emit(traceback.format_exc())


class MultiPipelineWorker(QThread):
    progress=Signal(str)
    finished=Signal(dict,object,object,object)
    error=Signal(str)

    def __init__(self,observations,parent=None,*,refine=True):
        super().__init__(parent)
        self.observations=observations
        self.refine=refine

    def run(self):
        try:
            from core.pipeline import run_multiview
            from core.render import auto_render_mesh
            result=run_multiview(self.observations,progress=self.progress.emit,refine=self.refine)
            used=result['used_view_ids'];excluded=result['excluded_views']
            self.capture_summary=f'实际采用 {len(used)} 个视角。张口指标使用张口正面。'
            if excluded:
                names={'open_left':'张口左侧','open_right':'张口右侧'}
                self.capture_summary+='\n未采用：'+ '、'.join(names.get(r['view_id'],r['view_id']) for r in excluded)+'。侧面不可见或一致性不足；原始数据已保留。'
            c,o=result['meshes']
            self.finished.emit(result['metrics'],auto_render_mesh(c.vertices,c.faces,800,800),
                auto_render_mesh(o.vertices,o.faces,800,800),result['aruco_corners'])
        except Exception:self.error.emit(traceback.format_exc())
