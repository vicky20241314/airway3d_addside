import os
import traceback
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal as Signal


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
                self.frame_ready.emit(rgb.copy(), depth.copy(), camera)
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
