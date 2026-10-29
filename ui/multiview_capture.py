"""Guided multi-view capture, retaining the original paired mode and signal."""
from datetime import datetime
from pathlib import Path
import time
import cv2
import numpy as np
from PyQt6.QtWidgets import QComboBox,QCheckBox
from PyQt6.QtCore import pyqtSignal as Signal
from ui.capture_widget import CaptureWidget
from core.camera import Camera,backproject_window,coordinate_transform
from core.face_test import detect_face_bbox
from core.aruco import detect_aruco
from core.multiview_data import Observation,fuse_sequence,save_session
from core.pipeline import write_json
from third_party.tufa.Test import get_lmk_single


class MultiCaptureWidget(CaptureWidget):
    session_done=Signal(object)
    FOUR=(('closed','front'),('closed','left'),('closed','right'),('open','front'))
    SIX=FOUR+(('open','left'),('open','right'))

    def __init__(self,parent=None):
        self._observations=[];self._mv_frames=[];self._capture_folder=None;self._view_started_at=0.
        super().__init__(parent)

    def _build_ui(self):
        super()._build_ui()
        self.mode=QComboBox()
        self.mode.addItems(['原双帧模式','闭口三视角 + 张口正面','闭口三视角 + 张口三视角'])
        self.mode.setCurrentIndex(2)
        self.btn_capture.parent().layout().insertWidget(1,self.mode)
        self.refine_check=QCheckBox('联合细化网格')
        self.refine_check.setToolTip('联合各视角细化网格；关闭可保留基础重建结果作对照。指标始终使用基础重建结果。')
        self.refine_check.setChecked(True)
        self.btn_capture.parent().layout().insertWidget(2,self.refine_check)
        self.mode.currentIndexChanged.connect(self.reset)
        self._guidance()

    def sequence(self):return self.SIX if self.mode.currentIndex()==2 else self.FOUR

    def _guidance(self):
        if self.mode.currentIndex()==0:return
        index=min(len(self._observations),len(self.sequence())-1)
        state,view=self.sequence()[index]
        name={'front':'正面','left':'左约40°','right':'右约40°'}[view]
        title=('闭口' if state=='closed' else '张口')+name
        self.lbl_title.setText(f'{index+1}/{len(self.sequence())} 采集{title}\n保持头颈和表情不动，移动相机；停稳后采集。侧面无需 ArUco。')
        self.btn_capture.setText('采集'+title)
        self._stage=0 if state=='closed' else 1
        self._view_started_at=time.monotonic()

    def reset(self,*args):
        self._observations=[];self._mv_frames=[];self._capture_folder=None
        super().reset()
        self.mode.setEnabled(True)
        self._guidance()

    def on_frame(self,rgb,depth,camera):
        captured=getattr(camera,'captured_at_monotonic',None)
        if captured is None:captured=time.monotonic()
        if self.mode.currentIndex()!=0 and captured<self._view_started_at:return
        super().on_frame(rgb,depth,camera)
        if self.mode.currentIndex()==0 or self._warning_active:return
        index=len(self._observations)
        if index>=len(self.sequence()):return
        state,view=self.sequence()[index]
        intr=np.array([camera.fx,camera.fy,camera.cx,camera.cy])
        self._mv_frames.append((state,view,rgb.copy(),depth.copy(),captured,intr))
        self._mv_frames=self._mv_frames[-5:]

    def _show_warning(self,message):
        self._mv_frames=[]
        try:super()._show_warning(message)
        finally:self._view_started_at=time.monotonic()

    def _on_capture(self):
        if self.mode.currentIndex()==0:return super()._on_capture()
        if self._camera is None or len(self._observations)>=len(self.sequence()):return
        attempt=None;attempt_info={}
        try:
            state,view=self.sequence()[len(self._observations)]
            if self._capture_folder is None:
                self._capture_folder=Path(__file__).resolve().parents[1]/'output'/('capture_mv_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            attempt=self._capture_folder/('attempt_'+datetime.now().strftime('%H%M%S_%f'))
            attempt.mkdir(parents=True,exist_ok=False)
            frames=list(self._mv_frames)
            np.savez_compressed(attempt/'raw_sequence.npz',rgb=np.asarray([f[2] for f in frames]),
                depth=np.asarray([f[3] for f in frames]),timestamps_s=np.asarray([f[4] for f in frames]),
                intrinsics=np.asarray([f[5] for f in frames]))
            attempt_info=dict(status='started',state=state,view=view,view_id=f'{state}_{view}',
                frames=len(frames),motion_mode='camera_moved',depth_unit='mm')
            write_json(attempt/'attempt.json',attempt_info)
            camera=Camera(self._camera.fx,self._camera.fy,self._camera.cx,self._camera.cy)
            rgb,depth,quality=fuse_sequence(frames,camera)
            attempt_info['quality']=quality
            bbox=detect_face_bbox(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if bbox is None:raise ValueError('未检测到人脸，请调整相机后重采')
            uv=np.asarray(get_lmk_single(rgb,bbox),np.float32)
            xyz,valid=backproject_window(uv,depth,camera,window=2,trim=0,keep_shape=True)
            obs=Observation(f'{state}_{view}',state,view,rgb,depth,camera,uv,coordinate_transform(xyz),valid,quality)
            # Preserve raw attempts before checking strict front requirements.
            save_session(attempt,[obs])
            obs.validate()
            if state=='closed' and view=='front':
                _,_,throat=detect_aruco(rgb,depth,camera)
                if throat is None or not np.isfinite(throat).all():raise ValueError('正面闭口需要有效 ArUco 喉结点，请重采')
            self._observations.append(obs)
            attempt_info['status']='accepted';write_json(attempt/'attempt.json',attempt_info)
            self._mv_frames=[];self._frame_buffer=[];self.mode.setEnabled(False)
            if len(self._observations)==len(self.sequence()):
                save_session(self._capture_folder/'complete',self._observations)
                self.btn_capture.setEnabled(False)
                self.session_done.emit(self._observations)
            else:self._guidance()
        except Exception as exc:
            if attempt is not None and attempt.exists():
                attempt_info.update(status='failed',error=str(exc))
                try:write_json(attempt/'attempt.json',attempt_info)
                except OSError:pass  # Preserve the original capture/storage error.
            self._show_warning(str(exc))
