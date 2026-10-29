"""Guided multi-view capture, retaining the original paired mode and signal."""
from datetime import datetime
from pathlib import Path
import time
import cv2
import numpy as np
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import pyqtSignal as Signal
from ui.capture_widget import CaptureWidget
from core.camera import Camera,backproject_window,coordinate_transform
from core.face_test import detect_face_bbox
from core.aruco import detect_aruco
from core.multiview_data import Observation,fuse_sequence,save_session
from third_party.tufa.Test import get_lmk_single


class MultiCaptureWidget(CaptureWidget):
    session_done=Signal(object)
    FOUR=(('closed','front'),('closed','left'),('closed','right'),('open','front'))
    SIX=FOUR+(('open','left'),('open','right'))

    def __init__(self,parent=None):
        self._observations=[];self._mv_frames=[];self._capture_folder=None
        super().__init__(parent)

    def _build_ui(self):
        super()._build_ui()
        self.mode=QComboBox()
        self.mode.addItems(['原双帧模式','闭口三视角 + 张口正面','闭口三视角 + 张口三视角'])
        self.mode.setCurrentIndex(2)
        self.btn_capture.parent().layout().insertWidget(1,self.mode)
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

    def reset(self,*args):
        self._observations=[];self._mv_frames=[];self._capture_folder=None
        super().reset()
        self.mode.setEnabled(True)
        self._guidance()

    def on_frame(self,rgb,depth,camera):
        super().on_frame(rgb,depth,camera)
        if self.mode.currentIndex()==0 or self._warning_active:return
        index=len(self._observations)
        if index>=len(self.sequence()):return
        state,view=self.sequence()[index]
        intr=np.array([camera.fx,camera.fy,camera.cx,camera.cy])
        self._mv_frames.append((state,view,rgb.copy(),depth.copy(),time.monotonic(),intr))
        self._mv_frames=self._mv_frames[-5:]

    def _show_warning(self,message):
        self._mv_frames=[]
        super()._show_warning(message)

    def _on_capture(self):
        if self.mode.currentIndex()==0:return super()._on_capture()
        if self._camera is None or len(self._observations)>=len(self.sequence()):return
        try:
            camera=Camera(self._camera.fx,self._camera.fy,self._camera.cx,self._camera.cy)
            rgb,depth,quality=fuse_sequence(self._mv_frames,camera)
            state,view=self.sequence()[len(self._observations)]
            bbox=detect_face_bbox(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if bbox is None:raise ValueError('未检测到人脸，请调整相机后重采')
            uv=np.asarray(get_lmk_single(rgb,bbox),np.float32)
            xyz,valid=backproject_window(uv,depth,camera,window=2,trim=0,keep_shape=True)
            obs=Observation(f'{state}_{view}',state,view,rgb,depth,camera,uv,coordinate_transform(xyz),valid,quality)
            # Preserve raw attempts before checking strict front requirements.
            if self._capture_folder is None:
                self._capture_folder=Path(__file__).resolve().parents[1]/'output'/('capture_mv_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            save_session(self._capture_folder/('attempt_'+datetime.now().strftime('%H%M%S_%f')),[obs])
            obs.validate()
            if state=='closed' and view=='front':
                _,_,throat=detect_aruco(rgb,depth,camera)
                if throat is None or not np.isfinite(throat).all():raise ValueError('正面闭口需要有效 ArUco 喉结点，请重采')
            self._observations.append(obs)
            self._mv_frames=[];self._frame_buffer=[];self.mode.setEnabled(False)
            if len(self._observations)==len(self.sequence()):
                save_session(self._capture_folder/'complete',self._observations)
                self.btn_capture.setEnabled(False)
                self.session_done.emit(self._observations)
            else:self._guidance()
        except Exception as exc:self._show_warning(str(exc))
