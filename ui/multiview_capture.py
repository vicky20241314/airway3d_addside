"""Guided multi-view capture, retaining the original paired mode and signal."""
from datetime import datetime
from pathlib import Path
import time
import cv2
import numpy as np
from PyQt6.QtWidgets import QComboBox,QCheckBox,QLabel,QApplication
from PyQt6.QtCore import pyqtSignal as Signal,QEventLoop
from ui.capture_widget import CaptureWidget
from core.camera import Camera
from core.face_test import detect_face_bbox
from core.aruco import detect_aruco
from core.multiview_data import save_session
from core.temporal_capture import prepare_burst
from core.pipeline import write_json
from third_party.tufa.Test import get_lmk_single


class MultiCaptureWidget(CaptureWidget):
    session_done=Signal(object)
    FOUR=(('closed','front'),('closed','left'),('closed','right'),('open','front'))
    SIX=FOUR+(('open','left'),('open','right'))

    def __init__(self,parent=None):
        self._observations=[];self._mv_frames=[];self._capture_folder=None;self._view_started_at=0.;self._processing=False
        super().__init__(parent)

    def _build_ui(self):
        super()._build_ui()
        self.mode=QComboBox()
        self.mode.addItems(['原双帧模式','闭口三视角 + 张口正面（推荐）','额外采集张口侧面（可选，不一致时不采用）'])
        self.mode.setCurrentIndex(1)
        self.btn_capture.parent().layout().insertWidget(1,self.mode)
        self.frame_count_choice=QComboBox()
        for count in (5,9,15):self.frame_count_choice.addItem(f'每视角 {count} 帧'+('（手持推荐）' if count==5 else ''),count)
        self.frame_count_choice.setCurrentIndex(0)
        self.btn_capture.parent().layout().insertWidget(2,self.frame_count_choice)
        self.burst_status=QLabel();self.burst_status.setWordWrap(True)
        self.burst_status.setStyleSheet('color: #555; font-size: 12px;')
        self.btn_capture.parent().layout().insertWidget(3,self.burst_status)
        self.refine_check=QCheckBox('联合细化网格')
        self.refine_check.setToolTip('联合各视角细化网格；关闭可保留基础重建结果作对照。指标始终使用基础重建结果。')
        self.refine_check.setChecked(True)
        self.btn_capture.parent().layout().insertWidget(4,self.refine_check)
        self.mode.currentIndexChanged.connect(self.reset)
        self.frame_count_choice.currentIndexChanged.connect(self._reset_burst)
        self._guidance()

    def sequence(self):return self.SIX if self.mode.currentIndex()==2 else self.FOUR

    def _reset_burst(self,*args):
        self._mv_frames=[];self._view_started_at=time.monotonic();self._burst_status()

    def _burst_status(self):
        if self.mode.currentIndex()==0:
            self.frame_count_choice.hide();self.burst_status.hide();return
        self.frame_count_choice.show();self.burst_status.show()
        count=int(self.frame_count_choice.currentData())
        self.burst_status.setText(f'已缓存 {len(self._mv_frames)}/{count} 帧。短暂停留后采集；自动补偿小幅手抖，至少使用 {max(3,int(np.ceil(count*.6)))} 帧。')
        self.btn_capture.setEnabled(len(self._mv_frames)>=count and len(self._observations)<len(self.sequence()))

    def _guidance(self):
        if self.mode.currentIndex()==0:self._burst_status();return
        index=min(len(self._observations),len(self.sequence())-1)
        state,view=self.sequence()[index]
        name={'front':'正面','left':'左约40°','right':'右约40°'}[view]
        title=('闭口' if state=='closed' else '张口')+name
        self.lbl_title.setText(f'{index+1}/{len(self.sequence())} 采集{title}\n移动相机或自然小幅转头，到位后短暂停留；尽量保持表情。侧面只补充头面部，无需 ArUco。')
        self.btn_capture.setText('采集'+title)
        self._stage=0 if state=='closed' else 1
        self._view_started_at=time.monotonic()
        self._burst_status()

    def reset(self,*args):
        self._observations=[];self._mv_frames=[];self._capture_folder=None
        super().reset()
        self.mode.setEnabled(True)
        self.frame_count_choice.setEnabled(True)
        self._guidance()

    def on_frame(self,rgb,depth,camera):
        if self._processing:return
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
        self._mv_frames=[f for f in self._mv_frames if captured-f[4]<=1.][-int(self.frame_count_choice.currentData()):]
        self._burst_status()

    def _show_warning(self,message):
        self._mv_frames=[]
        try:super()._show_warning(message)
        finally:self._view_started_at=time.monotonic();self._burst_status()

    def _on_capture(self):
        if self._processing:return
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
            if len(frames)<int(self.frame_count_choice.currentData()):raise ValueError('帧数尚未达到所选数量，请停稳后等待缓存')
            camera=Camera(self._camera.fx,self._camera.fy,self._camera.cx,self._camera.cy)
            self._processing=True
            self.btn_capture.setEnabled(False);self.burst_status.setText(f'正在检查 {len(frames)} 帧，请保持姿态…')
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            def detector(rgb):
                bbox=detect_face_bbox(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                return bbox,None if bbox is None else get_lmk_single(rgb,bbox)
            def progress(done,total):
                self.burst_status.setText(f'正在检查第 {done}/{total} 帧，请保持姿态…')
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            obs=prepare_burst(frames,camera,detector=detector,progress=progress)
            attempt_info['quality']=obs.quality
            # Preserve raw attempts before checking strict front requirements.
            save_session(attempt,[obs])
            obs.validate()
            if state=='closed' and view=='front':
                _,_,throat=detect_aruco(obs.rgb,obs.depth,camera)
                if throat is None or not np.isfinite(throat).all():raise ValueError('正面闭口需要有效 ArUco 喉结点，请重采')
            self._observations.append(obs)
            attempt_info['status']='accepted';write_json(attempt/'attempt.json',attempt_info)
            self._mv_frames=[];self._frame_buffer=[];self.mode.setEnabled(False);self.frame_count_choice.setEnabled(False)
            if len(self._observations)==len(self.sequence()):
                save_session(self._capture_folder/'complete',self._observations)
                self.btn_capture.setEnabled(False)
                self.burst_status.setText('多帧采集完成，正在重建。')
                self.session_done.emit(self._observations)
            else:self._guidance()
        except Exception as exc:
            if attempt is not None and attempt.exists():
                attempt_info.update(status='failed',error=str(exc))
                if hasattr(exc,'diagnostics'):attempt_info['temporal_diagnostics']=exc.diagnostics
                try:write_json(attempt/'attempt.json',attempt_info)
                except OSError:pass  # Preserve the original capture/storage error.
            self._show_warning(str(exc))
        finally:
            self._processing=False
