"""Model and GUI smoke test with explicit saved RGB-D, no hardware claim."""
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['UNIFACE_CACHE_DIR']=str(ROOT/'models/uniface')
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
 p.add_argument('--output',type=Path,default=ROOT/'output/frontend_validation');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 from core.camera import Camera
 from core.face_test import detect_face_bbox
 from core.mask_detection import get_parser,get_face_data_single
 from third_party.tufa.Test import _get_tufa,get_lmk_single
 import cv2
 data=np.load(a.capture);cam=Camera(*map(float,data['intrinsics']))
 get_parser();_get_tufa()
 result={}
 for name in ['closed','open']:
  rgb=data['rgb_'+name];depth=data['depth_'+name]
  bbox=detect_face_bbox(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
  assert bbox is not None
  lmk=get_lmk_single(rgb,bbox);assert lmk.shape==(68,2) and np.isfinite(lmk).all()
  d=get_face_data_single(cam,rgb,depth,bbox_xywh=bbox);assert d['valid']
  result[name]={'landmarks':len(lmk),'points':len(d['pts_face_neck'])}
 from unittest.mock import patch
 from PyQt6.QtWidgets import QApplication
 from ui.workers import CameraThread
 from ui.main_window import MainWindow
 app=QApplication([])
 from PyQt6.QtGui import QFont,QFontDatabase
 QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
 app.setFont(QFont('Microsoft YaHei',10))
 with patch.object(CameraThread,'start',lambda self:None):
  w=MainWindow();w.show();app.processEvents()
  assert w._stack.currentIndex()==0 and w._capture._stage==0
  w._capture.on_frame(data['rgb_closed'],data['depth_closed'],cam)
  app.processEvents();w.grab().save(str(a.output/'capture_page.png'))
  w.close();app.processEvents()
 result.update(gui_capture_page=True,hardware_capture=False)
 (a.output/'frontend.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(result)
if __name__=='__main__':main()
