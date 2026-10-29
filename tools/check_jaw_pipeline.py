"""Actual GUI worker integration using a supplied paired capture for offline testing."""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('UNIFACE_CACHE_DIR',str(ROOT/'models/uniface'))
import numpy as np
from core.camera import Camera

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cap=np.load(a.run/'capture_pair.npz');prep=np.load(a.run/'prepared.npz');camera=Camera(*cap['intrinsics'])
    import core.mask_detection as mask
    import core.render as render
    import core.pipeline as pipeline
    calls=[];outputs=[];renders=[]
    def segment(cam,rgb,depth,bbox_xywh=None):
        state=('closed','open')[len(calls)];calls.append(state)
        assert np.array_equal(rgb,cap['rgb_'+state]) and np.array_equal(depth,cap['depth_'+state])
        return dict(valid=True,pts_face_neck=prep['points_'+state],parsing=prep['parsing_'+state])
    mask.get_face_data_single=segment
    def renderer(vertices,faces,*args):
        assert len(vertices)==20018 and len(faces)==39904
        renders.append(True);return np.zeros((64,64,3),np.uint8)
    render.auto_render_mesh=renderer
    real=pipeline.run_pair
    def invoke(*args,**kwargs):
        value=real(*args,**kwargs,output_root=ROOT/'diagnostics/jaw_integrated');outputs.append(value);return value
    pipeline.run_pair=invoke
    from PyQt6.QtWidgets import QApplication
    from ui.workers import PipelineWorker
    app=QApplication.instance() or QApplication([])
    worker=PipelineWorker(cap['rgb_closed'],cap['rgb_open'],cap['depth_closed'],cap['depth_open'],camera,
        landmarks={k:prep[k] for k in prep.files if k.startswith(('lmk','mask_'))})
    errors=[];finished=[]
    worker.error.connect(errors.append);worker.finished.connect(lambda *args:finished.append(True));worker.progress.connect(lambda s:print(s,flush=True))
    worker.run()
    if errors:raise RuntimeError(errors[0])
    assert finished and calls==['closed','open'] and len(renders)==2
    folder=outputs[0]['folder'];report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    assert report['algorithm']=='cross-v4-jaw-supported' and report['status']=='complete'
    assert report['fitting']['loss_profile']=='jaw_supported'
    assert all(d['initial_stage2_yaw_deg'] is not None and d['reason']=='ok' for d in report['fitting']['jaw_constraints'])
    result=dict(gui_worker_passed=True,real_aruco_and_stage1_stage2_stage3=True,
        default_jaw_constraints_enabled=True,paired_inputs_separate=True,hardware_tested=False,
        mocked='cached segmentation/landmarks and renderer',run=str(folder))
    (ROOT/'diagnostics/jaw_integrated/checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
