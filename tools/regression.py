"""Explicit offline regression; never used as the application's default input."""
import argparse,os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('UNIFACE_CACHE_DIR',str(ROOT/'models/uniface'))
import numpy as np
import trimesh
from core.camera import Camera
from core.cross_refine import refine_mesh,subdivide_with_anchors,ALGORITHMS,DEFAULT_PROFILE
from core.provenance import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);args=p.parse_args()
 run=args.source/'output/20261023_101953_469487';ref=args.source/'jiaochastage3'
 out=ROOT/f'validation/cross_{DEFAULT_PROFILE}_full';out.mkdir(parents=True,exist_ok=True)
 cap=np.load(run/'capture_pair.npz');prep=np.load(run/'prepared.npz')
 camera=Camera(*map(float,cap['intrinsics']))
 manifest=json.loads((ref/'generation_manifest.json').read_text(encoding='utf-8'))
 assert sha(run/'closed/mesh_closed_stage2.obj')==manifest['stage2_sha256']
 base=trimesh.load(run/'closed/mesh_closed_stage2.obj',process=False,force='mesh')
 saved=base.vertices.copy()
 final,reference,info=refine_mesh(base,cap['depth_closed'],camera,prep['parsing_closed'],prep['lmk2d_closed'])
 expected=trimesh.load(ref/'mesh_closed_cross_stage3.obj',process=False,force='mesh')
 error=np.linalg.norm(final.vertices-expected.vertices,axis=1)
 # V2 intentionally changes stage3. V1 remains a provenance/reference output.
 from core.conservative_refine import ShapeGuard,RefineOptions
 assert not len(ShapeGuard(reference,final.faces,RefineOptions()).inspect(final.vertices)[0])
 assert info['algorithm']==ALGORITHMS[DEFAULT_PROFILE]
 assert np.linalg.norm(final.vertices-reference,axis=1).max()<=.002+1e-12
 assert np.array_equal(saved,base.vertices)
 assert np.isclose(trimesh.Trimesh(reference,final.faces,process=False).area,base.area,rtol=1e-12)
 print('Intentional difference from legacy v1, max m:',error.max(),flush=True)
 results=dict(algorithm=ALGORITHMS[DEFAULT_PROFILE],legacy_comparison_max_vertex_difference_m=float(error.max()),surface_area_preserved=True,anchors_checked=True)
 (out/'checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 # Exercise the actual UI worker and run_pair, substituting only cached image
 # preprocessing and renderer. Both FLAME fitting and stage3 run for real.
 import core.mask_detection as mask
 import core.render as render
 import core.pipeline as pipeline
 calls=[];renders=[]
 def segmentation(cam,rgb,depth,bbox_xywh=None):
  name='closed' if len(calls)==0 else 'open';calls.append(name)
  assert np.array_equal(depth,cap['depth_'+name])
  assert np.array_equal(rgb,cap['rgb_'+name])
  return dict(valid=True,pts_face_neck=prep['points_'+name],parsing=prep['parsing_'+name])
 mask.get_face_data_single=segmentation
 def renderer(v,f,*a):
  assert len(v)==20018 and len(f)==39904
  renders.append(len(v));return np.zeros((64,64,3),np.uint8)
 render.auto_render_mesh=renderer
 original=pipeline.run_pair;outputs=[]
 def invoke(*a,**kw):
  result=original(*a,**kw,output_root=out/'runs');outputs.append(result);return result
 pipeline.run_pair=invoke
 from PyQt6.QtWidgets import QApplication
 app=QApplication.instance() or QApplication([])
 from ui.workers import PipelineWorker
 landmarks={k:prep[k] for k in prep.files if k.startswith(('lmk','mask_'))}
 worker=PipelineWorker(cap['rgb_closed'],cap['rgb_open'],cap['depth_closed'],cap['depth_open'],camera,landmarks=landmarks)
 failures=[];finished=[]
 worker.error.connect(failures.append);worker.finished.connect(lambda *a:finished.append(True));worker.progress.connect(lambda s:print(s,flush=True))
 worker.run()
 if failures:raise RuntimeError(failures[0])
 assert finished and calls==['closed','open'] and len(renders)==2
 actual=outputs[0];results['run']=str(actual['folder'])
 results['stage2_differences_m']={}
 for i,name in enumerate(('closed','open')):
  previous=trimesh.load(run/name/f'mesh_{name}_stage2.obj',process=False,force='mesh')
  delta=np.linalg.norm(actual['stage2'][i].vertices-previous.vertices,axis=1)
  results['stage2_differences_m'][name]=dict(max=float(delta.max()),mean=float(delta.mean()))
  with np.load(actual['folder']/name/'vertex_offsets.npz') as d:
   assert np.allclose(d['reference_vertices']+d['offsets'],actual['meshes'][i].vertices,rtol=0,atol=1e-12)
   assert np.array_equal(d['reference_vertices'][:5023],actual['stage2'][i].vertices)
 results.update(worker_full_pipeline_passed=True,renderer_receives_stage3=True,hardware_capture_tested=False,
                preprocessing='cached segmentation and detected landmarks; actual ArUco check and fitting',
                real_renderer_tested=False)
 (out/'checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 print(json.dumps(results),flush=True)

if __name__=='__main__':main()
