"""Controlled diagnosis; writes new artifacts and never edits captured meshes."""
import argparse,json,sys,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from core.camera import Camera
from core.student.student_refine import refine_depth

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 out=ROOT/'diagnostics'/a.run.name;out.mkdir(parents=True,exist_ok=True)
 cap=np.load(a.run/'capture_pair.npz');prep=np.load(a.run/'prepared.npz');cam=Camera(*cap['intrinsics'])
 paths=[a.run/'closed/mesh_closed_stage2.obj',a.run/'closed/mesh_closed.obj']
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 hashes={str(p):digest(p) for p in paths}
 base=trimesh.load(paths[0],process=False,force='mesh');actual=trimesh.load(paths[1],process=False,force='mesh')
 v,f=trimesh.remesh.subdivide(base.vertices,base.faces)
 offsets=np.load(a.run/'closed/vertex_offsets.npz')
 assert np.allclose(v,offsets['reference_vertices'],atol=1e-8,rtol=0)
 assert np.allclose(offsets['reference_vertices']+offsets['offsets'],actual.vertices,atol=1e-8,rtol=0)
 coarse,info=refine_depth(base.vertices,base.faces,cap['depth_closed'],cam,prep['parsing_closed'],landmarks_2d=prep['lmk2d_closed'])
 coarse_mesh=trimesh.Trimesh(coarse,base.faces,process=False);coarse_mesh.export(out/'same_stage2_student_no_subdivision.obj')
 cv,cf=trimesh.remesh.subdivide(coarse,base.faces);assert np.array_equal(cf,f)
 ref=trimesh.Trimesh(v,f,process=False)
 dense_coarse=trimesh.Trimesh(cv,cf,process=False)
 edges=base.edges_unique
 dist,edge_id=cKDTree(base.vertices[edges].mean(1)).query(v[len(base.vertices):])
 assert dist.max()<1e-7
 endpoints=edges[edge_id]
 residual=actual.vertices[len(base.vertices):]-actual.vertices[endpoints].mean(1)
 mid_moved=np.linalg.norm(offsets['offsets'][len(base.vertices):],axis=1)>1e-6
 front=ref.vertex_normals[len(base.vertices):,2]>.2
 selected=mid_moved&front
 def summary(x):return dict(count=len(x),mean=float(np.mean(x)),p95=float(np.percentile(x,95)),max=float(np.max(x)))
 def geometry(mesh):
  angles=np.degrees(mesh.face_adjacency_angles)
  initial=np.degrees(ref.face_adjacency_angles)
  pair=ref.face_adjacency
  move=np.linalg.norm(mesh.vertices-v,axis=1)
  active=(move[ref.faces[pair]].max(axis=(1,2))>1e-6)&(ref.face_normals[pair][:,:,2].min(1)>.2)
  flat=active&(initial<.01)
  return dict(displacement_mm=summary(move[move>1e-6]*1000),
              initially_coplanar_internal_edges_angle_deg=summary(angles[flat]),
              internal_edges_above_10deg=int((angles[flat]>10).sum()),
              internal_edges_above_20deg=int((angles[flat]>20).sum()))
 result=dict(input_hashes=hashes,subdivision_surface_area_difference_mm2=float((ref.area-base.area)*1e6),
             saved_displacement_applied_once=True,
             no_subdivision_control=geometry(dense_coarse),current_subdivision=geometry(actual),
             midpoint_departure_from_endpoint_average_mm=summary(np.linalg.norm(residual[selected],axis=1)*1000),
             control_solver=info,current_solver=json.loads((a.run/'closed/stage3_report.json').read_text(encoding='utf-8')),
             interpretation='Same stage2, captured depth, parsing, landmarks and student defaults; only subdivision and its required anchor adapter differ. Roughness is not scanner accuracy.')
 plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
 fig,axes=plt.subplots(1,4,figsize=(18,7))
 light=np.array([.3,.3,1]);light/=np.linalg.norm(light)
 for ax,mesh,title in zip(axes,[base,ref,dense_coarse,actual],['本次 stage2','仅中点细分（不位移）','同一 stage2 + student（无细分）','当前：细分 + student']):
  ids=np.flatnonzero(mesh.face_normals[:,2]>.01);ids=ids[np.argsort(mesh.triangles_center[ids,2])]
  shade=np.clip(.35+.65*(mesh.face_normals[ids]@light),0,1)
  ax.add_collection(PolyCollection(mesh.triangles[ids,:,:2]*1000,facecolors=np.repeat(shade[:,None],3,axis=1),edgecolors='none'))
  ax.set_xlim(base.bounds[:,0]*1000);ax.set_ylim(base.bounds[:,1]*1000);ax.set_aspect('equal');ax.set_title(title);ax.axis('off')
 fig.tight_layout();fig.savefig(out/'geometry_comparison.png',dpi=170);plt.close(fig)
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 assert all(digest(Path(p))==h for p,h in hashes.items())
 print(json.dumps({k:v for k,v in result.items() if 'solver' not in k},ensure_ascii=True),flush=True)
if __name__=='__main__':main()
