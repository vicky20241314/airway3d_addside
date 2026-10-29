"""Explicit cached-data validation of the guarded stage3; no source overwrite."""
import argparse, os, sys, json, hashlib, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from core.camera import Camera
from core.cross_refine import refine_mesh, subdivide_with_anchors
from core.student.arap_refine import landmark_matrix
from core.student.offset_refine import depth_inputs
from core.student.initial_surface import closest_surface_numpy
from core.conservative_refine import ShapeGuard,RefineOptions

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(a):
    a=np.asarray(a)
    return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()))
def draw(meshes,titles,path):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,len(meshes),figsize=(5*len(meshes),8))
    light=np.array([.3,.3,1]);light/=np.linalg.norm(light)
    for ax,mesh,title in zip(axes,meshes,titles):
        ids=np.flatnonzero(mesh.face_normals[:,2]>.01)
        ids=ids[np.argsort(mesh.triangles_center[ids,2])]
        shade=np.clip(.35+.65*(mesh.face_normals[ids]@light),0,1)
        ax.add_collection(PolyCollection(mesh.triangles[ids,:,:2]*1000,facecolors=np.repeat(shade[:,None],3,axis=1),edgecolors='none'))
        ax.set_xlim(meshes[0].bounds[:,0]*1000);ax.set_ylim(meshes[0].bounds[:,1]*1000)
        ax.set_aspect('equal');ax.axis('off');ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=ROOT/'validation/cross_v2'/a.run.name;out.mkdir(parents=True,exist_ok=True)
    cap=np.load(a.run/'capture_pair.npz');prep=np.load(a.run/'prepared.npz');cam=Camera(*cap['intrinsics'])
    inputs=[a.run/'capture_pair.npz',a.run/'prepared.npz']
    inputs += [a.run/s/f'mesh_{s}{suffix}.obj' for s in ('closed','open') for suffix in ('_stage2','')]
    hashes={str(p):digest(p) for p in inputs}
    report=dict(algorithm='cross-v2-guarded',input_hashes=hashes,hardware_tested=False,states={})
    for state in ('closed','open'):
        print('START',state,flush=True);started=time.perf_counter()
        folder=out/state;folder.mkdir(exist_ok=True)
        mesh=trimesh.load(a.run/state/f'mesh_{state}_stage2.obj',process=False,force='mesh')
        old=trimesh.load(a.run/state/f'mesh_{state}.obj',process=False,force='mesh')
        saved=mesh.vertices.copy();sv,sf,J=subdivide_with_anchors(mesh)
        assert np.allclose(J@sv,landmark_matrix(mesh.faces,len(saved))@saved,rtol=0,atol=1e-12)
        # Every anchor now belongs to an actual child face, also after arbitrary deformation.
        face_sets={tuple(sorted(t)) for t in sf}
        assert all(tuple(sorted(J.getrow(i).indices)) in face_sets for i in range(J.shape[0]))
        rng=np.random.default_rng(42);deformed=sv+rng.normal(0,.0001,sv.shape)
        assert np.isfinite(J@deformed).all()
        final,base,info=refine_mesh(mesh,cap['depth_'+state],cam,prep['parsing_'+state],prep['lmk2d_'+state],profile='v2')
        assert np.array_equal(saved,mesh.vertices)
        assert np.array_equal(base[:len(saved)],saved)
        ref=trimesh.Trimesh(base,sf,process=False)
        assert np.isclose(ref.area,mesh.area,rtol=1e-12)
        delta=final.vertices-base
        assert np.max(np.linalg.norm(delta,axis=1))<=.002+1e-12
        assert all(h['step_m']<=.00025+1e-12 for h in info['history'])
        guard=ShapeGuard(base,sf,RefineOptions());bad,quality=guard.inspect(final.vertices)
        assert not len(bad)
        assert np.max(np.linalg.norm(delta,axis=1))>1e-6,'No meaningful displacement'
        # Same fixed, reliable depth sample for all variants. Diagnostic only, not VEGA accuracy.
        points,_,_=depth_inputs(base,sf,cap['depth_'+state],cam,prep['parsing_'+state],prep['lmk2d_'+state])
        if len(points)>16000: points=points[np.sort(np.random.default_rng(42).choice(len(points),16000,replace=False))]
        fit={}
        for name,m in [('stage2',ref),('v1',old),('v2',final)]:
            _,dist,_,_=closest_surface_numpy(m.vertices,m.faces,points)
            fit[name]=dict(rmse_mm=float(np.sqrt(np.mean(dist**2))*1000),**summary(dist*1000))
        old_bad,old_quality=guard.inspect(old.vertices)
        edges=mesh.edges_unique
        _,ix=cKDTree(saved[edges].mean(1)).query(base[len(saved):]);ends=edges[ix]
        def midpoint(m):return summary(np.linalg.norm(m.vertices[len(saved):]-m.vertices[ends].mean(1),axis=1)*1000)
        final.export(folder/f'mesh_{state}.obj')
        np.savez_compressed(folder/'vertex_offsets.npz',reference_vertices=base,faces=sf,offsets=delta)
        with np.load(folder/'vertex_offsets.npz') as d:
            assert np.array_equal(d['reference_vertices']+d['offsets'],final.vertices)
        (folder/'stage3_report.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        draw([mesh,old,final],['原始 stage2','修改前：细分 + student','修改后：受保护 student'],folder/'comparison.png')
        item=dict(elapsed_s=time.perf_counter()-started,stage2_unchanged=True,subdivision_surface_preserved=True,
                  child_surface_anchors=True,offset_applied_once=True,geometry_v1=old_quality,geometry_v2=quality,
                  bad_vertices_v1=len(old_bad),bad_vertices_v2=len(bad),midpoint_v1_mm=midpoint(old),midpoint_v2_mm=midpoint(final),
                  fixed_depth_sample=len(points),depth_to_mesh_diagnostic=fit,
                  displacement_mm=summary(np.linalg.norm(delta,axis=1)*1000),iterations=len(info['history']))
        report['states'][state]=item
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({state:item}),flush=True)
    assert all(digest(Path(p))==h for p,h in hashes.items())
    report['all_inputs_unchanged']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS',out,flush=True)
if __name__=='__main__':main()
