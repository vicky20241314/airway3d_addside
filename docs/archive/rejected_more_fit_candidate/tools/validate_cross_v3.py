"""Compare v2/v3 with identical RGB-D inputs and unchanged stage2 meshes."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from core.cross_refine import refine_mesh
from core.camera import Camera
from core.student.offset_refine import depth_inputs,_operators
from core.student.initial_surface import closest_surface_numpy
from core.conservative_refine import ShapeGuard,RefineOptions
from tools.validate_cross_v2 import draw,summary,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--render',action='store_true');a=p.parse_args()
    out=ROOT/'validation/cross_v3'/a.run.name;out.mkdir(parents=True,exist_ok=True)
    cap=np.load(a.run/'capture_pair.npz');prep=np.load(a.run/'prepared.npz');cam=Camera(*cap['intrinsics'])
    paths=[a.run/'capture_pair.npz',a.run/'prepared.npz']+[a.run/s/f'mesh_{s}_stage2.obj' for s in ('closed','open')]
    hashes={str(p):digest(p) for p in paths}
    report=dict(input_hashes=hashes,code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'core').rglob('*.py')},
                notes='Fixed depth-to-triangle distances are RGB-D agreement, not independent anatomical accuracy.',states={})
    for state in ('closed','open'):
        print('START',state,flush=True);dest=out/state;dest.mkdir(exist_ok=True)
        mesh=trimesh.load(a.run/state/f'mesh_{state}_stage2.obj',force='mesh',process=False)
        saved=mesh.vertices.copy()
        v2,base,i2=refine_mesh(mesh,cap['depth_'+state],cam,prep['parsing_'+state],prep['lmk2d_'+state],profile='v2')
        legacy=ROOT/'validation/cross_v2'/a.run.name/state/f'mesh_{state}.obj'
        if legacy.exists():
            old=trimesh.load(legacy,force='mesh',process=False)
            assert np.max(np.abs(old.vertices-v2.vertices))<1e-8,'V2 behavior changed'
        v3,base3,i3=refine_mesh(mesh,cap['depth_'+state],cam,prep['parsing_'+state],prep['lmk2d_'+state],profile='v3')
        assert np.array_equal(base,base3) and np.array_equal(saved,mesh.vertices)
        guard=ShapeGuard(base,v3.faces,RefineOptions());bad,quality=guard.inspect(v3.vertices)
        assert not len(bad)
        assert np.linalg.norm(v3.vertices-base,axis=1).max()<=.002+1e-12
        assert all(h['step_m']<=.00025+1e-12 for h in i3['history'])
        points,_,observed=depth_inputs(base,v3.faces,cap['depth_'+state],cam,prep['parsing_'+state],prep['lmk2d_'+state])
        if len(points)>16000:points=points[np.sort(np.random.default_rng(42).choice(len(points),16000,replace=False))]
        ref=trimesh.Trimesh(base,v3.faces,process=False)
        pair=ref.face_adjacency
        region=observed[ref.faces[pair]].all(axis=(1,2))
        flat=region&(np.degrees(ref.face_adjacency_angles)<.01)
        L,_,_=_operators(base,v3.faces)
        vertex_ids=np.flatnonzero(observed)
        anchors={}
        from core.cross_refine import subdivide_with_anchors
        _,_,J=subdivide_with_anchors(mesh)
        metrics={}
        for name,m,info in [('stage2',ref,None),('v2',v2,i2),('v3',v3,i3)]:
            _,dist,_,_=closest_surface_numpy(m.vertices,m.faces,points)
            displacement=m.vertices-base
            angles=np.degrees(m.face_adjacency_angles)
            metrics[name]=dict(depth_rmse_mm=float(np.sqrt(np.mean(dist**2))*1000),
                depth_distance_mm=summary(dist*1000),
                displacement_laplacian_mm=summary(np.linalg.norm(L@displacement,axis=1)[vertex_ids]*1000),
                initially_flat_face_angle_deg=summary(angles[flat]),
                max_displacement_mm=float(np.linalg.norm(displacement,axis=1).max()*1000))
            if info:metrics[name].update(iterations=len(info['history']),elapsed_s=info['elapsed_s'],quality=info['quality'])
            uv=np.column_stack((cam.fx*(J@m.vertices)[:,0]/(-(J@m.vertices)[:,2])+cam.cx,
                                -cam.fy*(J@m.vertices)[:,1]/(-(J@m.vertices)[:,2])+cam.cy))
            anchors[name]=summary(np.linalg.norm(uv-prep['lmk2d_'+state][17:68],axis=1))
        v2.export(dest/'mesh_v2.obj');v3.export(dest/f'mesh_{state}.obj')
        np.savez_compressed(dest/'vertex_offsets.npz',reference_vertices=base,faces=v3.faces,offsets=v3.vertices-base)
        with np.load(dest/'vertex_offsets.npz') as d:assert np.array_equal(d['reference_vertices']+d['offsets'],v3.vertices)
        for name,info in [('v2',i2),('v3',i3)]:
            (dest/f'{name}_report.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        draw([mesh,v2,v3],['原始 stage2','v2：已有几何保护','v3：位移平滑与切向约束'],dest/'flat_geometry_comparison.png')
        render_status='not_requested'
        if a.render:
            from core.render import auto_render_mesh
            import cv2
            for name,m in [('v2',v2),('v3',v3)]:
                before=m.vertices.copy();before_faces=m.faces.copy()
                for smooth in (False,True):
                    im=auto_render_mesh(m.vertices,m.faces,800,800,smooth=smooth)
                    ok,encoded=cv2.imencode('.png',im);assert ok
                    (dest/f'{name}_{"smooth" if smooth else "flat"}_render.png').write_bytes(encoded.tobytes())
                assert np.array_equal(before,m.vertices) and np.array_equal(before_faces,m.faces)
            render_status='real_pyrender_passed_inputs_unchanged'
        report['states'][state]=dict(metrics=metrics,landmark_reprojection_px=anchors,
            region='All observed vertices and adjacency pairs fully inside observed region, fixed before comparison',
            n_region_vertices=len(vertex_ids),n_initially_flat_pairs=int(flat.sum()),n_depth_samples=len(points),
            stage2_unchanged=True,geometry_guard_passed=True,render_status=render_status)
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({state:report['states'][state]}),flush=True)
    assert all(digest(Path(p))==h for p,h in hashes.items())
    report['all_inputs_unchanged']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS',str(out).encode('ascii','backslashreplace').decode(),flush=True)
if __name__=='__main__':main()
