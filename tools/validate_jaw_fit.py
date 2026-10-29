"""Paired RGB-D regression of opt-in jaw-supported stage2, never edits input runs."""
import argparse,os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from core.camera import Camera
from core.fit_options import FitOptions
from core.integrated_fitting import train
from core.pipeline import finish_meshes,write_json
from core.jaw_constraints import prepare_jaw
from core.student.initial_surface import closest_surface_numpy
from tools.diagnose_jaw import project,stats
from dataclasses import asdict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,nargs='+',required=True);a=ap.parse_args()
    root=ROOT/'diagnostics/jaw_candidate';root.mkdir(parents=True,exist_ok=True)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    results={};options=FitOptions(jaw_contour_weight=.5,jaw_region_balance=True)
    for run in a.runs:
        print('START',run.name,flush=True);out=root/run.name;out.mkdir(exist_ok=True)
        cap=np.load(run/'capture_pair.npz');prep=np.load(run/'prepared.npz');oldlm=np.load(run/'stage2_landmarks.npz')
        k=cap['intrinsics'];camera=Camera(*k);names=('closed','open')
        files=[run/'capture_pair.npz',run/'prepared.npz']+[run/s/f'mesh_{s}_stage2.obj' for s in names]
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p):sha(p) for p in files}
        l3=[prep['lmk3d_'+s] for s in names];l2=[prep['lmk2d_'+s] for s in names]
        masks=[prep['mask_'+s] for s in names];points=[prep['points_'+s] for s in names];parsing=[prep['parsing_'+s] for s in names]
        new=train(*l3,*l2,*masks,*points,camera,parsing_closed=parsing[0],parsing_open=parsing[1],
            options=options,debug_dir=out,progress=lambda m:print(m,flush=True))
        metrics,snapshots,final,diagnostics=finish_meshes(new,[cap['depth_'+s] for s in names],camera,
            parsing,l2,prep['adam_3d'],l3,out,progress=lambda m:print(m,flush=True))
        np.savez_compressed(out/'stage2_landmarks.npz',**{s:new[i].metadata['flame_landmarks_3d'] for i,s in enumerate(names)})
        item=dict(input_hashes=hashes,options=asdict(options),states={},fitting=new[0].metadata['fit_report'])
        for b,state in enumerate(names):
            old=trimesh.load(run/state/f'mesh_{state}_stage2.obj',force='mesh',process=False)
            sample=points[b];rng=np.random.default_rng(42)
            if len(sample)>8000:sample=sample[np.sort(rng.choice(len(sample),8000,replace=False))]
            observation=prepare_jaw(parsing[b],l2[b],l3[b],masks[b],camera,sample)
            ids=observation['ids'];normals=observation['normals'];groups=observation.get('regions',np.ones(len(sample),int))
            comparison={}
            for name,mesh,landmarks in [('before',old,oldlm[state]),('after',new[b],new[b].metadata['flame_landmarks_3d'])]:
                pred=project(landmarks,k);normal=np.abs(((pred[ids]-l2[b][ids])*normals).sum(1))
                _,dist,_,_=closest_surface_numpy(mesh.vertices,mesh.faces,sample)
                surface={}
                for gi,gn in [(0,'jaw'),(1,'other_face'),(2,'neck')]:
                    d=dist[groups==gi]*1000
                    surface[gn]=dict(n=len(d),rmse_mm=float(np.sqrt(np.mean(d**2))),**stats(d)) if len(d) else None
                comparison[name]=dict(jaw_normal_px=stats(normal),jaw_all_reprojection_px=stats(np.linalg.norm(pred[:17]-l2[b][:17],axis=1)),
                    internal_reprojection_px=stats(np.linalg.norm(pred[17:]-l2[b][17:],axis=1)),surface=surface)
            comparison['accepted_contour_ids']=ids.tolist()
            comparison['stage2_change_mm']=stats(np.linalg.norm(new[b].vertices-old.vertices,axis=1)*1000)
            comparison['stage3_quality']=diagnostics[b].get('quality')
            item['states'][state]=comparison
            fig,axes=plt.subplots(1,4,figsize=(20,8))
            for ax,title,landmarks in zip(axes[:2],['原图 + 原 stage2 轮廓','原图 + 候选 stage2 轮廓'],[oldlm[state],new[b].metadata['flame_landmarks_3d']]):
                pred=project(landmarks,k);uv=l2[b]
                ax.imshow(cap['rgb_'+state]);ax.plot(uv[:17,0],uv[:17,1],color='lime',label='观测')
                ax.plot(pred[:17,0],pred[:17,1],color='red',label='拟合');ax.legend()
                ax.set_xlim(uv[:,0].min()-40,uv[:,0].max()+40);ax.set_ylim(uv[:,1].max()+45,uv[:,1].min()-55)
                ax.axis('off');ax.set_title(title)
            light=np.array([.3,.3,1]);light/=np.linalg.norm(light)
            for ax,m,title in zip(axes[2:],[old,new[b]],['原 stage2 几何','候选 stage2 几何']):
                vis=np.flatnonzero(m.face_normals[:,2]>.01);vis=vis[np.argsort(m.triangles_center[vis,2])]
                shade=np.clip(.35+.65*(m.face_normals[vis]@light),0,1)
                ax.add_collection(PolyCollection(m.triangles[vis,:,:2],facecolors=np.repeat(shade[:,None],3,axis=1),edgecolors='none'))
                ax.set_xlim(old.bounds[:,0]);ax.set_ylim(old.bounds[:,1]);ax.set_aspect('equal');ax.axis('off');ax.set_title(title)
            fig.tight_layout();fig.savefig(out/state/'comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
        assert all(sha(Path(p))==h for p,h in hashes.items());item['inputs_unchanged']=True
        write_json(out/'comparison.json',item);results[run.name]=item['states'];write_json(root/'summary.json',results)
        print(json.dumps({run.name:item['states']}),flush=True)
    print('PASS',flush=True)
if __name__=='__main__':main()
