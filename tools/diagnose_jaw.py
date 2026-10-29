"""Read-only diagnosis of jaw observations, fitted stage2 and final meshes."""
import os,sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def project(v,k):
    fx,fy,cx,cy=k;z=-v[:,2]
    return np.column_stack((fx*v[:,0]/z+cx,-fy*v[:,1]/z+cy))
def stats(v):
    v=np.asarray(v);return dict(mean=float(v.mean()),p95=float(np.percentile(v,95)),max=float(v.max())) if len(v) else None
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,nargs='+',required=True);a=ap.parse_args()
    out=ROOT/'diagnostics/jaw';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    report={}
    for run in a.runs:
        cap=np.load(run/'capture_pair.npz');prep=np.load(run/'prepared.npz');lm=np.load(run/'stage2_landmarks.npz')
        item={}
        for state in ('closed','open'):
            k=cap['intrinsics'];uv=prep['lmk2d_'+state];l3=prep['lmk3d_'+state];valid=prep['mask_'+state].astype(bool)
            fitted=lm[state];pred=project(fitted,k);parsing=prep['parsing_'+state];dep=cap['depth_'+state]
            base=trimesh.load(run/state/f'mesh_{state}_stage2.obj',force='mesh',process=False)
            final=trimesh.load(run/state/f'mesh_{state}.obj',force='mesh',process=False)
            offset=np.load(run/state/'vertex_offsets.npz')
            ref=offset['reference_vertices'];delta=offset['offsets'];proj=project(ref,k)
            low=(proj[:,1]>uv[33,1])&(proj[:,1]<uv[8,1]+5)&(proj[:,0]>uv[:17,0].min())&(proj[:,0]<uv[:17,0].max())
            ids=np.arange(17);good=valid[:17]&np.isfinite(l3[:17]).all(1)
            depths=[]
            for i in ids:
                x,y=np.rint(uv[i]).astype(int);patch=dep[max(y-2,0):y+3,max(x-2,0):x+3]
                depths.append(dict(id=int(i),valid=bool(good[i]),pixel_label=int(parsing[y,x]),
                    patch_nonzero=int((patch>0).sum()),patch_span_mm=float(np.ptp(patch.astype(float))),
                    reprojection_px=float(np.linalg.norm(pred[i]-uv[i])),
                    landmark_distance_mm=float(np.linalg.norm(fitted[i]-l3[i])*1000) if good[i] else None))
            points=prep['points_'+state];pp=project(points,k);pix=np.rint(pp).astype(int)
            inside=(pix[:,0]>=0)&(pix[:,0]<parsing.shape[1])&(pix[:,1]>=0)&(pix[:,1]<parsing.shape[0])
            labels=parsing[pix[inside,1],pix[inside,0]]
            item[state]=dict(jaw_reprojection_px=stats(np.linalg.norm(pred[:17]-uv[:17],axis=1)),
                jaw_distance_mm=stats(np.linalg.norm(fitted[:17][good]-l3[:17][good],axis=1)*1000),
                lower_face_stage3_move_mm=stats(np.linalg.norm(delta[low],axis=1)*1000),
                input_point_labels={str(int(l)):int((labels==l).sum()) for l in np.unique(labels)},jaw_points=depths,
                input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'capture_pair.npz',run/state/f'mesh_{state}_stage2.obj',run/state/f'mesh_{state}.obj']})
            if state!='closed':continue
            fig,axs=plt.subplots(1,4,figsize=(20,8))
            axs[0].imshow(cap['rgb_'+state]);axs[0].plot(uv[:17,0],uv[:17,1],'o-',color='lime',label='观测脸缘 0–16')
            axs[0].plot(pred[:17,0],pred[:17,1],'o-',color='red',label='stage2 脸缘')
            for i in ids:axs[0].text(uv[i,0]+2,uv[i,1],str(i),color='yellow',fontsize=8)
            axs[0].legend(loc='upper right');axs[0].set_title('原图与下颌观测／拟合轮廓')
            bounds=[uv[:,0].min()-45,uv[:,0].max()+45,uv[:,1].max()+50,uv[:,1].min()-60]
            axs[0].set_xlim(bounds[:2]);axs[0].set_ylim(bounds[2:]);axs[0].axis('off')
            axs[1].imshow(parsing,cmap='tab20',vmin=0,vmax=19);axs[1].plot(uv[:17,0],uv[:17,1],'w.-')
            axs[1].set_xlim(bounds[:2]);axs[1].set_ylim(bounds[2:]);axs[1].axis('off');axs[1].set_title('分割：检查脸／颈边界')
            light=np.array([.3,.3,1]);light/=np.linalg.norm(light)
            for ax,m,title in zip(axs[2:],[base,final],['stage2：原始拟合','stage3：最终网格']):
                visible=np.flatnonzero(m.face_normals[:,2]>.01);visible=visible[np.argsort(m.triangles_center[visible,2])]
                shade=np.clip(.35+.65*(m.face_normals[visible]@light),0,1)
                ax.add_collection(PolyCollection(m.triangles[visible,:,:2],facecolors=np.repeat(shade[:,None],3,axis=1),edgecolors='none'))
                ax.set_xlim(base.bounds[:,0]);ax.set_ylim(base.bounds[:,1]);ax.set_aspect('equal');ax.axis('off');ax.set_title(title)
            fig.suptitle(run.name);fig.tight_layout();fig.savefig(out/f'{run.name}.png',dpi=140);plt.close(fig)
        report[run.name]=item
        print(json.dumps({run.name:{s:{k:v for k,v in data.items() if k not in ('jaw_points','input_hashes')} for s,data in item.items()}}),flush=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
