"""All-frame in-sample diagnostics, matched front controls and geometry-only QA.

The fitting mask is deliberately NOT used to remove difficult evaluation pixels.
Baseline metrics use the original four representative raw frames only, because
that algorithm does not provide per-frame poses for the remaining burst frames.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial.transform import Rotation

from audit import dump,sha
from evaluate import raster_depth
from fit_b import normals


def load_fit(path):
    report=json.loads((path/'report.json').read_text(encoding='utf-8'))
    with np.load(path/'parameters.npz',allow_pickle=False) as arrays:p={k:arrays[k] for k in arrays.files}
    return report,p,{r['id']:i for i,r in enumerate(report['frames'])}


def refined(p,i):
    if 'refined_vertices' in p:return p['refined_vertices'][i]
    v=p['posed_vertices'][i]
    return v+normals(v,p['faces'])*p['deviation_m'][:,None]


def region_masks(d,stride):
    h,w=d['depth'].shape;uv=d['uv']
    mouth=np.zeros((h,w),np.uint8)
    if np.isfinite(uv[48:60]).all():
        p=uv[48:60];x0,y0=np.floor(p.min(0)-15).astype(int);x1,y1=np.ceil(p.max(0)+15).astype(int)
        mouth[max(y0,0):min(y1+1,h),max(x0,0):min(x1+1,w)]=1
    yy,xx=np.mgrid[:h:stride,:w:stride]
    return dict(full=np.ones_like(yy,bool),mouth=mouth[::stride,::stride].astype(bool),
        lower_face=yy>uv[33,1],left_cheek=(xx<uv[33,0]-20)&(yy>uv[27,1])&(yy<uv[57,1]),
        right_cheek=(xx>uv[33,0]+20)&(yy>uv[27,1])&(yy<uv[57,1]))


def metrics(rendered,d,stride,regions):
    observed=d['depth'][::stride,::stride]*.001
    valid=d['face_mask'][::stride,::stride]&(observed>.1)&(observed<1.5)
    result={}
    for name,region in regions.items():
        mask=valid&region;covered=mask&np.isfinite(rendered)
        error=np.abs(rendered[covered]-observed[covered])*1000
        result[name]=dict(valid_pixels=int(mask.sum()),covered_pixels=int(covered.sum()),
            ray_coverage_fraction=float(covered.sum()/max(mask.sum(),1)),
            median_mm=float(np.median(error)) if len(error) else None,
            p95_mm=float(np.percentile(error,95)) if len(error) else None,
            over_5mm_fraction=float(np.count_nonzero(error>5)/max(mask.sum(),1)),
            over_10mm_fraction=float(np.count_nonzero(error>10)/max(mask.sum(),1)))
        if name=='full':
            bad=(covered & (np.abs(rendered-observed)>.005)).astype(np.uint8)
            _,_,stats,_=cv2.connectedComponentsWithStats(bad,connectivity=8)
            components=sorted(stats[1:].tolist(),key=lambda x:x[4],reverse=True)
            result[name]['largest_over_5mm_components']=[dict(
                raw_pixel_bbox=[int(x*stride),int(y*stride),int(w*stride),int(h*stride)],
                evaluated_pixels=int(area)) for x,y,w,h,area in components[:5] if area>=4]
    return result


def geometry_sheet(fits,output):
    fig,axes=plt.subplots(len(fits),5,figsize=(16,4*len(fits)),squeeze=False)
    for row,(name,(report,p,_)) in enumerate(fits.items()):
        reference=report['references']['closed'];i=next(i for i,r in enumerate(report['frames']) if r['id']==reference['frame_id'])
        canonical=(refined(p,i)-p['t'][i])@p['R'][i];f=p['faces']
        # Crop only the plotting window, never modify or decimate the mesh.
        for col,angle in enumerate((0,-30,30,-90,90)):
            R=Rotation.from_euler('y',angle,degrees=True).as_matrix();v=canonical@R.T
            tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
            shade=.25+.75*np.maximum(0,n@np.array([-.3,.4,.866]))
            color=np.column_stack((shade*.78,shade*.82,shade*.86,np.ones(len(shade))))
            order=np.argsort(tri[:,:,2].mean(1));ax=axes[row,col]
            ax.add_collection(PolyCollection(tri[order,:,:2]*1000,facecolors=color[order],edgecolors='none',antialiased=False))
            ax.set_xlim(-105,105);ax.set_ylim(-110,120);ax.set_aspect('equal');ax.set_facecolor('#20252b')
            ax.set_title(f'{name}: {angle:+} deg'+(' | prior profile' if abs(angle)==90 else ''),fontsize=10)
            ax.set_xlabel('mm');ax.set_ylabel('mm')
    fig.suptitle('Untextured metric geometry; +/-90 views include unobserved model completion',fontsize=13)
    fig.tight_layout();fig.savefig(output/'geometry_closed.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,len(fits),figsize=(5*len(fits),5),squeeze=False)
    for col,(name,(report,p,_)) in enumerate(fits.items()):
        reference=report['references']['open'];i=next(i for i,r in enumerate(report['frames']) if r['id']==reference['frame_id'])
        v=refined(p,i);tri=v[p['faces']];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
        color=np.repeat((.2+.8*np.maximum(n[:,2],0))[:,None],3,1);order=np.argsort(tri[:,:,2].mean(1))
        ax=axes[0,col];ax.add_collection(PolyCollection(tri[order,:,:2]*1000,facecolors=color[order],edgecolors='none',antialiased=False))
        landmarks=(v[p['faces'][p['landmark_faces'][i]]]*p['landmark_barycentric'][i,:,:,None]).sum(1) if 'landmark_faces' in p else p['landmarks'][i]
        ax.scatter(landmarks[48:,0]*1000,landmarks[48:,1]*1000,s=6,c='#e94f37')
        ax.autoscale();ax.set_xlim(landmarks[48:,0].mean()*1000-60,landmarks[48:,0].mean()*1000+60)
        ax.set_ylim(landmarks[48:,1].mean()*1000-45,landmarks[48:,1].mean()*1000+45);ax.set_aspect('equal');ax.set_title(name+' open mouth / mm')
    fig.tight_layout();fig.savefig(output/'geometry_open_mouth.png',dpi=160);plt.close(fig)


def support_sheet(fit,labels,output):
    report,p,_=fit;i=next(i for i,r in enumerate(report['frames']) if r['id']==report['references']['closed']['frame_id'])
    canonical=(refined(p,i)-p['t'][i])@p['R'][i];f=p['faces']
    # A triangle has repeated support only if all of its vertices have it.
    face_label=labels[f].min(1)
    palette=np.array([[.45,.45,.48],[.90,.68,.2],[.20,.68,.85]])
    fig,axes=plt.subplots(1,5,figsize=(16,4))
    for ax,angle in zip(axes,[0,-30,30,-90,90]):
        v=canonical@Rotation.from_euler('y',angle,degrees=True).as_matrix().T;tri=v[f]
        order=np.argsort(tri[:,:,2].mean(1))
        ax.add_collection(PolyCollection(tri[order,:,:2]*1000,facecolors=palette[face_label[order]],edgecolors='none',antialiased=False))
        ax.set_xlim(-105,105);ax.set_ylim(-110,120);ax.set_aspect('equal');ax.set_title(f'{angle:+} deg');ax.set_xlabel('mm')
    fig.suptitle('Blue: repeated measured support; yellow: insufficient/model prior; grey: unobserved completion')
    fig.tight_layout();fig.savefig(output/'support_regions.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--original-comparison',type=Path,required=True)
    p.add_argument('--b',type=Path,required=True);p.add_argument('--b-front',type=Path,required=True)
    p.add_argument('--c',type=Path,required=True);p.add_argument('--c-front',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();sequence=json.loads(a.sequence.read_text(encoding='utf-8'))
    original=json.loads(a.original_comparison.read_text(encoding='utf-8'))
    representatives={r['frame_id']:r for r in original['frames']}
    fits={name:load_fit(path) for name,path in [('B',a.b),('B_front',a.b_front),('C',a.c),('C_front',a.c_front)]}
    frames=[];heatmaps=[];stride=4
    for source in sequence['frames']:
        fid=source['id']
        with np.load(a.sequence.parent/source['file'],allow_pickle=False) as d:data={k:d[k] for k in d.files}
        regions=region_masks(data,stride);results={};renders={}
        for name,(report,params,index) in fits.items():
            if fid not in index:continue
            i=index[fid];image=raster_depth(refined(params,i),params['faces'],data['intrinsics'],data['depth'].shape,stride)
            renders[name]=image;results[name]=metrics(image,data,stride,regions)
            if name=='C':
                coarse=raster_depth(params['posed_vertices'][i],params['faces'],data['intrinsics'],data['depth'].shape,stride)
                results['C_coarse']=metrics(coarse,data,stride,regions)
                if report.get('front_anchor') and source['view']!='front':
                    before=raster_depth(params['before_side_detail_vertices'][i],params['faces'],data['intrinsics'],data['depth'].shape,stride)
                    results['C_before_side_detail']=metrics(before,data,stride,regions)
        cindex=fits['C'][2][fid];cparams=fits['C'][1]
        row=dict(frame_id=fid,state=source['state'],view=source['view'],role='training data diagnosis',methods=results,
            fitted_yaw_deg=float(Rotation.from_matrix(cparams['R'][cindex]).as_euler('xyz',degrees=True)[1]))
        if fid in representatives:
            row['baseline']=representatives[fid]['baseline'];heatmaps.append((fid,data,renders))
        if source['view']=='front':
            row['front_regression']={}
            for name in ['B','C']:
                row['front_regression'][name]={region:{metric:results[name][region][metric]-results[name+'_front'][region][metric]
                    for metric in ['median_mm','p95_mm','ray_coverage_fraction']} for region in regions}
            # Matched coverage diagnostic avoids a method improving by missing rays.
            common=np.isfinite(renders['C'])&np.isfinite(renders['C_front'])
            common&=data['face_mask'][::stride,::stride]&(data['depth'][::stride,::stride]>100)
            obs=data['depth'][::stride,::stride]*.001
            row['matched_front_full']={name:dict(median_mm=float(np.median(np.abs(renders[name][common]-obs[common]))*1000),
                p95_mm=float(np.percentile(np.abs(renders[name][common]-obs[common]),95)*1000),pixels=int(common.sum())) for name in ['C','C_front']}
        frames.append(row)
        print(json.dumps(dict(frame=fid,C=results['C']['full'],front_change=row.get('front_regression',{}).get('C',{}).get('full'))),flush=True)
    fig,axes=plt.subplots(4,3,figsize=(14,15))
    for ri,(fid,data,renders) in enumerate(heatmaps):
        obs=data['depth'][::stride,::stride]*.001;valid=data['face_mask'][::stride,::stride]&(obs>.1)
        for ci,name in enumerate(['B','C','C_front']):
            ax=axes[ri,ci];ax.set_title(fid+' / '+name);ax.axis('off')
            if name not in renders:continue
            error=np.abs(renders[name]-obs)*1000;error[~valid]=np.nan
            shown=ax.imshow(error,vmin=0,vmax=10,cmap='magma');yy,xx=np.where(valid)
            ax.set_xlim(xx.min()-5,xx.max()+5);ax.set_ylim(yy.max()+5,yy.min()-5)
    fig.colorbar(shown,ax=axes.ravel().tolist(),label='absolute camera-depth residual / mm',shrink=.5)
    fig.savefig(a.output/'depth_residuals.png',dpi=140);plt.close(fig)
    geometry_sheet({name:fits[name] for name in ['B','C']},a.output)
    with np.load(a.c/'support.npz',allow_pickle=False) as evidence:
        support_sheet(fits['C'],evidence['region_label'],a.output)
    mesh_checks={}
    for state in ['closed','open']:
        mesh=trimesh.load(a.c/f'{state}_refined_m.obj',process=False)
        mesh_checks[state]=dict(finite=bool(np.isfinite(mesh.vertices).all()),vertices=len(mesh.vertices),triangles=len(mesh.faces),
            extent_m=mesh.extents,mesh_unit='m',watertight=bool(mesh.is_watertight),export_roundtrip=True)
    controls={}
    for name in ['B','C']:
        front=[r for r in frames if r['view']=='front']
        controls[name]=dict(max_p95_regression_mm=max(r['front_regression'][name]['full']['p95_mm'] for r in front),
            mean_p95_regression_mm=float(np.mean([r['front_regression'][name]['full']['p95_mm'] for r in front])),
            worse_p95_frames=[r['frame_id'] for r in front if r['front_regression'][name]['full']['p95_mm']>0],
            scope='training-frame diagnosis; any positive difference is reported without tolerance redefinition')
    report=dict(schema='head_rgbd.comparison.c.v1',sequence_sha256=sha(a.sequence),baseline=original['baseline'],frames=frames,
        timing={name:{k:r[k] for k in ['elapsed_s','fitting_s','cuda_peak_allocated_mib','cuda_peak_reserved_mib','steps','device']} for name,(r,_,_) in fits.items()},
        front_controls=controls,mesh_checks=mesh_checks,evaluation_elapsed_s=time.perf_counter()-start,
        evaluation_mask='all original semantic face pixels with valid 0.1-1.5m depth; no fitting erosion/outlier masking',
        metric='camera-axis depth residual on training data; coverage is fitted-camera ray coverage',
        acceptance=dict(surface_error_mm=None,silhouette_error_px=None,independent_coverage=None,real_side_accuracy='unverified',
            reason='no independent extrinsics, manual silhouettes or independent scan'),
        limitations=['baseline evaluated only on four representatives; its burst fusion differs from single-frame input',
            'depth quantisation, semantic errors and mouth-cavity ambiguity remain','profile views at 90 degrees show model completion, not measured evidence'])
    report['guarded_identity']=fits['C'][0].get('front_anchor')
    dump(a.output/'comparison.json',report)
    print(json.dumps(dict(front_controls=controls,mesh_checks=mesh_checks),default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':main()
