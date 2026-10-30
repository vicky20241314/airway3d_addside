"""Same-frame depth diagnostics. These are NOT independent scan accuracy metrics."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import trimesh
from audit import dump


def raster_depth(vertices,faces,intr,shape,stride=4):
    """Perspective-correct triangle z buffer at actual pixel centres (CPU)."""
    h,w=shape; hh=(h+stride-1)//stride;ww=(w+stride-1)//stride
    z=-vertices[:,2];fx,fy,cx,cy=intr
    uv=np.column_stack((fx*vertices[:,0]/np.maximum(z,1e-8)+cx,-fy*vertices[:,1]/np.maximum(z,1e-8)+cy))/stride
    image=np.full((hh,ww),np.inf)
    for face in faces:
        if (z[face]<=.01).any():continue
        tri=uv[face];x0,y0=np.maximum(np.ceil(tri.min(0)).astype(int),0)
        x1,y1=np.minimum(np.floor(tri.max(0)).astype(int),[ww-1,hh-1])
        if x1<x0 or y1<y0:continue
        x,y=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1))
        a,b,c=tri
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10:continue
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        inside=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1.0000001)
        inv=u/z[face[0]]+v/z[face[1]]+(1-u-v)/z[face[2]]
        value=np.where(inside,1/np.maximum(inv,1e-12),np.inf)
        image[y0:y1+1,x0:x1+1]=np.minimum(image[y0:y1+1,x0:x1+1],value)
    return image


def diagnostic(v,f,d,stride=4):
    rendered=raster_depth(v,f,d['intrinsics'],d['depth'].shape,stride)
    observed=d['depth'][::stride,::stride]*.001
    valid=d['face_mask'][::stride,::stride]&(observed>.1)&(observed<1.5)
    covered=valid&np.isfinite(rendered)
    error=np.abs(rendered[covered]-observed[covered])*1000
    return dict(valid_face_pixels=int(valid.sum()),rendered_face_pixels=int(covered.sum()),
        ray_coverage_fraction=float(covered.sum()/max(valid.sum(),1)),
        depth_residual_median_mm=float(np.median(error)) if len(error) else None,
        depth_residual_p95_mm=float(np.percentile(error,95)) if len(error) else None,
        supported_within_5mm_fraction=float(np.count_nonzero(error<5)/max(valid.sum(),1)),
        surface_error_mm=None,contour_error_px=None,
        surface_error_status='unverified: no independent reference surface/extrinsics',
        contour_error_status='unverified: no human face silhouette',
        metric_type='in-sample camera-axis depth residual, not Euclidean surface error')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--capture-session',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--b',type=Path,required=True)
    p.add_argument('--b-front',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    sequence=json.loads(a.sequence.read_text(encoding='utf-8'))
    session=json.loads(a.capture_session.read_text(encoding='utf-8'))
    benchmark=json.loads((a.baseline/'benchmark.json').read_text(encoding='utf-8'))
    folder=Path(benchmark['result_folder']);extrinsics=json.loads((folder/'extrinsics.json').read_text(encoding='utf-8'))
    baseline_mesh={state:trimesh.load(folder/state/('mesh_'+state+'_stage2.obj'),process=False) for state in ('closed','open')}
    b_report=json.loads((a.b/'report.json').read_text(encoding='utf-8'))
    arrays=np.load(a.b/'parameters.npz',allow_pickle=False)
    front_report=json.loads((a.b_front/'report.json').read_text(encoding='utf-8')) if a.b_front else None
    front_arrays=np.load(a.b_front/'parameters.npz',allow_pickle=False) if a.b_front else None
    rows=[]
    for obs in session['observations']:
        idx=obs['quality']['temporal']['representative_rgb_index'];fid=obs['view_id']+'_'+str(idx)
        source=next(r for r in sequence['frames'] if r['id']==fid)
        with np.load(a.sequence.parent/source['file'],allow_pickle=False) as data:d={k:data[k] for k in data.files}
        bi=next(i for i,r in enumerate(b_report['frames']) if r['id']==fid)
        r=dict(frame_id=fid,view=obs['view'],state=obs['state'],role='training observation',
            independent_extrinsics=False,baseline=None,B=None,A=None,A_status='not yet available',
            b_initial_yaw_deg=b_report['frames'][bi]['initial_yaw_deg'])
        if obs['view_id'] in extrinsics:
            T=extrinsics[obs['view_id']];mesh=baseline_mesh[obs['state']]
            vertices=np.asarray(mesh.vertices)@np.array(T['R']).T+np.array(T['t'])
            # Representative RGB frame shares the observation coordinate frame.
            r['baseline']=diagnostic(vertices,np.array(mesh.faces),d)
        else:r['baseline_status']='view excluded by baseline'
        vertices=arrays['posed_vertices'][bi]
        from fit_b import normals
        vertices=vertices+normals(vertices,arrays['faces'])*arrays['deviation_m'][:,None]
        r['B']=diagnostic(vertices,arrays['faces'],d)
        if front_arrays is not None and obs['view']=='front':
            fi=next(i for i,row in enumerate(front_report['frames']) if row['id']==fid)
            v=front_arrays['posed_vertices'][fi]
            v=v+normals(v,front_arrays['faces'])*front_arrays['deviation_m'][:,None]
            r['B_front_only']=diagnostic(v,front_arrays['faces'],d)
            r['B_front_change_median_mm']=r['B']['depth_residual_median_mm']-r['B_front_only']['depth_residual_median_mm']
        rows.append(r);print(fid+': '+json.dumps(r,ensure_ascii=False),flush=True)
    dump(a.output/'comparison.json',dict(baseline=benchmark,B={k:b_report[k] for k in ['elapsed_s','cuda_peak_allocated_mib','cuda_peak_reserved_mib','device','steps']},
        frames=rows,elapsed_s=time.perf_counter()-started,
        acceptance=dict(front_surface_error='unverified',side_surface_error='unverified',
            silhouette_error='unverified',independent_effective_coverage='unverified',front_degradation='unverified'),
        conclusion='No claim of real-side accuracy improvement is supported by this in-sample diagnostic.',
        missing=['held-out oblique observations with independently calibrated external poses','human face silhouettes','independent head scan'],
        frame_policy='same four raw representative frames from one real capture; baseline fit uses fused depths from those bursts',
        independence_warning='Both methods estimate poses using fitting data. B and baseline use different expression models. Residuals cannot establish scan accuracy.'))


if __name__=='__main__':main()
