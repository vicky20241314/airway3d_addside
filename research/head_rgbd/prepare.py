"""Adapt saved raw bursts to per-frame metric observations, without depth averaging."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from audit import dump, sha


def backproject(uv, depth, intr):
    uv=np.asarray(uv,float)
    pix=np.rint(np.nan_to_num(uv)).astype(int)
    h,w=depth.shape
    valid=np.isfinite(uv).all(1)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    z=np.zeros(len(uv)); ids=np.flatnonzero(valid)
    z[ids]=depth[pix[ids,1],pix[ids,0]]*.001
    valid&=(z>.1)&(z<1.5)
    fx,fy,cx,cy=intr
    xyz=np.column_stack(((uv[:,0]-cx)*z/fx,-(uv[:,1]-cy)*z/fy,-z))
    return xyz.astype(np.float32),valid


def rigid(source,target):
    x=source-source.mean(0); y=target-target.mean(0)
    u,s,vh=np.linalg.svd(x.T@y)
    if s[1]<1e-10: raise ValueError('degenerate rigid alignment')
    fix=np.eye(3);fix[2,2]=np.linalg.det(vh.T@u.T)
    r=vh.T@fix@u.T
    return r,target.mean(0)-source.mean(0)@r.T


def track_overlap(previous_rgb,rgb,previous_depth,depth,intr,face_mask,upper_fraction=.6,previous_intr=None):
    """Upper-face KLT with forward/backward check; robust metric rigid estimate."""
    gray0=cv2.cvtColor(previous_rgb,cv2.COLOR_RGB2GRAY)
    gray1=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    mask=face_mask.astype(np.uint8)*255
    ys=np.nonzero(mask)[0]
    if not len(ys): return dict(status='lost',reason='no_face_mask')
    if upper_fraction<1:mask[int(np.percentile(ys,upper_fraction*100)):]=0
    p=cv2.goodFeaturesToTrack(gray0,160,.01,6,mask=mask)
    if p is None: return dict(status='lost',reason='no_tracks')
    q,ok,_=cv2.calcOpticalFlowPyrLK(gray0,gray1,p,None)
    back,good,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,q,None)
    keep=ok[:,0].astype(bool)&good[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.)
    a,va=backproject(p[:,0],previous_depth,intr if previous_intr is None else previous_intr); b,vb=backproject(q[:,0],depth,intr)
    keep&=va&vb
    if keep.sum()<8: return dict(status='lost',reason='insufficient_overlap',tracks=int(keep.sum()))
    a,b=a[keep],b[keep]
    for _ in range(3):
        r,t=rigid(a,b); residual=np.linalg.norm(a@r.T+t-b,axis=1)
        keep=residual<max(.003,float(np.median(residual)*2.5))
        if keep.sum()<8: break
        a,b=a[keep],b[keep]
    r,t=rigid(a,b); error=float(np.median(np.linalg.norm(a@r.T+t-b,axis=1)))
    return dict(status='tracked' if error<.008 else 'lost',tracks=len(a),median_m=error,
                previous_to_current_R=r,previous_to_current_t=t,
                direction='p_current = R @ p_previous + t',independent_extrinsics=False)


def input_chunks(args):
    if args.capture:
        for raw in sorted(args.capture.glob('attempt_*/raw_sequence.npz')):
            info=json.loads((raw.parent/'session.json').read_text(encoding='utf-8'))
            if info.get('depth_unit')!='mm' or info.get('coordinate_convention')!='X_right_Y_up_Z_backward':
                raise ValueError('Unsupported capture depth/coordinate declaration')
            entry=info['observations'][0]
            with np.load(raw.parent/entry['file'],allow_pickle=False) as obs:cached={k:obs[k] for k in obs.files}
            with np.load(raw,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
            provenance=dict(path=str(raw.resolve()),sha256=sha(raw),
                observation_sha256=sha(raw.parent/entry['file']),manifest_sha256=sha(raw.parent/'session.json'))
            yield raw,entry,cached,arrays,raw.parent.name,provenance
    else:
        manifest=args.frame_manifest.resolve();info=json.loads(manifest.read_text(encoding='utf-8'))
        if info.get('schema')!='head_rgbd.raw.v1' or info.get('depth_unit')!='mm' or info.get('rgb_depth_aligned') is not True:
            raise ValueError('General chunks require explicit schema, mm depth and declared RGB/depth alignment')
        for entry in info['chunks']:
            raw=(manifest.parent/entry['file']).resolve()
            if entry['state'] not in ('closed','open') or entry['view'] not in ('front','left','right'):
                raise ValueError('Reference state/view label unsupported; expression is still fitted per frame')
            for key in ('view_id','segment'):
                if not entry[key] or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in entry[key]):
                    raise ValueError('Unsafe chunk identity')
            with np.load(raw,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
            count=len(arrays['rgb'])
            if arrays['uv'].shape!=(count,68,2):raise ValueError('Provide independent per-frame 68-point detections as uv[N,68,2]')
            cached=dict(temporal_frame_uv=arrays['uv'],temporal_frame_valid=arrays.get('landmark_valid',np.isfinite(arrays['uv']).all(2)))
            if arrays['intrinsics'].shape==(4,):arrays['intrinsics']=np.tile(arrays['intrinsics'],(count,1))
            provenance=dict(path=str(raw),sha256=sha(raw),manifest_sha256=sha(manifest))
            yield raw,entry,cached,arrays,entry['segment'],provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--capture',type=Path)
    inputs.add_argument('--frame-manifest',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-cache',type=Path,required=True)
    p.add_argument('--copy-cache-from',type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    if a.copy_cache_from:
        shutil.copytree(a.copy_cache_from,a.model_cache,dirs_exist_ok=True)
    os.environ['UNIFACE_CACHE_DIR']=str(a.model_cache)
    from core import mask_detection
    mask_detection._ONNX_PROVIDERS=['CPUExecutionProvider']
    from core.camera import Camera
    rows=[];sources=[];start=time.perf_counter();last=None;last_segment=None;last_stamp=None;counters={}
    for raw,entry,cached,data,segment,provenance in input_chunks(a):
        sources.append(provenance)
        count=len(data['rgb'])
        if any(len(data[key])!=count for key in ['depth','intrinsics','timestamps_s']) or len(cached['temporal_frame_uv'])!=count:
            raise ValueError('RGB, depth, intrinsics, timestamps and landmarks must have equal frame counts')
        for i,(rgb,depth,intr,stamp) in enumerate(zip(data['rgb'],data['depth'],data['intrinsics'],data['timestamps_s'])):
            index=counters.get(entry['view_id'],0);counters[entry['view_id']]=index+1
            fid=entry['view_id']+'_'+str(index)
            if segment!=last_segment or last_stamp is None or not 0<float(stamp)-last_stamp<.3:last=None
            row=dict(id=fid,segment=segment,state=entry['state'],view=entry['view'],index=index,
                timestamp_s=float(stamp),source=str(raw.resolve()),source_frame=i,
                status='excluded',reason=None,independent_extrinsics=False)
            rows.append(row)
            try:
                if not np.isfinite(stamp):raise ValueError('invalid_timestamp')
                if last_segment==segment and last_stamp is not None and float(stamp)<=last_stamp:raise ValueError('nonmonotonic_timestamp')
                if rgb.shape!=(*depth.shape,3): raise ValueError('unaligned_rgb_depth_sizes')
                if not np.isfinite(depth).all() or (depth<0).any():raise ValueError('invalid_depth_values')
                if not np.isfinite(intr).all() or min(intr[:2])<=0: raise ValueError('invalid_intrinsics')
                uv=cached['temporal_frame_uv'][i]
                xyz,valid=backproject(uv,depth,intr)
                valid&=cached['temporal_frame_valid'][i]
                if valid[27:48].sum()<6: raise ValueError('insufficient_upper_face_depth')
                segmentation=mask_detection.get_face_data_single(Camera(*intr),rgb,depth)['parsing']
                face=np.isin(segmentation,[1,2,3,4,5,7,8,10,12,13])
                # Erode unreliable depth/silhouette boundaries, preserve jaw geometry inside.
                face=cv2.erode(face.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                ys,xs=np.where(face&(depth>100)&(depth<1500))
                if len(xs)<200: raise ValueError('insufficient_face_depth')
                crop=cv2.cvtColor(rgb[max(0,ys.min()):ys.max()+1,max(0,xs.min()):xs.max()+1],cv2.COLOR_RGB2GRAY)
                sharp=float(cv2.Laplacian(cv2.resize(crop,(128,128)),cv2.CV_64F).var())
                overlap=track_overlap(last[0],rgb,last[1],depth,intr,last[2],previous_intr=last[4]) if last else dict(status='segment_start')
                neck_overlap=track_overlap(last[0],rgb,last[1],depth,intr,last[3],upper_fraction=1.,previous_intr=last[4]) if last else dict(status='segment_start')
                # A lost adjacent track triggers per-frame landmark reinitialization, not global failure.
                row.update(status='candidate',sharpness=sharp,face_depth_pixels=len(xs),
                    overlap=overlap,neck_overlap=neck_overlap,reinitialized=overlap['status']=='lost',file=fid+'.npz',
                    landmark_source='stored independent per-frame detections; XYZ recomputed from raw depth',
                    alignment_evidence='capture schema claims alignment; physical calibration not independently verified')
                np.savez_compressed(a.output/row['file'],rgb=rgb,depth=depth,intrinsics=intr,
                    uv=uv,xyz=xyz,valid=valid,face_mask=face,neck_mask=segmentation==14)
                last=(rgb,depth,face,segmentation==14,intr)
                last_segment=segment;last_stamp=float(stamp)
                print(fid+': candidate '+overlap['status'],flush=True)
            except (ValueError,KeyError) as exc:
                row['reason']=str(exc);print(fid+': excluded '+str(exc),flush=True)

    if not rows: raise ValueError('No raw bursts found')
    dump(a.output/'sequence.json',dict(schema='head_rgbd.sequence.v1',depth_unit='mm',mesh_unit='m',
        coordinate_convention='X_right_Y_up_Z_backward',sources=sources,frames=rows,
        continuous_full_turn=False,elapsed_s=time.perf_counter()-start,
        camera_model='Orbbec Gemini 215 (PnP verified on research host)' if a.capture else 'declared in external acquisition metadata; not hardware-verified',
        independent_extrinsics=False,human_silhouettes=False))
    print('Prepared '+str(sum(r['status']=='candidate' for r in rows))+'/'+str(len(rows))+' frames',flush=True)


if __name__=='__main__': main()
