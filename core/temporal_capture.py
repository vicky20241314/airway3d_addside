"""Stationary per-view RGB-D bursts with temporal landmark/depth consensus."""
import math
import warnings
import cv2
import numpy as np
from core.camera import backproject_window,coordinate_transform
from core.multiview_data import Observation


class BurstQualityError(ValueError):
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics


def _median(a,axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(a,axis=axis)


def detect_frame(rgb):
    from core.face_test import detect_face_bbox
    from third_party.tufa.Test import get_lmk_single
    box=detect_face_bbox(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    return box,None if box is None else get_lmk_single(rgb,box)


def prepare_burst(frames,camera,*,detector=None,progress=lambda done,total:None):
    """Fuse only a majority stationary cluster; never average moving cameras.

    One representative sharp RGB remains for parsing/display. Depth and all
    landmark constraints use the accepted temporal observations, not that RGB
    alone. Temporal agreement does not prove side landmark visibility.
    """
    if not 5<=len(frames)<=15:raise ValueError('每视角需要5–15帧，请停稳后等待缓存')
    if len({(f[0],f[1]) for f in frames})!=1:raise ValueError('禁止跨状态或跨视角融合')
    intr=np.array([camera.fx,camera.fy,camera.cx,camera.cy])
    if any(not np.allclose(f[5],intr,rtol=0,atol=1e-5) for f in frames):raise ValueError('采集期间内参变化，请重采')
    times=np.array([f[4] for f in frames],float)
    if not np.isfinite(times).all() or not (np.diff(times)>0).all() or times[-1]-times[0]>1.:
        raise ValueError('短序列必须按时间递增且不超过1秒，请重新停稳采集')
    state,view=frames[0][:2];shape=frames[0][3].shape
    for f in frames:
        if f[3].shape!=shape or f[2].shape!=(*shape,3) or not np.isfinite(f[3]).all() or (f[3]<0).any():
            raise ValueError('短序列RGB/深度尺寸或数值无效')
    n=len(frames);required=max(5,math.ceil(.6*n));detector=detector or detect_frame
    uv=np.full((n,68,2),np.nan);xyz=np.full((n,68,3),np.nan);valid=np.zeros((n,68),bool)
    rows=[];sharp=np.zeros(n);boxes=[];candidate=np.zeros(n,bool);stable=np.arange(27,48)
    for i,f in enumerate(frames):
        progress(i+1,n)
        row=dict(index=i,status='rejected',reason='detection_failed');rows.append(row);boxes.append(None)
        box,points=detector(f[2])
        if box is None or points is None:continue
        points=np.asarray(points,float)
        if points.shape!=(68,2):continue
        x,y,w,h=map(int,box);height,width=shape
        x0,y0=max(0,x),max(0,y);x1,y1=min(width,x+w),min(height,y+h)
        if x1<=x0 or y1<=y0:continue
        boxes[i]=(x0,y0,x1,y1)
        crop=cv2.cvtColor(f[2][y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
        crop=cv2.resize(crop,(128,128),interpolation=cv2.INTER_AREA)
        sharp[i]=float(cv2.Laplacian(crop,cv2.CV_64F).var());row['sharpness']=sharp[i]
        uv[i]=points
        finite=np.isfinite(points).all(1)
        p,good=backproject_window(points[finite],f[3],camera,window=2,trim=0,keep_shape=True)
        xyz[i,finite]=coordinate_transform(p);valid[i,finite]=good
        candidate[i]=valid[i,stable].sum()>=6
        row['reason']='pending' if candidate[i] else 'insufficient_stable_depth'
    if candidate.any():
        threshold=.3*float(np.median(sharp[candidate]))
        blurry=candidate&(sharp<threshold);candidate[blurry]=False
        for i in np.flatnonzero(blurry):rows[i]['reason']='blurred_relative_to_burst'
    info=dict(input_frames=n,minimum_accepted=required,frames=rows)
    if candidate.sum()<required:raise BurstQualityError('清晰且有可靠深度的帧不足，请停稳后重采',info)
    masked=np.where(valid[...,None],uv,np.nan)
    center=_median(masked[candidate]);scores=np.full(n,np.inf)
    for i in np.flatnonzero(candidate):
        ids=stable[valid[i,stable]&np.isfinite(center[stable]).all(1)]
        scores[i]=float(np.median(np.linalg.norm(uv[i,ids]-center[ids],axis=1)))
        lip=np.arange(48,68);lip=lip[valid[i,lip]&np.isfinite(center[lip]).all(1)]
        if len(lip):scores[i]+=.2*float(np.mean(np.linalg.norm(uv[i,lip]-center[lip],axis=1)))
    reference=int(np.lexsort((-sharp,scores))[0]);accepted=np.zeros(n,bool)
    pairs=((50,58),(51,57),(52,56))
    for i in np.flatnonzero(candidate):
        shared=stable[valid[i,stable]&valid[reference,stable]]
        if len(shared)<6:rows[i]['reason']='insufficient_shared_stable_depth';continue
        motion_px=float(np.median(np.linalg.norm(uv[i,shared]-uv[reference,shared],axis=1)))
        motion_m=float(np.median(np.linalg.norm(xyz[i,shared]-xyz[reference,shared],axis=1)))
        rows[i].update(motion_px=motion_px,motion_m=motion_m)
        if motion_px>2. or motion_m>.003:rows[i]['reason']='camera_or_head_motion';continue
        lip_pairs=[(a,b) for a,b in pairs if valid[i,a] and valid[i,b] and valid[reference,a] and valid[reference,b]]
        if len(lip_pairs)>=2:
            gap=np.median([np.linalg.norm(xyz[i,a]-xyz[i,b]) for a,b in lip_pairs])
            ref_gap=np.median([np.linalg.norm(xyz[reference,a]-xyz[reference,b]) for a,b in lip_pairs])
            if abs(gap-ref_gap)>max(.002,.1*ref_gap):rows[i]['reason']='mouth_state_changed';continue
        accepted[i]=True;rows[i].update(status='accepted',reason=None)
    count=int(accepted.sum());info.update(accepted_frames=count,reference_index=reference)
    if count<required:raise BurstQualityError(f'稳定一致帧只有{count}/{n}，至少需要{required}帧；请保持头颈/张口状态并停稳相机重采',info)
    depths=np.stack([frames[i][3] for i in np.flatnonzero(accepted)]).astype(np.float32)
    depths[depths<=0]=np.nan
    med=_median(depths);support=np.isfinite(depths).sum(0)
    mad=_median(np.abs(depths-med))
    keep=(support>=max(3,math.ceil(count*.5)))&np.isfinite(med)&(mad<=2.)
    depth=np.where(keep,np.rint(np.nan_to_num(med)),0).clip(0,65535).astype(np.uint16)
    # Landmark voting uses each frame's independently detected UV and depth.
    uvs=masked[accepted];points=np.where(valid[...,None],xyz,np.nan)[accepted]
    consensus=_median(uvs);center3=_median(points)
    votes=valid[accepted].sum(0)
    spread_px=_median(np.linalg.norm(uvs-consensus,axis=2))
    spread_m=_median(np.linalg.norm(points-center3,axis=2))
    good=(votes>=max(3,math.ceil(.6*count)))&(spread_px<=2.)&(spread_m<=.004)
    z=-center3[:,2]
    result3=np.column_stack(((consensus[:,0]-camera.cx)*z/camera.fx,-(consensus[:,1]-camera.cy)*z/camera.fy,-z))
    good&=np.isfinite(result3).all(1)&np.isfinite(consensus).all(1)&(z>0)
    # Representative RGB must itself be a sharp accepted frame near consensus.
    choices=np.flatnonzero(accepted&(sharp>=np.median(sharp[accepted])))
    representative=int(choices[np.argmin(scores[choices])])
    x0,y0,x1,y1=boxes[representative];region=keep[y0:y1,x0:x1]
    if int(region.sum())<100:raise BurstQualityError('融合后可靠人脸深度不足，请重采',info)
    jitter=float(np.median(mad[y0:y1,x0:x1][region]))
    info.update(representative_rgb_index=representative,landmark_votes=votes.tolist(),
        landmark_spread_px=np.nan_to_num(spread_px,nan=-1.).tolist(),
        landmark_spread_mm=np.nan_to_num(spread_m*1000,nan=-1.).tolist(),
        depth_mad_median_mm=jitter,reliable_face_depth_fraction=float(region.mean()))
    quality=dict(motion_mode='camera_moved',frames=count,input_frames=n,timestamps_s=times.tolist(),
        weight=float(np.clip((count/n)/(1+jitter/2),.25,1)),temporal=info,
        source='stationary_multiframe_consensus')
    temporal=dict(frame_uv=uv.astype(np.float32),frame_xyz=xyz.astype(np.float32),frame_valid=valid,
        accepted=accepted,timestamps_s=times,depth_support=support.astype(np.uint8),
        depth_mad_mm=np.nan_to_num(mad,nan=-1.).astype(np.float32))
    return Observation(f'{state}_{view}',state,view,frames[representative][2].copy(),depth,camera,
        consensus.astype(np.float32),result3.astype(np.float32),good,quality,temporal=temporal)
