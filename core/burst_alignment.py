"""Small camera-motion compensation, source camera -> reference camera (metres).

Use upper-face RGB tracks with depth, then verify held-out tracks. No scale,
mouth/neck fit, depth-hole filling, or cross-view fusion is allowed here.
"""
import cv2
import numpy as np
from core.camera import backproject_window, coordinate_transform
from core.multiview_geometry import project
from core.multiview_registration import rigid_svd


def warp_depth(depth, camera, transform, reference_depth):
    h,w=depth.shape;y,x=np.nonzero(depth>0);z=depth[y,x]*.001
    p=np.column_stack(((x-camera.cx)*z/camera.fx,-(y-camera.cy)*z/camera.fy,-z))
    uv,z=project(transform.to_view(p),camera);pix=np.rint(uv).astype(int)
    keep=(z>0)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    index=pix[keep,1]*w+pix[keep,0]
    out=np.full(h*w,np.inf);np.minimum.at(out,index,z[keep]*1000)
    out=out.reshape(h,w)
    # Explicit z-buffer and reference-depth check: no foreground/background mix.
    good=np.isfinite(out)&(reference_depth>0)&(np.abs(out-reference_depth)<=5.)
    return np.where(good,out,0).astype(np.float32)


def align_frame(rgb,depth,reference_rgb,reference_depth,camera,box):
    gray=lambda a:cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    a,b=gray(reference_rgb),gray(rgb);mask=np.zeros_like(a)
    x0,y0,x1,y1=box
    mask[y0:int(y0+.58*(y1-y0)),x0:x1]=255
    corners=cv2.goodFeaturesToTrack(a,maxCorners=220,qualityLevel=.01,minDistance=5,mask=mask)
    if corners is None or len(corners)<24:raise ValueError('上半脸可追踪纹理不足')
    tracked,status,_=cv2.calcOpticalFlowPyrLK(a,b,corners,None,winSize=(21,21),maxLevel=3)
    if tracked is None:raise ValueError('帧间追踪失败')
    back,reverse,_=cv2.calcOpticalFlowPyrLK(b,a,tracked,None,winSize=(21,21),maxLevel=3)
    if back is None:raise ValueError('反向追踪失败')
    keep=status.ravel().astype(bool)&reverse.ravel().astype(bool)
    keep&=np.linalg.norm(back[:,0]-corners[:,0],axis=1)<.75
    u,v=corners[keep,0],tracked[keep,0]
    finite=np.isfinite(u).all(1)&np.isfinite(v).all(1);u,v=u[finite],v[finite]
    target,ok1=backproject_window(u,reference_depth,camera,window=2,trim=0,keep_shape=True)
    source,ok2=backproject_window(v,depth,camera,window=2,trim=0,keep_shape=True)
    target=coordinate_transform(target[ok1&ok2]);source=coordinate_transform(source[ok1&ok2])
    if len(source)<24:raise ValueError('共同纹理深度不足24点')
    rng=np.random.default_rng(42);order=rng.permutation(len(source));split=len(order)*2//3
    train,test=order[:split],order[split:];best=None
    for _ in range(80):
        ids=rng.choice(train,3,replace=False)
        try:T=rigid_svd(source[ids],target[ids])
        except ValueError:continue
        err=np.linalg.norm(T.to_view(source[train])-target[train],axis=1);inside=err<.0025
        score=(int(inside.sum()),-float(np.minimum(err,.01).sum()))
        if best is None or score>best[0]:best=(score,inside)
    if best is None or best[0][0]<max(10,.6*len(train)):raise ValueError('小幅刚性运动估计不可靠')
    ids=train[best[1]];T=rigid_svd(source[ids],target[ids])
    error=np.linalg.norm(T.to_view(source[test])-target[test],axis=1)
    angle=float(np.degrees(np.arccos(np.clip((np.trace(T.R)-1)/2,-1,1))))
    motion=float(np.median(np.linalg.norm(T.to_view(source)-source,axis=1)))
    if np.median(error)>.002 or (error<.003).mean()<.7:raise ValueError('独立追踪点未通过运动补偿检查')
    if angle>3. or motion>.015:raise ValueError('运动超过小幅手抖范围（3度/15mm）')
    return T,dict(method='upper_face_rgbd_tracks',direction='reference = R @ frame + t',
        rotation_deg=angle,motion_m=motion,validation_points=len(test),validation_median_m=float(np.median(error)),
        R=T.R.tolist(),t=T.t.tolist())
