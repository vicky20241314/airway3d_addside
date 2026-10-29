"""Observed image jaw contours and fixed regional surface weights.

No anatomical target shape, scanner target or noisy jaw-edge 3-D point loss.
Contour residuals act only normal to the observed curve, allowing along-curve
sliding. Pixel/depth scale and semantic gates are fixed before optimization.
"""
import cv2
import numpy as np


def prepare_jaw(parsing, landmarks2d, landmarks3d, valid3d, camera, points):
    lm=np.asarray(landmarks2d,float);xyz=np.asarray(landmarks3d,float)
    valid=np.asarray(valid3d,bool);pts=np.asarray(points,float)
    uniform=np.full(len(pts),1/max(len(pts),1))
    result=dict(ids=np.empty(0,int),normals=np.empty((0,2)),pixel_to_m=0.,
                weights=uniform,reason='missing_parsing',groups={})
    if parsing is None:return result
    parsing=np.asarray(parsing);h,w=parsing.shape
    if lm.shape!=(68,2) or not np.isfinite(lm[:17]).all():
        result['reason']='invalid_2d_contour';return result
    iod=float(np.linalg.norm(lm[45]-lm[36]));width=float(np.ptp(lm[:17,0]))
    if iod<10 or width<30:
        result['reason']='face_too_small';return result
    depth_ids=np.arange(17,68)
    good=valid[depth_ids]&np.isfinite(xyz[depth_ids]).all(1)&(xyz[depth_ids,2]<0)
    if good.sum()<8:
        result['reason']='insufficient_internal_depth';return result
    result['pixel_to_m']=float(np.median(-xyz[depth_ids[good],2])/np.sqrt(camera.fx*camera.fy))
    center=lm[33]
    accepted=[];normals=[]
    for i in range(2,15):  # exclude upper temples / ears
        tangent=lm[i+1]-lm[i-1];length=np.linalg.norm(tangent)
        inward=center-lm[i];ilen=np.linalg.norm(inward)
        if length<2 or ilen<2:continue
        inward/=ilen;normal=np.array([-tangent[1],tangent[0]])/length
        steps=np.array([.015,.03,.05])*iod
        inside=np.rint(lm[i]+steps[:,None]*inward).astype(int)
        outside=np.rint(lm[i]-steps[:,None]*inward).astype(int)
        allpix=np.vstack((inside,outside))
        if not ((allpix[:,0]>=0)&(allpix[:,0]<w)&(allpix[:,1]>=0)&(allpix[:,1]<h)).all():continue
        inn=parsing[inside[:,1],inside[:,0]];outer=parsing[outside[:,1],outside[:,0]]
        # Support from face skin on one side and background/neck/clothes on
        # the other. Hair and ears must not become jaw targets.
        if (inn==1).sum()<2 or not np.isin(outer,[0,14,16]).any():continue
        accepted.append(i);normals.append(normal)
    result['ids']=np.asarray(accepted,int);result['normals']=np.asarray(normals).reshape(-1,2)
    result['reason']='ok' if len(accepted)>=6 else 'too_few_supported_jaw_points'
    if len(accepted)<6:
        result['ids']=np.empty(0,int);result['normals']=np.empty((0,2))
    # Fixed, observation-defined lower-face band; do not use prediction errors.
    line=np.ones((h,w),np.uint8)
    cv2.polylines(line,[np.rint(lm[:17]).astype(np.int32)],False,0,1)
    distance=cv2.distanceTransform(line,cv2.DIST_L2,5)
    polygon=np.zeros((h,w),np.uint8)
    cv2.fillPoly(polygon,[np.rint(lm[:17]).astype(np.int32)],1)
    z=-pts[:,2]
    pix=np.rint(np.column_stack((camera.fx*pts[:,0]/z+camera.cx,-camera.fy*pts[:,1]/z+camera.cy))).astype(int)
    inside=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    labels=np.full(len(pts),-1);dist=np.zeros(len(pts));inpoly=np.zeros(len(pts),bool)
    labels[inside]=parsing[pix[inside,1],pix[inside,0]]
    dist[inside]=distance[pix[inside,1],pix[inside,0]]
    inpoly[inside]=polygon[pix[inside,1],pix[inside,0]]>0
    jaw=(labels==1)&inpoly&(pix[:,1]>lm[33,1])&(dist>.008*width)&(dist<.10*width)
    neck=labels==14;other=~jaw&~neck
    groups=[jaw,other,neck];budgets=np.array([.30,.55,.15])
    counts=np.array([g.sum() for g in groups])
    result['groups']={name:int(count) for name,count in zip(('jaw','other_face','neck'),counts)}
    result['regions']=np.where(jaw,0,np.where(neck,2,1))
    if counts[0]>=20 and len(accepted)>=6:
        budgets[counts==0]=0;budgets/=budgets.sum()
        weights=np.zeros(len(pts))
        for g,budget,count in zip(groups,budgets,counts):
            if count:weights[g]=budget/count
        result['weights']=weights
        result['group_budgets']=budgets.tolist()
    else:result['group_budgets']='uniform_fallback'
    return result
