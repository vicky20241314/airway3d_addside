"""Conservative semantic RGB-D rigid seeds; no extra detector or free scale."""
import itertools
import cv2
import numpy as np
from core.multiview_geometry import RigidCamera


def stable_surface(o, *, fallback=True):
    """Return camera-space points/normals and the independent image ROI.

    Landmark ROI is retained when valid. Otherwise semantic eye/brow and nose
    anchors must both exist; skin alone never defines a rigid face ROI.
    """
    if o.parsing is None:raise ValueError('missing semantic parsing')
    p=o.parsing;depth=np.asarray(o.depth,float);uv=np.asarray(o.uv)
    anchor=[8,33,*range(19,25)]
    reliable=np.asarray(o.valid)[anchor].all() and np.isfinite(uv[anchor]).all()
    top=np.median(uv[19:25,1]) if reliable else np.nan
    bottom=uv[33,1]+.25*(uv[8,1]-uv[33,1]) if reliable else np.nan
    method='landmark_roi'
    yy,xx=np.mgrid[:p.shape[0],:p.shape[1]]
    if not np.isfinite([top,bottom]).all() or bottom<=top:
        if not fallback:raise ValueError('stable landmark ROI unavailable')
        eyes=np.isin(p,(2,3,4,5,6));nose=p==10
        if eyes.sum()<20 or nose.sum()<20:raise ValueError('semantic upper-face anchors insufficient')
        top=float(np.quantile(yy[eyes],.1));bottom=float(np.quantile(yy[nose],.85))
        if bottom-top<8:raise ValueError('degenerate semantic upper-face anchors')
        method='semantic_depth_normal_roi'
    labels=(p==1) if method=='landmark_roi' else np.isin(p,(1,2,3,4,5,6,10))
    mask=cv2.erode(labels.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    mask&=(yy>top)&(yy<bottom)
    z=depth*.001
    points=np.stack(((xx-o.camera.cx)*z/o.camera.fx,-(yy-o.camera.cy)*z/o.camera.fy,-z),-1)
    normal=np.zeros_like(points)
    normal[1:-1,1:-1]=np.cross(points[1:-1,2:]-points[1:-1,:-2],points[2:,1:-1]-points[:-2,1:-1])
    length=np.linalg.norm(normal,axis=-1);normal/=np.maximum(length[...,None],1e-15)
    facing=(normal*(-points)).sum(-1);normal[facing<0]*=-1
    low=cv2.erode(depth,np.ones((3,3),np.uint8));high=cv2.dilate(depth,np.ones((3,3),np.uint8))
    mask&=(low>0)&(high-low<=8)&(length>1e-12)&(np.abs(facing)/np.maximum(np.linalg.norm(points,axis=-1),1e-15)>.2)
    mask[[0,-1]]=False;mask[:,[0,-1]]=False
    return points[mask],normal[mask],mask,dict(roi_method=method,roi_points=int(mask.sum()))


def ransac_seed(x,y,rigid_svd, *, minimum=6):
    if len(x)<minimum:raise ValueError('insufficient rigid seed pairs')
    rng=np.random.default_rng(42);best=None
    if len(x)<=12:triples=list(itertools.combinations(range(len(x)),3))
    else:triples=[rng.choice(len(x),3,replace=False) for _ in range(300)]
    for ids in triples:
        try:T=rigid_svd(x[list(ids)],y[list(ids)])
        except ValueError:continue
        error=np.linalg.norm(T.to_view(x)-y,axis=1);inside=error<.006
        score=(int(inside.sum()),-float(np.minimum(error,.02).sum()))
        if best is None or score>best[0]:best=score,T,inside
    if best is None or best[0][0]<minimum:raise ValueError('rigid seed consensus insufficient')
    inside=best[2];T=rigid_svd(x[inside],y[inside])
    # Collinear/small supports are not observable even with many inliers.
    spread=np.linalg.svd(x[inside]-x[inside].mean(0),compute_uv=False)/np.sqrt(inside.sum())
    if spread[1]<.002:raise ValueError('rigid seed spatial support degenerate')
    return T,int(inside.sum())


def rgbd_feature_seed(front,side,front_mask,side_mask,rigid_svd):
    """Mutual SIFT matches lifted by actual depth, verified by 3-D RANSAC.

    Features initialize only; independent dense normal-compatible overlap is
    still required by the caller. Textureless observations fail explicitly.
    """
    sift=cv2.SIFT_create(nfeatures=1200)
    features=[]
    for o,mask in ((side,side_mask),(front,front_mask)):
        gray=cv2.cvtColor(o.rgb,cv2.COLOR_RGB2GRAY)
        k,d=sift.detectAndCompute(gray,mask.astype(np.uint8)*255)
        if d is None or len(k)<6:raise ValueError('insufficient stable RGB-D features')
        features.append((k,d))
    (ks,ds),(kf,df)=features;matcher=cv2.BFMatcher()
    def ratio(a,b):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2)
                if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
    sf=ratio(ds,df);fs=ratio(df,ds);pairs=[(a,b) for a,b in sf.items() if fs.get(b)==a]
    x=[];y=[]
    for a,b in pairs:
        points=[]
        for o,k in ((side,ks[a]),(front,kf[b])):
            u,v=np.rint(k.pt).astype(int);z=float(o.depth[v,u])*.001
            points.append([(u-o.camera.cx)*z/o.camera.fx,-(v-o.camera.cy)*z/o.camera.fy,-z])
        x.append(points[0]);y.append(points[1])
    T,n=ransac_seed(np.asarray(x),np.asarray(y),rigid_svd)
    return T,dict(seed_method='mutual_sift_rgbd_ransac',seed_matches=len(pairs),seed_inliers=n)


def geometric_feature_seed(source,snormal,target,tnormal,rigid_svd):
    """Global rigid seed from local point/normal histograms (SPFH-style).

    Rotation-invariant point-pair Darboux coordinates, three 11-bin histograms,
    reciprocal ratio matches and 3-D consensus. This is a candidate generator,
    not permission to accept a textureless or nearly planar overlap blindly.
    """
    from scipy.spatial import cKDTree
    def describe(points,normals):
        _,ids=np.unique(np.floor(points/.004).astype(int),axis=0,return_index=True)
        ids=np.sort(ids)
        if len(ids)>1200:ids=ids[np.linspace(0,len(ids)-1,1200,dtype=int)]
        p=points[ids];n=normals[ids];tree=cKDTree(points);descriptors=[];valid=[]
        for i in range(len(p)):
            neighbors=np.asarray(tree.query_ball_point(p[i],.020));neighbors=neighbors[neighbors!=ids[i]]
            if len(neighbors)<8:continue
            delta=points[neighbors]-p[i];distance=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(distance[:,None],1e-12)
            v=np.cross(direction,n[i]);length=np.linalg.norm(v,axis=1);good=length>.05
            if good.sum()<8:continue
            v=v[good]/length[good,None];w=np.cross(n[i],v);other=normals[neighbors[good]]
            alpha=(v*other).sum(1);phi=direction[good]@n[i]
            angle=np.arctan2((w*other).sum(1),other@n[i])
            h=np.r_[np.histogram(alpha,11,(-1,1))[0],np.histogram(phi,11,(-1,1))[0],np.histogram(angle,11,(-np.pi,np.pi))[0]].astype(float)
            h/=max(np.linalg.norm(h),1e-12);descriptors.append(h);valid.append(i)
        if len(valid)<6:raise ValueError('insufficient geometric feature support')
        return p[valid],np.asarray(descriptors)
    x,dx=describe(np.asarray(source),np.asarray(snormal));y,dy=describe(np.asarray(target),np.asarray(tnormal))
    distance,nn=cKDTree(dy).query(dx,k=2);_,reverse=cKDTree(dx).query(dy)
    keep=(distance[:,0]<.85*np.maximum(distance[:,1],1e-12))&(reverse[nn[:,0]]==np.arange(len(x)))
    # Equal descriptors have no distinctiveness, even if both distances are zero.
    keep&=distance[:,1]>1e-8
    T,n=ransac_seed(x[keep],y[nn[keep,0]],rigid_svd)
    return T,dict(seed_method='point_normal_histogram_ransac',seed_matches=int(keep.sum()),seed_inliers=n)
