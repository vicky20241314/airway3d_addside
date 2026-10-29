"""Stable upper-face, multi-scale rigid ICP, with explicit side -> front output."""
import numpy as np
from scipy.spatial import cKDTree
from core.multiview_geometry import RigidCamera, project


class RegistrationConsistencyError(ValueError):
    """Keep accepted face alignment when a later consistency check fails."""
    def __init__(self,message,transform,diagnostics,source,target):
        super().__init__(message)
        self.transform=transform;self.diagnostics=diagnostics
        self.source=source;self.target=target


def _neck_surface(o):
    """Reliable interior pixels and camera-facing normals in this camera."""
    import cv2
    depth=np.asarray(o.depth,np.float64)
    y,x=np.mgrid[:depth.shape[0],:depth.shape[1]];z=depth*.001
    p=np.stack(((x-o.camera.cx)*z/o.camera.fx,-(y-o.camera.cy)*z/o.camera.fy,-z),axis=-1)
    normal=np.zeros_like(p)
    normal[1:-1,1:-1]=np.cross(p[1:-1,2:]-p[1:-1,:-2],p[2:,1:-1]-p[:-2,1:-1])
    length=np.linalg.norm(normal,axis=-1)
    normal/=np.maximum(length[...,None],1e-15)
    facing=(normal*(-p)).sum(-1)
    normal[facing<0]*=-1
    cosine=np.abs(facing)/np.maximum(np.linalg.norm(p,axis=-1),1e-15)
    kernel=np.ones((3,3),np.uint8)
    mask=cv2.erode((o.parsing==14).astype(np.uint8),kernel).astype(bool)
    # Do not compare segmentation boundaries, depth holes, or depth jumps.
    low=cv2.erode(depth,kernel);high=cv2.dilate(depth,kernel)
    mask&=(low>0)&(high-low<=8.)&(length>1e-12)&(cosine>.2)
    mask[[0,-1],:]=False;mask[:,[0,-1]]=False
    return p,normal,mask


def _projective_neck_direction(source,target,transform,target_camera):
    p,n,mask=source;q,m,target_mask=target
    moved=transform.to_view(p[mask][::2]);normal=n[mask][::2]@transform.R.T
    uv,z=project(moved,target_camera);pix=np.rint(uv).astype(int)
    h,w=target_mask.shape
    inside=(z>0)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    ids=np.flatnonzero(inside);pix=pix[ids]
    keep=target_mask[pix[:,1],pix[:,0]]
    ids=ids[keep];pix=pix[keep]
    normal_target=m[pix[:,1],pix[:,0]]
    toward_camera=-moved[ids]/np.maximum(np.linalg.norm(moved[ids],axis=1,keepdims=True),1e-15)
    # A point seen in the side camera need not face the front camera. Require
    # compatible surfaces rather than deleting rows by their residual size.
    keep=((normal[ids]*toward_camera).sum(1)>.2)&((normal[ids]*normal_target).sum(1)>.5)
    delta=moved[ids[keep]]-q[pix[keep,1],pix[keep,0]]
    signed=(delta*normal_target[keep]).sum(1)
    # No distance, depth-error, or nearest-neighbour gate: genuine neck motion
    # must remain in the statistic, including motion away from the camera.
    error=np.abs(signed)
    return dict(source_samples=len(moved),overlap_samples=len(error),
        median_m=float(np.median(error)) if len(error) else None,
        p90_m=float(np.quantile(error,.9)) if len(error) else None,
        signed_median_m=float(np.median(signed)) if len(error) else None)


def check_neck_consistency(front,side,transform):
    """Compare only projectively shared neck surfaces, in BOTH cameras.

    transform maps side -> front. Sparse/non-overlapping neck observations
    are explicitly unverified; they are never counted as a passed check.
    """
    a=_neck_surface(front);b=_neck_surface(side)
    forward=_projective_neck_direction(b,a,transform,front.camera)
    reverse=_projective_neck_direction(a,b,transform.inverse(),side.camera)
    reliable=[row['median_m'] for row in (forward,reverse) if row['overlap_samples']>=100]
    error=max(reliable) if reliable else None
    status=('inconsistent' if error is not None and error>.010 else
            'passed' if len(reliable)==2 else 'partial_overlap' if reliable else 'insufficient_overlap')
    return dict(method='bidirectional_projective_neck_point_to_plane',status=status,
        threshold_m=.010,minimum_overlap_samples=100,median_m=error,
        side_to_front=forward,front_to_side=reverse,
        warning=None if status in ('passed','inconsistent') else '颈部共同可见区域不足，头颈一致性未完整验证；请检查叠加点云及采集姿态')


def save_registration_diagnostics(folder,view_id,transform,diagnostics,source,target,*,failed=False):
    from pathlib import Path
    import trimesh
    from core.pipeline import write_json
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    suffix='_failed' if failed else ''
    write_json(folder/f'registration_{view_id}{suffix}.json',dict(view_id=view_id,**diagnostics,
        R_side_to_front=transform.R,t_side_to_front=transform.t))
    overlay=trimesh.points.PointCloud(np.vstack((target,transform.to_view(source))),
        colors=np.vstack((np.tile([60,160,255,255],(len(target),1)),np.tile([255,110,60,255],(len(source),1)))))
    overlay.export(folder/f'registration_{view_id}{suffix}_front.ply')


def rigid_svd(source, target):
    x=np.asarray(source);y=np.asarray(target)
    u,s,vh=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
    if s[1]<1e-10: raise ValueError('Registration landmarks are degenerate')
    fix=np.eye(3);fix[-1,-1]=np.linalg.det(vh.T@u.T)
    R=vh.T@fix@u.T
    return RigidCamera(R,y.mean(0)-R@x.mean(0))


def stable_points(o):
    from core.camera import backproject, coordinate_transform
    import cv2
    uv=np.asarray(o.uv);p=o.parsing
    if p is None: raise ValueError('Registration requires semantic parsing')
    skin=cv2.erode((p==1).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    yy,xx=np.where(skin)
    top=np.nanmedian(uv[19:25,1]);bottom=uv[33,1]+.25*(uv[8,1]-uv[33,1])
    if not np.isfinite([top,bottom]).all() or bottom<=top:
        raise ValueError(f'{o.view_id}: stable upper-face ROI unavailable; 请重采')
    keep=(yy>top)&(yy<bottom)
    points,_=backproject(np.column_stack((xx[keep],yy[keep])),o.depth,o.camera)
    return coordinate_transform(points)


def _voxel(points, size):
    _,ids=np.unique(np.floor(points/size).astype(int),axis=0,return_index=True)
    return points[np.sort(ids)]


def check_expression(front,side,T):
    gaps=[]
    for o in (front,side):
        pairs=[(i,j) for i,j in ((50,58),(51,57),(52,56)) if o.valid[i] and o.valid[j]]
        gaps.append(float(np.median([np.linalg.norm(o.xyz[i]-o.xyz[j]) for i,j in pairs])) if len(pairs)>=2 else None)
    result=dict(lip_gap_m=gaps,expression_check='lip_distances' if all(g is not None for g in gaps) else 'insufficient_lip_observation')
    if all(g is not None for g in gaps) and abs(gaps[0]-gaps[1])>max(.003,.15*gaps[0]):
        raise ValueError(f'{side.view_id}: 张闭口程度变化，请保持同一表情重采；lip_gap_m={gaps}')
    if side.state=='open':
        if any(g is None for g in gaps):raise ValueError(f'{side.view_id}: 无足够可靠唇点检查张口保持，请重采')
        ids=np.array([50,51,52,56,57,58]);ids=ids[front.valid[ids]&side.valid[ids]]
        error=np.linalg.norm(T.to_view(side.xyz[ids])-front.xyz[ids],axis=1)
        result['open_lip_consistency_m']=float(np.median(error)) if len(error) else None
        if len(error)<4 or np.median(error)>.006:
            raise ValueError(f'{side.view_id}: 张口形态或下颌位置不一致/唇点不可靠，请保持张口后重采')
    return result


def register_to_front(front, side):
    """Initialize from RANSAC internal points, then ICP on stable overlap only.

    Internal detector points are provisional SE(3) seeds, never direct side
    fitting targets. Require independent dense overlap to accept a seed.
    """
    import itertools
    source=stable_points(side);target=stable_points(front)
    if min(len(source),len(target))<100: raise ValueError(f'{side.view_id}: stable face overlap too small; 请重采')
    ids=np.arange(27,48)
    ids=ids[front.valid[ids]&side.valid[ids]]
    if len(ids)<6: raise ValueError(f'{side.view_id}: stable internal depth seeds不足6个，请重采')
    x=side.xyz[ids];y=front.xyz[ids]
    rng=np.random.default_rng(42)
    triples=list(itertools.combinations(range(len(ids)),3))
    if len(triples)>160: triples=[triples[i] for i in rng.choice(len(triples),160,replace=False)]
    best=None
    for tri in triples:
        try:T=rigid_svd(x[list(tri)],y[list(tri)])
        except ValueError:continue
        error=np.linalg.norm(T.to_view(x)-y,axis=1);inside=error<.006
        score=(int(inside.sum()),-float(np.minimum(error,.02).sum()))
        if best is None or score>best[0]:best=(score,T,inside)
    if best is None or best[0][0]<6: raise ValueError(f'{side.view_id}: landmark seed rejected; 请重采')
    T=rigid_svd(x[best[2]],y[best[2]])
    history=[]
    for voxel,gate in ((.008,.030),(.004,.015),(.002,.008)):
        a=_voxel(source,voxel);b=_voxel(target,voxel);tree=cKDTree(b)
        for iteration in range(30):
            moved=T.to_view(a);distance,nn=tree.query(moved)
            keep=distance<min(gate,float(np.quantile(distance,.8)))
            if keep.sum()<30: raise ValueError(f'{side.view_id}: ICP overlap lost; 请重采')
            update=rigid_svd(moved[keep],b[nn[keep]])
            T=update.compose(T)
            if np.linalg.norm(update.t)<1e-6 and np.linalg.norm(update.R-np.eye(3))<1e-5:break
        history.append(dict(voxel_m=voxel,gate_m=gate,iterations=iteration+1))
    moved=T.to_view(source);distance,_=cKDTree(target).query(moved)
    good=distance<.008;coverage=float(good.mean())
    rmse=float(np.sqrt(np.mean(distance[good]**2))) if good.any() else float('inf')
    reverse=cKDTree(moved).query(target)[0];reverse_coverage=float((reverse<.008).mean())
    diag=dict(direction='front = R_side_to_front @ side + t_side_to_front',rmse_m=rmse,
        source_overlap=coverage,target_overlap=reverse_coverage,seed_inliers=int(best[0][0]),history=history)
    if min(coverage,reverse_coverage)<.25 or rmse>.004:
        raise ValueError(f'{side.view_id}: 配准失败，请重采；{diag}')
    # Rigid-invariant lip separation diagnoses expression changes without
    # adding expression DOFs per view. Occluded/invalid measurements are skipped.
    try:
        diag.update(check_expression(front,side,T))
        neck=check_neck_consistency(front,side,T);diag['neck_check']=neck
        if neck['status']=='inconsistent':
            raise ValueError(f"{side.view_id}: 共同可见颈部与面部变换不一致，请保持头颈姿态并移动相机重采；neck={neck['median_m']:.4f}m")
    except ValueError as exc:
        diag.update(consistency_error=str(exc))
        raise RegistrationConsistencyError(str(exc),T,diag,source,target) from exc
    return T,diag,source,target
