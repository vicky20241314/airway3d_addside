"""Stable upper-face, multi-scale rigid ICP, with explicit side -> front output."""
import numpy as np
from scipy.spatial import cKDTree
from core.multiview_geometry import RigidCamera, project


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
    diag.update(check_expression(front,side,T))
    # Neck must obey camera motion too: head-turn captures otherwise appear
    # well registered on the face and corrupt a shared neck pose.
    neck=[]
    from core.camera import backproject,coordinate_transform
    for o in (front,side):
        yy,xx=np.where(o.parsing==14);p,_=backproject(np.column_stack((xx,yy)),o.depth,o.camera)
        neck.append(coordinate_transform(p)[::4])
    if min(map(len,neck))>=100:
        d=cKDTree(neck[0]).query(T.to_view(neck[1]))[0]
        neck_error=float(np.median(d));diag['neck_median_m']=neck_error
        if neck_error>.010:raise ValueError(f'{side.view_id}: 头颈关系变化或颈部配准失败，请移动相机重采；neck={neck_error:.4f}m')
    return T,diag,source,target
