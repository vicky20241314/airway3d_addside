"""Frozen outer-loop robust landmarks and same-state relative SE(3) priors."""
import numpy as np
from scipy.spatial.transform import Rotation
from core.multiview_geometry import RigidCamera,ray_visible


SIDE_STATIC_IDS=np.array([21,22,*range(27,48)])


def soft_side_landmarks(o,vertices,faces,model,delta=.004,gross=.020):
    import cv2
    valid=np.zeros(68,bool);reasons=dict(invalid_observation=0,semantic_or_depth=0,occluded=0,gross_outlier=0)
    ids=SIDE_STATIC_IDS[np.asarray(o.valid)[SIDE_STATIC_IDS]&np.isfinite(o.xyz[SIDE_STATIC_IDS]).all(1)&np.isfinite(o.uv[SIDE_STATIC_IDS]).all(1)]
    reasons['invalid_observation']=len(SIDE_STATIC_IDS)-len(ids)
    uv=np.rint(np.nan_to_num(o.uv,nan=-1e6)).astype(int);h,w=o.depth.shape
    depth=np.asarray(o.depth,float);low=cv2.erode(depth,np.ones((3,3),np.uint8));high=cv2.dilate(depth,np.ones((3,3),np.uint8))
    for i in ids:
        x,y=uv[i]
        if not (1<=x<w-1 and 1<=y<h-1) or low[y,x]<=0 or high[y,x]-low[y,x]>8:
            reasons['semantic_or_depth']+=1;continue
        if o.parsing is None or o.parsing[y,x] not in (1,2,3,4,5,6,10) or abs(depth[y,x]*.001+o.xyz[i,2])>.004:
            reasons['semantic_or_depth']+=1;continue
        valid[i]=True
    candidates=np.flatnonzero(valid)
    if len(candidates):
        visible=ray_visible(vertices,faces,model[candidates])
        valid[candidates[~visible]]=False;reasons['occluded']=int((~visible).sum())
    raw=np.linalg.norm(np.nan_to_num(o.xyz)-model,axis=1)
    reasons['gross_outlier']=int((valid&(raw>=gross)).sum());valid&=raw<gross
    ids=np.flatnonzero(valid);weight=np.minimum(1.,delta/np.maximum(raw[ids],1e-12))
    return ids,weight,valid,dict(landmark_rejection_counts=reasons,landmark_raw_residual_m=raw[ids].tolist(),
        landmark_irls_weights=weight.tolist(),legacy_6mm_rejected=int((raw[ids]>=.006).sum()),
        landmark_policy='observed_static_internal_huber',landmark_huber_m=delta,landmark_gross_m=gross)


def relative_transforms(rot,trans,front_indices,observations):
    return {b:RigidCamera(rot[b],trans[b]).compose(RigidCamera(rot[front_indices[o.state]],trans[front_indices[o.state]]).inverse())
        for b,o in enumerate(observations) if o.view!='front'}


def se3_log(T):
    """Left discrepancy log: T_rel @ inv(T_rel_initial); [rad, metres]."""
    omega=Rotation.from_matrix(T.R).as_rotvec();angle=np.linalg.norm(omega)
    x,y,z=omega;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    c=1/12+angle**2/720 if angle<1e-4 else (1-.5*angle/np.tan(.5*angle))/angle**2
    return np.r_[omega,(np.eye(3)-.5*K+c*K@K)@T.t]


def pose_prior_values(rot,trans,front_indices,observations,initial,options):
    rel=relative_transforms(rot,trans,front_indices,observations)
    scales=np.r_[np.full(3,np.deg2rad(options.pose_prior_rotation_std_deg)),np.full(3,options.pose_prior_translation_std_m)]
    return np.concatenate([se3_log(T.compose(initial[b].inverse()))/scales for b,T in rel.items()]) if rel else np.empty(0)


def pose_drift(rot,trans,front_indices,observations,initial):
    rel=relative_transforms(rot,trans,front_indices,observations)
    return {observations[b].view_id:dict(rotation_deg=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(T.R@initial[b].R.T).as_rotvec()))),
        translation_mm=float(np.linalg.norm(T.t-initial[b].t)*1000)) for b,T in rel.items()}
