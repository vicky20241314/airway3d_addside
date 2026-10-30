"""Derived head-only constraints; raw observations are never modified."""
from dataclasses import replace
import numpy as np
from core.multiview_geometry import project


def constraint_observation(o):
    if o.view=='front':return o
    if o.parsing is None:raise ValueError(f'{o.view_id}: missing face segmentation')
    # BiSeNet head labels, excluding neck (14), clothing and hair/background.
    head=np.isin(o.parsing,np.arange(1,14))
    depth=o.depth.copy();depth[~head]=0
    parsing=o.parsing.copy();parsing[~head]=255  # not transition-mask background
    points=np.asarray(o.points) if o.points is not None else np.empty((0,3))
    uv,z=project(points,o.camera);pix=np.rint(np.nan_to_num(uv,nan=-1e6)).astype(int)
    h,w=head.shape
    keep=np.isfinite(points).all(1)&(z>0)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    ids=np.flatnonzero(keep)
    keep[ids]&=head[pix[ids,1],pix[ids,0]]
    return replace(o,depth=depth,parsing=parsing,points=points[keep].copy(),
        quality={**o.quality,'constraint_region':'head_only','neck_used':False})
