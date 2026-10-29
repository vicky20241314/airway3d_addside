"""Camera-origin ray broad phase in projection space; exact 3-D narrow phase.

The grid only selects triangle candidates. Rays are never snapped to pixels,
surfaces are never rasterized/downsampled, and all front/back triangles remain.
Unsupported near-plane geometry returns None for the reference fallback.
"""
import numpy as np
from trimesh.intersections import planes_lines
from trimesh.triangles import points_to_barycentric, normals as triangle_normals
from trimesh.constants import tol


def projected_visibility(vertices,faces,samples,tolerance):
    v=np.asarray(vertices,np.float64);f=np.asarray(faces,np.int64);s=np.asarray(samples,np.float64)
    if not np.isfinite(v).all() or (v[:,2]>=-1e-5).any():return None
    result=np.zeros(len(s),bool)
    if not len(f) or not len(s):return result
    uv=v[:,:2]/(-v[:,2,None]);tri=v[f];projected=uv[f]
    lo=projected.min(1)-1e-12;hi=projected.max(1)+1e-12
    origin=lo.min(0);extent=hi.max(0)-origin
    if (extent<1e-12).any():return None
    size=64;scale=size/extent
    lower=np.clip(np.floor((lo-origin)*scale).astype(int),0,size-1)
    upper=np.clip(np.floor((hi-origin)*scale).astype(int),0,size-1)
    widths=upper[:,0]-lower[:,0]+1;counts=widths*(upper[:,1]-lower[:,1]+1)
    if counts.sum()>4_000_000:return None  # bounded storage for pathological triangles
    offsets=np.r_[0,np.cumsum(counts)]
    face_ids=np.repeat(np.arange(len(f)),counts)
    local=np.arange(len(face_ids))-np.repeat(offsets[:-1],counts)
    cells=(lower[face_ids,1]+local//widths[face_ids])*size+lower[face_ids,0]+local%widths[face_ids]
    order=np.argsort(cells,kind='stable');table=face_ids[order]
    starts=np.r_[0,np.cumsum(np.bincount(cells,minlength=size*size))]
    # Match Trimesh's edge differences, normalization and degenerate-face rule.
    unit,nondegenerate=triangle_normals(tri)
    normals=np.zeros((len(f),3));normals[nondegenerate]=unit
    good=np.isfinite(s).all(1)&(s[:,2]<-1e-5)
    ids=np.flatnonzero(good);q=s[ids,:2]/(-s[ids,2,None])
    inside=((q>=origin)&(q<=origin+extent)).all(1);ids=ids[inside];q=q[inside]
    cell=np.clip(np.floor((q-origin)*scale).astype(int),0,size-1)
    bins=cell[:,1]*size+cell[:,0]
    # Limit candidate memory as well as the number of rays per chunk.
    count=starts[bins+1]-starts[bins];cumulative=np.r_[0,np.cumsum(count)]
    boundary=[]
    begin=0
    while begin<len(ids):
        end=min(begin+1024,int(np.searchsorted(cumulative,cumulative[begin]+200_000,side='right')-1))
        end=max(begin+1,end)
        n=count[begin:end];repeats=np.repeat(np.arange(end-begin),n)
        local=np.arange(n.sum())-np.repeat(np.r_[0,np.cumsum(n)[:-1]],n)
        candidates=table[np.repeat(starts[bins[begin:end]],n)+local]
        ray_ids=ids[begin:end][repeats]
        points,valid=planes_lines(tri[candidates,0],normals[candidates],np.zeros((len(ray_ids),3)),s[ray_ids])
        distance=np.full(end-begin,np.inf)
        if valid.any():
            bary=points_to_barycentric(tri[candidates[valid]],points)
            hit=(bary>-tol.zero).all(1)&(bary<1+tol.zero).all(1)
            hit&=(points*s[ray_ids[valid]]).sum(1)>-1e-6
            hits=points[hit];rays=repeats[valid][hit]
            along=(hits*s[ids[begin:end][rays]]).sum(1)
            first=np.full(end-begin,np.inf);np.minimum.at(first,rays,along)
            nearest=along==first[rays]
            pick=np.full(end-begin,len(hits),dtype=int)
            np.minimum.at(pick,rays[nearest],np.flatnonzero(nearest))
            found=pick<len(hits)
            distance[found]=np.linalg.norm(hits[pick[found]]-s[ids[begin:end][found]],axis=1)
        result[ids[begin:end]]=distance<=tolerance
        boundary.extend(ids[begin:end][np.abs(distance-tolerance)<=1e-12])
        begin=end
    if boundary:
        # Preserve reference semantics at floating-point threshold/tie boundaries.
        from core.multiview_geometry import ray_visible
        boundary=np.asarray(boundary,int)
        result[boundary]=ray_visible(v,f,s[boundary],tolerance=tolerance,backend='reference')
    return result
