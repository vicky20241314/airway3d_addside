"""Observation-supported profile fragments; never fixed 3-D jaw identities."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from core.multiview_geometry import project,ray_visible


def observed_profile(o):
    if o.parsing is None:return dict(status='missing_segmentation',uv=np.empty((0,2)),normal=np.empty((0,2)))
    labels=o.parsing;head=np.isin(labels,np.arange(1,14)).astype(np.uint8)
    contours,_=cv2.findContours(head,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    nose=np.argwhere(labels==10);eyes=np.argwhere(np.isin(labels,(2,3,4,5,6)))
    empty=dict(uv=np.empty((0,2)),normal=np.empty((0,2)))
    if not contours or len(nose)<20 or len(eyes)<20:return dict(status='insufficient_profile_anchors',**empty)
    contour=max(contours,key=cv2.contourArea)[:,0,:].astype(float)
    yx=np.argwhere(head);center=yx[:,1].mean();tip=nose[:,1].mean()
    if abs(tip-center)<2:return dict(status='near_frontal_or_ambiguous_profile',**empty)
    direction=1 if tip>center else -1
    signed=cv2.distanceTransform(head,cv2.DIST_L2,5)-cv2.distanceTransform(1-head,cv2.DIST_L2,5)
    gy,gx=np.gradient(signed);xy=contour.astype(int);x,y=xy.T
    normal=-np.column_stack((gx[y,x],gy[y,x]));normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    h,w=labels.shape
    invalid=np.isin(labels,(7,8,9,14,16,17,18,255)).astype(np.uint8)
    invalid=cv2.dilate(invalid,np.ones((7,7),np.uint8)).astype(bool)
    keep=(x>=4)&(x<w-4)&(y>=4)&(y<h-4)&~invalid[y,x]
    counts=dict(raw_boundary_points=len(x),semantic_and_image_safe=int(keep.sum()))
    keep&=(direction*(x-center)>0)&(y>=np.quantile(eyes[:,0],.1))
    counts['anterior_profile']=int(keep.sum())
    # Sample inside the face, not at mixed foreground/background depth pixels.
    inside=np.rint(contour-normal*4).astype(int);inside[:,0]=np.clip(inside[:,0],1,w-2);inside[:,1]=np.clip(inside[:,1],1,h-2)
    ix,iy=inside.T;depth=np.asarray(o.depth,float)
    low=cv2.erode(depth,np.ones((3,3),np.uint8));high=cv2.dilate(depth,np.ones((3,3),np.uint8))
    keep&=(low[iy,ix]>0)&(high[iy,ix]-low[iy,ix]<=8)&(head[iy,ix]>0)
    counts['reliable_interior_depth']=int(keep.sum())
    if o.quality.get('upper_face_only'):
        keep&=y<np.quantile(nose[:,0],.85)
    ids=np.flatnonzero(keep)[::2]
    counts['sampled_fragment_points']=len(ids)
    if len(ids)<12:return dict(status='insufficient_reliable_profile',filter_counts=counts,**empty)
    return dict(status='accepted',uv=contour[ids],normal=normal[ids],profile_direction=direction,filter_counts=counts)


def silhouette_topology(faces):
    faces=np.asarray(faces);edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    unique,inverse=np.unique(edges,axis=0,return_inverse=True)
    adjacency=[[] for _ in unique]
    for edge,face in zip(inverse,np.repeat(np.arange(len(faces)),3)):adjacency[edge].append(face)
    pairs=np.full((len(unique),2),-1,int)
    for i,faces in enumerate(adjacency):pairs[i,:min(len(faces),2)]=faces[:2]
    return unique,pairs


def freeze_profile(vertices,faces,o,observed,topology):
    empty=dict(edges=np.empty((0,2),int),target=np.empty((0,2)),normal=np.empty((0,2)),irls=np.empty(0))
    if observed['status']!='accepted':return empty
    edges,adj=topology;tri=vertices[faces];centers=tri.mean(1)
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);facing=(normals*(-centers)).sum(1)>0
    a=facing[adj[:,0]];b=np.where(adj[:,1]>=0,facing[np.maximum(adj[:,1],0)],False)
    ids=np.flatnonzero(a!=b);mid=vertices[edges[ids]].mean(1)
    visible=ray_visible(vertices,faces,mid);ids=ids[visible];mid=mid[visible]
    uv,z=project(mid,o.camera)
    if not len(ids):return empty
    distance,nn=cKDTree(observed['uv']).query(uv);keep=(z>0)&(distance<20)
    ids=ids[keep];nn=nn[keep];uv=uv[keep]
    # Two-way fragment matching uses only this supported contour subset.
    # Both terms use projected silhouette edges, with frozen outer-loop pairs.
    if len(ids):
        rev_distance,rev_nn=cKDTree(uv).query(observed['uv']);use=rev_distance<20
        rows=np.r_[np.arange(len(ids)),rev_nn[use]];targets=np.r_[nn,np.flatnonzero(use)]
        return dict(edges=edges[ids[rows]],target=observed['uv'][targets],normal=observed['normal'][targets])
    return empty
