"""Fixed anatomical barycentric anchors shared across initialization/refinement."""
import cv2
import numpy as np
from scipy import sparse
from core.student.arap_refine import landmark_matrix
from core.student.detail_snap import _project_model_to_uv, _sample_labels
from core.student.depth_refine import depth_observation, fragile_pixel_mask


def ray_rows(J, uv, camera):
    # Model convention: X right, Y up, Z negative in front of camera.
    r=J.tocoo();n=J.shape[0];v=J.shape[1]
    ku=(uv[:,0]-camera.cx)/camera.fx;kv=-(uv[:,1]-camera.cy)/camera.fy
    rows=np.r_[2*r.row,2*r.row,2*r.row+1,2*r.row+1]
    cols=np.r_[3*r.col,3*r.col+2,3*r.col+1,3*r.col+2]
    vals=np.r_[r.data,r.data*ku[r.row],r.data,r.data*kv[r.row]]
    return sparse.coo_matrix((vals,(rows,cols)),shape=(2*n,3*v)).tocsr()


class SemanticConstraints:
    def __init__(self,vertices,faces,target,camera,depth,parsing,weight=40.,landmark_operator=None):
        self.J=landmark_matrix(faces,len(vertices)) if landmark_operator is None else landmark_operator
        if self.J.shape != (51,len(vertices)) or not np.isfinite(self.J.data).all():
            raise ValueError("Invalid landmark operator")
        self.camera=camera;self.parsing=parsing
        target=np.asarray(target,float)[17:68]
        h,w=depth.shape
        good=np.isfinite(target).all(1)&(target[:,0]>=2)&(target[:,0]<w-2)&(target[:,1]>=2)&(target[:,1]<h-2)
        self.good=good;self.target=target
        J=self.J[good];uv=target[good]
        C=ray_rows(J,uv,camera)
        # Balance each anatomical group; preserve independent left/right sides.
        counts=np.array([5]*10+[9]*9+[6]*12+[20]*20)
        weights=np.sqrt(weight*6/counts[good])
        C=sparse.diags(np.repeat(weights,2))@C
        t=np.zeros(C.shape[0])
        z,confidence=depth_observation(uv,depth)
        pix=np.rint(uv).astype(int)
        fragile=fragile_pixel_mask(parsing,depth,np.asarray(target_to_68(target)))
        reliable=(confidence>.2)&~fragile[pix[:,1],pix[:,0]]&np.isin(parsing[pix[:,1],pix[:,0]],[1,2,3,10,12,13])
        r=J[reliable].tocoo();scale=np.sqrt(4*confidence[reliable])
        Cz=sparse.coo_matrix((r.data*scale[r.row],(r.row,3*r.col+2)),shape=(reliable.sum(),3*len(vertices))).tocsr()
        self.C=sparse.vstack([C,Cz],format='csr');self.t=np.r_[t,-z[reliable]*scale]
        self.info=dict(n_2d=int(good.sum()),n_reliable_depth=int(reliable.sum()),weight=weight)
        # Keep model lip triangles tied to model landmarks, not to observed UV.
        pred,_=_project_model_to_uv(self.J@vertices,camera)
        polygon=np.round(pred[48-17:60-17]).astype(np.int32)
        mask=np.zeros(depth.shape,np.uint8);cv2.fillPoly(mask,[polygon],1)
        mask=cv2.dilate(mask,np.ones((7,7),np.uint8))
        centers=vertices[faces].mean(1);centers_uv,_=_project_model_to_uv(centers,camera)
        labels,inside=_sample_labels(centers_uv,mask)
        self.lip_faces=inside&(labels==1)

    def linear_system(self,base,ids):
        columns=(3*ids[:,None]+np.arange(3)).ravel()
        return self.C[:,columns],self.t-self.C@base.ravel()

    def activate(self,active,vertices,faces):
        # Anatomical eyelid/lip supports may lie over a wrongly aligned image
        # label. Allow them to move under landmark supervision, not depth holes.
        mask=active.copy();mask[np.unique(self.J[self.good].indices)]=True
        edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        seeds=np.zeros(len(mask),bool);seeds[np.unique(self.J[self.good].indices)]=True
        for _ in range(2):
            expanded=seeds.copy();expanded[edges[seeds[edges[:,0]],1]]=True;expanded[edges[seeds[edges[:,1]],0]]=True;seeds=expanded
        return mask|seeds

    def allowed_correspondences(self,triangle_ids,points):
        uv,_=_project_model_to_uv(points,self.camera);labels,inside=_sample_labels(uv,self.parsing)
        observed_lip=np.isin(labels,[12,13])
        return inside & (observed_lip==self.lip_faces[triangle_ids])


def target_to_68(target):
    return np.vstack([np.full((17,2),np.nan),target])
