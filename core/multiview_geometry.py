"""Explicit rigid cameras and ray-tested visibility; no free scale."""
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class RigidCamera:
    """column convention: p_view = R @ p_common + t; arrays use p @ R.T + t."""
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.R = np.asarray(self.R, float).copy()
        self.t = np.asarray(self.t, float).copy()
        if (self.R.shape != (3,3) or self.t.shape != (3,) or not np.isfinite(self.t).all()
                or not np.allclose(self.R.T@self.R, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(self.R), 1, atol=1e-5)):
            raise ValueError('Expected a finite SE(3) transform, without scale')

    def to_view(self, points): return np.asarray(points)@self.R.T+self.t
    def to_common(self, points): return (np.asarray(points)-self.t)@self.R
    def inverse(self): return RigidCamera(self.R.T, -self.R.T@self.t)
    def compose(self, other): return RigidCamera(self.R@other.R, self.R@other.t+self.t)


def project(points, camera):
    p = np.asarray(points)
    z = -p[:,2]
    safe = np.maximum(z, 1e-8)
    return np.column_stack((camera.fx*p[:,0]/safe+camera.cx, -camera.fy*p[:,1]/safe+camera.cy)), z


def ray_visible(vertices, faces, samples, *, tolerance=.0015):
    samples = np.asarray(samples)
    good = np.isfinite(samples).all(1) & (samples[:,2] < -1e-5)
    result = np.zeros(len(samples), bool)
    ids = np.flatnonzero(good)
    if not len(ids): return result
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    hits, ray_ids, _ = mesh.ray.intersects_location(np.zeros((len(ids),3)), samples[ids], multiple_hits=False)
    result[ids[ray_ids]] = np.linalg.norm(hits-samples[ids[ray_ids]],axis=1) <= tolerance
    return result


def visible_faces(vertices, faces, camera, depth, *, depth_tolerance=.020):
    """Face-centre rays prevent back/contralateral matches before nearest search."""
    v = np.asarray(vertices); f = np.asarray(faces)
    tri = v[f]; centers = tri.mean(1)
    normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-15)
    direction = -centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-15)
    uv,z = project(centers,camera)
    pix = np.rint(uv).astype(int);h,w = depth.shape
    valid = (z>0)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)&((normal*direction).sum(1)>.10)
    ids = np.flatnonzero(valid)
    if len(ids):
        observed = depth[pix[ids,1],pix[ids,0]]*.001
        valid[ids] &= (observed>0)&(np.abs(observed-z[ids])<depth_tolerance)
    ids = np.flatnonzero(valid)
    valid[ids] &= ray_visible(v,f,centers[ids])
    return valid


def visible_landmarks(vertices, faces, landmarks, uv, valid, camera, depth, tolerance=.006):
    """Detector visibility is NOT trusted; verify model visibility + measured depth."""
    mask = np.asarray(valid,bool).copy()
    mask[:17] = False  # contours are view-dependent, never 3-D material points
    mask &= ray_visible(vertices,faces,landmarks)
    pred,z = project(landmarks,camera)
    safe_uv = np.nan_to_num(uv, nan=-1000.)
    pix = np.rint(safe_uv).astype(int);h,w=depth.shape
    mask &= (pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)&(z>0)
    ids=np.flatnonzero(mask)
    if len(ids):
        d=depth[pix[ids,1],pix[ids,0]]*.001
        mask[ids] &= (d>0)&(np.abs(d-z[ids])<tolerance)&(np.linalg.norm(pred[ids]-uv[ids],axis=1)<10)
    return mask
