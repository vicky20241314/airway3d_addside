"""One exact midpoint subdivision followed by bounded normal displacement.

No FLAME basis or metric landmarks are modified. Each active vertex has one
scalar degree of freedom along its fixed reference normal; no tangential drift.
Sparse weighted least squares is solved repeatedly with projective depth data,
displacement smoothing, small-step backtracking and reference-shape guards.
"""
from dataclasses import dataclass, asdict
import copy
import time
import cv2
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class RefineOptions:
    smooth_weights: tuple = (4., 2.)
    iterations_per_level: int = 8
    anchor_weight: float = .2
    edge_weight: float = .25
    max_offset_m: float = .002
    max_step_m: float = .00025
    max_distance_m: float = .006
    huber_m: float = .001
    min_edge_ratio: float = .8
    max_edge_ratio: float = 1.25
    min_area_ratio: float = .7
    max_area_ratio: float = 1.4
    max_normal_angle_deg: float = 20.
    max_dihedral_change_deg: float = 10.
    step_tol_m: float = 1e-5

    def validate(self):
        values = [v for k,v in asdict(self).items() if k != 'smooth_weights']
        if (not self.smooth_weights or not np.isfinite(values).all()
                or not np.isfinite(self.smooth_weights).all()
                or min(*values, *self.smooth_weights) <= 0
                or not 0 < self.min_edge_ratio <= 1 <= self.max_edge_ratio
                or not 0 < self.min_area_ratio <= 1 <= self.max_area_ratio
                or self.max_normal_angle_deg >= 90 or self.max_dihedral_change_deg >= 90):
            raise ValueError('Invalid conservative refinement settings')


def _normals(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    area2 = np.linalg.norm(cross, axis=1)
    return cross/np.maximum(area2[:,None],1e-15), area2


class ShapeGuard:
    """Bounds relative to the subdivided reference, not just the previous step."""
    def __init__(self, vertices, faces, options):
        self.base = np.asarray(vertices).copy()
        self.faces = np.asarray(faces)
        self.options = options
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        self.edges = mesh.edges_unique
        self.adjacency = mesh.face_adjacency
        self.adjacency_edges = mesh.face_adjacency_edges
        self.n0, self.area0 = _normals(vertices, faces)
        self.length0 = np.linalg.norm(vertices[self.edges[:,0]]-vertices[self.edges[:,1]], axis=1)
        self.valid_faces = self.area0 > 1e-14
        self.valid_edges = self.length0 > 1e-12
        self.angle0 = self._dihedral(vertices,self.n0)

    def _dihedral(self, vertices, normals):
        a,b = self.adjacency.T
        edges = vertices[self.adjacency_edges[:,1]]-vertices[self.adjacency_edges[:,0]]
        edges /= np.maximum(np.linalg.norm(edges,axis=1,keepdims=True),1e-15)
        return np.arctan2(np.einsum('ij,ij->i',np.cross(normals[a],normals[b]),edges),
                          np.einsum('ij,ij->i',normals[a],normals[b]))

    def inspect(self, vertices):
        if not np.isfinite(vertices).all():
            return np.arange(len(vertices)), {'finite':False}
        o=self.options
        normals,area=_normals(vertices,self.faces)
        lengths=np.linalg.norm(vertices[self.edges[:,0]]-vertices[self.edges[:,1]],axis=1)
        ratios=lengths/np.maximum(self.length0,1e-15)
        area_ratios=area/np.maximum(self.area0,1e-15)
        cosine=np.clip(np.einsum('ij,ij->i',normals,self.n0),-1,1)
        normal_angle=np.rad2deg(np.arccos(cosine))
        change=self._dihedral(vertices,normals)-self.angle0
        change=np.rad2deg(np.abs(np.arctan2(np.sin(change),np.cos(change))))
        bad_faces=self.valid_faces & ((area_ratios<o.min_area_ratio-1e-9)
            | (area_ratios>o.max_area_ratio+1e-9) | (normal_angle>o.max_normal_angle_deg+1e-8))
        bad_edges=self.valid_edges & ((ratios<o.min_edge_ratio-1e-9)|(ratios>o.max_edge_ratio+1e-9))
        valid_pairs=self.valid_faces[self.adjacency].all(1)
        bad_pairs=valid_pairs & (change>o.max_dihedral_change_deg+1e-8)
        bad=np.unique(np.r_[self.faces[bad_faces].ravel(),self.edges[bad_edges].ravel(),
                            self.faces[self.adjacency[bad_pairs]].ravel()])
        report=dict(finite=True, min_edge_ratio=float(ratios[self.valid_edges].min(initial=1)),
                    max_edge_ratio=float(ratios[self.valid_edges].max(initial=1)),
                    min_area_ratio=float(area_ratios[self.valid_faces].min(initial=1)),
                    max_area_ratio=float(area_ratios[self.valid_faces].max(initial=1)),
                    max_normal_angle_deg=float(normal_angle[self.valid_faces].max(initial=0)),
                    max_dihedral_change_deg=float(change[valid_pairs].max(initial=0)))
        return bad,report


def _project(vertices, camera, shape):
    z=-vertices[:,2]
    uv=np.column_stack((camera.fx*vertices[:,0]/np.maximum(z,1e-8)+camera.cx,
                        -camera.fy*vertices[:,1]/np.maximum(z,1e-8)+camera.cy))
    pixels=np.rint(uv).astype(int)
    h,w=shape
    inside=(z>0)&(pixels[:,0]>=2)&(pixels[:,0]<w-2)&(pixels[:,1]>=2)&(pixels[:,1]<h-2)
    x=np.clip(pixels[:,0],2,w-3);y=np.clip(pixels[:,1],2,h-3)
    return uv,x,y,inside


def _safe_pixels(depth, parsing, landmarks):
    # Eyes, lips, nostrils, silhouette boundaries and neck stay fixed. No extra
    # 51-point fitting force: landmarks only protect uncertain feature regions.
    skin=np.isin(parsing,(1,2,3,10)).astype(np.uint8)
    safe=cv2.erode(skin,np.ones((7,7),np.uint8)).astype(bool)
    protected=np.zeros(depth.shape,np.uint8)
    if landmarks is not None:
        for p in np.asarray(landmarks)[list(range(31,36))+list(range(36,68))]:
            if np.isfinite(p).all():
                cv2.circle(protected,tuple(np.rint(p).astype(int)),6,1,-1)
    return safe & ~protected.astype(bool)


def _observations(vertices, depth, safe, camera):
    uv,x,y,inside=_project(vertices,camera,depth.shape)
    patch=np.stack([depth[y+dy,x+dx] for dy in (-1,0,1) for dx in (-1,0,1)],axis=1).astype(float)
    span=np.ptp(patch,axis=1)
    valid=inside & safe[y,x] & (patch>0).all(1) & (span<8.)
    # Range-preserving median, with a local plane subpixel correction.
    coords=np.array([(dx,dy,1.) for dy in (-1,0,1) for dx in (-1,0,1)])
    plane=patch@np.linalg.pinv(coords).T
    z=(np.median(patch,axis=1)+np.sum(plane[:,:2]*(uv-np.column_stack((x,y))),axis=1))/1000.
    points=np.column_stack(((uv[:,0]-camera.cx)*z/camera.fx,-(uv[:,1]-camera.cy)*z/camera.fy,-z))
    confidence=np.exp(-(span/6.)**2)*valid
    return points,confidence,y*depth.shape[1]+x


def refine_mesh(mesh, depth, camera, parsing, landmarks_2d=None, options=None, *, subdivide=True):
    options=options or RefineOptions();options.validate()
    started=time.perf_counter()
    depth=np.asarray(depth);parsing=np.asarray(parsing)
    if (depth.ndim!=2 or depth.shape!=parsing.shape or min(depth.shape)<5
            or not np.isfinite(depth).all() or (depth<0).any()):
        raise ValueError('Expected finite nonnegative depth and matching parsing')
    if not np.isfinite([camera.fx,camera.fy,camera.cx,camera.cy]).all() or min(camera.fx,camera.fy)<=0:
        raise ValueError('Invalid camera intrinsics')
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces)==0:
        raise ValueError('Invalid input mesh')
    # The default path is unchanged. The no-subdivision variant solves the same
    # guarded normal offsets directly on the original stage-2 vertices.
    if subdivide:
        base,faces=trimesh.remesh.subdivide(np.asarray(mesh.vertices),np.asarray(mesh.faces))
    else:
        base=np.asarray(mesh.vertices).copy();faces=np.asarray(mesh.faces).copy()
    reference=trimesh.Trimesh(base,faces,process=False)
    normals=np.asarray(reference.vertex_normals).copy()
    guard=ShapeGuard(base,faces,options)
    edges=guard.edges
    rows=np.repeat(np.arange(len(edges)),2)
    E=sparse.coo_matrix((np.tile([1.,-1.],len(edges)),(rows,edges.ravel())),shape=(len(edges),len(base))).tocsr()
    adjacency=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(base),len(base))).tocsr()
    degree=np.asarray(adjacency.sum(1)).ravel()
    L=sparse.eye(len(base))-sparse.diags(1/np.maximum(degree,1))@adjacency
    safe=_safe_pixels(depth,parsing,landmarks_2d)
    points,confidence,_=_observations(base,depth,safe,camera)
    view=-base/np.maximum(np.linalg.norm(base,axis=1,keepdims=True),1e-12)
    eligible=(confidence>.1)&(np.einsum('ij,ij->i',normals,view)>.4)
    eligible &= np.linalg.norm(points-base,axis=1)<options.max_distance_m
    # Test visibility once on V0. Subsequent motion is small and remains guarded.
    candidates=np.flatnonzero(eligible)
    if len(candidates):
        hit,ray_ids,_=reference.ray.intersects_location(np.zeros((len(candidates),3)), -view[candidates],multiple_hits=False)
        visible=np.zeros(len(candidates),bool)
        visible[ray_ids]=np.linalg.norm(hit-base[candidates[ray_ids]],axis=1)<.0005
        eligible[candidates[~visible]]=False
    ids=np.flatnonzero(eligible)
    info=dict(method=('subdivided_normal_displacement' if subdivide else 'normal_displacement_without_subdivision'),
              subdivision_enabled=bool(subdivide),options=asdict(options),
              original_vertices=len(mesh.vertices),subdivided_vertices=len(base),
              active_vertices=len(ids),history=[],metric_geometry_stage=2)
    d=np.zeros(len(ids));current=base.copy()
    status='no_observations' if not len(ids) else 'max_iterations'
    if len(ids):
        # Smooth the vector displacement N*d rather than absolute mesh positions.
        K=sparse.csr_matrix((len(ids),len(ids)))
        G=sparse.csr_matrix((len(ids),len(ids)))
        scaled_E=sparse.diags(.002/np.maximum(guard.length0,.0002))@E
        for axis in range(3):
            N=sparse.diags(normals[ids,axis])
            LN=L[:,ids]@N;EN=scaled_E[:,ids]@N
            K+=LN.T@LN;G+=EN.T@EN
        boundary=np.asarray(adjacency@(~eligible).astype(float)).ravel()[ids]>0
        anchor=options.anchor_weight*(1+9*boundary)
        for smooth in options.smooth_weights:
            R=smooth*K+options.edge_weight*G+sparse.diags(anchor)
            for iteration in range(options.iterations_per_level):
                observed,confidence,pixel=_observations(current,depth,safe,camera)
                delta=observed-current
                w=confidence[ids]
                w*=np.linalg.norm(delta[ids],axis=1)<options.max_distance_m
                target=np.einsum('ij,ij->i',observed[ids]-base[ids],normals[ids])
                residual=target-d
                w*=np.minimum(1.,options.huber_m/np.maximum(np.abs(residual),1e-12))
                unique,inv,counts=np.unique(pixel[ids],return_inverse=True,return_counts=True)
                w/=counts[inv]  # multiple mesh vertices cannot amplify one pixel
                if not (w>0).any():
                    status='no_correspondences';break
                H=R+sparse.diags(w)
                proposal=spsolve(H.tocsc(),w*target)
                if not np.isfinite(proposal).all():
                    status='nonfinite_proposal';break
                proposal=np.clip(proposal,-options.max_offset_m,options.max_offset_m)
                direction=np.clip(proposal-d,-options.max_step_m,options.max_step_m)
                # First damp only the vertices involved in a threatened feature.
                for _ in range(12):
                    trial=base.copy();trial[ids]+=normals[ids]*(d+direction)[:,None]
                    bad,_=guard.inspect(trial)
                    if not len(bad):break
                    direction[np.isin(ids,bad)]*=.5
                def energy(value):
                    return float(np.sum(w*(value-target)**2)+value@(R@value))
                initial=energy(d);accepted=False
                for _ in range(16):
                    candidate=d+direction
                    trial=base.copy();trial[ids]+=normals[ids]*candidate[:,None]
                    bad,quality=guard.inspect(trial)
                    if not len(bad) and energy(candidate)<=initial+1e-16:
                        accepted=True;break
                    direction*=.5
                step=float(np.abs(direction).max()) if accepted else 0.
                if accepted:d,current=candidate,trial
                info['history'].append(dict(smooth=smooth,iteration=iteration+1,
                    accepted=accepted,max_step_m=step,observations=int((w>0).sum()),
                    energy_before=initial,energy_after=energy(d)))
                if not accepted:status='shape_or_energy_guard';break
                if step<options.step_tol_m:status='small_step';break
    bad,quality=guard.inspect(current)
    if len(bad):raise RuntimeError('Refinement violated reference-shape bounds')
    info.update(status=status,quality=quality,max_offset_m=float(np.linalg.norm(current-base,axis=1).max()),
                elapsed_s=time.perf_counter()-started)
    result=trimesh.Trimesh(current,faces,process=False)
    result.metadata=copy.deepcopy(mesh.metadata)
    result.metadata.update(geometry_stage=3,metrics_geometry_stage=2,refinement=info)
    # Stored separately: stage-2 landmark metadata must never masquerade as a
    # correspondence to the newly subdivided/refined mesh.
    result.metadata.pop('flame_landmarks_3d',None)
    return result,base,info
