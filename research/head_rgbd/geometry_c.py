"""Measured-ray triangle correspondences and bounded, evidence-labelled refinement.

All distances are metres in X-right/Y-up/Z-backward camera coordinates.
No nearest-vertex correspondence or interpolated depth is used here.
"""
import gc
import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.ndimage import minimum_filter, maximum_filter, binary_erosion
import trimesh

from fit_b import normals
from prepare import backproject


def mouth_pixels(d, margin=8):
    mask = np.zeros(d['depth'].shape, np.uint8)
    uv = d['uv'][48:60]
    if np.isfinite(uv).all():
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
        mask = cv2.dilate(mask, np.ones((2*margin+1, 2*margin+1), np.uint8))
    return mask.astype(bool)


def sample_depth(d, count, rng, exclude_mouth=False, stride=None):
    depth = d['depth'].astype(float)*.001
    valid = d['face_mask'] & (depth > .1) & (depth < 1.5)
    # Reject semantic edges and measured depth jumps; never invent cavity depth.
    valid = binary_erosion(valid, iterations=2)
    lo = minimum_filter(np.where(depth > .1, depth, np.inf), size=3)
    hi = maximum_filter(depth, size=3)
    valid &= (hi-lo < .010)
    if exclude_mouth:
        valid &= ~mouth_pixels(d)
    if stride:
        grid = np.zeros_like(valid); grid[::stride, ::stride] = True
        valid &= grid
    ys, xs = np.where(valid)
    take = rng.choice(len(xs), min(count, len(xs)), replace=False)
    uv = np.column_stack((xs[take], ys[take]))
    pts, ok = backproject(uv, d['depth'], d['intrinsics'])
    return pts[ok], uv[ok], dict(valid_pixels=int(valid.sum()), sampled=int(ok.sum()))


def first_hits(mesh,points,chunk=192):
    """Bound transient rtree intersection arrays; retain original ray numbering."""
    hits=[];rays=[];faces=[]
    for start in range(0,len(points),chunk):
        block=points[start:start+chunk]
        h,r,f=mesh.ray.intersects_location(np.zeros_like(block),block,multiple_hits=False)
        hits.append(h);rays.append(r+start);faces.append(f)
    if not hits:return np.empty((0,3)),np.empty(0,int),np.empty(0,int)
    return np.concatenate(hits),np.concatenate(rays),np.concatenate(faces)


def ray_surface(v, f, points, max_depth=.025, max_plane=None):
    """Exact first-hit triangle query for each real observation ray (rtree).

    Occluded triangles cannot receive a correspondence. The visibility query is
    two-sided; a separate front-normal/incidence gate rejects back-facing hits.
    Barycentric coordinates are recomputed in 3D at the intersection.
    """
    if not len(points):return None,dict(rays=0,hits=0,accepted=0,reason='no_reliable_depth')
    mesh = trimesh.Trimesh(v, f, process=False)
    hit, ray, face = first_hits(mesh,points)
    # trimesh ray/cache objects have reference cycles; do not accumulate trees
    # over hundreds of offline ICP updates on a 16GB shared machine.
    del mesh
    gc.collect()
    if not len(hit):
        return None, dict(rays=len(points), hits=0, accepted=0)
    tri = v[f[face]]
    bc = trimesh.triangles.points_to_barycentric(tri, hit)
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    direction = -hit / np.maximum(np.linalg.norm(hit, axis=1, keepdims=True), 1e-12)
    incidence = (normal*direction).sum(1)
    plane = ((hit-points[ray])*normal).sum(1)
    depth = np.abs(hit[:, 2]-points[ray, 2])
    finite = np.isfinite(bc).all(1) & np.isfinite(plane)
    front = incidence > .15
    keep = finite & front & (depth < max_depth)
    if max_plane is not None:
        keep &= np.abs(plane) < max_plane
    stats = dict(rays=len(points), hits=len(hit), no_surface_hit=len(points)-len(hit),
                 invalid=int((~finite).sum()), backface_or_grazing=int((~front).sum()),
                 depth_rejected=int((depth >= max_depth).sum()),
                 plane_rejected=int((np.abs(plane) >= max_plane).sum()) if max_plane else 0,
                 accepted=int(keep.sum()))
    if not keep.any():
        return None, stats
    return dict(face=face[keep], bary=bc[keep], normal=normal[keep],
                target=points[ray[keep]], hit=hit[keep], ray=ray[keep],
                incidence=incidence[keep], plane=plane[keep]), stats


def landmark_validity(d):
    valid = d['valid'].copy() & np.isfinite(d['xyz']).all(1)
    rejected = []
    # Inner-lip depth is ambiguous at a cavity boundary. A >6mm jump relative
    # to other points on the same lip is unreliable, even if nonzero.
    for group in ([61, 62, 63], [65, 66, 67]):
        ids = np.array([j for j in group if valid[j]])
        if len(ids) >= 3:
            median = np.median(d['xyz'][ids, 2])
            bad = ids[np.abs(d['xyz'][ids, 2]-median) > .006]
            valid[bad] = False; rejected.extend(bad.tolist())
    return valid, rejected


def refine_shared(vertices, faces, data, identity_ids, mouth_vertex, rng, rows=None,
                  fixed_mask=None, fixed_deviation=None):
    """Bounded barycentric point-to-plane least squares with spatial smoothing.

    Only closed observations contribute. Directions are each frame's animated
    normals, so neither raw depth nor different expressions are averaged.
    """
    count = vertices.shape[1]
    matrices, targets, weights, records = [], [], [], []
    support = np.zeros((len(identity_ids), count), bool)
    attempted = np.zeros(count, int)
    for k, i in enumerate(identity_ids):
        points, _, sampling = sample_depth(data[i], 16000, rng, True, stride=3)
        corr, stats = ray_surface(vertices[i], faces, points, .008, .005)
        records.append(dict(frame_index=i, sampling=sampling, **stats))
        if corr is None:
            continue
        ids = faces[corr['face']]
        for j in range(3):
            np.add.at(attempted, ids[:, j], corr['bary'][:, j] > .05)
        good = ~mouth_vertex[ids].any(1)
        ids = ids[good]; bc = corr['bary'][good]
        vn = normals(vertices[i], faces)
        value = bc*(vn[ids]*corr['normal'][good, None]).sum(2)
        residual = -corr['plane'][good]
        weight = corr['incidence'][good]**2 / np.maximum(1, np.abs(residual)/.002)
        if rows is not None and rows[i]['view']=='front':weight*=3.
        support[k, np.unique(ids[bc > .05])] = True
        matrices.append(sparse.coo_matrix((value.ravel(),
            (np.repeat(np.arange(len(ids)), 3), ids.ravel())), shape=(len(ids), count)).tocsr())
        targets.append(residual); weights.append(weight)
    if not matrices:
        return (np.zeros(count) if fixed_deviation is None else fixed_deviation.copy()), support, attempted, records
    A = sparse.vstack(matrices); b = np.concatenate(targets); w = np.concatenate(weights)
    edges = np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [0, 2]])), axis=1), axis=0)
    L = sparse.coo_matrix((np.tile([1., -1.], len(edges)),
        (np.repeat(np.arange(len(edges)), 2), edges.ravel())), shape=(len(edges), count)).tocsr()
    supported = (support.sum(0) >= 2) & ~mouth_vertex
    if fixed_mask is not None:supported &= ~fixed_mask
    free = np.flatnonzero(supported)
    deviation = np.zeros(count) if fixed_deviation is None else fixed_deviation.copy()
    if not len(free):
        return deviation, support, attempted, records
    # Bounds are unchanged from B. Unsupported/mouth columns are exactly zero.
    fixed=deviation.copy();fixed[free]=0
    for _ in range(3):
        rw = np.sqrt(w / np.maximum(1., np.abs(A@deviation-b)/.002))
        system = sparse.vstack((A[:, free].multiply(rw[:, None]), .5*L[:, free],
                                .1*sparse.eye(count, format='csr')[:, free]))
        target = np.concatenate(((b-A@fixed)*rw, -.5*(L@fixed), -.1*fixed))
        fit = lsq_linear(system, target, bounds=(-.003, .003), tol=1e-6,
                         lsmr_tol=1e-6, max_iter=60)
        if not np.isfinite(fit.x).all():
            raise ValueError('nonfinite refinement')
        deviation[free] = fit.x
    records.append(dict(solver_status=int(fit.status), optimality=float(fit.optimality),
                        bound_m=.003, mouth_identity_refinement=False))
    return deviation, support, attempted, records


def front_protected_vertices(vertices,faces,data,front_ids):
    """Protect real frontal measurement rays, plus one topology ring.

    Uses all original valid semantic depth, including pixels rejected for ICP.
    Does not turn a frontal image into a side observation.
    """
    protected=np.zeros(vertices.shape[1],bool)
    for i in front_ids:
        d=data[i];valid=d['face_mask']&(d['depth']>100)&(d['depth']<1500)
        grid=np.zeros_like(valid);grid[::2,::2]=True;ys,xs=np.where(valid&grid)
        pts,ok=backproject(np.column_stack((xs,ys)),d['depth'],d['intrinsics']);pts=pts[ok]
        if not len(pts):continue
        mesh=trimesh.Trimesh(vertices[i],faces,process=False)
        _,_,fi=first_hits(mesh,pts)
        protected[np.unique(faces[fi])]=True
        del mesh
        gc.collect()
    protected[np.unique(faces[protected[faces].any(1)])]=True
    return protected


def safe_refinement(vertices,faces,deviation,locked_mask=None,locked_effective=None):
    """Deterministic pose-dependent orientation barrier, no identity parameters.

    Keep >=5% original signed triangle area. Only attenuate bounded normal
    offsets; never expand the displacement budget. Optional anchor values are
    fixed boundary conditions, including the anchor's own safety projection.
    """
    output=[];effective=[];records=[]
    for i,v in enumerate(vertices):
        delta=deviation.copy();locked=np.zeros(len(v),bool)
        if locked_effective is not None and i in locked_effective:
            locked=locked_mask;delta[locked]=locked_effective[i][locked]
        normal=normals(v,faces);tri=v[faces]
        original=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area2=(original**2).sum(1)
        initial_bad=0
        for iteration in range(80):
            vv=v+normal*delta[:,None];t=vv[faces]
            current=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
            bad=((current*original).sum(1)<.05*area2)&(area2>1e-24)
            if iteration==0:initial_bad=int(bad.sum())
            if not bad.any():break
            ids=np.unique(faces[bad]);ids=ids[~locked[ids]]
            if not len(ids):raise RuntimeError('unsafe locked anchor geometry')
            delta[ids]*=.5
        else:raise RuntimeError('orientation projection failed; no mesh exported')
        output.append(vv);effective.append(delta)
        records.append(dict(frame_index=i,initial_unsafe_triangles=initial_bad,final_unsafe_triangles=0,
            attenuated_vertices=int((np.abs(delta-deviation)>1e-9).sum()),iterations=iteration,
            minimum_signed_area_ratio=float(np.min((current*original).sum(1)[area2>1e-24]/area2[area2>1e-24]))))
    return np.asarray(output),np.asarray(effective),records


def neck_surfaces(rows, data, output):
    """Separate per-track neck surface; gaps/lost neck tracks create components.

    This is a measured open depth patch in a segment anchor camera, not a
    watertight neck identity model and not registration between capture gaps.
    """
    reports = []; groups = []
    for i, row in enumerate(rows):
        motion = row.get('neck_overlap', {})
        contiguous = i > 0 and row['segment'] == rows[i-1]['segment'] and (
            0 < row['timestamp_s']-rows[i-1]['timestamp_s'] < .2)
        if contiguous and motion.get('status') == 'tracked' and groups:
            oldR, oldt = groups[-1][-1][1:]
            R = np.asarray(motion['previous_to_current_R'])@oldR
            t = np.asarray(motion['previous_to_current_R'])@oldt+np.asarray(motion['previous_to_current_t'])
        else:
            groups.append([]); R = np.eye(3); t = np.zeros(3)
        groups[-1].append((i, R, t))
    for gi, group in enumerate(groups):
        anchor = group[0][0]; h, w = data[anchor]['depth'].shape; stride = 4
        hh, ww = (h+3)//4, (w+3)//4; all_pixels=[]; all_z=[]; all_frames=[]
        for i, R, t in group:
            d = data[i]; grid = np.zeros((h,w), bool); grid[::2,::2] = True
            ys,xs = np.where(d['neck_mask'] & (d['depth']>100) & (d['depth']<1500) & grid)
            cloud, ok = backproject(np.column_stack((xs,ys)), d['depth'], d['intrinsics'])
            cloud = (cloud[ok]-t)@R
            fx,fy,cx,cy = data[anchor]['intrinsics']; z = -cloud[:,2]
            pix = np.rint(np.column_stack((fx*cloud[:,0]/z+cx, -fy*cloud[:,1]/z+cy))/stride).astype(int)
            keep=(pix[:,0]>=0)&(pix[:,0]<ww)&(pix[:,1]>=0)&(pix[:,1]<hh)&(z>.1)
            all_pixels.extend((pix[keep,1]*ww+pix[keep,0]).tolist());all_z.extend(z[keep]);all_frames.extend([i]*int(keep.sum()))
        buckets={}
        for p,z,frame in zip(all_pixels,all_z,all_frames):buckets.setdefault(p,[]).append((z,frame))
        depth=np.zeros((hh,ww));support=np.zeros((hh,ww),int)
        for p,values in buckets.items():
            values=np.array(values);median=np.median(values[:,0]);consistent=np.abs(values[:,0]-median)<.005
            if not consistent.any():continue
            depth.flat[p]=np.median(values[consistent,0]);support.flat[p]=len(np.unique(values[consistent,1]))
        ys,xs=np.where(depth>.1);index=np.full((hh,ww),-1,int);index[ys,xs]=np.arange(len(xs))
        fx,fy,cx,cy=data[anchor]['intrinsics'];z=depth[ys,xs]
        vertices=np.column_stack(((xs*stride-cx)*z/fx,-(ys*stride-cy)*z/fy,-z))
        faces=[]
        for y in range(hh-1):
            for x in range(ww-1):
                for indices in ([index[y,x],index[y+1,x],index[y,x+1]],
                                [index[y,x+1],index[y+1,x],index[y+1,x+1]]):
                    if min(indices)<0:continue
                    tri=vertices[indices]
                    if max(np.linalg.norm(tri[0]-tri[1]),np.linalg.norm(tri[1]-tri[2]),np.linalg.norm(tri[2]-tri[0]))<.012:faces.append(indices)
        name=f'neck_track_{gi:02d}_observed_m.ply'
        if faces:
            mesh=trimesh.Trimesh(vertices,np.asarray(faces),process=False)
            mesh.export(output/name)
            np.savez_compressed(output/f'neck_track_{gi:02d}_support.npz',support_frames=support[ys,xs],vertices_m=vertices,faces=np.asarray(faces))
        reports.append(dict(anchor_frame=rows[anchor]['id'],frames=[rows[i]['id'] for i,_,_ in group],
            status='observed_patch_exported' if faces else 'insufficient_neck_depth',mesh=name if faces else None,
            vertices=len(vertices),triangles=len(faces),face_identity_feedback=False,
            coordinates='anchor camera X_right_Y_up_Z_backward; metres',
            limitation='independent neck tracking and measured open patch; no cross-gap neck identity registration'))
    return reports
