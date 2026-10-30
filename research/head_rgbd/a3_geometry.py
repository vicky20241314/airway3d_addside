"""Independent, model-free RGB-D surface reconstruction (A3, not hifi3dface).

Only NumPy/SciPy and the standard library. Camera convention throughout is
X right, Y down, Z forward, metres; T_reference_from_camera @ [p,1].
No learned landmarks, segmentation, template, texture or head completion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def transform(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def backproject(depth_m, K):
    y, x = np.indices(depth_m.shape)
    fx, fy, cx, cy = K
    return np.stack(((x - cx) * depth_m / fx,
                     (y - cy) * depth_m / fy, depth_m), axis=-1)


def project(points, K):
    fx, fy, cx, cy = K
    z = points[..., 2]
    return np.stack((fx * points[..., 0] / np.maximum(z, 1e-12) + cx,
                     fy * points[..., 1] / np.maximum(z, 1e-12) + cy), -1)


def polygon_mask(shape, polygon):
    """Even-odd fill at actual pixel centres; polygon is an ROI, not truth."""
    y, x = np.indices(shape)
    inside = np.zeros(shape, bool)
    p = np.asarray(polygon, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('ROI needs at least three finite pixel coordinates')
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        if a[1] == b[1]:
            continue
        crosses = ((a[1] > y) != (b[1] > y))
        inside ^= crosses & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    return inside


def rigid_fit(src, dst):
    """Kabsch SE(3), deliberately no scale parameter."""
    src, dst = np.asarray(src), np.asarray(dst)
    if len(src) < 4 or src.shape != dst.shape or not np.isfinite([src, dst]).all():
        raise ValueError('at least four finite paired XYZ landmarks required')
    a, b = src.mean(0), dst.mean(0)
    u, s, vt = np.linalg.svd((src-a).T @ (dst-b))
    if s[1] < 1e-8:
        raise ValueError('degenerate landmark arrangement')
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    T = np.eye(4)
    T[:3, :3] = vt.T @ correction @ u.T
    T[:3, 3] = b - T[:3, :3] @ a
    return T


def depth_normals(xyz, valid):
    n = np.zeros_like(xyz)
    dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    n[1:-1, 1:-1] = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1)
    good = binary_erosion(valid) & (norm > 1e-12)
    n /= np.maximum(norm[..., None], 1e-12)
    n[np.sum(n * xyz, axis=-1) > 0] *= -1
    n[~good] = 0
    return n, good


def load_frame(row, stride, base):
    path = Path(row['source'])
    if not path.is_absolute():
        path = base / path
    with np.load(path, allow_pickle=False) as data:
        idx = int(row['source_frame'])
        all_depth = data['depth']
        source_frame_count = len(all_depth)
        raw = all_depth[idx]
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError('raw depth must be uint16 millimetres')
        intr = np.asarray(data['intrinsics'], float)
        intr = intr[idx] if intr.ndim == 2 else intr
        if intr.shape != (4,) or not np.isfinite(intr).all() or min(intr[:2]) <= 0:
            raise ValueError('invalid per-frame intrinsic calibration')
        timestamp = float(data['timestamps_s'][idx])
        if data['rgb'].shape[1:3] != raw.shape:
            raise ValueError('RGB/depth raster mismatch')
        roi_full = polygon_mask(raw.shape, row['roi_polygon_px'])
        # Millimetres are a declared sensor encoding, NOT an inferred OBJ scale.
        depth = raw[::stride, ::stride].astype(float) / 1000.
        roi = roi_full[::stride, ::stride]
        valid = roi & np.isfinite(depth) & (depth > .1) & (depth < 1.5)
        xyz = backproject(depth, intr / stride)
        normal, normal_valid = depth_normals(xyz, valid)
        lm = {}
        for name, uv in row['landmarks_px'].items():
            x, y = np.rint(uv).astype(int)
            z = None
            if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0]:
                patch = raw[max(0,y-2):y+3, max(0,x-2):x+3]
                observed = patch[(patch > 100) & (patch < 1500)]
                if raw[y, x] > 100 and raw[y, x] < 1500 and len(observed) >= 9:
                    z = float(np.median(observed)) / 1000.
            lm[name] = None if z is None else np.array([
                (uv[0]-intr[2])*z/intr[0], (uv[1]-intr[3])*z/intr[1], z])
    return dict(row=row, path=path, K=intr/stride, raw_K=intr,
                depth=depth, valid=valid, roi=roi, xyz=xyz, normal=normal,
                normal_valid=normal_valid, landmarks=lm, timestamp_s=timestamp,
                source_shape=list(raw.shape), source_frame_count=source_frame_count, stride=stride)


def paired_landmarks(frame, ref, names):
    common = [n for n in names if frame['landmarks'].get(n) is not None
              and ref['landmarks'].get(n) is not None]
    return (np.array([frame['landmarks'][n] for n in common]),
            np.array([ref['landmarks'][n] for n in common]), common)


def register(frame, ref, names, iterations):
    src_lm, dst_lm, common = paired_landmarks(frame, ref, names)
    T = rigid_fit(src_lm, dst_lm)
    init = T.copy()
    yy = np.indices(ref['depth'].shape)[0] * ref['stride']
    target_valid = ref['normal_valid'] & (yy < ref['row']['upper_limit_y_px'])
    target = ref['xyz'][target_valid]
    target_n = ref['normal'][target_valid]
    if len(target) < 300:
        raise ValueError('not enough reference upper-face observations')
    tree = cKDTree(target)
    sy = np.indices(frame['depth'].shape)[0] * frame['stride']
    source_valid = frame['normal_valid'] & (sy < frame['row']['upper_limit_y_px'])
    source = frame['xyz'][source_valid][::2]
    source_n = frame['normal'][source_valid][::2]
    history = []
    for gate in (.015, .008, .005):
        for iteration in range(iterations):
            q = transform(source, T)
            distances, indices = tree.query(q)
            n = target_n[indices]
            good = (distances < gate) & (np.sum((source_n @ T[:3,:3].T)*n, -1) > .65)
            if good.sum() < 200:
                raise ValueError('insufficient upper-face ICP overlap')
            p, v, normal = q[good], target[indices[good]], n[good]
            residual = np.sum((p-v)*normal, -1)
            keep = np.abs(residual) < np.percentile(np.abs(residual), 90) + 1e-8
            p, normal, residual = p[keep], normal[keep], residual[keep]
            # Huber influence; normalized dense term and explicitly weighted
            # three-dimensional landmark anchor. Neither can introduce scale.
            weight = np.minimum(1., .0015 / np.maximum(np.abs(residual), 1e-12))
            A = np.column_stack((np.cross(p, normal), normal))
            w = np.sqrt(weight / len(residual))
            A, b = A*w[:,None], -residual*w
            anchors = transform(src_lm, T)
            blocks = []
            for qlm in anchors:
                x,y,z = qlm
                blocks.append(np.column_stack((np.array([[0,z,-y],[-z,0,x],[y,-x,0]]), np.eye(3))))
            anchor_A = np.concatenate(blocks)
            anchor_w = np.sqrt(.08 / len(anchors))
            A = np.vstack((A, anchor_A*anchor_w))
            b = np.r_[b, (dst_lm-anchors).reshape(-1)*anchor_w]
            delta, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            if rank < 6:
                raise ValueError('underconstrained rigid registration')
            dr, dt = np.linalg.norm(delta[:3]), np.linalg.norm(delta[3:])
            delta *= min(1., np.deg2rad(2)/max(dr,1e-12), .005/max(dt,1e-12))
            update = np.eye(4)
            update[:3,:3] = Rotation.from_rotvec(delta[:3]).as_matrix()
            update[:3,3] = delta[3:]
            T = update @ T
            history.append(dict(gate_m=gate, iteration=iteration,
                                correspondences=len(residual),
                                median_abs_point_plane_mm=float(np.median(np.abs(residual))*1000)))
            if np.linalg.norm(delta) < 1e-7:
                break
    q = transform(source, T)
    distances, ix = tree.query(q)
    valid = distances < .005
    residual = np.abs(np.sum((q[valid]-target[ix[valid]])*target_n[ix[valid]], -1))
    median = float(np.median(residual)) if len(residual) else 1.
    # A failed side is recorded and excluded, never silently relabelled front.
    accepted = len(residual) >= 300 and median < .003
    return T, dict(accepted=accepted, scale=1., initial_T=init.tolist(),
                   rigid_landmarks=common, iterations_executed=len(history),
                   median_upper_point_plane_mm=median*1000,
                   landmarks_3d_rms_mm=float(np.sqrt(np.mean((transform(src_lm,T)-dst_lm)**2))*1000),
                   history=history, reason=None if accepted else 'fixed overlap/residual guard failed')


def splat_depth(points, K, shape):
    uv = project(points, K)
    finite = np.isfinite(points).all(1) & np.isfinite(uv).all(1) & (points[:,2] > .1)
    points, uv = points[finite], uv[finite]
    xy = np.rint(uv).astype(int)
    h,w = shape
    inside = (xy[:,0] >= 0) & (xy[:,0] < w) & (xy[:,1] >= 0) & (xy[:,1] < h)
    xy, points = xy[inside], points[inside]
    out = np.full(h*w, np.inf)
    np.minimum.at(out, xy[:,1]*w+xy[:,0], points[:,2])
    out[~np.isfinite(out)] = np.nan
    return out.reshape(shape)


def fuse(frames, transforms, ref, reference_id, state, gate=.003):
    """Common upper identity, reference-specific lower face; no invented samples.

    Other states/bursts are never averaged into a reference jaw or mouth.
    Same-burst lower-face samples must agree within a fixed 1.5 mm gate.
    """
    ids = [f['row']['id'] for f in frames]
    rindex = ids.index(reference_id)
    layers = []
    for frame,T in zip(frames,transforms):
        points = transform(frame['xyz'][frame['valid']], T)
        normal = frame['normal'][frame['valid']] @ T[:3,:3].T
        visible = np.sum(normal*points, -1) < -.15*np.linalg.norm(points,axis=-1)
        layers.append(splat_depth(points[visible], ref['K'], ref['depth'].shape))
    layers = np.asarray(layers)
    # The fixed closed reference is the upper-face identity anchor in all states.
    anchor = layers[0].copy()
    # No nanmedian warnings for unobserved rays.
    sorted_layers = np.sort(np.where(np.isfinite(layers), layers, np.inf), axis=0)
    count = np.isfinite(layers).sum(0)
    mid = np.maximum((count-1)//2, 0)
    consensus = np.take_along_axis(sorted_layers, mid[None], axis=0)[0]
    consensus[count == 0] = np.nan
    anchor = np.where(np.isfinite(anchor), anchor, consensus)
    accepted = np.isfinite(layers) & (np.abs(layers-anchor) <= gate)
    yy = np.indices(anchor.shape)[0]*ref['stride']
    upper = yy < ref['row']['upper_limit_y_px']
    # Lower face: only the same actual short burst as this expression reference.
    # The reference anchors depth; other-state/side bursts cannot move its jaw.
    eligible = np.array([f['path']==frames[rindex]['path'] and
                         f['row']['state']==state for f in frames])
    lower_layers = np.where(eligible[:,None,None],layers,np.nan)
    lower_count = np.isfinite(lower_layers).sum(0)
    lower_sorted = np.sort(np.where(np.isfinite(lower_layers),lower_layers,np.inf),axis=0)
    lower_mid = np.maximum((lower_count-1)//2,0)
    lower_consensus = np.take_along_axis(lower_sorted,lower_mid[None],axis=0)[0]
    # A reference hole is filled only by >=2 actual agreeing measurements.
    lower_consensus[lower_count<2] = np.nan
    lower_anchor = np.where(np.isfinite(layers[rindex]),layers[rindex],lower_consensus)
    accepted[:,~upper] = False
    for i in np.flatnonzero(eligible):
        accepted[i,~upper] = np.isfinite(layers[i,~upper]) & (np.abs(layers[i,~upper]-lower_anchor[~upper])<=.0015)
    # Unsupported single observations cannot fill a reference hole.
    reject = (~upper)&~np.isfinite(layers[rindex])&(accepted.sum(0)<2)
    accepted[:,reject] = False
    selected = np.sort(np.where(accepted,layers,np.inf), axis=0)
    counts = accepted.sum(0)
    middle = np.maximum((counts-1)//2, 0)
    depth = np.take_along_axis(selected,middle[None],axis=0)[0]
    # Even sample counts: symmetric median, not bias toward the front layer.
    high = np.take_along_axis(selected,(counts//2)[None],axis=0)[0]
    depth = np.where(counts > 0, (depth+high)/2, np.nan)
    # Keep only manually delimited support union; do not close depth holes.
    bits = np.zeros(anchor.shape,np.uint32)
    for i in range(len(frames)):
        bits |= accepted[i].astype(np.uint32) << i
    return depth, counts, bits, layers, upper


def refine_depth(depth, count, upper, passes=2):
    """Bounded bilateral prior within each independently anchored face region."""
    original = depth.copy()
    out = depth.copy()
    for _ in range(passes):
        total = np.zeros(depth.shape)
        weights = np.zeros(depth.shape)
        neighbours = np.zeros(depth.shape,int)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                shifted = np.roll(out,(dy,dx),(0,1))
                ok = np.isfinite(shifted) & np.isfinite(out) & (np.abs(shifted-out)<.003)
                # Identity smoothing may not see a reference-specific expression
                # neighbour across the seam. Otherwise open/closed identity drifts.
                ok &= upper == np.roll(upper,(dy,dx),(0,1))
                if dy == 1: ok[0] = False
                if dy == -1: ok[-1] = False
                if dx == 1: ok[:,0] = False
                if dx == -1: ok[:,-1] = False
                w = np.where(ok,np.exp(-np.nan_to_num((shifted-out)/.0015)**2),0.)
                total += np.nan_to_num(shifted)*w
                weights += w
                neighbours += ok
        update = np.isfinite(out) & (count>=2) & (neighbours>=5)
        candidate = total/np.maximum(weights,1e-12)
        out[update] = np.clip(candidate[update],original[update]-.0008,original[update]+.0008)
    return out


def mesh_from_depth(depth, K, max_edge=.006):
    valid = np.isfinite(depth) & (depth > .1)
    xyz = backproject(np.nan_to_num(depth),K)
    index = np.full(depth.shape,-1,int)
    index[valid] = np.arange(valid.sum())
    a,b,c,d = index[:-1,:-1],index[:-1,1:],index[1:,:-1],index[1:,1:]
    triangles = np.concatenate((np.stack((a,c,b),-1).reshape(-1,3),
                                np.stack((b,c,d),-1).reshape(-1,3)))
    triangles = triangles[(triangles>=0).all(1)]
    vertices = xyz[valid]
    pts = vertices[triangles]
    edges = np.stack((pts[:,1]-pts[:,0],pts[:,2]-pts[:,1],pts[:,0]-pts[:,2]),1)
    keep = (np.linalg.norm(edges,axis=-1).max(1)<max_edge)
    keep &= np.linalg.norm(np.cross(edges[:,0],-edges[:,2]),axis=-1)>1e-12
    triangles = triangles[keep]
    if not len(triangles):
        raise ValueError('no supported triangles')
    used = np.unique(triangles)
    remap = np.full(len(vertices),-1,int)
    remap[used] = np.arange(len(used))
    pixels = np.column_stack(np.where(valid))[used]
    return vertices[used], remap[triangles], pixels


def write_obj(path, vertices, faces):
    if not np.isfinite(vertices).all():
        raise ValueError('nonfinite mesh coordinate')
    with open(path,'w',encoding='ascii') as f:
        f.write('# A3 observed face; metres; X right Y down Z forward\n')
        f.write('# Open surface, no template completion, no texture\n')
        np.savetxt(f,vertices,fmt='v %.10f %.10f %.10f')
        np.savetxt(f,faces+1,fmt='f %d %d %d')


def principal_component(vertices, faces):
    """Conservative skin-surface export; detached teeth/cavity islands excluded."""
    i = np.concatenate([faces[:,0],faces[:,1],faces[:,2]])
    j = np.concatenate([faces[:,1],faces[:,2],faces[:,0]])
    graph = coo_matrix((np.ones(len(i)),(i,j)),shape=(len(vertices),len(vertices))).tocsr()
    n,labels = connected_components(graph,directed=False)
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    selected = np.flatnonzero(labels==largest)
    remap = np.full(len(vertices),-1,int)
    remap[selected] = np.arange(len(selected))
    kept = (labels[faces]==largest).all(1)
    stats = dict(components=int(n),discarded_vertices=int(len(vertices)-len(selected)),
                 discarded_triangles=int((~kept).sum()),policy='largest connected observed skin surface; no hole fill')
    return selected,remap[faces[kept]],stats


def observation_support(vertices_common, faces, frames, transforms, tolerance=.003):
    normals = np.zeros_like(vertices_common)
    p = vertices_common[faces]
    fn = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    for i in range(3):
        np.add.at(normals,faces[:,i],fn)
    normals /= np.maximum(np.linalg.norm(normals,axis=-1)[:,None],1e-12)
    bits = np.zeros(len(vertices_common),np.uint32)
    residuals = np.full((len(frames),len(vertices_common)),np.nan,np.float32)
    for i,(frame,T) in enumerate(zip(frames,transforms)):
        inv = np.linalg.inv(T)
        p = transform(vertices_common,inv)
        uv = project(p,frame['K'])
        xy = np.rint(uv).astype(int)
        h,w = frame['depth'].shape
        ok = (p[:,2]>.1)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        ix = np.where(ok)[0]
        x,y = xy[ix].T
        ok[ix] &= frame['valid'][y,x]
        # Facing and sensor-depth tests jointly reject backside and occlusion.
        n = normals @ inv[:3,:3].T
        ok &= np.sum(n*p,axis=-1)<-.1*np.linalg.norm(p,axis=-1)
        ix = np.where(ok)[0]
        x,y = xy[ix].T
        residuals[i,ix] = p[ix,2]-frame['depth'][y,x]
        supported = np.isfinite(residuals[i]) & (np.abs(residuals[i])<=tolerance)
        bits |= supported.astype(np.uint32) << i
    counts = np.array([int(v).bit_count() for v in bits],np.uint8)
    return counts,bits,residuals


def raster_depth(vertices, faces, K, shape, stride=4):
    """Independent CPU triangle z buffer, perspective-correct reciprocal Z."""
    h,w = shape
    image = np.full(((h+stride-1)//stride,(w+stride-1)//stride),np.inf)
    uv = project(vertices,K)/stride
    for face in faces:
        z = vertices[face,2]
        if np.any(z<=.1):
            continue
        tri = uv[face]
        lo = np.maximum(np.ceil(tri.min(0)).astype(int),0)
        hi = np.minimum(np.floor(tri.max(0)).astype(int),[image.shape[1]-1,image.shape[0]-1])
        if np.any(hi<lo):
            continue
        x,y = np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        a,b,c = tri
        det = np.cross(b-a,c-a)
        if abs(det)<1e-12:
            continue
        wb = ((x-a[0])*(c[1]-a[1])-(y-a[1])*(c[0]-a[0]))/det
        wc = ((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))/det
        wa = 1-wb-wc
        covered = (wa>=-1e-7)&(wb>=-1e-7)&(wc>=-1e-7)
        zz = np.where(covered,1/np.maximum(wa/z[0]+wb/z[1]+wc/z[2],1e-12),np.inf)
        patch = image[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        np.minimum(patch,zz,out=patch)
    return image


def diagnostic(vertices, faces, frame, stride=2):
    rendered = raster_depth(vertices,faces,frame['K'],frame['depth'].shape,stride)
    observed = frame['depth'][::stride,::stride]
    mask = frame['valid'][::stride,::stride]
    hit = mask & np.isfinite(rendered)
    err = np.abs(rendered-observed)*1000
    y = np.indices(mask.shape)[0]*stride*frame['stride']
    split = frame['row']['upper_limit_y_px']
    regions = {'face_roi':mask,'upper_shared':mask&(y<split),
               'lower_reference_state':mask&(y>=split)}
    lm = frame['row']['landmarks_px']
    if 'chin' in lm:
        regions['chin_jaw_band'] = mask & (y >= lm['chin'][1]-70)
    if 'upper_lip_inner' in lm and 'lower_lip_inner' in lm:
        centre = np.mean([lm['upper_lip_inner'],lm['lower_lip_inner']],axis=0)
        x = np.indices(mask.shape)[1]*stride*frame['stride']
        regions['mouth_box'] = mask & (np.abs(x-centre[0])<50) & (np.abs(y-centre[1])<45)
    result = {}
    for name,roi in regions.items():
        e = err[roi&hit]
        result[name] = dict(valid_pixels=int(roi.sum()),hit_pixels=int((roi&hit).sum()),
            median_axial_mm=float(np.median(e)) if len(e) else None,
            p95_axial_mm=float(np.percentile(e,95)) if len(e) else None,
            ray_overlap=float((roi&hit).sum()/max(roi.sum(),1)),
            within_3mm_fraction=float((roi&hit&(err<=3)).sum()/max(roi.sum(),1)))
    return result
