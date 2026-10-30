"""Separate in-fit diagnostics from independent, calibrated held-out views."""
import hashlib
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from core.multiview_geometry import project,visible_faces


def input_digest(o):
    h=hashlib.sha256()
    for a in (o.rgb,o.depth):
        h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def assert_held_out(training,held_out):
    ids={o.view_id for o in training};hashes={input_digest(o) for o in training}
    for o in held_out:
        if o.view_id in ids or input_digest(o) in hashes:
            raise ValueError(f'{o.view_id}: held-out input duplicates a fitted view/ID')


def silhouette(vertices,faces,camera,shape):
    uv,z=project(vertices,camera);mask=np.zeros(shape,np.uint8);h,w=shape
    for f in faces:
        if not (z[f]>0).all() or not np.isfinite(uv[f]).all():continue
        if (uv[f,0]<0).all() or (uv[f,0]>=w).all() or (uv[f,1]<0).all() or (uv[f,1]>=h).all():continue
        poly=np.rint(np.clip(uv[f],[-2*w,-2*h],[3*w,3*h])).astype(np.int32)
        cv2.fillConvexPoly(mask,poly,1)
    return mask.astype(bool)


def contour_errors(predicted,reference):
    predicted=np.asarray(predicted,bool);reference=np.asarray(reference,bool)
    if predicted.shape!=reference.shape:raise ValueError('Reviewed reference mask/image shape mismatch')
    kernel=np.ones((3,3),np.uint8)
    a=predicted & ~cv2.erode(predicted.astype(np.uint8),kernel).astype(bool)
    b=reference & ~cv2.erode(reference.astype(np.uint8),kernel).astype(bool)
    pa=np.column_stack(np.where(a));pb=np.column_stack(np.where(b))
    if not len(pa) or not len(pb):return dict(status='empty_contour')
    da,ia=cKDTree(pa).query(pb);db,_=cKDTree(pb).query(pa)
    signed=cv2.distanceTransform((~reference).astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform(reference.astype(np.uint8),cv2.DIST_L2,5)
    gy,gx=np.gradient(signed);normals=np.column_stack((gy[b],gx[b]))
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    normal_error=np.abs(np.sum((pa[ia]-pb)*normals,axis=1))
    return dict(status='reviewed_mask',chamfer_mean_px=float((da.mean()+db.mean())/2),
        normal_mean_px=float(normal_error.mean()),normal_p95_px=float(np.percentile(normal_error,95)))


def evaluate_view(mesh,o,transform,reference_mask=None,max_points=16000):
    from core.side_constraints import constraint_observation
    o=constraint_observation(o)
    from core.student.offset_refine import depth_inputs
    from core.student.detail_snap import SNAP_LABELS
    v=transform.to_view(mesh.vertices);faces=np.asarray(mesh.faces)
    p,confidence,_=depth_inputs(v,faces,o.depth,o.camera,o.parsing,o.uv)
    reliable_count=len(p)
    yy,xx=np.mgrid[2:o.depth.shape[0]-2:2,2:o.depth.shape[1]-2:2]
    candidates=int(np.isin(o.parsing[yy,xx],SNAP_LABELS).sum())
    if len(p)>max_points:p=p[np.sort(np.random.default_rng(42).choice(len(p),max_points,replace=False))]
    # Evaluation never gates model faces by agreement with measured depth.
    # Visibility is geometric, errors are reported BEFORE any distance trimming.
    visible=visible_faces(v,faces,o.camera,o.depth,depth_tolerance=None)
    distance=np.empty(0)
    if len(p) and visible.any():
        surface=trimesh.Trimesh(v,faces[visible],process=False)
        chunks=[trimesh.proximity.closest_point(surface,p[start:start+256])[1] for start in range(0,len(p),256)]
        distance=np.concatenate(chunks)
    result=dict(view_id=o.view_id,state=o.state,view=o.view,constraint_region='front_face_neck' if o.view=='front' else 'head_only',n_candidate_pixels=candidates,
        n_reliable_depth=reliable_count,n_surface_samples=len(distance),visible_faces=int(visible.sum()),
        reliable_depth_fraction=float(reliable_count/max(candidates,1)),
        correspondence_threshold_m=.020,effective_coverage=0.,surface=None,
        contour=dict(status='reviewed_reference_mask_missing'))
    if len(distance):
        result['surface']=dict(mean_mm=float(distance.mean()*1000),rmse_mm=float(np.sqrt(np.mean(distance**2))*1000),
            p95_mm=float(np.percentile(distance,95)*1000),max_mm=float(distance.max()*1000),distance_trimming=False)
        result['effective_coverage']=float((distance<.020).mean()*reliable_count/max(candidates,1))
    if reference_mask is not None:result['contour']=contour_errors(silhouette(v,faces,o.camera,o.depth.shape),reference_mask)
    return result


def summarize(rows):
    result={}
    for role in ('in_fit','held_out'):
        for kind in ('front','side'):
            subset=[r for r in rows if r['role']==role and (r['view']=='front')==(kind=='front')]
            if not subset:continue
            counts=np.array([r['n_surface_samples'] for r in subset]);total=int(counts.sum())
            result[role+'_'+kind]=dict(views=len(subset),surface_samples=total,
                mean_mm=float(sum(r['surface']['mean_mm']*n for r,n in zip(subset,counts) if n)/total) if total else None,
                rmse_mm=float(np.sqrt(sum(r['surface']['rmse_mm']**2*n for r,n in zip(subset,counts) if n)/total)) if total else None,
                mean_effective_coverage=float(np.mean([r['effective_coverage'] for r in subset])),
                contour_normal_mean_px=float(np.mean([r['contour']['normal_mean_px'] for r in subset if r['contour']['status']=='reviewed_mask']))
                    if any(r['contour']['status']=='reviewed_mask' for r in subset) else None)
    return result
