"""Explicit numerical adapter: authorized existing full-head output + A3 surface.

No model/weight imports. Frozen unseen prior, bounded regularized face correction.
All coordinates in this module are common-camera metres (right/down/forward).
"""
import numpy as np
from scipy.sparse import coo_matrix,diags,eye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from a3_geometry import project,transform,raster_depth
from check_a3 import vertex_normals


def subdivide(vertices,faces,levels=1):
    v,f = vertices.copy(),faces.copy()
    for _ in range(levels):
        edges = np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        unique,inv = np.unique(edges,axis=0,return_inverse=True)
        ab,bc,ca = (inv.reshape(3,-1)+len(v))
        v = np.vstack([v,(v[unique[:,0]]+v[unique[:,1]])/2])
        a,b,c = f.T
        f = np.vstack([np.stack([a,ab,ca],1),np.stack([ab,b,bc],1),
                       np.stack([ca,bc,c],1),np.stack([ab,bc,ca],1)])
    return v,f


def graph_matrices(n,faces):
    a = np.r_[faces[:,0],faces[:,1],faces[:,2]]
    b = np.r_[faces[:,1],faces[:,2],faces[:,0]]
    adj = coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    adj.data[:] = 1.
    degree = np.asarray(adj.sum(1)).ravel()
    L = eye(n,format='csr')-diags(1/np.maximum(degree,1))@adj
    return adj,L


def fragile_vertices(v,f):
    p = v[f]
    edge = np.linalg.norm(p[:,[1,2,0]]-p,axis=-1)
    area2 = np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    thin = (edge.min(1)<.0002)|(area2/np.maximum(edge.max(1)**2,1e-20)<.025)
    locked = np.zeros(len(v),bool)
    locked[np.unique(f[thin])] = True
    # One-ring buffer, so neighbouring corrections cannot tear a thin eyelid.
    adj,_ = graph_matrices(len(v),f)
    locked |= np.asarray(adj@locked.astype(float)).ravel()>0
    return locked


def topology(v,f):
    edges = np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    unique,count = np.unique(edges,axis=0,return_counts=True)
    adj,_ = graph_matrices(len(v),f)
    components = connected_components(adj,directed=False)[0]
    boundary = unique[count==1]
    loops = 0
    if len(boundary):
        ids,indices = np.unique(boundary,return_inverse=True)
        pairs = indices.reshape(-1,2)
        g = coo_matrix((np.ones(len(pairs)*2),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(ids),len(ids))).tocsr()
        loops = connected_components(g,directed=False)[0]
    return dict(vertices=len(v),triangles=len(f),components=int(components),
                boundary_edges=int((count==1).sum()),boundary_components=int(loops),
                nonmanifold_edges=int((count>2).sum()),watertight=bool(np.all(count==2)))


def geometry_guard(base,current,faces):
    old,new = base[faces],current[faces]
    on = np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
    nn = np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
    oa,na = np.linalg.norm(on,axis=1),np.linalg.norm(nn,axis=1)
    meaningful = oa>1e-12
    cos = np.sum(on*nn,axis=1)/np.maximum(oa*na,1e-30)
    ratio = na/np.maximum(oa,1e-30)
    old_edges = np.linalg.norm(old[:,[1,2,0]]-old,axis=-1)
    new_edges = np.linalg.norm(new[:,[1,2,0]]-new,axis=-1)
    er = new_edges/np.maximum(old_edges,1e-20)
    max_offset = float(np.max(np.linalg.norm(current-base,axis=1)))
    passed = (np.isfinite(current).all() and max_offset<=.0030000001
              and np.all(cos[meaningful]>.2) and np.all((ratio[meaningful]>.3)&(ratio[meaningful]<3.))
              and np.all((er[old_edges>1e-10]>.5)&(er[old_edges>1e-10]<2.)))
    return dict(passed=bool(passed),max_offset_mm=max_offset*1000,
                inverted_triangles=int((cos[meaningful]<=0).sum()),
                min_normal_cos=float(cos[meaningful].min()),
                min_area_ratio=float(ratio[meaningful].min()),max_area_ratio=float(ratio[meaningful].max()))


def correspondences(v,faces,target_v,target_n,tree,target_valid,K,shape,to_camera):
    distances,ix = tree.query(v)
    normals = vertex_normals(v,faces)
    n = target_n[ix]
    uv = np.rint(project(v,K)).astype(int)
    h,w = shape
    inside = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)&(v[:,2]>.1)
    good = np.zeros(len(v),bool)
    ii = np.flatnonzero(inside)
    good[ii] = target_valid[uv[ii,1],uv[ii,0]]
    good &= (distances<.008)&(np.sum(normals*n,axis=1)>.6)
    # Cull hidden eye/back surfaces using the full head's own z buffer.
    cam = transform(v,to_camera)
    cam_n = normals@to_camera[:3,:3].T
    good &= np.sum(cam_n*cam,axis=1)<-.1*np.linalg.norm(cam,axis=1)
    self_depth = raster_depth(v,faces,K,shape,2)
    xy = np.rint(project(v,K)/2).astype(int)
    in_buffer = (xy[:,0]>=0)&(xy[:,0]<self_depth.shape[1])&(xy[:,1]>=0)&(xy[:,1]<self_depth.shape[0])
    visible = np.zeros(len(v),bool)
    ii = np.flatnonzero(in_buffer)
    zz = self_depth[xy[ii,1],xy[ii,0]]
    visible[ii] = np.isfinite(zz)&(v[ii,2]<=zz+.002)
    good &= visible
    # Normal projection avoids snapping template vertices onto the A3 pixel grid.
    q = v-n*np.sum((v-target_v[ix])*n,axis=1)[:,None]
    return q,good,distances


def fit_surface(base,faces,target_v,target_faces,target_valid,K,shape,to_camera,
                iterations=8,fixed_mask=None,fixed_delta=None,protected_mask=None):
    n = len(base)
    adj,L = graph_matrices(n,faces)
    reg = (L.T@L).tocsr()*4.+eye(n,format='csr')*.03
    target_n = vertex_normals(target_v,target_faces)
    tree = cKDTree(target_v)
    _,direct,_ = correspondences(base,faces,target_v,target_n,tree,target_valid,K,shape,to_camera)
    if direct.sum()<100:
        raise ValueError('insufficient registered frontal overlap; no similarity-scale rescue')
    active = direct.copy()
    for _ in range(4):
        active |= np.asarray(adj@active.astype(float)).ravel()>0
    normals = vertex_normals(base,faces)
    camera = transform(base,to_camera)
    active &= np.sum((normals@to_camera[:3,:3].T)*camera,axis=1)<0
    protected = fragile_vertices(base,faces) if protected_mask is None else protected_mask
    active &= ~protected
    locked = ~active
    locked_delta = np.zeros_like(base)
    if fixed_mask is not None:
        if np.any(np.linalg.norm(fixed_delta[fixed_mask&protected],axis=1)>1e-14):
            raise ValueError('shared correction conflicts with protected original geometry')
        locked |= fixed_mask
        locked_delta[fixed_mask] = fixed_delta[fixed_mask]
        active |= fixed_mask
    free = np.flatnonzero(~locked)
    fixed = np.flatnonzero(locked)
    delta = locked_delta.copy()
    history = []
    for iteration in range(iterations):
        current = base+delta
        q,good,distances = correspondences(current,faces,target_v,target_n,tree,target_valid,K,shape,to_camera)
        good &= active
        weights = good.astype(float)*12.
        target_delta = delta+(q-current)
        H = reg+diags(weights)
        rhs = weights[:,None]*target_delta
        rhs_free = rhs[free]-H[free][:,fixed]@locked_delta[fixed]
        proposal = locked_delta.copy()
        proposal[free] = spsolve(H[free][:,free].tocsc(),rhs_free)
        norms = np.linalg.norm(proposal,axis=1)
        proposal[free] *= np.minimum(1.,.003/np.maximum(norms[free],1e-12))[:,None]
        step = proposal-delta
        length = np.linalg.norm(step[free],axis=1)
        step[free] *= np.minimum(1.,.00075/np.maximum(length,1e-12))[:,None]
        accepted = False
        for power in range(12):
            trial = delta+step*(.5**power)
            trial[fixed] = locked_delta[fixed]
            guard = geometry_guard(base,base+trial,faces)
            if guard['passed']:
                delta = trial
                accepted = True
                break
        history.append(dict(iteration=iteration,accepted=accepted,
            constraints=int(good.sum()),median_nearest_surface_mm=float(np.median(distances[good])*1000),guard=guard))
        if not accepted:
            break
    guard = geometry_guard(base,base+delta,faces)
    if not guard['passed']:
        raise ValueError('final full-head geometry guard failed')
    assert np.array_equal(delta[~active],np.zeros_like(delta[~active]))
    if fixed_mask is not None:
        np.testing.assert_array_equal(delta[fixed_mask],fixed_delta[fixed_mask])
    return base+delta,dict(active=active,direct=direct,delta=delta,history=history,
                          guard=guard,frozen_count=int((~active).sum()),protected=protected)


def write_grouped_obj(path,v,f,category):
    with open(path,'w',encoding='ascii') as out:
        out.write('# Full-head prior + bounded A3 correction; metres X right Y down Z forward\n')
        out.write('# Rear/scalp/ears contain inferred prior, not full-head RGB-D scan\n')
        np.savetxt(out,v,fmt='v %.10f %.10f %.10f')
        face_tag = np.max(category[f],axis=1)
        for tag,name in [(0,'prior_only'),(1,'regularized_transition'),(2,'data_constrained')]:
            out.write('g '+name+'\n')
            np.savetxt(out,f[face_tag==tag]+1,fmt='f %d %d %d')
