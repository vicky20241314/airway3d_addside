"""Known synthetic surface and guard/parameter regression for v3."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import trimesh
from scipy import sparse
from scipy.spatial import cKDTree
from core.cross_refine import PROFILES
from core.conservative_refine import ShapeGuard,RefineOptions
from core.student.offset_refine import refine_pointcloud
from core.student.early_stop import StudentEarlyStop

def main():
    n=13
    xy=np.array([(x,y) for y in np.linspace(-.018,.018,n) for x in np.linspace(-.018,.018,n)])
    # Preserve a known curved baseline feature; only D is regularized.
    base=np.column_stack((xy,-.5+.003*np.exp(-np.sum(xy**2,axis=1)/(.008**2))))
    f=[]
    for y in range(n-1):
        for x in range(n-1):
            i=y*n+x;f.extend([[i,i+1,i+n+1],[i,i+n+1,i+n]])
    mesh=trimesh.Trimesh(base,np.array(f),process=False)
    sv,sf=trimesh.remesh.subdivide(base,mesh.faces)
    edges=mesh.edges_unique;_,ix=cKDTree(base[edges].mean(1)).query(sv[len(base):]);end=edges[ix]
    count=len(sv)-len(base)
    C=sparse.coo_matrix((np.tile([1.,-.5,-.5],count),
        (np.repeat(np.arange(count),3),np.column_stack((np.arange(len(base),len(sv)),end)).ravel())),shape=(count,len(sv))).tocsr()
    truth=sv.copy();truth[:,2]+=.0007*np.exp(-np.sum(sv[:,:2]**2,axis=1)/(.020**2))
    rng=np.random.default_rng(20261025)
    fi=rng.integers(len(sf),size=4000);bary=rng.dirichlet([1,1,1],size=len(fi))
    points=np.einsum('ij,ijk->ik',bary,truth[sf[fi]])
    points[:,2]+=rng.normal(0,.0004,len(points))
    guard=ShapeGuard(sv,sf,RefineOptions())
    metrics={}
    for name,cfg in PROFILES.items():
        result,info=refine_pointcloud(sv,sf,points,shape_guard=guard,max_offset_m=.002,max_step_m=.00025,
            stages=cfg['stages'],iterations_per_stage=8,edge_weight=.05,anchor_weight=.01,
            consistency_operator=C,consistency_weight=cfg['consistency'],edge_length_scale_m=.002,
            tangent_weight=cfg['tangent'],continue_after_small_step=cfg['continue_after_small_step'],
            early_stop=StudentEarlyStop(step_tol_m=cfg['step_tol_m']))
        error=np.linalg.norm(result-truth,axis=1)*1000
        assert not len(guard.inspect(result)[0])
        assert np.max(np.linalg.norm(result-sv,axis=1))<=.002+1e-12
        assert max(h['step_m'] for h in info['history'])<=.00025+1e-12
        assert np.linalg.norm(result-sv,axis=1).max()>.0001
        metrics[name]=dict(truth_vertex_rmse_mm=float(np.sqrt(np.mean(error**2))),iterations=len(info['history']),
                          max_move_mm=float(np.linalg.norm(result-sv,axis=1).max()*1000))
    bad=sv.copy();bad[len(base)+10,2]+=.008
    assert len(guard.inspect(bad)[0])
    try:refine_pointcloud(sv,sf,points,tangent_weight=-1)
    except ValueError:pass
    else:raise AssertionError('Invalid tangent weight accepted')
    report=dict(seed=20261025,synthetic_noise_std_mm=.4,n_points=len(points),metrics=metrics,
        all_bounds_passed=True,nonzero_motion=True,spike_rejected=True,
        note='Known synthetic vertex truth only; no claim about real anatomical correspondence.')
    out=ROOT/'validation/cross_v3';out.mkdir(parents=True,exist_ok=True)
    (out/'synthetic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
