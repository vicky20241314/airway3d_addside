"""Small synthetic safety checks and source audit, separate from capture."""
import sys,os,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
from core.student.offset_refine import refine_pointcloud
from core.conservative_refine import ShapeGuard,RefineOptions

def main():
    p=argparse.ArgumentParser();p.add_argument('--new-source',type=Path,required=True);a=p.parse_args()
    xy=np.array([(x,y) for y in range(5) for x in range(5)],float)*.005
    base=np.column_stack((xy,np.full(25,-1.)))
    faces=[]
    for y in range(4):
        for x in range(4):
            k=y*5+x;faces.extend([[k,k+1,k+6],[k,k+6,k+5]])
    faces=np.array(faces);guard=ShapeGuard(base,faces,RefineOptions())
    assert not len(guard.inspect(base)[0])
    spike=base.copy();spike[12,2]+=.008
    assert len(guard.inspect(spike)[0]),'Spike was not detected'
    points=base.copy();points[:,2]+=.001
    result,info=refine_pointcloud(base,faces,points,shape_guard=guard,max_offset_m=.002,
        max_step_m=.00025,stages=(4.,),iterations_per_stage=8,early_stop=None)
    assert not len(guard.inspect(result)[0])
    assert np.linalg.norm(result-base,axis=1).max()>.0005
    assert np.linalg.norm(result-base,axis=1).max()<=.002+1e-12
    assert max(h['step_m'] for h in info['history'])<=.00025+1e-12
    assert np.array_equal(base[:,2],np.full(25,-1.))
    empty,empty_info=refine_pointcloud(base,faces,np.empty((0,3)),shape_guard=guard)
    assert np.array_equal(base,empty) and empty_info['status']=='no_observations'
    fixed=np.zeros(25,bool);fixed[6:19]=True
    partial,_=refine_pointcloud(base,faces,points,active=fixed,shape_guard=guard,
        max_offset_m=.002,max_step_m=.00025,stages=(4.,),iterations_per_stage=3)
    assert np.array_equal(partial[~fixed],base[~fixed])
    for kw in ({'max_step_m':-1},{'consistency_weight':float('nan')}):
        try:refine_pointcloud(base,faces,points,**kw)
        except ValueError:pass
        else:raise AssertionError('Invalid parameter accepted')
    files=['joint_lm.py','fit_options.py','integrated_fitting.py','lm_jacobians.py','landmark_policy.py','initial_surface.py']
    audit={}
    for name in files:
        src=a.new_source/'core'/name;dst=ROOT/'core'/name
        hs=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        audit[name]=dict(source_sha256=hs(src),integrated_sha256=hs(dst))
        assert audit[name]['source_sha256']==audit[name]['integrated_sha256'],name
    result=dict(synthetic_guard_rejects_spike=True,smooth_translation_is_not_frozen=True,
        bounds_and_step_checked=True,empty_observations_unchanged=True,inactive_vertices_fixed=True,
        invalid_parameters_rejected=True,stage1_stage2_source_audit=audit)
    out=ROOT/'validation/cross_v2';out.mkdir(parents=True,exist_ok=True)
    (out/'contracts.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
