"""Synthetic mathematical contracts, not real side reconstruction evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from scipy import sparse
from scipy.spatial.transform import Rotation
from core.camera import Camera
from core.multiview_geometry import RigidCamera
from core.multiview_data import Observation
from core.multiview_refine import camera_equations,JointSemantic,prepare_state,normalize_view_equations
from core.student.offset_refine import refine_pointcloud
from core.conservative_refine import RefineOptions,ShapeGuard


def plane_fixture():
    n=7;x,y=np.meshgrid(np.linspace(-.06,.06,n),np.linspace(-.06,.06,n))
    base=np.column_stack((x.ravel(),y.ravel(),np.full(n*n,-.6)))
    faces=[]
    for row in range(n-1):
        for col in range(n-1):
            a=row*n+col;faces.extend([[a,a+1,a+n],[a+1,a+n+1,a+n]])
    return base,np.array(faces)


def plane_observation(angle,view_id):
    R=Rotation.from_euler('y',angle,degrees=True).as_matrix();center=np.array([0,0,-.6])
    T=RigidCamera(R,center-R@center)
    camera=Camera(400,400,64,64);yy,xx=np.mgrid[:128,:128]
    rays=np.column_stack(((xx.ravel()-64)/400,-(yy.ravel()-64)/400,-np.ones(xx.size)))
    origin=T.to_common(np.zeros((1,3)))[0];direction=rays@T.R
    length=(-.5995-origin[2])/direction[:,2]
    common=origin+length[:,None]*direction
    valid=(np.abs(common[:,:2])<.06).all(1)&(length>0)
    depth=np.where(valid,length*1000,0).reshape(128,128)
    labels=valid.reshape(128,128).astype(np.uint8)
    obs=Observation(view_id,'closed','front' if angle==0 else ('left' if angle>0 else 'right'),
        np.zeros((128,128,3),np.uint8),depth,camera,np.full((68,2),np.nan),np.full((68,3),np.nan),
        np.zeros(68,bool),dict(weight=1.,source='synthetic_plane_contract'),labels,np.empty((0,3)))
    return obs,T


class JointRefineTests(unittest.TestCase):
    def test_semantic_row_count_quality_and_single_view_scale(self):
        C=sparse.csr_matrix([[1.,2.,3.]])
        t=np.array([.25]);many=sparse.vstack([C]*100,format='csr')
        rows=normalize_view_equations([(C,t),(many,np.tile(t,100))],[1.,.25])
        residual=np.array([.2,.3,.4])
        energy=[np.linalg.norm(A@residual-b)**2 for A,b in rows]
        self.assertAlmostEqual(energy[0]/energy[1],4.)
        reverse=normalize_view_equations([(many,np.tile(t,100)),(C,t)],[.25,1.])
        for (A,b),(D,e) in zip(rows,reversed(reverse)):
            np.testing.assert_allclose(A.toarray(),D.toarray());np.testing.assert_allclose(b,e)
        single=normalize_view_equations([(C,t)],[.25])
        np.testing.assert_array_equal(single[0][0].toarray(),C.toarray())
        np.testing.assert_array_equal(single[0][1],t)

    def test_multiscale_registration_known_transform_and_failure(self):
        from core.multiview_registration import register_to_front
        from core.multiview_geometry import project
        front,A=plane_observation(0,'synthetic_front');side,B=plane_observation(30,'synthetic_side')
        ids=np.arange(68);common=np.column_stack((.025*np.sin(ids*.8),.015*np.cos(ids*.7),np.full(68,-.5995)))
        common[19:25,1]=.035;common[8,1]=-.055;common[33,1]=-.005
        common[[50,51,52],1]=-.025;common[[56,57,58],1]=-.035
        for o,T in ((front,A),(side,B)):
            o.xyz=T.to_view(common);o.uv=project(o.xyz,o.camera)[0];o.valid=np.ones(68,bool)
        estimated,info,_,_=register_to_front(front,side)
        np.testing.assert_allclose(estimated.to_view(side.xyz),front.xyz,atol=.002,rtol=0)
        self.assertGreater(info['source_overlap'],.25);self.assertLess(info['rmse_m'],.004)
        from core.multiview_registration import RegistrationConsistencyError,save_registration_diagnostics
        with patch('core.multiview_registration.check_neck_consistency',return_value=dict(status='inconsistent',median_m=.025)):
            with self.assertRaises(RegistrationConsistencyError) as failure:
                register_to_front(front,side)
        exc=failure.exception
        self.assertIn('neck_check',exc.diagnostics)
        np.testing.assert_allclose(exc.transform.to_view(side.xyz),front.xyz,atol=.002,rtol=0)
        with tempfile.TemporaryDirectory() as folder:
            save_registration_diagnostics(folder,side.view_id,exc.transform,exc.diagnostics,exc.source,exc.target,failed=True)
            self.assertTrue((Path(folder)/f'registration_{side.view_id}_failed.json').exists())
            self.assertTrue((Path(folder)/f'registration_{side.view_id}_failed_front.ply').exists())
        side.valid[27:48]=False
        with self.assertRaises(ValueError):register_to_front(front,side)

    def test_translated_camera_equations_and_jacobian(self):
        rng=np.random.default_rng(5);n=7
        C=sparse.random(12,3*n,density=.25,random_state=5,format='csr');target=rng.normal(size=12)
        T=RigidCamera(Rotation.from_euler('xyz',[8,35,4],degrees=True).as_matrix(),np.array([.1,-.02,-.6]))
        v=rng.normal(size=(n,3));delta=rng.normal(size=(n,3))*.001
        A,t=camera_equations(C,target,T,n)
        np.testing.assert_allclose(A@v.ravel()-t,C@T.to_view(v).ravel()-target,atol=1e-12)
        np.testing.assert_allclose(A@delta.ravel(),C@(delta@T.R.T).ravel(),atol=1e-12)
        joint=JointSemantic([(A,t)],n);J,rhs=joint.linear_system(v,np.arange(n))
        np.testing.assert_allclose(J@delta.ravel()-rhs,C@T.to_view(v+delta).ravel()-target,atol=1e-12)

    def solve_plane(self,reverse=False):
        base,faces=plane_fixture();original=base.copy()
        pairs=[plane_observation(angle,'test_'+str(i)) for i,angle in enumerate((0,30,-30))]
        if reverse:pairs.reverse()
        obs=[p[0] for p in pairs];transforms={p[0].view_id:p[1] for p in pairs}
        J=sparse.coo_matrix((np.ones(51),(np.arange(51),np.full(51,24))),shape=(51,len(base))).tocsr()
        p,w,g,q,active,semantic,provider,diag=prepare_state(base,faces,J,obs,transforms,max_points=5000)
        guard=ShapeGuard(base,faces,RefineOptions())
        result,info=refine_pointcloud(base,faces,p,active=active,confidence=w,
            point_groups=g,group_quality=q,correspondence_provider=provider,semantic_constraints=semantic,
            stages=(1.,),iterations_per_stage=4,early_stop=None,max_points=len(p),
            max_offset_m=.002,max_step_m=.00025,shape_guard=guard,
            edge_weight=.05,anchor_weight=.01,fairing_strength=1.,fairing_max_change_m=.0001)
        np.testing.assert_array_equal(base,original)
        self.assertEqual(len(guard.inspect(result)[0]),0)
        self.assertLessEqual(np.linalg.norm(result-base,axis=1).max(),.002+1e-12)
        self.assertTrue(all(row['step_m']<=.00025+1e-12 for row in info['history']))
        self.assertTrue(all(len(row)==3 for row in provider.history))
        self.assertTrue(all(d['reliable_points']>100 for d in diag))
        return result,info

    def test_joint_D_order_independence_and_guards(self):
        first,info=self.solve_plane();second,_=self.solve_plane(True)
        np.testing.assert_allclose(first,second,atol=2e-7,rtol=0)
        self.assertGreater(info['max_move_m'],1e-5)
        base,faces=plane_fixture();bad=base.copy();bad[24,2]+=.01
        self.assertGreater(len(ShapeGuard(base,faces,RefineOptions()).inspect(bad)[0]),0)

    def test_view_point_count_does_not_dominate(self):
        base=np.array([[0,0,-.6],[.1,0,-.6],[0,.1,-.6]])
        face=np.array([[0,1,2]])
        def solve(n):
            target=np.tile(base.mean(0),(n+1,1));target[:n,2]+=.001;target[n:,2]-=.001
            groups=np.r_[np.zeros(n,int),1]
            def provider(current,faces,points,indices):
                bary=np.full((len(points),3),1/3);q=np.tile(current.mean(0),(len(points),1))
                return q,np.linalg.norm(q-points,axis=1),np.zeros(len(points),int),bary,np.ones(len(points),bool)
            return refine_pointcloud(base,face,target,point_groups=groups,group_quality={0:1.,1:1.},
                correspondence_provider=provider,stages=(0.,),iterations_per_stage=2,max_points=len(target),early_stop=None,
                max_offset_m=.002,max_step_m=.00025)[0]
        np.testing.assert_allclose(solve(1),solve(100),rtol=0,atol=1e-10)

    def test_pipeline_calls_one_solver_per_state_and_applies_D_once(self):
        from core.pipeline import finish_multiview_refinement
        base,faces=plane_fixture();mesh=trimesh.Trimesh(base,faces,process=False)
        observations=[plane_observation(a,s+'_'+str(i))[0] for s in ('closed','open') for i,a in enumerate((0,30,-30))]
        for o in observations:o.state=o.view_id.split('_')[0]
        def refine(mesh,views,transforms):
            result=mesh.copy();reference=mesh.vertices.copy();result.vertices=reference+np.array([0,0,.0002])
            return result,reference,dict(shared_displacement=True)
        with tempfile.TemporaryDirectory() as d,patch('core.cross_refine.refine_multiview_mesh',side_effect=refine) as solve:
            outputs,_=finish_multiview_refinement((mesh,mesh),observations,{},d)
            self.assertEqual(solve.call_count,2)
            self.assertEqual([len(call.args[1]) for call in solve.call_args_list],[3,3])
            for state,result in zip(('closed','open'),outputs):
                saved=trimesh.load(Path(d)/state/f'mesh_{state}.obj',process=False,force='mesh')
                np.testing.assert_allclose(saved.vertices,result.vertices,rtol=0,atol=1e-14)
                with np.load(Path(d)/state/'vertex_offsets.npz') as a:
                    np.testing.assert_allclose(result.vertices,a['reference_vertices']+a['offsets'],atol=1e-12)
                    np.testing.assert_allclose(a['offsets'],np.tile([0,0,.0002],(len(base),1)),atol=1e-12)


if __name__=='__main__':unittest.main()
