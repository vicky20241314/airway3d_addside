"""Numerical contracts for candidate algorithms; no real accuracy claims."""
import copy
from dataclasses import replace
import tempfile
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from core.fit_options import FitOptions
from core.multiview_geometry import RigidCamera
from core.multiview_registration import rigid_svd,check_expression,register_to_front,RegistrationError
from core.registration_fallback import stable_surface,rgbd_feature_seed,geometric_feature_seed
from core.multiview_constraints import soft_side_landmarks,SIDE_STATIC_IDS,se3_log,relative_transforms
from test_multiview import fixture
import test_multiview as contracts


class FallbackTests(unittest.TestCase):
    def test_failed_overlap_diagnostic_cannot_abort_other_views(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from core.multiview_registration import RegistrationConsistencyError,select_registered_views
        observations=[fixture('front','closed'),fixture('front','open'),fixture('left','closed')]
        points=np.array([[0.,0.,-.6],[.01,0.,-.6],[0.,.01,-.6]])
        error=RegistrationConsistencyError('no overlap',RigidCamera(np.eye(3),np.zeros(3)),
            dict(rmse_m=float('inf'),source_overlap=0.,target_overlap=0.),points,points)
        with tempfile.TemporaryDirectory() as d,patch('core.multiview_registration.register_to_front',side_effect=error):
            kept,_,_,excluded=select_registered_views(observations,d)
            r=json.loads((Path(d)/'registration_closed_left_failed.json').read_text(encoding='utf-8'))
            self.assertIsNone(r['rmse_m']);self.assertEqual(len(kept),2);self.assertEqual(len(excluded),1)

    def surface(self):
        o=fixture('left');o.valid[:]=False;o.uv[:]=np.nan;o.parsing[:]=1
        o.parsing[8:15,10:54]=2;o.parsing[35:47,25:39]=10
        o.rgb[:]=np.random.default_rng(2).integers(0,256,o.rgb.shape,dtype=np.uint8)
        return o

    def test_roi_works_without_landmarks_but_not_without_anchors(self):
        o=self.surface();p,n,m,d=stable_surface(o)
        self.assertGreater(len(p),100);self.assertEqual(d['roi_method'],'semantic_depth_normal_roi')
        self.assertTrue((o.parsing[m]<11).all());self.assertTrue((n[:,2]>0).all())
        o.parsing[:]=1
        with self.assertRaises(ValueError):stable_surface(o)

    def test_rgbd_seed_and_textureless_failure(self):
        import cv2
        o=self.surface();front=copy.deepcopy(o);front.view='front'
        for view in (o,front):
            view.rgb=cv2.resize(view.rgb,(256,256),interpolation=cv2.INTER_CUBIC)
            view.depth=cv2.resize(view.depth,(256,256),interpolation=cv2.INTER_NEAREST)
            view.parsing=cv2.resize(view.parsing,(256,256),interpolation=cv2.INTER_NEAREST)
            from core.camera import Camera
            view.camera=Camera(320,320,128,128)
        _,_,m,_=stable_surface(o)
        T,d=rgbd_feature_seed(front,o,m,m,rigid_svd)
        np.testing.assert_allclose(T.R,np.eye(3),atol=1e-10);self.assertGreaterEqual(d['seed_inliers'],6)
        T,d,_,_=register_to_front(front,o,options=FitOptions())
        self.assertEqual(d['seed_method'],'mutual_sift_rgbd_ransac')
        self.assertGreater(d['source_overlap'],.99);self.assertLess(d['rmse_m'],1e-9)
        o.rgb[:]=0
        with self.assertRaises(ValueError):rgbd_feature_seed(front,o,m,m,rigid_svd)

    def test_missing_lips_are_unverified_and_inconsistent_lower_face_disabled(self):
        f=fixture('front','open');s=fixture('left','open');s.valid[48:]=False
        d=check_expression(f,s,RigidCamera(np.eye(3),np.zeros(3)),'upper_face')
        self.assertEqual(d['expression_status'],'unverified');self.assertFalse(d['lower_face_used'])
        s.valid[:]=True;s.xyz[[50,51,52],1]=.03
        d=check_expression(f,s,RigidCamera(np.eye(3),np.zeros(3)),'upper_face')
        self.assertEqual(d['expression_status'],'inconsistent');self.assertFalse(d['lower_face_used'])

    def test_geometric_global_seed_direction_and_degenerate_failure(self):
        rng=np.random.default_rng(17);p=rng.uniform(-.04,.04,(900,3))
        # A curved asymmetric surface supplies distinctive pair geometry.
        p[:,2]=.25*p[:,0]+3*p[:,0]**2+5*p[:,1]**2+2*p[:,0]*p[:,1]
        n=np.column_stack((-(.25+6*p[:,0]+2*p[:,1]),-(10*p[:,1]+2*p[:,0]),np.ones(len(p))))
        n/=np.linalg.norm(n,axis=1,keepdims=True)
        known=RigidCamera(Rotation.from_euler('xyz',[7,30,-9],degrees=True).as_matrix(),np.array([.02,.01,-.5]))
        T,d=geometric_feature_seed(p,n,known.to_view(p),n@known.R.T,rigid_svd)
        np.testing.assert_allclose(T.to_view(p),known.to_view(p),atol=.001)
        self.assertEqual(d['seed_method'],'point_normal_histogram_ransac')
        with self.assertRaises(ValueError):geometric_feature_seed(np.zeros((100,3)),np.tile([0.,0.,1.],(100,1)),p,n,rigid_svd)

    def test_soft_landmarks_above_6mm_are_not_deleted(self):
        from unittest.mock import patch
        o=fixture('left');model=o.xyz.copy();model[SIDE_STATIC_IDS,0]=.008
        with patch('core.multiview_constraints.ray_visible',return_value=np.ones(len(SIDE_STATIC_IDS),bool)):
            ids,w,_,d=soft_side_landmarks(o,np.empty((0,3)),np.empty((0,3),int),model)
        np.testing.assert_array_equal(ids,SIDE_STATIC_IDS);np.testing.assert_allclose(w,.5)
        self.assertEqual(d['legacy_6mm_rejected'],len(ids));self.assertFalse(np.isin(ids,np.r_[:17,48:68]).any())
        model[SIDE_STATIC_IDS,0]=.021
        with patch('core.multiview_constraints.ray_visible',return_value=np.ones(len(SIDE_STATIC_IDS),bool)):
            ids,w,_,d=soft_side_landmarks(o,np.empty((0,3)),np.empty((0,3),int),model)
        self.assertEqual(len(ids),0)

    def test_profile_is_extracted_before_head_mask_discards_background(self):
        import cv2
        from core.side_silhouette import observed_profile
        from core.side_constraints import constraint_observation
        o=fixture('left');o.parsing[:]=0
        cv2.ellipse(o.parsing,(32,32),(24,29),0,0,360,1,-1)
        o.parsing[12:17,22:45]=2;o.parsing[24:35,44:52]=10;o.depth[o.parsing==0]=0
        profile=observed_profile(o)
        self.assertEqual(profile['status'],'accepted');self.assertGreater(len(profile['uv']),12)
        masked=constraint_observation(o)
        self.assertNotEqual(observed_profile(masked)['status'],'accepted')
        # Regression contract: the fitting constructor receives the RAW profile.
        from core.multiview_fitting import MultiProblem
        from core.integrated_fitting import make_flame
        options=FitOptions(shape_components=3,expression_components=2,side_contour_weight=.3)
        observations=[fixture('front','closed'),fixture('front','open'),masked]
        flame=make_flame(options,'cpu',batch=3)
        initial={s:dict(shape=np.zeros(3),expression=np.zeros(2),pose=np.zeros(6)) for s in ('closed','open')}
        transforms=[RigidCamera(np.eye(3),np.array([0.,0.,-.6])) for _ in observations]
        p=MultiProblem(flame,observations,initial,transforms,options,raw_profiles={o.view_id:profile})
        self.assertIs(p.profiles[2],profile);self.assertEqual(p.profiles[2]['status'],'accepted')


class PoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.integrated_fitting import make_flame
        cls.options=FitOptions(shape_components=3,expression_components=2,basis_levels=((3,2),),max_points=50)
        cls.flame=make_flame(cls.options,'cpu',batch=4,dtype=torch.float64)

    def problem(self,strategy):
        p=contracts.FlameTests.problem(self);p.options=replace(p.options,pose_strategy=strategy)
        p.begin_shape_fit(p.theta)
        v,lm=p.forward(p.theta);p.embedding=p._embedding(p.theta,v);p.context=[]
        for b in range(4):
            tri=p.faces[100:103];bc=p.theta.new_full((3,3),1/3)
            p.context.append(dict(tri=tri,bary=bc,points=(v[b][tri]*bc[:,:,None]).sum(1)+.0002,
                normals=p.theta.new_tensor([[0,0,1.]]*3),weights=p.theta.new_full((3,),.25),ids=np.array([33]),
                valid=np.zeros(68,bool),jaw_irls=p.theta.new_empty(0)))
            p.jaw[b]['ids']=np.empty(0,int)
        return p

    def test_all_modes_residual_jacobian(self):
        for strategy in ('joint','prior','fixed'):
            p=self.problem(strategy);r,j=p.residual(p.theta,2);active=p.layout.active(2)
            for col,i in enumerate(active):
                a=p.theta.clone();b=a.clone();a[i]+=1e-5;b[i]-=1e-5
                fd=(p.residual(a,2,False)[0]-p.residual(b,2,False)[0])/2e-5
                torch.testing.assert_close(j[:,col],fd,rtol=3e-3,atol=3e-5)
            self.assertEqual(j.shape[1],p.layout.size-(12 if strategy=='fixed' else 0))

    def test_prior_direction_invariance_and_lie_log(self):
        from scipy.linalg import logm
        p=self.problem('prior');np.testing.assert_allclose(p.pose_prior(p.theta),0,atol=1e-10)
        T=RigidCamera(Rotation.from_euler('xyz',[5,8,-3],degrees=True).as_matrix(),np.array([.01,-.02,.03]))
        M=np.eye(4);M[:3,:3]=T.R;M[:3,3]=T.t;L=logm(M)
        np.testing.assert_allclose(se3_log(T),np.r_[L[2,1],L[0,2],L[1,0],L[:3,3]],atol=1e-12)
        rot=p.rotations.numpy();trans=np.stack([p.theta[s].numpy() for s in p.layout.trans])
        original=relative_transforms(rot,trans,p.front_indices,p.observations)
        changed=[RigidCamera(r,t).compose(T) for r,t in zip(rot,trans)]
        updated=relative_transforms(np.stack([x.R for x in changed]),np.stack([x.t for x in changed]),p.front_indices,p.observations)
        for i in original:
            np.testing.assert_allclose(original[i].R,updated[i].R,atol=1e-12)
            np.testing.assert_allclose(original[i].t,updated[i].t,atol=1e-12)

    def test_fixed_relative_follows_global_front(self):
        p=self.problem('fixed');p.theta[p.layout.rot[0]][1]+=.05;p.theta[p.layout.trans[0]][0]+=.01
        *_,r,t=p.parameters(p.theta)
        rel=relative_transforms(r.numpy(),t.numpy(),p.front_indices,p.observations)
        for i in rel:
            np.testing.assert_allclose(rel[i].R,p.initial_relative[i].R,atol=1e-12)
            np.testing.assert_allclose(rel[i].t,p.initial_relative[i].t,atol=1e-12)

    def test_fixed_progressive_active_prefix_mapping(self):
        from unittest.mock import patch
        from core.joint_lm import solve_progressive
        p=self.problem('fixed');p.options=replace(p.options,basis_levels=((1,1),(3,2)),stage2_outer=1,inner_steps=1)
        # Keep these numerical correspondences/embedding frozen while testing
        # the real progressive solver's reduced active-column mapping.
        with patch.object(p,'refresh',return_value=None):theta,report=solve_progressive(p,p.theta)
        self.assertTrue(torch.isfinite(theta).all())
        counts=[v['active_parameter_count'] for v in report['passes']]
        self.assertLess(counts[0],counts[1]);self.assertEqual(counts[1],len(p.layout.active(2)))

    def test_profile_residual_jacobian_with_frozen_edges(self):
        p=self.problem('joint');p.options=replace(p.options,side_contour_weight=.3)
        edges=p.faces[100:104,:2].clone()
        from core.joint_lm import project
        v,_=p.forward(p.theta);target=project(v[2][edges].mean(1),p.observations[2].camera).numpy()+.3
        p.context[2]['profile']=dict(edges=edges.cpu().numpy(),target=target,normal=np.tile([1.,0.],(len(edges),1)),irls=p.theta.new_ones(len(edges)))
        _,j=p.residual(p.theta,2)
        for i in range(p.layout.size):
            a=p.theta.clone();b=a.clone();a[i]+=1e-5;b[i]-=1e-5
            fd=(p.residual(a,2,False)[0]-p.residual(b,2,False)[0])/2e-5
            torch.testing.assert_close(j[:,i],fd,rtol=3e-3,atol=3e-5)

    def test_shared_rest_correction_follows_pose_and_save_roundtrip(self):
        from core.canonical_shape_refine import posed_vertices
        p=self.problem('joint');offset=np.zeros_like(p.flame.v_template.numpy());offset[:,0]=.0005
        p.theta[p.layout.pose[1]][1]=.4
        base=posed_vertices(p,p.theta,np.zeros_like(offset));moved=posed_vertices(p,p.theta,offset)
        # Translation mode moves through each camera rotation, not in a common
        # camera axis. Rest joints are also regressed from the translated mesh.
        *_,r,t=p.parameters(p.theta)
        for b in range(4):np.testing.assert_allclose(moved[b]-base[b],np.broadcast_to(r[b].numpy()@offset[0],offset.shape),atol=1e-10)
        with tempfile.TemporaryDirectory() as d:
            path=d+'/shape.npz';np.savez(path,offset=offset)
            with np.load(path) as a:np.testing.assert_allclose(posed_vertices(p,p.theta,a['offset']),moved,atol=1e-12)

    def test_joint_free_correction_keeps_posed_protected_vertices_fixed(self):
        from core.canonical_shape_refine import posed_vertices,low_frequency_basis
        p=self.problem('joint');p.theta[p.layout.pose[1]][1]=.4
        v=p.flame.v_template.numpy();active=v[:,1]>np.median(v[:,1])
        B,_=low_frequency_basis(v,p.faces.numpy(),active,2,p.flame.J_regressor.numpy())
        offset=B[:,:,0]*.0004
        base=posed_vertices(p,p.theta,np.zeros_like(offset));moved=posed_vertices(p,p.theta,offset)
        np.testing.assert_allclose((moved-base)[:,~active],0,atol=1e-10)
        self.assertGreater(np.linalg.norm(moved-base,axis=-1).max(),1e-5)


class ShapeBasisTests(unittest.TestCase):
    def test_intentional_front_only_is_not_reported_as_registration_failure(self):
        from unittest.mock import patch
        from core.pipeline import run_multiview
        from pathlib import Path
        import json
        observations=[fixture('front','closed'),fixture('front','open'),fixture('left','closed')]
        with tempfile.TemporaryDirectory() as d,patch('core.aruco.detect_aruco',return_value=(None,None,np.array([0.,0.,-.6]))), \
                patch('core.multiview_fitting.fit_multiview',return_value=({'closed':'c','open':'o'},{},dict(used_view_ids=['closed_front','open_front'],degraded_to_front_only=True))), \
                patch('core.pipeline.finish_meshes',return_value=({},(),(),[])):
            result=run_multiview(observations,output_root=d,front_only=True,refine=False)
            r=json.loads((Path(result['folder'])/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(r['captured_view_ids']),3);self.assertEqual(len(r['requested_view_ids']),2)
        self.assertFalse(r['degraded_to_front_only']);side=r['view_participation'][2]
        self.assertIsNone(side['registration_accepted']);self.assertEqual(side['reason_code'],'front_only_requested')

    def test_low_modes_remove_rigid_motion_and_protect_inactive(self):
        import trimesh
        from core.canonical_shape_refine import low_frequency_basis
        mesh=trimesh.creation.icosphere(subdivisions=2);v=mesh.vertices;active=v[:,1]>-.5
        B,L=low_frequency_basis(v,mesh.faces,active,4)
        np.testing.assert_allclose(B[~active],0,atol=0)
        np.testing.assert_allclose(B.sum(0),0,atol=1e-11)
        center=v-v[active].mean(0)
        for axis in range(3):np.testing.assert_allclose(np.einsum('vc,vck->k',np.cross(np.eye(3)[axis],center),B),0,atol=1e-11)
        self.assertLessEqual(B.shape[-1],12)

    def test_joint_moments_are_removed_from_low_modes(self):
        import trimesh
        from core.canonical_shape_refine import low_frequency_basis
        mesh=trimesh.creation.icosphere(subdivisions=2);active=mesh.vertices[:,1]>-.5
        reg=np.random.default_rng(2).uniform(size=(5,len(mesh.vertices)));reg/=reg.sum(1,keepdims=True)
        B,_=low_frequency_basis(mesh.vertices,mesh.faces,active,4,reg)
        np.testing.assert_allclose(np.einsum('jv,vck->jck',reg,B),0,atol=1e-12)
        np.testing.assert_allclose(B[~active],0,atol=0)

    def test_frontal_cache_rejects_changed_input(self):
        from core.multiview_initialization import initialization_signature,save_initialization,load_initialization
        f={s:fixture('front',s) for s in ('closed','open')};options=FitOptions()
        a=initialization_signature(f,options)
        self.assertEqual(a,initialization_signature(f,replace(options,pose_strategy='prior',side_contour_weight=.3)))
        initial={s:dict(shape=np.zeros(3),expression=np.zeros(2),pose=np.zeros(6),rotation=np.eye(3),translation=np.zeros(3)) for s in f}
        with tempfile.TemporaryDirectory() as d:
            path=d+'/initial.npz';save_initialization(path,a,initial);load_initialization(path,a)
            f['closed'].xyz[33,0]+=.001
            with self.assertRaises(ValueError):load_initialization(path,initialization_signature(f,options))


if __name__=='__main__':unittest.main()
