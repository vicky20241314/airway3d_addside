"""Software contracts. Synthetic fixtures do not validate real side accuracy."""
import tempfile
import unittest
import os
from pathlib import Path
import numpy as np
import torch
import trimesh
from core.camera import Camera,backproject,coordinate_transform
from core.multiview_data import Observation,save_session,load_session,fuse_sequence
from core.multiview_geometry import RigidCamera,project,ray_visible,visible_faces,visible_landmarks
from core.multiview_registration import rigid_svd
from core.multiview_fitting import MultiLayout,MultiProblem
from core.fit_options import FitOptions
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


def fixture(view='front',state='closed'):
    uv=np.tile([32.,32.],(68,1));xyz=np.tile([0.,0.,-.6],(68,1))
    return Observation(state+'_'+view,state,view,np.zeros((64,64,3),np.uint8),np.full((64,64),600,np.uint16),
        Camera(80,80,32,32),uv,xyz,np.ones(68,bool),dict(weight=1.,motion_mode='camera_moved',source='synthetic_contract'),
        np.ones((64,64),np.uint8),xyz.copy())


class GeometryTests(unittest.TestCase):
    def test_transform_projection_and_scale(self):
        from scipy.spatial.transform import Rotation
        T=RigidCamera(Rotation.from_euler('y',40,degrees=True).as_matrix(),np.array([.05,.02,-.1]))
        p=np.array([[.02,.03,-.6],[-.02,.01,-.65]])
        np.testing.assert_allclose(T.to_common(T.to_view(p)),p,atol=1e-12)
        np.testing.assert_allclose(T.inverse().compose(T).to_view(p),p,atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(T.to_view(p)[0]-T.to_view(p)[1]),np.linalg.norm(p[0]-p[1]))
        cam=Camera(80,90,32,30);d=np.full((64,64),600,np.uint16);uv=np.array([[34,35],[30,28]])
        q,valid=backproject(uv,d,cam);out,z=project(coordinate_transform(q),cam)
        np.testing.assert_allclose(out,uv,atol=1e-5);np.testing.assert_allclose(z,.6)
        with self.assertRaises(ValueError):RigidCamera(np.eye(3)*2,np.zeros(3))

    def test_rigid_registration_direction(self):
        from scipy.spatial.transform import Rotation
        source=np.random.default_rng(2).normal(size=(50,3))*.04
        known=RigidCamera(Rotation.from_euler('xyz',[5,40,2],degrees=True).as_matrix(),np.array([.03,.01,-.6]))
        np.testing.assert_allclose(rigid_svd(source,known.to_view(source)).to_view(source),known.to_view(source),atol=1e-12)

    def test_occlusion_and_backside(self):
        mesh=trimesh.creation.box(extents=(.2,.2,.1));mesh.apply_translation([0,0,-.6])
        points=np.array([[0,0,-.55],[0,0,-.65],[0,0,.2]])
        self.assertEqual(ray_visible(mesh.vertices,mesh.faces,points).tolist(),[True,False,False])
        vis=visible_faces(mesh.vertices,mesh.faces,Camera(80,80,32,32),np.full((64,64),550,np.uint16))
        self.assertGreater(vis.sum(),0);self.assertTrue((mesh.triangles_center[vis,2]>-.6).all())
        lm=np.tile(points[1],(68,1));uv=np.tile([32,32],(68,1))
        self.assertFalse(visible_landmarks(mesh.vertices,mesh.faces,lm,uv,np.ones(68,bool),Camera(80,80,32,32),np.full((64,64),550)).any())


class SessionTests(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        obs=fixture()
        with tempfile.TemporaryDirectory() as d:
            path=save_session(d,[obs]);result=load_session(path)[0]
            np.testing.assert_array_equal(result.depth,obs.depth);self.assertEqual(result.camera.fx,80)
            with self.assertRaises(FileExistsError):save_session(d,[obs])

    def test_front_strict_side_partial_and_motion(self):
        o=fixture('left');o.valid[:]=False;o.xyz[:]=np.nan;o.validate();o.view='front'
        with self.assertRaises(ValueError):o.validate()
        o=fixture();o.quality['motion_mode']='head_turned'
        with self.assertRaises(ValueError):o.validate()

    def test_never_fuse_views_or_moving_sequence(self):
        o=fixture();intr=np.array([80,80,32,32])
        frames=[('closed','front',o.rgb,o.depth,1+i*.03,intr) for i in range(5)]
        _,d,q=fuse_sequence(frames,o.camera);np.testing.assert_array_equal(d,o.depth)
        frames[1]=('closed','left',o.rgb,o.depth,1.03,intr)
        with self.assertRaises(ValueError):fuse_sequence(frames,o.camera)
        frames=[('closed','front',o.rgb,o.depth+i*10,1+i*.03,intr) for i in range(5)]
        with self.assertRaises(ValueError):fuse_sequence(frames,o.camera)

    def test_parameters_shared_per_state(self):
        l=MultiLayout(100,30,(0,3),('closed','open','closed','closed'))
        self.assertEqual(l.expr[0],l.expr[2]);self.assertEqual(l.pose[0],l.pose[3]);self.assertNotEqual(l.expr[0],l.expr[1])
        self.assertNotEqual(l.rot[0],l.rot[2]);self.assertEqual(l.size,100+2*(30+2)+4*6)
        self.assertEqual(len(l.active(1)),24);self.assertEqual(len(set(l.active(2))),l.size)

    def test_six_views_have_only_two_state_slots(self):
        from core.multiview_data import validate_session
        obs=[fixture(v,s) for s in ('closed','open') for v in ('front','left','right')]
        self.assertEqual(len(validate_session(obs)),6)
        l=MultiLayout(100,30,(0,3),tuple(o.state for o in obs))
        self.assertEqual(l.size,200);self.assertEqual(len(l.state_slots),2)
        self.assertEqual(l.pose[3],l.pose[5]);self.assertEqual(l.expr[3],l.expr[4])

    def test_open_mouth_change_or_missing_support_rejected(self):
        from core.multiview_registration import check_expression
        import copy
        front=fixture('front','open');front.xyz[[50,51,52],1]=.02
        side=copy.deepcopy(front);side.view='left';side.view_id='open_left'
        T=RigidCamera(np.eye(3),np.zeros(3))
        check_expression(front,side,T)
        side.xyz[[56,57,58],1]=-.012
        with self.assertRaises(ValueError):check_expression(front,side,T)
        side=copy.deepcopy(front);side.valid[48:]=False
        with self.assertRaises(ValueError):check_expression(front,side,T)


class CaptureTests(unittest.TestCase):
    def test_guided_capture_state_machine_and_intrinsics(self):
        from unittest.mock import patch
        import itertools
        from PyQt6.QtWidgets import QApplication
        from ui.capture_widget import FaceBBoxThread
        import ui.multiview_capture as capture
        app=QApplication.instance() or QApplication([])
        sample=fixture();emitted=[]
        with tempfile.TemporaryDirectory() as d, patch.object(FaceBBoxThread,'start',lambda self:None), \
                patch.object(capture,'detect_face_bbox',return_value=np.array([8,8,48,48])), \
                patch.object(capture,'get_lmk_single',return_value=sample.uv), \
                patch.object(capture.MultiCaptureWidget,'_show_warning',side_effect=self.fail), \
                patch.object(capture.time,'monotonic',side_effect=itertools.count(1.,.03)), \
                patch.object(capture,'detect_aruco',return_value=(np.zeros((4,2)),0,np.array([0,0,-.6]))) as marker:
            w=capture.MultiCaptureWidget();w.mode.setCurrentIndex(getattr(self,'capture_mode',1));w._capture_folder=Path(d)/'capture'
            w.session_done.connect(emitted.append)
            count=len(w.sequence())
            for i in range(count):
                camera=Camera(80+i,90+i,32,32)
                for _ in range(5):w.on_frame(sample.rgb,sample.depth,camera)
                w._on_capture()
                self.assertEqual(len(w._observations),i+1)
                self.assertEqual(len(w._mv_frames),0)
            self.assertEqual(marker.call_count,1)
            self.assertEqual(len(emitted),1)
            self.assertEqual([o.camera.fx for o in emitted[0]],list(range(80,80+count)))
            w.stop_threads();w.close();app.processEvents()


class FlameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.integrated_fitting import make_flame
        cls.options=FitOptions(shape_components=3,expression_components=2,basis_levels=((3,2),),max_points=50)
        cls.flame=make_flame(cls.options,'cpu',batch=getattr(cls,'batch',4),dtype=torch.float64)

    def problem(self):
        from scipy.spatial.transform import Rotation
        observations=[fixture('front','closed'),fixture('front','open'),fixture('left','closed'),fixture('right','closed')]
        if self.flame.batch_size==6:observations.extend([fixture('left','open'),fixture('right','open')])
        angles=(0,0,35,-35,35,-35)[:len(observations)]
        transforms=[RigidCamera(Rotation.from_euler('y',a,degrees=True).as_matrix(),np.array([0.,0.,-.6])) for a in angles]
        param=dict(shape=np.array([.03,-.02,.01]),expression=np.array([.01,-.02]),pose=np.array([.02,0,0,.08,0,0]))
        return MultiProblem(self.flame,observations,{'closed':param,'open':param},transforms,self.options)

    def test_view_contour_matches_embedding_and_changes(self):
        p=self.problem();v,lm=p.forward(p.theta);tri,bc=p._embedding(p.theta,v)
        reconstructed=torch.stack([(v[b][tri[b]]*bc[b,:,:,None]).sum(1) for b in range(len(p.observations))])
        torch.testing.assert_close(lm,reconstructed)
        self.assertFalse(torch.equal(tri[0,:17],tri[2,:17]));self.assertTrue(torch.equal(tri[0,17:],tri[2,17:]))

    def test_multi_view_vertex_and_landmark_jacobians(self):
        p=self.problem();v,_=p.forward(p.theta);p.embedding=p._embedding(p.theta,v)
        jac=p.vertex_jacobians(p.theta,2);tri,bc=p.embedding
        l_jac=torch.stack([(jac[b][tri[b]]*bc[b,:,:,None,None]).sum(1) for b in range(len(p.observations))])
        for column in range(p.layout.size):
            plus=p.theta.clone();minus=p.theta.clone();eps=1e-5;plus[column]+=eps;minus[column]-=eps
            a,la=p.forward(plus);b,lb=p.forward(minus)
            torch.testing.assert_close(jac[:,:,:,column],(a-b)/(2*eps),rtol=2e-3,atol=3e-5)
            torch.testing.assert_close(l_jac[:,:,:,column],(la-lb)/(2*eps),rtol=2e-3,atol=3e-5)

    def test_full_residual_jacobian_and_single_state_priors(self):
        p=self.problem();v,lm=p.forward(p.theta);p.embedding=p._embedding(p.theta,v)
        p.context=[]
        for b in range(len(p.observations)):
            tri=p.faces[100:103];bc=p.theta.new_full((3,3),1/3)
            points=(v[b][tri]*bc[:,:,None]).sum(1)+.0002
            p.context.append(dict(tri=tri,bary=bc,points=points,normals=p.theta.new_tensor([[0,0,1.]]*3),
                weights=p.theta.new_full((3,),.25),ids=np.array([33]),valid=np.zeros(68,bool),jaw_irls=p.theta.new_empty(0)))
            p.jaw[b]['ids']=np.empty(0,int)
        residual,jac=p.residual(p.theta,2)
        # Last rows = shape once + expression once per STATE (not 4 views).
        expected=torch.cat((p.theta[p.layout.shape]*np.sqrt(p.options.shape_reg),
            *[p.theta[s[0]]*np.sqrt(p.options.expression_reg) for s in p.layout.state_slots.values()]))
        torch.testing.assert_close(residual[-7:],expected)
        for column in range(p.layout.size):
            plus=p.theta.clone();minus=p.theta.clone();plus[column]+=1e-5;minus[column]-=1e-5
            fd=(p.residual(plus,2,False)[0]-p.residual(minus,2,False)[0])/2e-5
            torch.testing.assert_close(jac[:,column],fd,rtol=3e-3,atol=3e-5)


class SixViewFlameTests(FlameTests):
    batch=6


class SixViewCaptureTests(CaptureTests):
    capture_mode=2


class MetricSourceTests(unittest.TestCase):
    def test_metrics_always_use_state_fronts_and_front_aruco(self):
        from unittest.mock import patch
        from core.pipeline import run_multiview
        obs=[fixture(v,s) for s in ('closed','open') for v in ('front','left','right')]
        for i,o in enumerate(obs):o.xyz[:,0]=i*.001
        c=obs[0];o=obs[3];adam=np.array([.1,.2,-.6])
        with tempfile.TemporaryDirectory() as d, \
                patch('core.aruco.detect_aruco',return_value=(None,None,adam)) as marker, \
                patch('core.multiview_fitting.fit_multiview',return_value=({'closed':'c_mesh','open':'o_mesh'},{},{})), \
                patch('core.pipeline.finish_meshes',return_value=({},(),(),[])) as finish:
            result=run_multiview(obs,output_root=d)
            args=finish.call_args.args
            self.assertEqual(args[0],('c_mesh','o_mesh'))
            np.testing.assert_array_equal(args[5],adam)
            np.testing.assert_array_equal(args[6][0],c.xyz);np.testing.assert_array_equal(args[6][1],o.xyz)
            self.assertEqual(marker.call_count,1)
            import json
            report=json.loads((result['folder']/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['metric_sources']['closed_front'],'closed_front')
            self.assertEqual(report['metric_sources']['open_front'],'open_front')


if __name__=='__main__':unittest.main()
