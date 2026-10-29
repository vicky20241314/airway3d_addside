import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from core.camera import Camera
from core.multiview_geometry import RigidCamera,project
from core.burst_alignment import align_frame,warp_depth
from core.temporal_capture import prepare_burst,BurstQualityError
from test_temporal_capture import burst_fixture
from test_multiview import fixture


class HandheldTests(unittest.TestCase):
    def test_pipeline_reports_exclusions_and_refines_only_fitted_views(self):
        from core.pipeline import run_multiview
        obs=[fixture(v,s) for s in ('closed','open') for v in ('front','left','right')]
        obs[4].parsing=None;obs[4].points=None
        used=[o.view_id for o in obs if o.state=='closed' or o.view=='front']
        fitting=dict(used_view_ids=used,excluded_views=[dict(view_id='open_left'),dict(view_id='open_right')])
        transforms={k:RigidCamera(np.eye(3),np.zeros(3)) for k in used}
        with tempfile.TemporaryDirectory() as d, \
                patch('core.aruco.detect_aruco',return_value=(None,None,np.array([0,0,-.6]))), \
                patch('core.mask_detection.get_face_data_single',return_value=dict(valid=False)), \
                patch('core.multiview_fitting.fit_multiview',return_value=({'closed':'c','open':'o'},transforms,fitting)), \
                patch('core.pipeline.finish_meshes',return_value=({},('c','o'),(),[])), \
                patch('core.pipeline.finish_multiview_refinement',return_value=(('c','o'),[])) as refine:
            result=run_multiview(obs,output_root=d)
            self.assertEqual(set(o.view_id for o in refine.call_args.args[1]),set(used))
            report=json.loads((result['folder']/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['captured_view_ids']),6)
            self.assertEqual(set(report['view_ids']),set(used))
            self.assertEqual(len(result['excluded_views']),2)

    def test_five_input_frames_allow_three_reliable_frames(self):
        frames,c,_,lm,det=burst_fixture(5)
        lm[0]+=10;lm[4]+=10
        o=prepare_burst(frames,c,detector=det)
        self.assertEqual(o.quality['frames'],3)
        self.assertFalse(o.temporal['accepted'][0]);self.assertFalse(o.temporal['accepted'][4])

    def test_rgbd_track_motion_and_reference_projection(self):
        rng=np.random.default_rng(6);c=Camera(200,200,80,80)
        rgb=rng.integers(0,255,(160,160,3),dtype=np.uint8)
        depth=np.full((160,160),600,np.uint16);box=(20,15,140,145)
        shifted=cv2.warpAffine(rgb,np.float32([[1,0,4],[0,1,1]]),(160,160))
        T,diag=align_frame(shifted,depth,rgb,depth,c,box)
        np.testing.assert_allclose(T.t,[-.012,.003,0],atol=.0003)
        self.assertGreaterEqual(diag['validation_points'],8)
        warped=warp_depth(depth,c,T,depth)
        np.testing.assert_allclose(warped[20:130,20:130],600,atol=.1)
        base_uv=np.column_stack((np.linspace(45,115,68),80+10*np.sin(np.arange(68))))
        frames=[];landmarks=[]
        for i,shift in enumerate((-4,-2,0,2,4)):
            color=cv2.warpAffine(rgb,np.float32([[1,0,shift],[0,1,0]]),(160,160));color[0,0,0]=i
            frames.append(('closed','front',color,depth,1+i*.03,np.array([200,200,80,80])))
            landmarks.append(base_uv+[shift,0])
        detector=lambda color:(np.array([20,15,120,130]),landmarks[int(color[0,0,0])])
        with self.assertRaises(BurstQualityError):prepare_burst(frames,c,detector=detector,compensate_motion=False)
        o=prepare_burst(frames,c,detector=detector)
        self.assertEqual(o.quality['frames'],5)
        self.assertEqual(o.quality['temporal']['compensated_frames'],4)
        np.testing.assert_allclose(project(o.xyz,c)[0],o.uv,atol=1e-5)
        np.testing.assert_allclose(o.uv,base_uv,atol=.1)

    def test_warp_does_not_fill_reference_holes_or_mix_depth_edges(self):
        c=Camera(100,100,5,5);ref=np.full((10,10),600,np.uint16);ref[0,0]=0
        source=ref.copy();source[0,0]=600;source[1,1]=620
        result=warp_depth(source,c,RigidCamera(np.eye(3),np.zeros(3)),ref)
        self.assertEqual(result[0,0],0);self.assertEqual(result[1,1],0)

    def test_optional_open_view_exclusion_is_explicit_and_closed_failure_stops(self):
        from core.multiview_registration import select_registered_views
        obs=[fixture('front','closed'),fixture('front','open'),fixture('left','open')]
        with tempfile.TemporaryDirectory() as d,patch('core.multiview_registration.register_to_front',side_effect=ValueError('missing lips')):
            kept,transforms,_,excluded=select_registered_views(obs,d)
            self.assertEqual([o.view_id for o in kept],['closed_front','open_front'])
            self.assertEqual(transforms,{})
            self.assertEqual(excluded[0]['view_id'],'open_left')
            report=json.loads((Path(d)/'view_selection.json').read_text(encoding='utf-8'))
            self.assertIn('open_left',report['captured_view_ids'])
            obs[-1]=fixture('left','closed')
            with self.assertRaises(ValueError):select_registered_views(obs,d)
        obs[-1]=fixture('left','open')
        with tempfile.TemporaryDirectory() as d,patch('core.multiview_registration.register_to_front',side_effect=RuntimeError('implementation error')):
            with self.assertRaises(RuntimeError):select_registered_views(obs,d)


if __name__=='__main__':unittest.main()
