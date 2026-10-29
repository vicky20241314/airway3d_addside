"""Independent-evaluation bookkeeping and synthetic error-measure contracts."""
import copy
import unittest
import numpy as np
import trimesh
from core.multiview_evaluation import assert_held_out,contour_errors,evaluate_view,summarize
from test_multiview_refine import plane_fixture,plane_observation


class EvaluationTests(unittest.TestCase):
    def test_duplicate_capture_cannot_be_held_out(self):
        training,_=plane_observation(0,'train');held=copy.deepcopy(training);held.view_id='held'
        with self.assertRaises(ValueError):assert_held_out([training],[held])
        held.rgb=held.rgb.copy();held.rgb[0,0,0]=1
        assert_held_out([training],[held])

    def test_reviewed_contour_errors(self):
        a=np.zeros((64,64),bool);a[16:48,16:48]=True
        self.assertEqual(contour_errors(a,a)['chamfer_mean_px'],0.)
        self.assertGreater(contour_errors(np.roll(a,2,axis=1),a)['normal_mean_px'],.5)
        self.assertEqual(contour_errors(a,np.zeros_like(a))['status'],'empty_contour')

    def test_untrimmed_surface_error_and_explicit_missing_contour(self):
        base,faces=plane_fixture();mesh=trimesh.Trimesh(base,faces,process=False)
        o,T=plane_observation(30,'test_side')
        value=evaluate_view(mesh,o,T,max_points=1000)
        self.assertGreater(value['n_surface_samples'],100)
        self.assertAlmostEqual(value['surface']['mean_mm'],.5,delta=.08)
        self.assertFalse(value['surface']['distance_trimming'])
        self.assertEqual(value['contour']['status'],'reviewed_reference_mask_missing')
        value['role']='held_out';summary=summarize([value])
        self.assertIn('held_out_side',summary);self.assertNotIn('in_fit_side',summary)


class ReplayOptionTests(unittest.TestCase):
    def test_stage2_comparison_never_refines(self):
        from unittest.mock import patch
        from tools.replay_multiview import main
        with patch('sys.argv',['replay','--session','synthetic.json','--compare-front']), \
                patch('tools.replay_multiview.load_session',return_value=[]), \
                patch('tools.replay_multiview.run_multiview',return_value={'folder':'synthetic_contract'}) as run:
            main()
            self.assertEqual(run.call_count,2)
            self.assertEqual([c.kwargs['front_only'] for c in run.call_args_list],[True,False])
            self.assertTrue(all(c.kwargs['refine'] is False for c in run.call_args_list))


if __name__=='__main__':unittest.main()
