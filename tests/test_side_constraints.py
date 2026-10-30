import unittest
import copy
from unittest.mock import patch
import numpy as np
from test_multiview import fixture
from core.side_constraints import constraint_observation


class SideConstraintTests(unittest.TestCase):
    def test_side_neck_excluded_without_mutating_capture_or_front(self):
        o=fixture('left');o.parsing[:32]=1;o.parsing[32:]=14
        from core.camera import backproject,coordinate_transform
        uv=np.array([[25.,20.],[25.,40.]])
        p,_=backproject(uv,o.depth,o.camera);o.points=coordinate_transform(p)
        raw=copy.deepcopy(o);head=constraint_observation(o)
        self.assertEqual(len(head.points),1)
        self.assertFalse(head.depth[32:].any())
        self.assertTrue((head.parsing[32:]==255).all())
        np.testing.assert_array_equal(o.depth,raw.depth)
        np.testing.assert_array_equal(o.parsing,raw.parsing)
        np.testing.assert_array_equal(o.points,raw.points)
        o.view='front';self.assertIs(constraint_observation(o),o)

    def test_refinement_receives_same_head_mask_as_fitting(self):
        from core.multiview_refine import prepare_state
        from test_multiview_refine import plane_fixture,plane_observation
        from scipy import sparse
        base,faces=plane_fixture();o,T=plane_observation(25,'side')
        o.parsing[70:]=14
        seen=[]
        def inputs(v,f,d,c,p,u):
            seen.append((d.copy(),p.copy()))
            return np.empty((0,3)),np.empty(0),np.zeros(len(v),bool)
        with patch('core.student.offset_refine.depth_inputs',side_effect=inputs):
            prepare_state(base,faces,sparse.csr_matrix((51,len(base))),[o],{o.view_id:T})
        self.assertFalse(seen[0][0][70:].any())
        self.assertTrue((seen[0][1][70:]==255).all())


if __name__=='__main__':unittest.main()
