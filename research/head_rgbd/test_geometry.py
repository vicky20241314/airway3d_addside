"""Contract tests on constructed geometry, not evidence of real scan accuracy."""
import unittest
import numpy as np
from adapt_a import encode_depth,decode_depth
from prepare import backproject,rigid
from evaluate import raster_depth


class GeometryContracts(unittest.TestCase):
    def test_packed_depth_matches_author_decoder(self):
        d=np.array([[0,100,255,500,1000,1500,6527.9]])
        np.testing.assert_allclose(decode_depth(encode_depth(d)),d,atol=.051)

    def test_packed_depth_overflow_rejected(self):
        with self.assertRaises(ValueError):encode_depth(np.array([[7000.]]))

    def test_camera_units_and_axis_directions(self):
        depth=np.full((3,3),500,np.uint16)
        xyz,valid=backproject([[1,1],[2,0],[-1,0]],depth,[100,100,1,1])
        np.testing.assert_allclose(xyz[:2],[[0,0,-.5],[.005,.005,-.5]],atol=1e-7)
        self.assertEqual(valid.tolist(),[True,True,False])

    def test_rigid_transform_has_no_scale(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
        expected=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        b=a@expected.T+[.1,.2,.3]
        r,t=rigid(a,b)
        np.testing.assert_allclose(a@r.T+t,b,atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(r),1.)

    def test_degenerate_motion_rejected(self):
        with self.assertRaises(ValueError):rigid(np.zeros((8,3)),np.zeros((8,3)))

    def test_zbuffer_keeps_nearest_surface(self):
        v=np.array([[-.1,-.1,-.5],[.1,-.1,-.5],[0,.1,-.5],[-.2,-.2,-1],[.2,-.2,-1],[0,.2,-1.]])
        d=raster_depth(v,np.array([[3,4,5],[0,1,2]]),[100,100,30,30],(60,60),stride=1)
        self.assertAlmostEqual(d[30,30],.5)

    def test_behind_camera_does_not_render(self):
        v=np.array([[-.1,-.1,.5],[.1,-.1,.5],[0,.1,.5]])
        d=raster_depth(v,np.array([[0,1,2]]),[100,100,30,30],(60,60),stride=1)
        self.assertFalse(np.isfinite(d).any())


if __name__=='__main__':unittest.main()
