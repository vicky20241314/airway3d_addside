"""Geometry contracts, not evidence of real reconstruction accuracy."""
import unittest
import numpy as np
import trimesh
from geometry_c import ray_surface, landmark_validity, refine_shared,first_hits,safe_refinement


class SurfaceContracts(unittest.TestCase):
    def test_pose_safety_projection_prevents_small_triangle_foldover(self):
        v=np.array([[0.,0.,0.],[.001,0.,0.],[0.,.001,0.],[0.,0.,.001]])
        f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        vv,delta,log=safe_refinement(v[None],f,np.array([.003,-.003,.003,-.003]))
        self.assertGreater(log[0]['initial_unsafe_triangles'],0)
        self.assertEqual(log[0]['final_unsafe_triangles'],0)
        self.assertLessEqual(np.abs(delta).max(),.003)
        self.assertTrue(np.isfinite(vv).all())
    def test_chunked_visibility_preserves_measured_ray_ids(self):
        mesh=trimesh.Trimesh([[-.2,-.2,-.4],[.2,-.2,-.4],[0,.2,-.4]],[[0,1,2]],process=False)
        points=np.column_stack((np.linspace(-.05,.05,257),np.zeros(257),np.full(257,-.401)))
        hit,ray,face=first_hits(mesh,points,chunk=13)
        np.testing.assert_array_equal(np.sort(ray),np.arange(257))
        np.testing.assert_array_equal(face,0)
        np.testing.assert_allclose(hit[:,0],points[ray,0]*(.4/.401),atol=1e-10)

    def test_nearer_triangle_occludes_farther_and_barycentric_is_metric(self):
        v=np.array([[-.2,-.2,-.4],[.2,-.2,-.4],[0,.2,-.4],
                    [-.2,-.2,-.5],[.2,-.2,-.5],[0,.2,-.5]])
        f=np.array([[0,1,2],[3,4,5]])
        points=np.array([[0.,0.,-.401],[.01,0.,-.402]])
        c,s=ray_surface(v,f,points)
        self.assertEqual(s['accepted'],2)
        np.testing.assert_array_equal(c['face'],[0,0])
        np.testing.assert_allclose((v[f[c['face']]]*c['bary'][:,:,None]).sum(1),c['hit'],atol=1e-10)
        np.testing.assert_allclose(c['hit'][:,2],-.4)

    def test_invalid_depth_does_not_become_origin_and_lip_spike_is_rejected(self):
        xyz=np.zeros((68,3));xyz[:,2]=-.4;xyz[66,2]=-.412
        valid=np.ones(68,bool);valid[62]=False;xyz[62]=np.nan
        mask,rejected=landmark_validity(dict(valid=valid,xyz=xyz))
        self.assertFalse(mask[62]);self.assertFalse(mask[66]);self.assertEqual(rejected,[66])

    def test_no_observation_leaves_prior_and_no_unsupported_deformation(self):
        v=np.array([[[-.1,-.1,-.4],[.1,-.1,-.4],[0,.1,-.4]]])
        f=np.array([[0,1,2]])
        delta,support,_,_=refine_shared(v,f,[],[],np.zeros(3,bool),np.random.default_rng(1))
        np.testing.assert_array_equal(delta,0);self.assertEqual(support.shape,(0,3))

    def test_lost_depth_preserves_protected_anchor(self):
        v=np.array([[[-.1,-.1,-.4],[.1,-.1,-.4],[0,.1,-.4]]])
        anchor=np.array([.001,-.002,0.])
        delta,_,_,_=refine_shared(v,np.array([[0,1,2]]),[],[],np.zeros(3,bool),np.random.default_rng(1),
            fixed_mask=np.ones(3,bool),fixed_deviation=anchor)
        np.testing.assert_array_equal(delta,anchor)


if __name__=='__main__':unittest.main()
