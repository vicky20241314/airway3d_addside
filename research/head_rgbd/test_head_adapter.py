"""Synthetic contracts; not real-data head-accuracy evidence."""
import unittest
import numpy as np
from head_adapter import subdivide,topology,geometry_guard,fit_surface,fragile_vertices
from run_head_combined import validate_authorization,MODEL_SHA


class HeadContracts(unittest.TestCase):
    def test_subdivision_keeps_closed_topology_and_prior_positions(self):
        v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        f = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
        vv,ff = subdivide(v,f,1)
        self.assertEqual(len(vv),10)
        self.assertEqual(len(ff),16)
        np.testing.assert_array_equal(vv[:4],v)
        self.assertTrue(topology(vv,ff)['watertight'])
        self.assertEqual(topology(vv,ff)['nonmanifold_edges'],0)

    def test_open_boundary_is_reported_not_silently_capped(self):
        v = np.array([[0,0,.5],[.001,0,.5],[0,.001,.5]])
        f = np.array([[0,2,1]])
        vv,ff = subdivide(v,f,1)
        self.assertFalse(topology(vv,ff)['watertight'])
        self.assertEqual(topology(vv,ff)['boundary_edges'],6)

    def test_cap_and_inverted_triangle_guards(self):
        v = np.array([[0,0,.5],[.001,0,.5],[0,.001,.5]])
        f = np.array([[0,2,1]])
        shifted = v.copy();shifted[:,2]+=.004
        self.assertFalse(geometry_guard(v,shifted,f)['passed'])
        flipped = v.copy();flipped[2,1] = -.001
        self.assertFalse(geometry_guard(v,flipped,f)['passed'])
        self.assertEqual(geometry_guard(v,flipped,f)['inverted_triangles'],1)

    def test_fragile_geometry_is_locked_instead_of_relaxing_guards(self):
        v = np.array([[0,0,.5],[.001,0,.5],[0,.00001,.5]])
        f = np.array([[0,2,1]])
        self.assertTrue(fragile_vertices(v,f).all())

    def test_frozen_unseen_component_and_shared_correction(self):
        y,x = np.indices((15,15))
        v = np.stack(((x-7)*.001,(y-7)*.001,np.full(x.shape,.5)),-1).reshape(-1,3)
        a = (y[:-1,:-1]*15+x[:-1,:-1]).ravel()
        f = np.concatenate([np.stack((a,a+15,a+1),1),np.stack((a+1,a+15,a+16),1)])
        # An unseen disconnected patch is a synthetic stand-in for locked prior.
        rear = np.array([[.05,0,.6],[.051,0,.6],[.05,.001,.6]])
        base = np.vstack([v,rear]);faces = np.vstack([f,[[225,227,226]]])
        target = v.copy();target[:,2] -= .001
        valid = np.ones((50,50),bool);K = np.array([500,500,25,25])
        result,info = fit_surface(base,faces,target,f,valid,K,(50,50),np.eye(4),iterations=3)
        np.testing.assert_array_equal(result[-3:],rear)
        self.assertTrue(info['guard']['passed'])
        self.assertLessEqual(np.max(np.linalg.norm(info['delta'],axis=1)),.0030000001)
        shared = np.zeros(len(base),bool);shared[:80] = True
        other,other_info = fit_surface(base,faces,target,f,valid,K,(50,50),np.eye(4),
            iterations=3,fixed_mask=shared,fixed_delta=info['delta'])
        np.testing.assert_array_equal(other_info['delta'][shared],info['delta'][shared])

    def test_permission_is_specific_and_noncommercial(self):
        valid = dict(model_sha256=MODEL_SHA,accepted_official_academic_license=True,
                     purpose='local_noncommercial_research',redistribution_authorized=False)
        validate_authorization(valid)
        for wrong in [dict(valid,purpose='commercial'),dict(valid,model_sha256='unknown'),
                      dict(valid,accepted_official_academic_license=False)]:
            with self.assertRaises(ValueError):validate_authorization(wrong)


if __name__=='__main__':
    unittest.main()
