"""Synthetic algorithm contracts only; these tests are not real-side evidence."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from a3_geometry import (rigid_fit,transform,project,backproject,polygon_mask,
    mesh_from_depth,depth_normals,splat_depth,fuse,refine_depth,raster_depth,principal_component,load_frame)


class GeometryContracts(unittest.TestCase):
    def test_rigid_recovers_metres_without_scale(self):
        p = np.random.default_rng(7).normal(size=(12,3))*.03+[0,0,.5]
        truth = np.eye(4)
        truth[:3,:3] = Rotation.from_euler('y',29,degrees=True).as_matrix()
        truth[:3,3] = [.04,.02,-.01]
        T = rigid_fit(p,transform(p,truth))
        np.testing.assert_allclose(T,truth,atol=1e-12)
        np.testing.assert_allclose(np.linalg.svd(T[:3,:3])[1],1.,atol=1e-12)
        # Deliberately scaled input cannot be explained as zero-residual rigid fit.
        bad = rigid_fit(p,transform(p*1.1,truth))
        self.assertGreater(np.linalg.norm(transform(p,bad)-transform(p*1.1,truth)),.001)

    def test_intrinsic_stride_and_camera_round_trip(self):
        K = np.array([849.2,849.1,640.1,361.0])
        z = np.ones((360,640))*.55
        xyz = backproject(z,K/2)
        uv = project(xyz,K/2)
        y,x = np.indices(z.shape)
        np.testing.assert_allclose(uv,np.stack((x,y),-1),atol=1e-10)
        np.testing.assert_allclose(project(xyz,K),uv*2,atol=1e-10)

    def test_raw_loader_uses_actual_frame_K_and_retains_invalid_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'synthetic_contract_only.npz'
            depth = np.full((2,8,8),500,np.uint16)
            depth[1,4,4] = 0
            intr = np.array([[100,110,3,4],[200,220,2,3]],float)
            np.savez(p,depth=depth,rgb=np.zeros((2,8,8,3),np.uint8),
                     intrinsics=intr,timestamps_s=np.array([1.,2.]))
            row = dict(source=str(p),source_frame=1,roi_polygon_px=[[0,0],[8,0],[8,8],[0,8]],
                       landmarks_px=dict(nose=[4,4]))
            f = load_frame(row,2,Path(tmp))
            np.testing.assert_array_equal(f['K'],intr[1]/2)
            self.assertFalse(f['valid'][2,2])
            self.assertIsNone(f['landmarks']['nose'])
            self.assertEqual(f['depth'][1,1],.5)

    def test_invalid_depth_not_meshed_or_bridged(self):
        z = np.full((8,8),.5)
        z[:,3:5] = np.nan
        vertices,faces,pixels = mesh_from_depth(z,[500,500,3,3])
        self.assertTrue(np.isfinite(vertices).all())
        self.assertFalse(np.isin(pixels[:,1],[3,4]).any())
        tri_pixels = pixels[faces]
        self.assertTrue(np.all(np.ptp(tri_pixels[:,:,1],axis=1)<=1))

    def test_outside_projection_and_backfacing(self):
        layer = splat_depth(np.array([[0,0,.5],[50,0,.5],[0,0,-.5]]),[100,100,2,2],(5,5))
        self.assertEqual(np.isfinite(layer).sum(),1)
        xyz = backproject(np.full((9,9),.5),[100,100,4,4])
        normal,valid = depth_normals(xyz,np.ones((9,9),bool))
        self.assertTrue((np.sum(normal[valid]*xyz[valid],-1)<0).all())

    def fake_frame(self,name,state,depth):
        shape = (12,12)
        z = np.full(shape,depth)
        xyz = backproject(z,[200,200,5,5])
        n,v = depth_normals(xyz,np.ones(shape,bool))
        return dict(row=dict(id=name,state=state,upper_limit_y_px=6),
            path=Path(name+'.npz'),xyz=xyz,normal=n,valid=np.ones(shape,bool),stride=1,K=np.array([200,200,5,5]),depth=z)

    def test_variable_views_order_and_expression_reference(self):
        a = self.fake_frame('ref','closed',.5)
        b = self.fake_frame('side','closed',.501)
        c = self.fake_frame('open','open',.502)
        f = [a,b,c]
        z,cnt,bits,_,upper = fuse(f,[np.eye(4)]*3,a,'ref','closed')
        zo,_,_,_,_ = fuse(f,[np.eye(4)]*3,a,'open','open')
        permuted = fuse([a,c,b],[np.eye(4)]*3,a,'ref','closed')[0]
        np.testing.assert_allclose(z,permuted,equal_nan=True)
        np.testing.assert_allclose(z[upper],zo[upper],equal_nan=True)
        self.assertAlmostEqual(z[8,5],.5)
        self.assertAlmostEqual(zo[8,5],.502)
        self.assertEqual(cnt[8,5],1)
        self.assertEqual(bits[8,5],1)
        # Any actual N; no artificial up view required.
        self.assertTrue(np.isfinite(fuse([a],[np.eye(4)],a,'ref','closed')[0][5,5]))

    def test_refinement_bound_and_hole_preservation(self):
        d = np.full((10,10),.5)
        d[5,5] = .502
        d[4,4] = np.nan
        out = refine_depth(d,np.full(d.shape,3),np.ones(d.shape,bool),passes=10)
        self.assertTrue(np.isnan(out[4,4]))
        self.assertLessEqual(np.nanmax(np.abs(out-d)),.000800001)

    def test_refinement_does_not_mix_expression_into_shared_identity(self):
        closed = np.full((12,12),.5)
        opened = closed.copy()
        opened[6:] += .002
        upper = np.indices(closed.shape)[0]<6
        n = np.full(closed.shape,3)
        c = refine_depth(closed,n,upper)
        o = refine_depth(opened,n,upper)
        np.testing.assert_array_equal(c[upper],o[upper])
        # A constant eligible lower region can acquire IEEE rounding in its mean;
        # the cross-state shared upper equality above remains bit-exact.
        np.testing.assert_allclose(o[~upper],opened[~upper],atol=1e-14,rtol=0)

    def test_lower_fusion_requires_real_same_burst_and_consistency(self):
        a = self.fake_frame('ref','closed',.5)
        b = self.fake_frame('adjacent','closed',.501)
        b['path'] = a['path']
        c = self.fake_frame('different_state','open',.501)
        c['path'] = a['path']
        z,count,bits,_,_ = fuse([a,b,c],[np.eye(4)]*3,a,'ref','closed')
        self.assertEqual(count[8,5],2)
        self.assertEqual(bits[8,5],3)
        self.assertAlmostEqual(z[8,5],.5005)
        # An expression/depth mismatch in the same burst must be rejected.
        b = self.fake_frame('adjacent','closed',.503)
        b['path'] = a['path']
        self.assertEqual(fuse([a,b],[np.eye(4)]*2,a,'ref','closed')[1][8,5],1)

    def test_classical_patch_match_tracks_actual_translation(self):
        from expand_a3_frames import match_patch
        rng = np.random.default_rng(9)
        a = rng.random((100,100))
        b = np.roll(a,(3,-2),(0,1))
        xy,quality = match_patch(a,b,[50,50])
        np.testing.assert_array_equal(xy,[48,53])
        self.assertGreater(quality,.999)

    def test_triangle_depth_render_units(self):
        v = np.array([[-.02,-.02,.5],[.02,-.02,.5],[-.02,.02,.5],[.02,.02,.5]])
        f = np.array([[0,2,1],[1,2,3]])
        d = raster_depth(v,f,[100,100,5,5],(11,11),1)
        self.assertAlmostEqual(d[5,5],.5)
        self.assertTrue(np.isinf(d[0,0]))

    def test_polygon_and_degenerate_landmarks(self):
        m = polygon_mask((10,10),[[2,2],[8,2],[8,8],[2,8]])
        self.assertEqual(m.sum(),36)
        with self.assertRaises(ValueError):
            rigid_fit(np.ones((4,3)),np.ones((4,3)))

    def test_disconnected_island_is_excluded_without_filling(self):
        v = np.array([[0,0,.5],[.001,0,.5],[0,.001,.5],[.001,.001,.5],
                      [.2,0,.9],[.201,0,.9],[.2,.001,.9]])
        f = np.array([[0,2,1],[1,2,3],[4,6,5]])
        selected,faces,stats = principal_component(v,f)
        self.assertEqual(stats['discarded_vertices'],3)
        self.assertEqual(len(faces),2)
        np.testing.assert_array_equal(selected,[0,1,2,3])


if __name__=='__main__':
    unittest.main()
