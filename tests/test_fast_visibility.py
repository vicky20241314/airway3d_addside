import unittest
import numpy as np
import trimesh
from core.multiview_geometry import ray_visible


class FastVisibilityTests(unittest.TestCase):
    def test_exact_tolerance_boundary_matches_reference(self):
        mesh=trimesh.Trimesh([[-1,-1,-.6],[1,-1,-.6],[0,1,-.6]],[[0,1,2]],process=False)
        rng=np.random.default_rng(42)
        points=np.column_stack((rng.uniform(-.1,.1,(200,2)),np.full(200,-.6)))
        for tolerance in (.001,.0015):
            samples=points+points/np.linalg.norm(points,axis=1,keepdims=True)*tolerance
            np.testing.assert_array_equal(
                ray_visible(mesh.vertices,mesh.faces,samples,tolerance=tolerance,backend='reference'),
                ray_visible(mesh.vertices,mesh.faces,samples,tolerance=tolerance,backend='projected'))

    def compare(self,mesh,samples):
        old=ray_visible(mesh.vertices,mesh.faces,samples,backend='reference')
        new=ray_visible(mesh.vertices,mesh.faces,samples,backend='projected')
        np.testing.assert_array_equal(new,old)

    def test_occluders_edges_vertices_empty_and_misses(self):
        front=trimesh.creation.box(extents=(.12,.12,.02));front.apply_translation([.02,0,-.5])
        back=trimesh.creation.box(extents=(.3,.3,.04));back.apply_translation([0,0,-.65])
        mesh=trimesh.util.concatenate([front,back])
        self.compare(mesh,np.vstack((mesh.vertices,mesh.triangles_center,[[0,0,-.8],[1,1,-.6],[0,0,.1],[np.nan,0,-.5]])))
        self.compare(mesh,np.empty((0,3)))

    def test_curved_surface_random_rays_and_offset_tolerance(self):
        mesh=trimesh.creation.icosphere(subdivisions=3,radius=.12);mesh.apply_translation([0,0,-.6])
        rng=np.random.default_rng(15)
        samples=rng.uniform([-.2,-.2,-.85],[.2,.2,-.4],(2100,3))
        self.compare(mesh,np.vstack((samples,mesh.vertices,mesh.triangles_center[::3],
            mesh.vertices+np.array([0,0,.003]),mesh.vertices+np.array([0,0,.0005]))))

    def test_camera_plane_crossing_uses_reference_fallback(self):
        mesh=trimesh.creation.box(extents=(.2,.2,.2))
        self.compare(mesh,np.array([[0,0,-.1],[0,0,-.2],[.01,.02,-.1]]))

    def test_degenerate_tiny_triangle_does_not_become_an_occluder(self):
        vertices=np.array([[-.1,-.1,-.6],[.1,-.1,-.6],[0,.1,-.6],
            [-1e-8,-1e-8,-.4],[1e-8,-1e-8,-.4],[0,1e-8,-.4]])
        mesh=trimesh.Trimesh(vertices,[[0,1,2],[3,4,5]],process=False)
        self.compare(mesh,np.array([[0,0,-.6],[0,0,-.4]]))


if __name__=='__main__':unittest.main()
