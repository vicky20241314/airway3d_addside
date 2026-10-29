"""Visibility-overlap regression fixtures; not real reconstruction accuracy."""
import copy
import unittest
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from core.camera import Camera,backproject,coordinate_transform
from core.multiview_data import Observation
from core.multiview_geometry import RigidCamera
from core.multiview_registration import check_neck_consistency


def neck_plane(angle,view_id):
    camera=Camera(400,400,64,64)
    R=Rotation.from_euler('y',angle,degrees=True).as_matrix()
    center=np.array([0.,0.,-.6]);front_to_view=RigidCamera(R,center-R@center)
    yy,xx=np.mgrid[:128,:128]
    rays=np.column_stack(((xx.ravel()-64)/400,-(yy.ravel()-64)/400,-np.ones(xx.size)))
    origin=front_to_view.to_common(np.zeros((1,3)))[0];direction=rays@R
    length=(-.6-origin[2])/direction[:,2]
    common=origin+length[:,None]*direction
    inside=(np.abs(common[:,:2])<.06).all(1)&(length>0)
    depth=np.where(inside,length*1000,0).reshape(128,128)
    o=Observation(view_id,'closed','front' if angle==0 else 'left',np.zeros((128,128,3),np.uint8),
        depth,camera,np.full((68,2),np.nan),np.full((68,3),np.nan),np.zeros(68,bool),
        dict(weight=1.,motion_mode='camera_moved'),inside.reshape(128,128).astype(np.uint8)*14)
    return o,front_to_view


class NeckOverlapTests(unittest.TestCase):
    def test_unobserved_neck_extent_does_not_cause_false_motion(self):
        front,_=neck_plane(0,'front');side,T=neck_plane(35,'left')
        front.parsing[:,:58]=0;front.parsing[:,70:]=0
        clouds=[]
        for o in (front,side):
            y,x=np.where(o.parsing==14)
            p,_=backproject(np.column_stack((x,y)),o.depth,o.camera)
            clouds.append(coordinate_transform(p))
        old=np.median(cKDTree(clouds[0]).query(T.to_common(clouds[1]))[0])
        self.assertGreater(old,.010)
        result=check_neck_consistency(front,side,T.inverse())
        self.assertEqual(result['status'],'passed')
        self.assertLess(result['median_m'],.0001)
        self.assertGreaterEqual(result['front_to_side']['overlap_samples'],100)
        # Swapping cameras must retain the same physical result.
        swapped=check_neck_consistency(side,front,T)
        self.assertAlmostEqual(result['median_m'],swapped['median_m'])

    def test_motion_toward_and_away_is_not_filtered_out(self):
        front,_=neck_plane(0,'front')
        for movement_mm in (-25.,25.):
            side=copy.deepcopy(front);side.view_id='left';side.view='left'
            side.depth[side.depth>0]+=movement_mm
            result=check_neck_consistency(front,side,RigidCamera(np.eye(3),np.zeros(3)))
            self.assertEqual(result['status'],'inconsistent')
            self.assertAlmostEqual(result['median_m'],.025,places=6)
            self.assertEqual(result['threshold_m'],.010)

    def test_missing_neck_is_unverified_not_passed(self):
        front,_=neck_plane(0,'front');side=copy.deepcopy(front)
        side.parsing[:]=0
        result=check_neck_consistency(front,side,RigidCamera(np.eye(3),np.zeros(3)))
        self.assertEqual(result['status'],'insufficient_overlap')
        self.assertIsNone(result['median_m']);self.assertTrue(result['warning'])

    def test_out_of_view_neck_is_unverified_not_passed(self):
        front,_=neck_plane(0,'front');side=copy.deepcopy(front)
        result=check_neck_consistency(front,side,RigidCamera(np.eye(3),np.array([2.,0.,0.])))
        self.assertEqual(result['status'],'insufficient_overlap')
        self.assertEqual(result['front_to_side']['overlap_samples'],0)
        self.assertEqual(result['side_to_front']['overlap_samples'],0)


if __name__=='__main__':unittest.main()
