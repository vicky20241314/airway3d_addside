"""Independent synthetic frame noise/motion; no real multi-frame effect claim."""
import copy
import tempfile
import unittest
import cv2
import numpy as np
from core.camera import Camera
from core.multiview_data import save_session,load_session
from core.multiview_geometry import project
from core.temporal_capture import prepare_burst,BurstQualityError


def burst_fixture(n=15):
    rng=np.random.default_rng(9);camera=Camera(200,200,32,32)
    uv=np.column_stack((np.linspace(20,44,68),32+4*np.sin(np.arange(68))))
    uv[[50,51,52],1]=28;uv[[56,57,58],1]=36
    pattern=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    frames=[];landmarks=[]
    for i in range(n):
        rgb=pattern.copy();rgb[0,0,0]=i
        depth=(600+rng.integers(-2,3,(64,64))).astype(np.uint16)
        frames.append(('closed','front',rgb,depth,1+i*.03,np.array([200,200,32,32])))
        landmarks.append(uv.copy())
    def detect(rgb):return np.array([8,8,48,48]),landmarks[int(rgb[0,0,0])]
    return frames,camera,uv,landmarks,detect


class TemporalCaptureTests(unittest.TestCase):
    def test_multiple_frames_reduce_noise_and_reject_bad_middle_frame(self):
        frames,camera,truth,lm,detect=burst_fixture()
        lm[7]+=10
        obs=prepare_burst(frames,camera,detector=detect)
        self.assertEqual(obs.quality['frames'],14)
        self.assertFalse(obs.temporal['accepted'][7])
        self.assertNotEqual(obs.quality['temporal']['representative_rgb_index'],7)
        np.testing.assert_allclose(obs.uv,truth,atol=2e-6)
        self.assertTrue(obs.valid.all())
        self.assertLess(np.abs(obs.depth.astype(float)-600).mean(),np.abs(frames[0][3].astype(float)-600).mean())
        uv,_=project(obs.xyz,camera);np.testing.assert_allclose(uv,obs.uv,atol=1e-5)
        with tempfile.TemporaryDirectory() as folder:
            loaded=load_session(save_session(folder,[obs]))[0]
            for key,value in obs.temporal.items():np.testing.assert_array_equal(loaded.temporal[key],value)

    def test_camera_motion_without_a_stable_majority_is_rejected(self):
        frames,camera,_,lm,detect=burst_fixture()
        for i in range(15):lm[i][:,0]+=i*.7
        with self.assertRaises(BurstQualityError) as error:prepare_burst(frames,camera,detector=detect)
        self.assertLess(error.exception.diagnostics['accepted_frames'],9)

    def test_low_support_and_unstable_depth_are_not_fused(self):
        frames,camera,_,lm,detect=burst_fixture()
        for i,f in enumerate(frames):
            f[3][8:12,8:12]=600 if i<2 else 0
            f[3][12:16,8:12]=600+((-20,0,20)[i%3])
            f[3][16:20,8:12]=700 if i==7 else 600
        obs=prepare_burst(frames,camera,detector=detect)
        self.assertFalse(obs.depth[8:16,8:12].any())
        np.testing.assert_array_equal(obs.depth[16:20,8:12],600)

    def test_landmark_votes_preserve_strict_front_and_partial_side(self):
        frames,camera,_,lm,detect=burst_fixture()
        for i in range(10):lm[i][33]=np.nan
        obs=prepare_burst(frames,camera,detector=detect)
        self.assertFalse(obs.valid[33])
        with self.assertRaises(ValueError):obs.validate()
        obs.view='left';obs.view_id='closed_left';obs.validate()

    def test_blur_and_mouth_changes_are_excluded(self):
        frames,camera,_,lm,detect=burst_fixture()
        frames[0][2][8:56,8:56]=cv2.GaussianBlur(frames[0][2][8:56,8:56],(15,15),5)
        for i in range(1,5):lm[i][[56,57,58],1]+=4
        obs=prepare_burst(frames,camera,detector=detect)
        self.assertEqual(obs.quality['frames'],10)
        self.assertEqual(obs.quality['temporal']['frames'][0]['reason'],'blurred_relative_to_burst')
        self.assertTrue(all(obs.quality['temporal']['frames'][i]['reason']=='mouth_state_changed' for i in range(1,5)))

    def test_frame_choices_and_cross_view_intrinsics_time_guards(self):
        for n in (5,9,15):
            frames,camera,_,_,detect=burst_fixture(n)
            self.assertEqual(prepare_burst(frames,camera,detector=detect).quality['frames'],n)
        frames,camera,_,_,detect=burst_fixture()
        for kind in ('view','intrinsics','time'):
            bad=copy.deepcopy(frames);row=list(bad[4])
            if kind=='view':row[1]='left'
            elif kind=='intrinsics':row[5][0]+=1
            else:row[4]=bad[3][4]
            bad[4]=tuple(row)
            with self.assertRaises(ValueError):prepare_burst(bad,camera,detector=detect)

    def test_offline_raw_burst_reprocessing_preserves_inputs(self):
        import hashlib,json,sys
        from pathlib import Path
        from unittest.mock import patch
        from tools.prepare_multiframe_capture import main
        frames,camera,_,_,detect=burst_fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);capture=root/'capture';hashes={}
            for state in ('closed','open'):
                attempt=capture/f'attempt_{state}';attempt.mkdir(parents=True)
                (attempt/'attempt.json').write_text(json.dumps(dict(status='accepted',state=state,view='front',motion_mode='camera_moved')),encoding='utf-8')
                path=attempt/'raw_sequence.npz'
                np.savez_compressed(path,rgb=np.stack([f[2] for f in frames]),depth=np.stack([f[3] for f in frames]),
                    timestamps_s=[f[4] for f in frames],intrinsics=np.stack([f[5] for f in frames]))
                hashes[path]=hashlib.sha256(path.read_bytes()).digest()
            with patch.object(sys,'argv',['prepare','--capture',str(capture),'--output',str(root/'results')]), \
                    patch('core.temporal_capture.detect_frame',side_effect=detect):main()
            manifest=next((root/'results').glob('*/complete/session.json'))
            observations=load_session(manifest)
            self.assertEqual({o.state for o in observations},{'closed','open'})
            self.assertTrue(all(o.temporal['accepted'].sum()==15 for o in observations))
            for path,digest in hashes.items():self.assertEqual(hashlib.sha256(path.read_bytes()).digest(),digest)


if __name__=='__main__':unittest.main()
