"""Exercise actual finish_meshes on a saved paired stage2; no fitting bypass in app."""
import sys,os,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import trimesh
from core.camera import Camera
from core.pipeline import finish_meshes
from core.cross_refine import DEFAULT_PROFILE,ALGORITHMS

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cap=np.load(a.run/'capture_pair.npz');prep=np.load(a.run/'prepared.npz')
    landmarks=np.load(a.run/'stage2_landmarks.npz');camera=Camera(*cap['intrinsics'])
    stage2=[]
    for state in ('closed','open'):
        m=trimesh.load(a.run/state/f'mesh_{state}_stage2.obj',force='mesh',process=False)
        m.metadata.update(geometry_stage=2,flame_landmarks_3d=landmarks[state])
        stage2.append(m)
    before=[m.vertices.copy() for m in stage2]
    out=ROOT/'validation/cross_v3_smooth/pipeline';out.mkdir(parents=True,exist_ok=True)
    metrics,snapshots,display,diagnostics=finish_meshes(stage2,
        [cap['depth_'+s] for s in ('closed','open')],camera,
        [prep['parsing_'+s] for s in ('closed','open')],
        [prep['lmk2d_'+s] for s in ('closed','open')],prep['adam_3d'],
        [prep['lmk3d_'+s] for s in ('closed','open')],out,
        progress=lambda x:print(x.encode('ascii','backslashreplace').decode(),flush=True))
    for i,s in enumerate(('closed','open')):
        assert np.array_equal(before[i],stage2[i].vertices)
        assert np.array_equal(before[i],snapshots[i].vertices)
        assert diagnostics[i]['algorithm']==ALGORITHMS[DEFAULT_PROFILE]
        assert display[i].metadata['geometry_stage']==3
        with np.load(out/s/'vertex_offsets.npz') as d:
            assert np.allclose(d['reference_vertices']+d['offsets'],display[i].vertices,atol=1e-12,rtol=0)
        expected=ROOT/'validation/cross_v3_smooth'/a.run.name/s/'vertex_offsets.npz'
        with np.load(expected) as d:
            assert np.allclose(d['reference_vertices']+d['offsets'],display[i].vertices,atol=1e-12,rtol=0)
    report=dict(algorithm=ALGORITHMS[DEFAULT_PROFILE],actual_finish_meshes_passed=True,
        stage2_unchanged=True,paired_inputs_match_independent_checks=True,
        metric_stage=2,display_stage=3,hardware_tested=False,
        scope='Actual stage2-to-stage3 pipeline and saving; stage1/stage2 not rerun in this check')
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
