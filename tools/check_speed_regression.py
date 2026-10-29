"""Compare saved full runs with identical inputs/options; no accuracy claim."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from core.multiview_data import load_session
from core.pipeline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    old=json.loads((a.baseline/'report.json').read_text(encoding='utf-8'))
    new=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    assert old['status']==new['status']=='complete'
    assert old['fit_configuration']==new['fit_configuration']
    assert old['stage3_enabled']==new['stage3_enabled']
    assert old['view_ids']==new['view_ids']
    before={o.view_id:o for o in load_session(a.baseline/'capture/session.json')}
    after={o.view_id:o for o in load_session(a.run/'capture/session.json')}
    assert before.keys()==after.keys()
    for key,b in before.items():
        c=after[key]
        for name in ('rgb','depth','uv','xyz','valid'):np.testing.assert_array_equal(getattr(b,name),getattr(c,name))
        np.testing.assert_array_equal([b.camera.fx,b.camera.fy,b.camera.cx,b.camera.cy],
                                      [c.camera.fx,c.camera.fy,c.camera.cx,c.camera.cy])
    rows=[]
    # Declared numerical-regression tolerances, well below depth sensor noise.
    for state in ('closed','open'):
        for stage,limit in (('stage2',.000005),('stage3',.00002)):
            if stage=='stage3' and not new['stage3_enabled']:continue
            name=f'mesh_{state}_stage2.obj' if stage=='stage2' else f'mesh_{state}.obj'
            b=trimesh.load(a.baseline/state/name,process=False,force='mesh')
            c=trimesh.load(a.run/state/name,process=False,force='mesh')
            np.testing.assert_array_equal(b.faces,c.faces)
            delta=np.linalg.norm(c.vertices-b.vertices,axis=1)
            rows.append(dict(state=state,stage=stage,max_vertex_delta_mm=float(delta.max()*1000),
                mean_vertex_delta_mm=float(delta.mean()*1000),limit_mm=limit*1000,passed=bool(delta.max()<=limit)))
    def compare_values(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:compare_values(x[key],y[key])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for b,c in zip(x,y):compare_values(b,c)
        elif isinstance(x,(int,float)) and not isinstance(x,bool):
            np.testing.assert_allclose(x,y,rtol=1e-4,atol=1e-4,equal_nan=True)
        else:assert x==y
    compare_values(json.loads((a.baseline/'metrics.json').read_text(encoding='utf-8')),
                   json.loads((a.run/'metrics.json').read_text(encoding='utf-8')))
    out=a.output/('speed_regression_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    result=dict(baseline=str(a.baseline.resolve()),run=str(a.run.resolve()),
        baseline_seconds=old['elapsed_s'],accelerated_seconds=new['elapsed_s'],
        observed_speedup=old['elapsed_s']/new['elapsed_s'],identical_inputs_and_options=True,
        metrics_within_tolerance=True,geometry=rows,passed=all(r['passed'] for r in rows),
        independent_accuracy_validated=False)
    write_json(out/'comparison.json',result);print(json.dumps(result),flush=True);print(out,flush=True)
    if not result['passed']:raise AssertionError('Mesh drift exceeds declared numerical tolerance')


if __name__=='__main__':main()
