"""Replay saved observations, or read a legacy pair without altering sources."""
import argparse
import os
import math
import json
import hashlib
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('UNIFACE_CACHE_DIR',str(ROOT/'models/uniface'))
from core.multiview_data import load_session,load_legacy_pair
from core.fit_options import FitOptions
from core.pipeline import run_multiview


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--session',type=Path);group.add_argument('--legacy-pair',type=Path)
    group.add_argument('--fitted-run',type=Path,help='replay only experimental shared canonical correction from saved stage2, on CPU')
    p.add_argument('--output',type=Path,default=ROOT/'output')
    p.add_argument('--front-only',action='store_true')
    p.add_argument('--compare-front',action='store_true',help='run both front-only and all observations, stage2 only')
    p.add_argument('--quick',action='store_true',help='integration smoke settings, NOT quality validation')
    p.add_argument('--stage2-only',action='store_true',help='disable joint refinement')
    p.add_argument('--shape-reg',type=float,help='positive identity prior weight; default stays 1e-6')
    p.add_argument('--config',type=Path,help='JSON object of FitOptions overrides')
    p.add_argument('--experiments',type=Path,help='JSON list: label, options, front_only, refine; one common frontal initialization')
    p.add_argument('--pose-strategy',choices=('joint','prior','fixed'))
    p.add_argument('--resume',action='store_true',help='resume completed labels from the same experiments/session/configuration')
    p.add_argument('--geometry-maps',action='store_true',help='save local geometry displacement PLY maps (0..6 mm), no accuracy claim')
    a=p.parse_args()
    if a.fitted_run:
        if a.config or a.quick or a.front_only or a.compare_front or a.pose_strategy or a.experiments or a.shape_reg is not None or a.resume:
            p.error('--fitted-run inherits fitted identity/state/cameras/configuration; fitting overrides are not accepted')
        from core.canonical_shape_refine import replay_saved_shape
        print(replay_saved_shape(a.fitted_run,a.output),flush=True);return
    if a.quick:print('注意：--quick 仅检查流程；迭代与采样大幅减少，结果不能用于判断相似度或精度。',flush=True)
    observations=load_session(a.session) if a.session else load_legacy_pair(a.legacy_pair)
    def input_digest(rows):
        h=hashlib.sha256()
        for o in sorted(rows,key=lambda o:o.view_id):
            h.update(json.dumps(dict(view_id=o.view_id,state=o.state,view=o.view,quality=o.quality),sort_keys=True).encode())
            for value in (o.rgb,o.depth,o.uv,o.xyz,o.valid,o.parsing,o.points,
                          np.array([o.camera.fx,o.camera.fy,o.camera.cx,o.camera.cy])):
                if value is None:h.update(b'None');continue
                a=np.ascontiguousarray(value);h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        return h.hexdigest()
    import numpy as np
    digest=input_digest(observations)
    options=FitOptions(stage1_outer=1,stage2_outer=1,inner_steps=2,max_points=500) if a.quick else FitOptions()
    if a.config:options=replace(options,**json.loads(a.config.read_text(encoding='utf-8-sig')))
    if a.pose_strategy:options=replace(options,pose_strategy=a.pose_strategy)
    if a.shape_reg is not None:
        if not math.isfinite(a.shape_reg) or a.shape_reg<=0:p.error('--shape-reg must be finite and positive')
        options=replace(options,shape_reg=a.shape_reg)
    experiments=json.loads(a.experiments.read_text(encoding='utf-8-sig')) if a.experiments else [dict(label='front_only' if f else 'all_views',front_only=f) for f in ([True,False] if a.compare_front else [a.front_only])]
    if a.compare_front or a.experiments:
        options=replace(options,front_initialization_cache=str(a.output.resolve()/'front_initialization.npz'))
    from core.pipeline import write_json
    runs=[];a.output.mkdir(parents=True,exist_ok=True)
    index=a.output/'replay_runs.json'
    if index.exists():
        if not a.resume:p.error('Output index already exists; use a new directory or --resume')
        saved=json.loads(index.read_text(encoding='utf-8-sig'))
        if saved['session']!=str(a.session or a.legacy_pair) or saved['smoke_test']!=a.quick:
            p.error('Resume input/mode mismatch')
        runs=saved['runs']
        if runs and input_digest(load_session(Path(runs[0]['folder'])/'capture/session.json'))!=digest:
            p.error('Resume source observations changed')
    for experiment in experiments:
        current=replace(options,**experiment.get('options',{}));current.schedule()
        previous=next((r for r in runs if r['label']==experiment['label']),None)
        if previous:
            from dataclasses import asdict
            report=json.loads((Path(previous['folder'])/'report.json').read_text(encoding='utf-8'))
            # JSON represents tuples as lists; compare the serialized form.
            expected=json.loads(json.dumps(asdict(current)))
            if report['status']!='complete' or report['fit_configuration']!=expected:p.error('Resume configuration mismatch')
            print('resume completed',experiment['label'],flush=True);continue
        result=run_multiview(observations,output_root=a.output,options=current,front_only=experiment.get('front_only',False),
            refine=experiment.get('refine',not (a.stage2_only or a.compare_front)),smoke_test=a.quick,progress=lambda s:print(s,flush=True))
        runs.append(dict(label=experiment['label'],folder=str(Path(result['folder']).resolve())))
        if Path(result['folder']).is_dir():
            write_json(a.output/'replay_runs.json',dict(session=str(a.session or a.legacy_pair),input_sha256=digest,runs=runs,smoke_test=a.quick))
            if a.geometry_maps:
                from core.multiview_diagnostics import save_geometry_changes
                save_geometry_changes(result['folder'])
        print(experiment['label'],result['folder'],flush=True)


if __name__=='__main__':main()
