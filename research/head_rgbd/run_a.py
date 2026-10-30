"""Python 3.6-compatible author sample geometry launcher, with isolated outputs."""
from __future__ import print_function
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frontend-only',action='store_true')
    p.add_argument('--resume-input',type=Path,help='Copy a prior task-owned prepared input, preserve that run')
    p.add_argument('--resume-from-step',type=int,choices=[2,3,4,5],default=2)
    p.add_argument('--train-steps',type=int,help='Explicit step5 flow-check budget; author default remains 100')
    p.add_argument('--input-dir',type=Path,help='Task-owned calibrated input after author sample check')
    p.add_argument('--camera-json',type=Path)
    p.add_argument('--preserve-upright',action='store_true')
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args();root=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(str(a.resume_input if a.resume_input else a.input_dir if a.input_dir else root/'test_data/RGBD/test1'),str(out/'input'))
    env=os.environ.copy();env['PATH']=str(Path(os.sys.prefix)/'bin')+os.pathsep+env.get('PATH','')
    env['CUDA_VISIBLE_DEVICES']='-1';env['OMP_NUM_THREADS']='4';env['TF_NUM_INTRAOP_THREADS']='4';env['TF_NUM_INTEROP_THREADS']='2'
    env['PYTHONUNBUFFERED']='1'
    if a.camera_json:env['HEAD_RGBD_CAMERA_JSON']=str(a.camera_json.resolve())
    if a.preserve_upright:env['HEAD_RGBD_ALLOW_AUTHOR_TRANSPOSE']='0'
    if a.frontend_only:
        cmd=[os.sys.executable,'-u','run_data_preparation.py','--GPU_NO=-1','--mode=test_RGBD',
            '--pb_path='+str(root/'resources/PB/'),'--img_dir='+str(out/'input')+'/', '--out_dir='+str(out/'input/prepare')]
        cwd=root/'data_prepare'
    else:
        text=(root/'run_opt_rgbd.sh').read_text()
        marker='\nif [ $is_bfm == "False" ];then\n    echo "start generate HD texture";'
        if marker not in text:raise RuntimeError('Author script changed; inspect before running')
        text=text[:text.index(marker)]
        text=text.replace('ROOT_DIR=$(pwd)/test_data/RGBD/test1/;',"ROOT_DIR='"+str(out/'input')+"/';")
        text=text.replace('GPU_NO=0;','GPU_NO=-1;')
        if a.train_steps is not None:
            if a.train_steps<1:raise ValueError('train-steps must be positive')
            text=text.replace('train_step=100','train_step='+str(a.train_steps))
        if a.resume_input:
            prefix=text[:text.index('echo "prepare datas";')]
            # Prefix contains only global path/asset variables, not data preparation.
            marker='echo "step '+str(a.resume_from_step)+':'
            body=text[text.index(marker):]
            text=prefix+'\nprefit_dir="$ROOT_DIR/prefit"\nprepare_dir="$ROOT_DIR/prepare"\ncd ./optimization/rgbd\n'+body
        text+='\necho GEOMETRY_STAGES_COMPLETE\n'
        (out/'geometry_only.sh').write_text(text,encoding='utf-8')
        cmd=['bash',str(out/'geometry_only.sh')];cwd=root
    print('Author invocation:',json.dumps(cmd),flush=True);started=time.perf_counter()
    author_code=None
    try:
        with (out/'author.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=str(cwd),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:author_code=process.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        code=author_code;error=None
    except subprocess.TimeoutExpired:
        code=124;error='Author process exceeded time budget'
    meshes=[str(f.relative_to(out)) for f in out.rglob('*.obj')]
    if code==0 and not a.frontend_only and not meshes:
        code=3;error='Author script returned zero but produced no geometry'
    report=dict(route='A author geometry path',author_commit='d5accd138359ad534662805cd710f58de4ee4c74',
        command=cmd,exit_code=code,author_exit_code=author_code,error=error,elapsed_s=time.perf_counter()-started,
        device='CPU TensorFlow 1.15.5 under WSL',frontend_only=a.frontend_only,
        status='author_frontend_complete' if code==0 and a.frontend_only else 'geometry_script_complete' if code==0 else 'blocked',
        geometry_files=meshes,cuda_peak_allocated_mib=0,
        resumed_from=str(a.resume_input) if a.resume_input else None,
        step5_train_steps=a.train_steps if a.train_steps is not None else 100,
        run_purpose='flow_check_only' if a.train_steps is not None else 'author_default_budget',
        real_data_inputs=bool(a.input_dir),real_data_adapted=bool(a.input_dir and code==0),real_side_effect_validated=False,
        camera_json=str(a.camera_json) if a.camera_json else None,preserve_upright=a.preserve_upright,
        log_sha256=hashlib.sha256((out/'author.log').read_bytes()).hexdigest())
    with (out/'report.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2));print((out/'author.log').read_text(errors='replace')[-6000:])
    return code


if __name__=='__main__':raise SystemExit(main())
