"""Python 3.6-compatible author sample geometry launcher, with isolated outputs."""
from __future__ import print_function
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frontend-only',action='store_true')
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args();root=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(str(root/'test_data/RGBD/test1'),str(out/'input'))
    env=os.environ.copy();env['PATH']=str(Path(os.sys.prefix)/'bin')+os.pathsep+env.get('PATH','')
    env['CUDA_VISIBLE_DEVICES']='-1';env['OMP_NUM_THREADS']='4';env['TF_NUM_INTRAOP_THREADS']='4';env['TF_NUM_INTEROP_THREADS']='2'
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
        text+='\necho GEOMETRY_STAGES_COMPLETE\n'
        (out/'geometry_only.sh').write_text(text,encoding='utf-8')
        cmd=['bash',str(out/'geometry_only.sh')];cwd=root
    print('Author invocation:',json.dumps(cmd),flush=True);started=time.perf_counter()
    try:
        with (out/'author.log').open('w',encoding='utf-8') as log:
            run=subprocess.run(cmd,cwd=str(cwd),env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
        code=run.returncode;error=None
    except subprocess.TimeoutExpired:
        code=124;error='Author process exceeded time budget'
    report=dict(route='A author geometry path',author_commit='d5accd138359ad534662805cd710f58de4ee4c74',
        command=cmd,exit_code=code,error=error,elapsed_s=time.perf_counter()-started,
        device='CPU TensorFlow 1.15.5 under WSL',frontend_only=a.frontend_only,
        status='author_frontend_complete' if code==0 and a.frontend_only else 'geometry_script_complete' if code==0 else 'blocked',
        geometry_files=[str(f.relative_to(out)) for f in out.rglob('*.obj')],
        real_data_adapted=False,real_side_effect_validated=False,
        log_sha256=hashlib.sha256((out/'author.log').read_bytes()).hexdigest())
    with (out/'report.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2));print((out/'author.log').read_text(errors='replace')[-6000:])
    return code


if __name__=='__main__':raise SystemExit(main())
