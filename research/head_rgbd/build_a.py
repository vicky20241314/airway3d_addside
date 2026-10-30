"""Python 3.6-compatible native author renderer build, with exact diagnostics."""
from __future__ import print_function
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import tensorflow as tf
    root=a.source.resolve();kernels=root/'third_party/kernels'
    compiler=Path(os.sys.prefix)/'bin/x86_64-conda-linux-gnu-c++'
    cmd=[str(compiler),'-std=c++11','-shared','-fPIC','-O2']
    cmd += [str(kernels/f) for f in ['rasterize_triangles_grad.cc','rasterize_triangles_op.cc','rasterize_triangles_impl.cc']]
    target=kernels/'rasterize_triangles_kernel.so'
    cmd += ['-o',str(target)] + tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    started=time.perf_counter();print('Build argv:',json.dumps(cmd),flush=True)
    run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    (a.output/'compiler.log').write_text(run.stdout,encoding='utf-8')
    report=dict(tensorflow=tf.__version__,python=os.sys.version,command=cmd,exit_code=run.returncode,
        elapsed_s=time.perf_counter()-started,source=str(root),renderer_loaded=False,
        platform='WSL Linux CPU; author GPU timing not reproduced')
    if run.returncode==0:
        try:
            tf.load_op_library(str(target));report['renderer_loaded']=True
            report['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
        except Exception as exc: report['load_error']=str(exc)
    with (a.output/'build.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2));print(run.stdout[-5000:])
    return 0 if report['renderer_loaded'] else 2


if __name__=='__main__':raise SystemExit(main())
