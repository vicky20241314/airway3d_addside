"""Replay the unchanged v0.12.0 baseline with isolated output and CUDA accounting."""
import argparse
import os
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from audit import dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-cache',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--front-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['UNIFACE_CACHE_DIR']=str(a.model_cache)
    torch.set_num_threads(4)
    from core.multiview_data import load_session
    from core.pipeline import run_multiview
    from core.fit_options import FitOptions
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    started=time.perf_counter()
    result=run_multiview(load_session(a.session),output_root=a.output,device=a.device,
        options=FitOptions(),refine=False,front_only=a.front_only,progress=lambda x:print(x,flush=True))
    if a.device=='cuda':torch.cuda.synchronize()
    dump(a.output/'benchmark.json',dict(baseline_commit='3e409af77a4686b0cf435fff2230b90a8d26e11d',
        baseline_version='v0.12.0',stage='stage2_only',device=a.device,front_only=a.front_only,
        elapsed_s=time.perf_counter()-started,result_folder=str(result['folder']),
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else 0,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if a.device=='cuda' else 0,
        session_sha256=sha(a.session),used_view_ids=result['used_view_ids'],excluded_views=result['excluded_views'],
        real_side_effect_validated=False))
    print('Baseline finished: '+str(result['folder']),flush=True)


if __name__=='__main__':main()
