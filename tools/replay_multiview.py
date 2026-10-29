"""Replay saved observations, or read a legacy pair without altering sources."""
import argparse
import os
import math
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
    p.add_argument('--output',type=Path,default=ROOT/'output')
    p.add_argument('--front-only',action='store_true')
    p.add_argument('--compare-front',action='store_true',help='run both front-only and all observations, stage2 only')
    p.add_argument('--quick',action='store_true',help='integration smoke settings, NOT quality validation')
    p.add_argument('--stage2-only',action='store_true',help='disable joint refinement')
    p.add_argument('--shape-reg',type=float,help='positive identity prior weight; default stays 1e-6')
    a=p.parse_args()
    if a.quick:print('注意：--quick 仅检查流程；迭代与采样大幅减少，结果不能用于判断相似度或精度。',flush=True)
    observations=load_session(a.session) if a.session else load_legacy_pair(a.legacy_pair)
    options=FitOptions(stage1_outer=1,stage2_outer=1,inner_steps=2,max_points=500) if a.quick else FitOptions()
    if a.shape_reg is not None:
        if not math.isfinite(a.shape_reg) or a.shape_reg<=0:p.error('--shape-reg must be finite and positive')
        options=replace(options,shape_reg=a.shape_reg)
    choices=[True,False] if a.compare_front else [a.front_only]
    for front_only in choices:
        result=run_multiview(observations,output_root=a.output,options=options,front_only=front_only,
            refine=not (a.stage2_only or a.compare_front),smoke_test=a.quick,progress=lambda s:print(s,flush=True))
        print(('front_only' if front_only else 'all_views'),result['folder'],flush=True)


if __name__=='__main__':main()
