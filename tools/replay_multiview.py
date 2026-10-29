"""Replay saved observations, or read a legacy pair without altering sources."""
import argparse
import os
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
    a=p.parse_args()
    observations=load_session(a.session) if a.session else load_legacy_pair(a.legacy_pair)
    options=FitOptions(stage1_outer=1,stage2_outer=1,inner_steps=2,max_points=500) if a.quick else FitOptions()
    choices=[True,False] if a.compare_front else [a.front_only]
    for front_only in choices:
        result=run_multiview(observations,output_root=a.output,options=options,front_only=front_only,progress=lambda s:print(s,flush=True))
        print(('front_only' if front_only else 'all_views'),result['folder'],flush=True)


if __name__=='__main__':main()
