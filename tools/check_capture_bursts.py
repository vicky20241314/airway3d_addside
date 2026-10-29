"""Recheck real raw capture attempts without changing their files."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from core.camera import Camera
from core.aruco import detect_aruco
from core.temporal_capture import prepare_burst
from core.multiview_data import save_session
from core.pipeline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    out=a.output/('burst_checks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for meta in sorted(a.capture.glob('attempt_*/attempt.json')):
        original=json.loads(meta.read_text(encoding='utf-8'));path=meta.parent/'raw_sequence.npz'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        old=original.get('temporal_diagnostics',original.get('quality',{}).get('temporal',{}))
        row=dict(attempt=meta.parent.name,view_id=original['view_id'],old_status=original['status'],
            old_accepted_frames=old.get('accepted_frames'),old_error=original.get('error'),input_sha256=digest)
        try:
            with np.load(path,allow_pickle=False) as raw:
                n=len(raw['rgb']);camera=Camera(*raw['intrinsics'][0])
                frames=[(original['state'],original['view'],raw['rgb'][i],raw['depth'][i],raw['timestamps_s'][i],raw['intrinsics'][i]) for i in range(n)]
                obs=prepare_burst(frames,camera)
            row.update(temporal_status='passed',quality=obs.quality)
            save_session(out/meta.parent.name,[obs]);obs.validate()
            if obs.state=='closed' and obs.view=='front':
                _,_,adam=detect_aruco(obs.rgb,obs.depth,obs.camera)
                if adam is None or not np.isfinite(adam).all():raise ValueError('正面闭口 ArUco 深度无效')
            row['capture_status']='passed'
        except ValueError as exc:
            row.update(capture_status='rejected',error=str(exc))
            if 'temporal_status' not in row:row.update(temporal_status='rejected',diagnostics=getattr(exc,'diagnostics',{}))
        assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
        print(row['attempt'],row['view_id'],row['temporal_status'],row['capture_status'],row.get('error',''),flush=True)
        write_json(out/'report.json',dict(source=str(a.capture.resolve()),attempts=rows,input_bytes_unchanged=True,
            temporal_counts=dict(Counter(r['temporal_status'] for r in rows)),
            capture_counts=dict(Counter(r['capture_status'] for r in rows)),independent_accuracy_validated=False))
    print(out,flush=True)


if __name__=='__main__':main()
