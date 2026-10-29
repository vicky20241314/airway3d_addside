"""Reprocess saved per-view raw bursts into a NEW consensus session."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from core.camera import Camera
from core.temporal_capture import prepare_burst
from core.multiview_data import validate_session,save_session
from core.pipeline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output');a=p.parse_args()
    chosen={}
    for path in sorted(a.capture.glob('attempt_*/attempt.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('status')=='accepted':chosen[(row['state'],row['view'])]=(path.parent,row)
    if not chosen:raise ValueError('没有保留原始短序列的成功采集；融合后的一帧不能恢复成真实多帧')
    folder=a.output/('temporal_capture_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',source=str(a.capture.resolve()),version=(ROOT/'VERSION').read_text().strip(),attempts=[])
    write_json(folder/'report.json',report);observations=[]
    try:
        for (state,view),(path,row) in chosen.items():
            if row.get('motion_mode')!='camera_moved':raise ValueError('原始采集必须声明移动相机、保持头颈姿态')
            with np.load(path/'raw_sequence.npz',allow_pickle=False) as raw:
                n=len(raw['rgb']);camera=Camera(*raw['intrinsics'][0])
                frames=[(state,view,raw['rgb'][i],raw['depth'][i],raw['timestamps_s'][i],raw['intrinsics'][i]) for i in range(n)]
                obs=prepare_burst(frames,camera)
            obs.quality['source_attempt']=str(path.resolve());observations.append(obs)
            report['attempts'].append(dict(source=str(path.resolve()),view_id=obs.view_id,quality=obs.quality))
            print(f'{obs.view_id}: {obs.quality["frames"]}/{n} stable frames',flush=True)
        observations=validate_session(observations)
        manifest=save_session(folder/'complete',observations)
        report.update(status='complete',session=str(manifest.resolve()))
        write_json(folder/'report.json',report);print(manifest,flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc),temporal_diagnostics=getattr(exc,'diagnostics',{}))
        write_json(folder/'report.json',report);raise


if __name__=='__main__':main()
