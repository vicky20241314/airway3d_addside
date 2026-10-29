"""Report front/side surface errors, reviewed silhouette errors and coverage.

Held-out extrinsics must be independently calibrated in state-front coordinates;
this tool never fits shape, displacement or camera pose to held-out data.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import trimesh
from core.multiview_data import load_session
from core.multiview_geometry import RigidCamera
from core.multiview_evaluation import assert_held_out,evaluate_view,summarize
from core.pipeline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--geometry',choices=['stage2','stage3'],default='stage2')
    p.add_argument('--held-out',type=Path);p.add_argument('--held-out-extrinsics',type=Path)
    p.add_argument('--reference-masks',type=Path,help='manually reviewed full model silhouette masks, VIEW_ID.png')
    p.add_argument('--output',type=Path,default=ROOT/'output')
    a=p.parse_args()
    if bool(a.held_out)!=bool(a.held_out_extrinsics):p.error('Held-out session and independently calibrated extrinsics must be supplied together')
    report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    if report['status']!='complete':raise ValueError('Only evaluate a completed run')
    source=Path(report['source']) if 'source' in report else a.run
    training=load_session(source/'prepared/session.json')
    ext=json.loads((source/'extrinsics.json').read_text(encoding='utf-8'))
    captured=training
    skipped=[o.view_id for o in captured if o.view_id not in ext]
    training=[o for o in captured if o.view_id in ext]
    held=load_session(a.held_out) if a.held_out else []
    assert_held_out(captured,held)
    if held:ext.update(json.loads(a.held_out_extrinsics.read_text(encoding='utf-8')))
    if any(t['direction']!='state_front_to_view' for t in ext.values()):raise ValueError('Expected state_front_to_view extrinsics')
    transforms={k:RigidCamera(t['R'],t['t']) for k,t in ext.items()}
    meshes={s:trimesh.load(a.run/s/(f'mesh_{s}_stage2.obj' if a.geometry=='stage2' else f'mesh_{s}.obj'),process=False,force='mesh') for s in ('closed','open')}
    rows=[]
    for role,observations in (('in_fit',training),('held_out',held)):
        for o in observations:
            if o.parsing is None:
                from core.mask_detection import get_face_data_single
                o.parsing=get_face_data_single(o.camera,o.rgb,o.depth)['parsing']
            mask=None
            if a.reference_masks and (a.reference_masks/(o.view_id+'.png')).exists():
                mask=cv2.imread(str(a.reference_masks/(o.view_id+'.png')),cv2.IMREAD_GRAYSCALE)
                if mask is None:raise ValueError('Cannot read reviewed reference mask')
                mask=mask>0
            row=evaluate_view(meshes[o.state],o,transforms[o.view_id],mask);row['role']=role;rows.append(row)
            print(role,o.view_id,row['surface'],flush=True)
    out=a.output/('mv_evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    value=dict(source=str(a.run.resolve()),geometry=a.geometry,views=rows,summary=summarize(rows),
        captured_but_not_fitted=skipped,
        independent_views_present=bool(held),reviewed_masks_complete=all(r['contour']['status']=='reviewed_mask' for r in rows),
        note='In-fit residuals are diagnostics. Independent calibrated oblique views and reviewed masks are required for real acceptance. No loss-based accuracy claim.')
    write_json(out/'evaluation.json',value);print(out,flush=True)


if __name__=='__main__':main()
