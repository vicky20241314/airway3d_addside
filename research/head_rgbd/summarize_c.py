"""Publish a small auditable milestone record; keep all large artifacts external."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
from audit import dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--final-dir',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.runtime
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    final=a.final_dir or run/'final-visibility';comparison=read(final/'comparison/comparison.json')
    c=read(final/'guarded/report.json');front_path=Path(c['front_anchor']['path']);front=read(front_path/'report.json')
    verified=read(final/'guarded/verification.json');fv=read(front_path/'verification.json')
    assert verified['status']==fv['status']=='passed'
    assert len(verified['front_guard_checks'])==10
    refs=read(final/'guarded/reference_geometry.json')['references']
    trials={}
    for folder in ['comparison-c-trial-01','comparison-c-v2','final-run/comparison']:
        r=read(run/folder/'comparison.json');trials[folder]=dict(front_controls=r['front_controls'],timing=r['timing'],accepted_for_final=False)
    representative=[]
    for row in comparison['frames']:
        if 'baseline' not in row:continue
        representative.append(dict(frame_id=row['frame_id'],view=row['view'],state=row['state'],fitted_yaw_deg=row['fitted_yaw_deg'],
            baseline=row['baseline'],methods=row['methods']))
    artifacts=[]
    for folder in [final/'guarded',front_path,final/'comparison']:
        for path in sorted(folder.iterdir()):
            if path.is_file():artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    versions={}
    for name in ['numpy','scipy','torch','trimesh','rtree','smplx','opencv-python','matplotlib']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='see environment freeze for distribution alias'
    frame_keys=['id','segment','state','view','timestamp_s','source','source_frame','initial_yaw_deg','fitted_yaw_deg',
        'role','identity_keyframe','selection_reason','pose_initialization','landmark_depth_invalid','landmark_depth_outlier_rejected',
        'motion_landmark_visibility_or_outlier_rejected']
    side_rows=[r for r in comparison['frames'] if 'baseline' in r and r['view']!='front']
    open_rows=[r for r in comparison['frames'] if r['state']=='open']
    output=dict(schema='head_rgbd.milestone.c.v1',date='2026-10-30',
        implementation='complete',real_data_pipeline='complete',real_accuracy_acceptance='unverified',
        branch='codex/rgbd-head-reconstruction-20261030',parent_research_commit='159a894dcfb0760633723092133a2964f1fd6490',
        baseline_commit='3e409af77a4686b0cf435fff2230b90a8d26e11d',baseline_core_third_party_version_diff='verified empty',
        independent_worktree=str(Path(__file__).resolve().parents[2]),runtime=str(run),final_directory=str(final),
        shared_environment_readonly=sys.executable,python=platform.python_version(),versions=versions,
        environment_freeze_sha256=sha(run/'environment-c-freeze.txt'),
        hardware=dict(cpu='Ryzen 7 5800H',gpu='RTX 3060 Laptop',vram_mib=6144,camera='Orbbec Gemini 215'),
        sequence_sha256=c['sequence_sha256'],model_sha256=c['model_sha256'],source_hashes=c['source_hashes'],
        final_device=c['device'],final_cold_pipeline_elapsed_s=c['front_anchor']['cold_pipeline_elapsed_s'],
        final_timing=comparison['timing'],baseline_timing=comparison['baseline'],
        timing_note='own GPU jobs ran sequentially; final safeguarded run used CPU while another agent used GPU; CPU wall time is not an exclusive-machine benchmark',
        front_controls=comparison['front_controls'],front_guard=c['front_anchor'],front_guard_verification=verified['front_guard_checks'],
        side_motion_landmark_policy=c.get('side_motion_landmark_policy'),
        triangle_foldovers_final=verified['refined_triangle_foldovers'],safety_projection=c['safety_log'],
        training_effect_gates=dict(
            same_algorithm_front_preservation=all(r['coverage_identical'] and r['maximum_depth_change_mm']<.0001 for r in verified['front_guard_checks']),
            representative_side_p95_better_than_B=all(r['methods']['C']['full']['p95_mm']<=r['methods']['B']['full']['p95_mm'] for r in side_rows),
            all_open_p95_better_than_original_B_front=all(r['methods']['C']['full']['p95_mm']<=r['methods']['B_front']['full']['p95_mm'] for r in open_rows),
            scope='in-sample diagnosis; failed effect gates remain failed and do not become independent accuracy acceptance'),
        frames=[{k:r[k] for k in frame_keys if k in r} for r in c['frames']],
        excluded_frames=[r['id'] for r in c['frames'] if r['status']=='excluded'],
        temporal_edges=c['temporal_edges'],neck=c['neck'],support_label_counts=c['support_label_counts'],
        representative_diagnostics=representative,
        reference_geometry={state:{k:r[k] for k in ['frame_id','coordinates','coarse_inner_lip_gap_mm','refined_inner_lip_gap_mm','observed_inner_lip_gap_mm']} for state,r in refs.items()},
        trials=trials,failed_attempts=[dict(output=str(run/'c-front-final'),exit_code=1,reason='host-memory allocation failure before bounded-ray/cache cleanup; no final artifact accepted')],
        artifacts=artifacts,acceptance=comparison['acceptance'],limitations=c['limitations'])
    dump(a.output,output)
    print(json.dumps(dict(output=str(a.output),front_guard=output['front_guard'],front_controls=output['front_controls'],
        representative=[dict(frame=r['frame_id'],B=r['methods']['B']['full'],C=r['methods']['C']['full'],before=r['methods'].get('C_before_side_detail',{}).get('full')) for r in representative],
        support_label_counts=c['support_label_counts'],neck_components=len(c['neck'])),indent=2))


if __name__=='__main__':main()
