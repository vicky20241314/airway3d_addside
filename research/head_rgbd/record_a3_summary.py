"""Publish only whitelisted non-geometric statistics, NEVER a private report copy."""
import argparse
import json
from pathlib import Path

from a3_geometry import dump,sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--front-control',type=Path,required=True)
    p.add_argument('--checks',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    a = p.parse_args()
    for f in (a.summary,a.comparison):
        if f.exists():
            raise FileExistsError(f)
    report = json.loads((a.result/'report.json').read_text(encoding='utf-8'))
    front = json.loads((a.front_control/'report.json').read_text(encoding='utf-8'))
    verification = json.loads((a.checks/'verification.json').read_text(encoding='utf-8'))
    comparison = json.loads((a.checks/'comparison.json').read_text(encoding='utf-8'))
    assert verification['full']['status']=='passed' and verification['front_control']['status']=='passed'
    # These are the exact source bytes used for reconstruction, not a guessed git ref.
    for name,want in report['code_sha256'].items():
        if sha(Path(__file__).with_name(name))!=want:
            raise ValueError('current code differs from recorded run: '+name)
    outputs = {}
    geometry_keys = ('vertices','triangles','mesh_sha256','finite',
        'supported_at_least_one_fraction','supported_at_least_two_fraction',
        'actual_multiframe_fusion_fraction','unobserved_completion_vertices','connectivity_filter')
    for state,info in report['outputs'].items():
        outputs[state] = dict(reference_frame_id=info['reference_frame_id'],
            missing_views=info['missing_views'],unit=info['unit'],coordinates=info['coordinates'],
            geometry={stage:{k:data[k] for k in geometry_keys}
                      for stage,data in info['geometry'].items()})
    summary = dict(schema='head_rgbd.a3_public_aggregate.v1',date='2026-10-30',
        branch='codex/rgbd-head-a-adapt-20261030',research_base='159a894dcfb0760633723092133a2964f1fd6490',
        route=report['method'],author_method_reproduction=False,
        status=dict(implementation='independent research prototype implemented',
            real_pipeline='passed with explicit frame exclusion',
            real_geometry_effect_acceptance='NOT PASSED: visible holes/seam, lower side overlap and no independent truth',
            license_audit='execution inventory and evidence recorded; unclear assets isolated',
            commercial_and_redistribution_clearance='not established'),
        models_loaded=[],source_code_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*a3*.py')},
        fitted_manifest_sha256=report['manifest_sha256'],outputs=outputs,budget=report['budget'],
        frames=[dict(id=r['id'],state=r['state'],view=r['view'],accepted=r['fit']['accepted'],
                     annotation_origin=r['annotation_origin'],
                     exclusion_reason=r['fit']['reason'] if not r['fit']['accepted'] else None,
                     icp_iterations=r['fit'].get('iterations_executed',0)) for r in report['frames']],
        selected_candidate_count=len(report['frames']),accepted_count=len(report['active_frame_ids']),
        frame_policy='20 real frames in 4 short bursts; manual seeds + classical RGB NCC; 19 accepted, no up views; no long-sequence claim',
        performance=dict(full_elapsed_s=report['elapsed_s'],front_only_elapsed_s=front['elapsed_s'],
            full_peak_process_working_set_mib=report['performance']['peak_process_working_set_mib'],
            front_peak_process_working_set_mib=front['performance']['peak_process_working_set_mib'],
            device='CPU',cuda_used=False,gpu_memory='not applicable; no CUDA context',
            timing_scope='raw I/O, mask/landmark lifting, fitting, fusion and export; excludes manual seeds, NCC expansion and separate diagnostics/QA',
            NCC_expansion_time_s=None,NCC_time_status='not separately instrumented',
            comparison_equal_accuracy=False),
        export_verification=dict(full='passed',front_only='passed',
            shared_identity=verification['full']['shared_identity']),
        environment=report['environment'],independent_acceptance=report['acceptance'],
        limitations=report['limitations']+['multi-frame support does not mean independent angular coverage',
            'visible holes at nose/eyes/boundaries and expression seam persist',
            'side-jaw identity not merged across bursts; lower face remains reference-specific',
            'same-frame lower medians cannot establish improved scan accuracy'],
        failure_history=[dict(experiment='smoke-v1',issue='detached oral/depth islands, open Z extent 319mm',
            resolution='largest connected surface, preserve excluded data; no oral-completion claim'),
            dict(experiment='formal-v2/checks-v2',issue='shared fine upper identity leaked across expression seam, max ~0.83mm',
                 resolution='separate smoothing neighbourhoods, regression test, final exported identity agreement <1e-8m'),
            dict(experiment='formal-v4',issue='closed_right_0 has fewer than four valid 3D seed correspondences',
                 resolution='excluded, four other actual right frames retained')],
        publication=dict(private_RGB_depth_landmarks_poses_meshes_previews_included=False,
            contents='source, notices and aggregate numerical diagnostics only; data/commercial rights unresolved'))
    # Comparison source contains statistics only; deliberately omit local paths and
    # source/session identifiers in the historical benchmark metadata.
    public_comparison = dict(schema='head_rgbd.a3_public_comparison.v1',metric=comparison['metric'],
        independent_extrinsics=False,independent_reference_scan=False,manual_silhouette_truth=False,
        original_pixel_stride=comparison['original_pixel_stride'],frames=comparison['frames'],
        baseline_commit=comparison['baseline_commit'],
        historical_budgets=dict(baseline={k:v for k,v in comparison['baseline_budget'].items()
            if k in ('baseline_commit','baseline_version','stage','elapsed_s','device',
                     'cuda_peak_allocated_mib','cuda_peak_reserved_mib')},B=comparison['B_budget']),
        A3_budget=comparison['A3_budget'],A3_elapsed_s=comparison['A3_elapsed_s'],
        front_only_elapsed_s=comparison['front_only_elapsed_s'],
        timing_comparable_as_equal_accuracy=False,caution=comparison['caution'])
    dump(a.summary,summary)
    dump(a.comparison,public_comparison)
    print('Wrote whitelisted aggregate summaries; no RGB/depth/landmark coordinates/poses/mesh arrays copied')


if __name__=='__main__':
    main()
