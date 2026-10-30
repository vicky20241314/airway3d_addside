"""Export public aggregate head-combination statistics, never private geometry."""
import argparse
import json
from pathlib import Path
from a3_geometry import dump,sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    results = []
    for root in (a.b,a.baseline):
        report = json.loads((root/'report.json').read_text(encoding='utf-8'))
        for name,want in report['code_sha256'].items():
            assert sha(root/'source_snapshot'/name)==want
            assert sha(Path(__file__).with_name(name))==want
        for state,info in report['states'].items():
            assert info['fit_guard']['passed']
            assert info['fit_guard']['inverted_triangles']==0
            assert info['frozen_region_max_change_m']==0.
            assert info['shared_correction_error_m']==0.
            assert info['topology']['nonmanifold_edges']==0
            assert sha(root/state/'head_combined_m.obj')==info['mesh_sha256']
        results.append({k:report[k] for k in ('method','head_source','legacy_baseline_commit',
            'model_sha256','authorization_evidence','code_sha256','source_model_weights_loaded',
            'TUFA_or_segmentation_executed','budget','states','diagnostics','elapsed_s',
            'device','cuda_used','full_head_shape_present','full_head_scan_accuracy_verified','limitations','publication')})
    dump(a.output,dict(schema='head_rgbd.head_combination_public_aggregate.v1',date='2026-10-30',
        purpose='local noncommercial research under user-confirmed academic model license',
        prototype_status='full-head geometric export and guards passed; full-head accuracy unverified',
        tests_passed=6,prior_A3_tests_passed=13,results=results,
        timing_scope='CPU input loading, fitting, export and same-data diagnostics; excludes historic model fitting, A3 preparation and preview rendering',
        unobserved_rear_scalp_ears='model prior, not measured full-head scanning',
        private_geometry_or_pose_arrays_included=False,
        commercialization_or_model_redistribution_clearance=False))
    print('Verified source snapshots, mesh hashes, guards and shared/frozen regions; wrote aggregate statistics only')


if __name__=='__main__':main()
