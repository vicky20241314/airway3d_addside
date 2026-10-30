"""Export a small, reviewable report; never copy RGB, depth, model or biometrics into Git."""
import argparse
import json
from pathlib import Path
from audit import dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.runtime
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    comparison=load(root/'comparison-final/comparison.json')
    b=load(root/'b-final/report.json');sequence=load(root/'sequence-v2/sequence.json')
    audit=load(root/'audit-initial/audit.json')
    data=dict(date='2026-10-30 Asia/Shanghai',hardware=dict(cpu='AMD Ryzen 7 5800H',
        gpu='NVIDIA GeForce RTX 3060 Laptop GPU',gpu_memory_mib=6144,driver='517.00',
        camera='Orbbec Gemini 215',ram_bytes=16540307456),comparison=comparison,
        preparation_s=sequence['elapsed_s'],input_hashes=sequence['sources'],
        selected_frames=[dict(id=r['id'],yaw_deg=r['initial_yaw_deg'],state=r['state'],view=r['view'],
            timestamp_s=r['timestamp_s'],selection_reason=r['selection_reason']) for r in b['frames'] if r['selected']],
        not_selected=[dict(id=r['id'],status=r['status'],reason=r['selection_reason']) for r in b['frames'] if not r['selected']],
        head_tracks=[dict(id=r['id'],status=r.get('overlap',{}).get('status')) for r in sequence['frames']],
        neck_tracks=[dict(id=r['id'],status=r.get('neck_overlap',{}).get('status')) for r in sequence['frames']],
        b_output_references={k:dict(frame_id=v['frame_id'],timestamp_s=v['timestamp_s'],source_frame=v['source_frame'],coordinate_system=v['coordinate_system']) for k,v in b['references'].items()},
        observed_refinement_vertices=b['observed_refinement_vertices'],total_vertices=b['total_vertices'],
        author_source_commit=audit['author_commit'],asset_archives={},a_runs=[],
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    # SHA256 was already computed while downloading, avoid re-reading 8GB here.
    downloads=load(root/'author-downloads/downloads.json')
    for name,row in downloads.items():
        if row['status']=='downloaded':data['asset_archives'][name]=row
    data['asset_archives']['model']=load(root/'model-resume.json')
    for name in ['a-build','a-build-v2','a-author-sample','a-author-sample-v2','a-author-sample-v3','a-author-sample-v4']:
        for file in ['build.json','report.json']:
            if (root/name/file).exists():data['a_runs'].append(dict(run=name,**load(root/name/file)))
    dump(a.output,data)
    print('Saved reviewable metrics: '+str(a.output))


if __name__=='__main__':main()
