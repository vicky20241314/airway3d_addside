"""Read-only source/data audit and bounded HTTP asset probes."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import urllib.request
import urllib.error

import numpy as np


ASSETS = {
    'model_drive': 'https://drive.google.com/uc?export=download&id=1vSb2EpduRJuIEUOc_aRsbjASSOUAk7tG',
    'resources_drive': 'https://drive.google.com/uc?export=download&id=1FoBbj1qWdf2LyJkxLYnMJ4h4woTQ6QL_',
    'sample_drive': 'https://drive.google.com/uc?export=download&id=1f5imMCO0ug_ozj0cDcNYtizom5lBu2PH',
    'model_weiyun': 'https://share.weiyun.com/ByrUzsJG',
    'resources_weiyun': 'https://share.weiyun.com/Hgov20It',
    'sample_weiyun': 'https://share.weiyun.com/5PCy1rID',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=encode, allow_nan=False)


def probe(url):
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-4095'})
        with urllib.request.urlopen(req, timeout=18) as response:
            body = response.read(4096)
            return dict(status=response.status, content_type=response.headers.get('Content-Type'),
                        content_length=response.headers.get('Content-Length'),
                        zip_signature=body[:4] == b'PK\x03\x04',
                        prefix=body[:350].decode('utf-8', errors='replace') if b'html' in body[:100].lower() else None,
                        elapsed_s=time.perf_counter()-start)
    except Exception as exc:
        return dict(error=type(exc).__name__ + ': ' + str(exc), elapsed_s=time.perf_counter()-start)


def describe_npz(path):
    with np.load(path, allow_pickle=False) as data:
        fields = {k: dict(shape=list(data[k].shape), dtype=str(data[k].dtype)) for k in data.files}
        extra = {}
        for k in data.files:
            if 'intrinsic' in k or 'timestamp' in k:
                extra[k] = data[k].tolist()
        return dict(path=str(path), sha256=sha(path), fields=fields, metadata=extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe-assets', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    result = dict(platform=platform.platform(), numpy=np.__version__,
                  author_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                  asset_present={n: (source/n).exists() for n in ['3DMM/files/AI-NEXT-Shape.mat', 'resources/PB', 'test_data/RGBD/test1', 'third_party/kernels/rasterize_triangles_kernel.so']},
                  source_hashes={str(f.relative_to(source)): sha(f) for f in source.rglob('*.py')},
                  assets={name: dict(url=url, **probe(url)) for name,url in ASSETS.items()} if args.probe_assets else {},
                  sessions=[], raw_sequences=[])
    for manifest in sorted(args.capture.rglob('session.json')):
        info = json.loads(manifest.read_text(encoding='utf-8'))
        result['sessions'].append(dict(path=str(manifest), sha256=sha(manifest),
            schema=info.get('schema'), depth_unit=info.get('depth_unit'), point_unit=info.get('point_unit'),
            coordinate_convention=info.get('coordinate_convention'),
            observations=[dict(view_id=r['view_id'], state=r['state'], view=r['view'],
                quality_keys=list(r.get('quality',{})), file=describe_npz(manifest.parent/r['file'])) for r in info['observations']]))
    for raw in sorted(args.capture.rglob('raw_sequence.npz')):
        result['raw_sequences'].append(describe_npz(raw))
    result['validation_status'] = dict(independent_extrinsics='not established by this audit',
        human_silhouette='not provided', real_side_improvement='unverified')
    dump(args.output/'audit.json', result)
    print(json.dumps(dict(output=str(args.output/'audit.json'), author_commit=result['author_commit'],
        asset_present=result['asset_present'], assets=result['assets'],
        raw_sequences=result['raw_sequences']), indent=2, ensure_ascii=False))


if __name__ == '__main__': main()
