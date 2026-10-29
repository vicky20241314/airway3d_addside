from pathlib import Path
import hashlib,json
from dataclasses import asdict
from core.fit_options import FitOptions
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def record_inputs(folder,options):
    from core.cross_refine import DEFAULT_PROFILE,ALGORITHMS
    data=dict(algorithm=(options or FitOptions()).algorithm_version(ALGORITHMS[DEFAULT_PROFILE]),mesh_unit='m',depth_unit='mm',
        capture_sha256=sha(folder/'capture_pair.npz'),fit_options=asdict(options or FitOptions()),
        assets={str(p.relative_to(ROOT)):sha(p) for d in ['assets','models','third_party']
                for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix.lower() in ['.pkl','.pth','.pt','.onnx','.npy']},
        code={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'core').rglob('*.py')})
    (folder/'provenance.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
