"""Hash-checked common frontal warm start for algorithm ablations."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np


def initialization_signature(fronts,options):
    from core.fit_options import FitOptions
    # Only legacy frontal fitting knobs affect this warm start.
    excluded=list(asdict(FitOptions()))
    excluded=excluded[excluded.index('registration_fallback'):]
    config={k:v for k,v in asdict(options).items() if k not in excluded}
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for state in ('closed','open'):
        o=fronts[state]
        for a in (o.xyz,o.uv,o.valid,o.points,o.parsing,
                  np.array([o.camera.fx,o.camera.fy,o.camera.cx,o.camera.cy])):
            a=np.ascontiguousarray(a);h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def save_initialization(path,signature,initial):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={f'{s}_{k}':v for s,p in initial.items() for k,v in p.items() if k in ('shape','expression','pose','rotation','translation')}
    with path.open('xb') as stream:np.savez_compressed(stream,signature=np.array(signature),**arrays)


def load_initialization(path,signature):
    with np.load(path,allow_pickle=False) as a:
        if str(a['signature'])!=signature:raise ValueError('Frontal initialization cache input/configuration mismatch')
        return {s:{k:a[f'{s}_{k}'].copy() for k in ('shape','expression','pose','rotation','translation')} for s in ('closed','open')}
