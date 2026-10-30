"""Snapshot actual local notices/versions without downloading or importing models."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import hashlib
import zipfile


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--notices',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    a.notices.mkdir(parents=True,exist_ok=False)
    root = Path(sys.prefix)
    source = a.shared/'hifi3dface'
    files = {'python':root/'LICENSE_PYTHON.txt',
        'python-distribution':root/'LICENSE',
        'hifi3dface':source/'LICENSE.md',
        'hifi3dface-model-terms':source/'3DMM/readme.md',
        'hifi3dface-sample-terms':source/'readme.txt'}
    for name in ('numpy','scipy','Pillow'):
        dist = metadata.distribution(name)
        f = next(f for f in dist.files if f.name in ('LICENSE.txt','LICENSE') and '.dist-info' in str(f))
        files[name] = Path(dist.locate_file(f))
    records = []
    for name,src in files.items():
        dest = a.notices/(name+'.txt')
        shutil.copyfile(src,dest)
        records.append(dict(name=name,path=str(src),sha256=sha(src),retained_notice=dest.name))
    archives = []
    expected = dict(sample='b5f9ec86a876e1c047f06a588b27254e76657b64f001716b838a058e4c58e4fd',
        model='f95800d1d4ea3853f8b87f7bf80736e8f54f8583352067d9c0a158773b8e6e6f',
        resources='b7b70dc2b02ad2a2947a74aa3926c702b0279546c6473dbaf85fe9ce07819109')
    for name,want in expected.items():
        f = a.shared/'author-downloads'/(name+'.zip')
        if not f.exists():
            archives.append(dict(name=name,status='archive missing at documented path',expected_sha256=want))
            continue
        got = sha(f)
        with zipfile.ZipFile(f) as z:
            licence_files = [n for n in z.namelist() if any(t in n.lower() for t in ('license','notice','readme'))]
            if licence_files:
                notice = z.read(licence_files[0])
                (a.notices/(name+'-archive-terms.txt')).write_bytes(notice)
        archives.append(dict(name=name,sha256=got,expected_sha256=want,
            matches=got==want,notice_entries=licence_files))
        if got != want:
            raise ValueError('archive hash mismatch: '+name)
    binaries = []
    for name in ('numpy.libs','scipy.libs'):
        for f in (root/'Lib/site-packages'/name).glob('*.dll'):
            binaries.append(dict(file=f.name,sha256=sha(f),bytes=f.stat().st_size))
    patches = []
    for f in Path(__file__).parent.glob('patch_a_*.py'):
        patches.append(dict(name=f.name,sha256=sha(f),used_in_new_A3=False))
    result = dict(python=sys.version,packages={n:metadata.version(n) for n in ('numpy','scipy','Pillow')},
        notices=records,archives=archives,bundled_binaries=binaries,existing_A_patches=patches,
        author_sha='d5accd138359ad534662805cd710f58de4ee4c74',
        A3_loaded_models=[],sample_reexecution=False,
        sample_evidence='prior research commit 159a894; a-author-sample-v5: ten-step export only; not full 100-step quality reproduction',
        commercial_clearance=False,reason='individual dependency grants are recorded; data consent, distribution contracts and other rights not cleared')
    (a.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
