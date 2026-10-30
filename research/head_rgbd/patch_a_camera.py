"""Add explicit task-only calibration/orientation hooks; author defaults unchanged."""
import argparse
from pathlib import Path
from audit import dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();pending=[]
    name='data_prepare/run_data_preparation.py';path=a.source/name;text=path.read_text(encoding='utf-8')
    old='if img.shape[0] < img.shape[1]:'
    if text.count(old)!=1:raise ValueError('Unexpected author transpose code')
    pending.append((name,path,text.replace(old,'if os.environ.get("HEAD_RGBD_ALLOW_AUTHOR_TRANSPOSE", "1") == "1" and img.shape[0] < img.shape[1]:')))
    name='optimization/rgbd/step0_prepare_frontend_data.py';path=a.source/name;text=path.read_text(encoding='utf-8')
    old='    K = {"fx": 592.7247, "fy": 593.7484, "cx": 239.7348, "cy": 319.4659}'
    if text.count(old)!=1:raise ValueError('Unexpected author intrinsics code')
    replacement=old+'''
    if os.environ.get("HEAD_RGBD_CAMERA_JSON"):
        import json
        with open(os.environ["HEAD_RGBD_CAMERA_JSON"], "r") as calibration:
            K = json.load(calibration)
        values = np.asarray([K[k] for k in ("fx", "fy", "cx", "cy")], dtype=float)
        if not np.isfinite(values).all() or min(values[:2]) <= 0:
            raise ValueError("Invalid explicitly supplied RGB-D calibration")'''
    pending.append((name,path,text.replace(old,replacement)))
    if any(path.with_suffix(path.suffix+'.research-original').exists() for _,path,_ in pending):raise ValueError('Backup exists; inspect before reapplying')
    rows=[]
    for name,path,patched in pending:
        before=sha(path);path.rename(path.with_suffix(path.suffix+'.research-original'));path.write_text(patched,encoding='utf-8')
        rows.append(dict(file=name,before_sha256=before,after_sha256=sha(path)))
    dump(a.report,dict(scope='opt-in real camera intrinsics and upright image preservation',changes=rows))
    print(rows)


if __name__=='__main__':main()
