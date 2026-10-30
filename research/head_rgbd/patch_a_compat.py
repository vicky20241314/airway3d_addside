"""Enable explicit pickle loading only for author's trusted legacy model assets."""
import argparse
from pathlib import Path
import re
from audit import dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();rows=[];pending=[]
    for name in ['utils/basis.py']:
        path=a.source/name;backup=path.with_suffix(path.suffix+'.research-original')
        text=(backup if backup.exists() else path).read_text(encoding='utf-8')
        patched,count=re.subn(r'''(np\.load\([^)]*encoding=["']latin1["'])(\s*\))''',r'\1, allow_pickle=True\2',text)
        if not count:raise ValueError('Expected legacy loader absent; source changed: '+name)
        if backup.exists() and path.read_text(encoding='utf-8')!=patched:
            raise ValueError('Unexpected edits after backup; preserve and inspect '+name)
        before=sha(backup if backup.exists() else path)
        pending.append((name,path,backup,patched,count,before))
    for name,path,backup,patched,count,before in pending:
        if not backup.exists():path.rename(backup);path.write_text(patched,encoding='utf-8')
        rows.append(dict(file=name,calls=count,before_sha256=before,after_sha256=sha(path)))
    dump(a.report,dict(scope='only explicit Latin-1 np.load calls in author model loaders',
        step5_note='author already explicitly enables pickle for head completion; unchanged',changes=rows))
    print(rows)


if __name__=='__main__':main()
