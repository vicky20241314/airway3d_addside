"""Follow public Google Drive download forms; never save an HTML page as ZIP."""
import argparse
import html
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from audit import ASSETS, dump, sha


def download(url, target, max_seconds):
    start = time.perf_counter()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    response = opener.open(url, timeout=25)
    if 'text/html' in response.headers.get('Content-Type', ''):
        body = response.read().decode('utf-8', errors='replace')
        form = re.search(r'<form[^>]+action="([^"]+)"', body)
        fields = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', body))
        if not form: raise RuntimeError('No public download confirmation form')
        action = html.unescape(form.group(1))
        if urllib.parse.urlparse(action).hostname != 'drive.usercontent.google.com':
            raise RuntimeError('Unexpected confirmation host')
        response = opener.open(action+'?'+urllib.parse.urlencode(fields), timeout=30)
    if 'text/html' in response.headers.get('Content-Type', ''):
        raise RuntimeError('Confirmation returned HTML, not archive')
    part = target.with_suffix('.partial')
    count = 0
    with part.open('xb') as stream:
        first = response.read(1024*1024)
        if first[:4] != b'PK\x03\x04': raise RuntimeError('Response is not a ZIP archive')
        stream.write(first); count += len(first)
        while True:
            block = response.read(1024*1024)
            if not block: break
            stream.write(block); count += len(block)
            if time.perf_counter()-start > max_seconds:
                raise TimeoutError('Download time budget exceeded; partial retained')
    if not zipfile.is_zipfile(part): raise RuntimeError('ZIP footer missing')
    part.rename(target)
    return dict(bytes=count, sha256=sha(target), elapsed_s=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',nargs='+',choices=['sample','model','resources'],default=['sample'])
    p.add_argument('--max-seconds',type=float,default=600)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    report={}
    for name in a.assets:
        print('Fetching '+name,flush=True)
        try: report[name]=dict(status='downloaded',**download(ASSETS[name+'_drive'],a.output/(name+'.zip'),a.max_seconds))
        except Exception as exc: report[name]=dict(status='blocked',error=type(exc).__name__+': '+str(exc))
        print(name+': '+str(report[name]),flush=True)
    dump(a.output/'downloads.json',report)
    return 0 if all(v['status']=='downloaded' for v in report.values()) else 2


if __name__=='__main__': raise SystemExit(main())
