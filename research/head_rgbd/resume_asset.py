"""Resume one task-local partial asset only after validating Content-Range."""
import argparse
import html
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from audit import ASSETS,dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial',type=Path,required=True)
    p.add_argument('--asset',choices=['model','resources','sample'],required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();size=a.partial.stat().st_size
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    response=opener.open(ASSETS[a.asset+'_drive'],timeout=30)
    body=response.read().decode('utf-8',errors='replace')
    action=html.unescape(re.search(r'<form[^>]+action="([^"]+)"',body).group(1))
    if urllib.parse.urlparse(action).hostname!='drive.usercontent.google.com':raise RuntimeError('Unexpected host')
    fields=dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',body))
    req=urllib.request.Request(action+'?'+urllib.parse.urlencode(fields),headers={'Range':'bytes='+str(size)+'-'})
    response=opener.open(req,timeout=60)
    cr=response.headers.get('Content-Range','')
    if response.status!=206 or not cr.startswith('bytes '+str(size)+'-'):
        dump(a.report,dict(status='resume_unsupported',http_status=response.status,content_range=cr,size=size))
        return 2
    total=int(cr.split('/')[-1]);print('Resuming '+str(size)+' / '+str(total),flush=True)
    with a.partial.open('ab') as stream:
        while True:
            block=response.read(1024*1024)
            if not block:break
            stream.write(block)
    if a.partial.stat().st_size!=total or not zipfile.is_zipfile(a.partial):raise RuntimeError('Incomplete archive')
    target=a.partial.with_suffix('.zip');a.partial.rename(target)
    result=dict(status='complete',path=str(target),bytes=total,sha256=sha(target))
    dump(a.report,result);print(result,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
