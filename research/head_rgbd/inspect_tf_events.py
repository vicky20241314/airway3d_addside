"""Read TFRecord event step headers without importing TensorFlow or large graphs."""
import argparse
import json
from pathlib import Path
import struct


def varint(data,pos):
    value=0;shift=0
    while pos<len(data) and shift<70:
        byte=data[pos];pos+=1;value|=(byte&127)<<shift
        if byte<128:return value,pos
        shift+=7
    raise ValueError('incomplete protobuf prefix')


def step_from_prefix(data):
    pos=0
    try:
        while pos<len(data):
            key,pos=varint(data,pos);field=key>>3;wire=key&7
            if wire==0:
                value,pos=varint(data,pos)
                if field==2:return value
            elif wire==1:pos+=8
            elif wire==2:
                size,pos=varint(data,pos);pos+=size
            elif wire==5:pos+=4
            else:break
    except ValueError:pass
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.parse_args();result=[]
    for path in a.root.rglob('events.out.tfevents*'):
        steps=[];size=path.stat().st_size
        with path.open('rb') as stream:
            while stream.tell()+12<=size:
                length=struct.unpack('<Q',stream.read(8))[0];stream.read(4)
                if stream.tell()+length+4>size:break
                prefix=stream.read(min(length,128));stream.seek(length-len(prefix)+4,1)
                steps.append(step_from_prefix(prefix))
        result.append(dict(path=str(path),bytes=size,complete_records=len(steps),max_step=max(steps,default=None)))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
