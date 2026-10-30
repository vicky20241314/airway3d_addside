"""Repackage real cached observations into bounded generic chunks, without synthesis."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from audit import dump
from prepare import input_chunks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=3)
    a=p.parse_args()
    if a.chunk_size<1:p.error('chunk-size must be positive')
    a.output.mkdir(parents=True,exist_ok=False);chunks=[];sources=[]
    for raw,entry,cached,data,segment,provenance in input_chunks(SimpleNamespace(capture=a.capture,frame_manifest=None)):
        sources.append(provenance)
        for begin in range(0,len(data['rgb']),a.chunk_size):
            end=begin+a.chunk_size;name=entry['view_id']+'_'+str(begin)+'.npz'
            np.savez_compressed(a.output/name,**{key:data[key][begin:end] for key in ['rgb','depth','intrinsics','timestamps_s']},
                uv=cached['temporal_frame_uv'][begin:end],landmark_valid=cached['temporal_frame_valid'][begin:end])
            chunks.append(dict(file=name,view_id=entry['view_id'],segment=segment,state=entry['state'],view=entry['view']))
    dump(a.output/'raw_manifest.json',dict(schema='head_rgbd.raw.v1',depth_unit='mm',rgb_depth_aligned=True,
        alignment_evidence='inherited source capture declaration; not independent physical verification',chunks=chunks,sources=sources))
    print('Repacked real data into '+str(len(chunks))+' chunks, without interpolated or synthetic frames')


if __name__=='__main__':main()
