"""Export calibrated aligned frames in author's packed-PNG format.

This produces inputs and calibration metadata, NOT a validated full A adapter.
The author's transpose and fixed camera need explicit handling before fitting.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from audit import dump,sha


def encode_depth(depth_mm):
    if not np.isfinite(depth_mm).all():raise ValueError('nonfinite depth')
    d=np.rint(np.asarray(depth_mm)*10).astype(np.int64)
    if (d<0).any() or (d>65279).any():raise ValueError('depth exceeds packed RGB encoding range')
    r=np.minimum(d//255,255);g=d-r*255
    return np.stack((np.zeros_like(d),g,r),-1).astype(np.uint8) # OpenCV BGR


def decode_depth(bgr):
    return (bgr[...,2].astype(float)*255+bgr[...,1].astype(float))/10


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state',choices=['closed','open'],default='closed')
    p.add_argument('--pixel-stride',type=int,default=1,help='Keep actual RGB/depth pixels at this stride; no interpolation')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.pixel_stride<1:raise ValueError('pixel-stride must be positive')
    info=json.loads(a.sequence.read_text(encoding='utf-8'));rows=[]
    for r in info['frames']:
        if r['status']!='candidate' or r['state']!=a.state:continue
        with np.load(a.sequence.parent/r['file'],allow_pickle=False) as d:
            stride=a.pixel_stride;depth=d['depth'][::stride,::stride];rgb=d['rgb'][::stride,::stride]
            encoded=encode_depth(depth)
            if not np.allclose(decode_depth(encoded),depth):raise ValueError('depth round trip failed')
            cv2.imwrite(str(a.output/(r['id']+'.JPG')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(a.output/(r['id']+'.PNG')),encoded)
            fx,fy,cx,cy=d['intrinsics']/stride
            rows.append(dict(id=r['id'],intrinsics_before_transpose=[fx,fy,cx,cy],
                intrinsics_after_author_transpose=[fy,fx,cy,cx],shape=list(depth.shape),source=r['file'],
                source_intrinsics=d['intrinsics'].tolist(),pixel_stride=stride))
    if rows and all(np.allclose(r['intrinsics_before_transpose'],rows[0]['intrinsics_before_transpose'],rtol=0,atol=1e-6) and r['shape']==rows[0]['shape'] for r in rows):
        fx,fy,cx,cy=rows[0]['intrinsics_before_transpose'];height,width=rows[0]['shape']
        dump(a.output/'camera.json',dict(fx=fx,fy=fy,cx=cx,cy=cy,width=width,height=height))
    dump(a.output/'calibration.json',dict(sequence_sha256=sha(a.sequence),frames=rows,
        depth_encoding='BGR: (R*255+G)/10 millimetres',aligned=True,
        coordinates_input='X_right_Y_up_Z_backward metres',author_depth_points='X_right_Y_down_Z_forward millimetres',
        to_author_xyz_matrix=np.diag([1000,-1000,-1000]),
        status='format_roundtrip_verified; author geometry integration pending sample success',
        required_changes=['remove author hardcoded iPhone intrinsics',
            'use author-detected 86 landmarks; do not substitute 68',
            'handle run_data_preparation image transpose consistently',
            'verify raster image dimensions and per-view calibration in step0/step5',
            'validate physical RGB/depth alignment independently']))
    print('Exported '+str(len(rows))+' real frames; calibration integration still pending')


if __name__=='__main__':main()
