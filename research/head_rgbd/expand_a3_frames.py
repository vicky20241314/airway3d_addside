"""Track manual seeds over real short bursts using normalized RGB patch matching.

No detector/network/model. Actual timestamps and source indices are retained.
Failed patches are omitted; no front frame is copied to invent another view.
Output contains private annotations and must remain outside the repository.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np
from scipy.signal import correlate2d


def match_patch(anchor, image, xy, radius=8, half=10):
    x,y = np.rint(xy).astype(int)
    h,w = anchor.shape
    margin = radius+half
    if x<margin or x>=w-margin or y<margin or y>=h-margin:
        return None,0.
    patch = anchor[y-half:y+half+1,x-half:x+half+1]
    patch = patch-patch.mean()
    energy = float(np.sum(patch**2))
    if energy<1e-6:
        return None,0.
    search = image[y-margin:y+margin+1,x-margin:x+margin+1]
    ones = np.ones(patch.shape)
    sums = correlate2d(search,ones,mode='valid')
    squares = correlate2d(search**2,ones,mode='valid')
    denominator = np.sqrt(np.maximum(squares-sums**2/patch.size,0)*energy)
    score = correlate2d(search,patch,mode='valid')/np.maximum(denominator,1e-12)
    iy,ix = np.unravel_index(np.argmax(score),score.shape)
    quality = float(score[iy,ix])
    # A peak on the search boundary may have left the admissible search region.
    if ix in (0,2*radius) or iy in (0,2*radius):
        return None,quality
    return [float(x+ix-radius),float(y+iy-radius)],quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-ncc',type=float,default=.85)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    if manifest['schema']!='head_rgbd.a3_manual.v1':
        raise ValueError('manual seed manifest required')
    if len({r['source'] for r in manifest['frames']})!=len(manifest['frames']):
        raise ValueError('one directly annotated seed per raw burst required')
    rows,excluded = [],[]
    for seed in manifest['frames']:
        path = Path(seed['source'])
        if not path.is_absolute():
            path = a.manifest.parent/path
        with np.load(path,allow_pickle=False) as d:
            rgb = d['rgb']
            times = d['timestamps_s']
            if len(times)>1 and (np.diff(times).max()>.3 or times[-1]-times[0]>.5):
                raise ValueError('patch propagation limited to <=0.5s short bursts, no gap interpolation')
            gray = rgb.astype(np.float64)@np.array([.299,.587,.114])/255.
        anchor = gray[seed['source_frame']]
        for index,image in enumerate(gray):
            row = copy.deepcopy(seed)
            row['source'] = str(path)
            row['id'] = seed['state']+'_'+seed['view']+'_'+str(index)
            row['source_frame'] = index
            row['timestamp_s'] = float(times[index])
            row['annotation_origin'] = 'manual_seed' if index==seed['source_frame'] else 'classical_NCC_from_manual_seed_on_actual_RGB'
            if index==seed['source_frame']:
                rows.append(row)
                continue
            landmarks,scores = {},{}
            for name,xy in seed['landmarks_px'].items():
                match,quality = match_patch(anchor,image,xy)
                scores[name] = dict(ncc=quality,accepted=match is not None and quality>=a.minimum_ncc)
                if scores[name]['accepted']:
                    landmarks[name] = match
            rigid = set(landmarks)&set(manifest['rigid_landmark_names'])
            if len(rigid)<4:
                excluded.append(dict(id=row['id'],reason='fewer than four accepted real RGB patch matches',scores=scores))
                continue
            shift = np.median([np.array(landmarks[n])-seed['landmarks_px'][n] for n in rigid],axis=0)
            row['landmarks_px'] = landmarks
            row['roi_polygon_px'] = (np.array(seed['roi_polygon_px'])+shift).tolist()
            row['upper_limit_y_px'] = seed['upper_limit_y_px']+float(shift[1])
            row['tracking'] = dict(seed_frame_id=seed['id'],patch_size=21,search_radius_px=8,
                                   minimum_ncc=a.minimum_ncc,scores=scores)
            rows.append(row)
    manifest['frames'] = rows
    manifest['tracking_excluded'] = excluded
    manifest['annotation'] += ' Other short-burst frames tracked by classical 21px NCC patches; no neural weights. Not independent geometric truth.'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(real_frames_selected=len(rows),excluded=[r['id'] for r in excluded],
                         roles={v:sum(r['view']==v for r in rows) for v in ('front','left','right','up')})))


if __name__=='__main__':
    main()
