"""Reload/verify exports, compare local legacy artifacts, optional untextured QA.

Does not import or run FLAME/TUFA/segmentation. Legacy outputs remain subject to
their original licenses; numerical summaries do not clear commercial rights.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from a3_geometry import (dump,sha,transform,load_frame,diagnostic,raster_depth,
                         backproject,depth_normals,principal_component)


def read_obj(path):
    vertices,faces = [],[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                face = [int(x.split('/')[0])-1 for x in line.split()[1:]]
                if len(face)!=3:
                    raise ValueError('triangular OBJ required')
                faces.append(face)
    return np.asarray(vertices),np.asarray(faces)


def vertex_normals(v,f):
    n = np.zeros_like(v)
    tri = v[f]
    face_normals = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    for corner in range(3):
        np.add.at(n,f[:,corner],face_normals)
    return n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)


def verify(root,report):
    checks = []
    for row in report['frames']:
        if not row['fit']['accepted']:
            assert row['T_reference_from_camera'] is None
            checks.append(dict(frame_id=row['id'],status='excluded with explicit failure record'))
            continue
        T = np.array(row['T_reference_from_camera'])
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(T[:3,:3]),1.,atol=1e-10)
        assert row['final_scale']==1.
        checks.append(dict(frame_id=row['id'],rotation_xyz_degrees=Rotation.from_matrix(T[:3,:3]).as_euler('xyz',degrees=True).tolist(),scale=1.))
    meshes = {}
    for state,info in report['outputs'].items():
        meshes[state] = {}
        for stage in ('coarse','refined'):
            path = root/state/('face_'+stage+'_m.obj')
            v,f = read_obj(path)
            a = np.load(root/state/('support_'+stage+'.npz'),allow_pickle=False)
            assert sha(path)==info['geometry'][stage]['mesh_sha256']
            np.testing.assert_allclose(v,a['vertices_m'],atol=5.1e-11,rtol=0)
            np.testing.assert_array_equal(f,a['faces'])
            assert np.isfinite(v).all() and len(f)>100
            assert f.min()>=0 and f.max()<len(v)
            assert np.all(a['observation_count']>0)
            bit_count = [int(b).bit_count() for b in a['observation_bits']]
            np.testing.assert_array_equal(a['observation_count'],bit_count)
            np.testing.assert_array_equal(a['fusion_contributor_count'],[int(b).bit_count() for b in a['fusion_contributor_bits']])
            tri = v[f]
            assert np.max(np.linalg.norm(tri[:,[1,2,0]]-tri,axis=-1))<.006+1e-9
            _,_,components = principal_component(v,f)
            assert components['components']==1
            common = transform(v,np.linalg.inv(np.array(info['T_output_from_common'])))
            meshes[state][stage] = (common,a)
            checks.append(dict(state=state,stage=stage,vertices=len(v),triangles=len(f),
                obj_reloaded=True,all_finite=True,valid_indices=True,all_vertices_supported=True,
                largest_edge_lt_6mm=True,one_connected_component=True))
    shared = []
    for stage in ('coarse','refined'):
        common,closed = meshes['closed'][stage]
        other,opened = meshes['open'][stage]
        first = {tuple(p):i for i,p in enumerate(closed['common_pixels_yx']) if closed['shared_identity_upper'][i]}
        pairs = [(first[tuple(p)],j) for j,p in enumerate(opened['common_pixels_yx'])
                 if opened['shared_identity_upper'][j] and tuple(p) in first]
        ii,jj = np.array(pairs).T
        error = np.linalg.norm(common[ii]-other[jj],axis=1)
        assert error.max()<1e-8
        shared.append(dict(stage=stage,shared_upper_vertices=len(pairs),max_identity_discrepancy_m=float(error.max())))
    return dict(checks=checks,shared_identity=shared,status='passed',
                scope='export integrity and coordinate contracts, NOT independent surface accuracy')


def common_ray_diagnostic(models,frame):
    rendered = {name:raster_depth(v,f,frame['K'],frame['depth'].shape,2)
                for name,(v,f) in models.items()}
    mask = frame['valid'][::2,::2]
    observed = frame['depth'][::2,::2]
    common = mask.copy()
    for d in rendered.values():
        common &= np.isfinite(d)
    result = {}
    for name,d in rendered.items():
        error = np.abs(d[common]-observed[common])*1000
        result[name] = dict(median_axial_mm=float(np.median(error)) if len(error) else None,
            p95_axial_mm=float(np.percentile(error,95)) if len(error) else None)
    return dict(common_pixels=int(common.sum()),input_valid_roi_pixels=int(mask.sum()),
                common_roi_fraction=float(common.sum()/max(mask.sum(),1)),methods=result,
                caution='all methods evaluated on the SAME jointly rendered rays; unrendered ROI remains missing, not accurate')


def preview(root,report,output):
    from PIL import Image,ImageDraw
    panel = 360
    canvas = Image.new('RGB',(panel*3,(panel+30)*2),(235,238,240))
    draw = ImageDraw.Draw(canvas)
    for row,state in enumerate(('closed','open')):
        vertices,faces = read_obj(root/state/'face_refined_m.obj')
        # Canonical common coordinates for both states, not texture or RGB.
        common = transform(vertices,np.linalg.inv(np.array(report['outputs'][state]['T_output_from_common'])))
        centre = np.array([0.,0.,.5])
        centre[:2] = np.median(common[:,:2],axis=0)
        for col,yaw in enumerate((-60,0,60)):
            R = Rotation.from_euler('y',yaw,degrees=True).as_matrix()
            v = (common-centre)@R.T+[0,0,1.2]
            K = np.array([1800,1800,panel/2,panel/2])
            depth = raster_depth(v,faces,K,(panel,panel),1)
            valid = np.isfinite(depth)
            xyz = backproject(np.where(valid,depth,0),K)
            normal,good = depth_normals(xyz,valid)
            light = np.array([-.35,-.4,-1.]); light/=np.linalg.norm(light)
            shade = np.clip(.28+.72*np.sum(normal*light,axis=-1),.15,1.)
            rgb = np.full((panel,panel,3),235,np.uint8)
            rgb[valid] = 100
            rgb[good] = (shade[good,None]*np.array([205,213,220])).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb),(col*panel,row*(panel+30)))
            draw.text((col*panel+8,row*(panel+30)+panel+8),state+' / yaw '+str(yaw)+' deg / geometry only',fill='black')
    canvas.save(output/'geometry_only.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--front-control',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--legacy-runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    report = json.loads((a.result/'report.json').read_text(encoding='utf-8'))
    front = json.loads((a.front_control/'report.json').read_text(encoding='utf-8'))
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    verified = verify(a.result,report)
    verified_front = verify(a.front_control,front)
    dump(a.output/'verification.json',dict(full=verified,front_control=verified_front))
    baseline_benchmark = json.loads((a.legacy_runtime/'baseline-v0120/benchmark.json').read_text(encoding='utf-8'))
    base = Path(baseline_benchmark['result_folder'])
    extrinsics = json.loads((base/'extrinsics.json').read_text(encoding='utf-8'))
    b_root = a.legacy_runtime/'b-final'
    b_report = json.loads((b_root/'report.json').read_text(encoding='utf-8'))
    b_arrays = np.load(b_root/'parameters.npz',allow_pickle=False)
    results = []
    for r in manifest['frames']:
        frame = load_frame(r,report['budget']['stride'],a.manifest.parent)
        state = r['state']
        view_id = state+'_'+r['view']
        baseline_v,baseline_f = read_obj(base/state/('mesh_'+state+'_stage2.obj'))
        be = extrinsics[view_id]
        baseline_v = baseline_v@np.array(be['R']).T+np.array(be['t'])
        # Explicit existing-project (right/up/back) -> sensor (right/down/forward).
        baseline_v = baseline_v*np.array([1,-1,-1])
        bi = next(i for i,row in enumerate(b_report['frames']) if row['id']==r['id'])
        b_v,b_f = b_arrays['posed_vertices'][bi],b_arrays['faces']
        b_v = (b_v+vertex_normals(b_v,b_f)*b_arrays['deviation_m'][:,None])*np.array([1,-1,-1])
        row = dict(frame_id=r['id'],state=state,view=r['view'],
            baseline=diagnostic(baseline_v,baseline_f,frame),B=diagnostic(b_v,b_f,frame))
        models = dict(baseline=(baseline_v,baseline_f),B=(b_v,b_f))
        pose = next(x for x in report['frames'] if x['id']==r['id'])
        output_T = np.array(report['outputs'][state]['T_output_from_common'])
        v,f = read_obj(a.result/state/'face_refined_m.obj')
        common = transform(v,np.linalg.inv(output_T))
        local = transform(common,np.linalg.inv(np.array(pose['T_reference_from_camera'])))
        row['A3'] = diagnostic(local,f,frame)
        models['A3'] = (local,f)
        if r['view']=='front':
            pose_front = next(x for x in front['frames'] if x['id']==r['id'])
            vf,ff = read_obj(a.front_control/state/'face_refined_m.obj')
            cf = transform(vf,np.linalg.inv(np.array(front['outputs'][state]['T_output_from_common'])))
            vf_local = transform(cf,np.linalg.inv(np.array(pose_front['T_reference_from_camera'])))
            row['A3_front_only'] = diagnostic(vf_local,ff,frame)
            models['A3_front_only'] = (vf_local,ff)
        row['common_ray_diagnostic'] = common_ray_diagnostic(models,frame)
        results.append(row)
        print(r['id'],json.dumps({k:v['face_roi'] for k,v in row.items() if isinstance(v,dict) and 'face_roi' in v}),flush=True)
    result = dict(metric='same-frame, same manual ROI, in-sample camera-axis depth diagnostic',
        independent_extrinsics=False,independent_reference_scan=False,manual_silhouette_truth=False,
        original_pixel_stride=report['budget']['stride']*2,frames=results,
        baseline_commit=baseline_benchmark['baseline_commit'],baseline_budget=baseline_benchmark,
        B_budget={k:b_report.get(k) for k in ('steps','elapsed_s','cuda_peak_allocated_mib','cuda_peak_reserved_mib')},
        A3_budget=report['budget'],A3_elapsed_s=report['elapsed_s'],front_only_elapsed_s=front['elapsed_s'],
        timing_comparable_as_equal_accuracy=False,
        caution='Different topology, number of frames, ROI construction, expression treatment and fitting budget; historical runs not repeated. More overlap/residual reduction is not proof of scan accuracy.',
        elapsed_check_s=time.perf_counter()-start)
    dump(a.output/'comparison.json',result)
    if a.preview:
        preview(a.result,report,a.output)
    print('Export verification passed; comparison saved locally; independent accuracy UNVERIFIED',flush=True)


if __name__=='__main__':
    main()
