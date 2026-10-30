"""Combine existing project full-head output with A3, explicitly research-only.

This adapter reads authorized existing outputs, not FLAME/TUFA/segmenter weights.
No default application algorithms or shared environments are modified.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import shutil
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from a3_geometry import (dump,sha,transform,load_frame,observation_support,diagnostic,
                         raster_depth,backproject,depth_normals)
from check_a3 import read_obj,vertex_normals
from head_adapter import subdivide,topology,fit_surface,write_grouped_obj,fragile_vertices
from run_a3 import landmarks_sidecar


MODEL_SHA = '8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0'


def validate_authorization(auth):
    if not (auth.get('model_sha256')==MODEL_SHA and auth.get('accepted_official_academic_license') is True
            and auth.get('purpose')=='local_noncommercial_research' and auth.get('redistribution_authorized') is False):
        raise ValueError('specific existing-model academic-license confirmation required; no blanket commercial consent')


def preview_meshes(meshes,output):
    from PIL import Image,ImageDraw
    size = 360
    canvas = Image.new('RGB',(size*3,(size+28)*2),(240,240,240))
    draw = ImageDraw.Draw(canvas)
    for row,state in enumerate(('closed','open')):
        v,f = meshes[state]
        centre = (v.min(0)+v.max(0))/2
        for col,yaw in enumerate((0,90,180)):
            vv = (v-centre)@Rotation.from_euler('y',yaw,degrees=True).as_matrix().T+[0,0,1.]
            K = np.array([800,800,size/2,size/2])
            d = raster_depth(vv,f,K,(size,size),1)
            valid = np.isfinite(d)
            n,good = depth_normals(backproject(np.where(valid,d,0),K),valid)
            light = np.array([-.35,-.4,-1.]);light/=np.linalg.norm(light)
            value = np.clip(.3+.7*np.sum(n*light,axis=-1),.15,1.)
            rgb = np.full((size,size,3),240,np.uint8)
            rgb[valid] = 110
            rgb[good] = (value[good,None]*np.array([205,215,225])).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb),(col*size,row*(size+28)))
            draw.text((col*size+5,row*(size+28)+size+6),state+' / '+str(yaw)+' deg / rear is PRIOR',fill='black')
    canvas.save(output/'whole_head_geometry.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a3',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--legacy-runtime',type=Path,required=True)
    p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--head-source',choices=['b','baseline'],default='b')
    p.add_argument('--subdivide',type=int,choices=[0,1,2],default=1)
    p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--preview',action='store_true')
    a = p.parse_args()
    auth = json.loads(a.authorization.read_text(encoding='utf-8'))
    validate_authorization(auth)
    report = json.loads((a.a3/'report.json').read_text(encoding='utf-8'))
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    if sha(a.manifest)!=report['manifest_sha256']:
        raise ValueError('A3 fitted manifest mismatch')
    b_report = json.loads((a.legacy_runtime/'b-final/report.json').read_text(encoding='utf-8'))
    if b_report['model_sha256']!=MODEL_SHA:
        raise ValueError('legacy model hash not covered by authorization')
    b_arrays = np.load(a.legacy_runtime/'b-final/parameters.npz',allow_pickle=False)
    baseline = json.loads((a.legacy_runtime/'baseline-v0120/benchmark.json').read_text(encoding='utf-8'))
    a.output.mkdir(parents=True,exist_ok=False)
    snapshot = a.output/'source_snapshot'
    snapshot.mkdir()
    for script in [Path(__file__),Path(__file__).with_name('head_adapter.py')]:
        shutil.copyfile(script,snapshot/script.name)
    started = time.perf_counter()
    common_reference = next(x for x in manifest['frames'] if x['id']==report['common_reference_frame'])
    common = load_frame(common_reference,report['budget']['stride'],a.manifest.parent)
    active_ids = report['active_frame_ids']
    frames = [load_frame(next(x for x in manifest['frames'] if x['id']==fid),
                         report['budget']['stride'],a.manifest.parent) for fid in active_ids]
    transforms = [np.array(next(x for x in report['frames'] if x['id']==fid)['T_reference_from_camera']) for fid in active_ids]
    summaries,meshes,comparisons = {},{},[]
    closed_field,shared_mask = None,None
    priors = {}
    protected_union = None
    for state in ('closed','open'):
        ref_id = report['outputs'][state]['reference_frame_id']
        frame_row = next(x for x in report['frames'] if x['id']==ref_id)
        T = np.array(frame_row['T_reference_from_camera'])
        if a.head_source=='b':
            idx = next(i for i,r in enumerate(b_report['frames']) if r['id']==ref_id)
            if (Path(b_report['frames'][idx]['source'])!=Path(frame_row['source']) or
                    b_report['frames'][idx]['source_frame']!=frame_row['source_frame']):
                raise ValueError('legacy head and A3 reference are not the same actual RGB-D frame')
            pv = b_arrays['posed_vertices'][idx]*[1,-1,-1]
            pf = b_arrays['faces'].copy()
            source = a.legacy_runtime/'b-final/parameters.npz'
        else:
            session = Path(frame_row['source']).parent.parent/'complete/session.json'
            if sha(session)!=baseline['session_sha256']:
                raise ValueError('baseline belongs to another capture session')
            data = json.loads(session.read_text(encoding='utf-8'))
            obs = next(x for x in data['observations'] if x['view_id']==state+'_front')
            if obs['quality']['temporal']['representative_rgb_index']!=frame_row['source_frame']:
                raise ValueError('baseline reference frame mismatch')
            source = Path(baseline['result_folder'])/state/('mesh_'+state+'_stage2.obj')
            pv,pf = read_obj(source)
            pv *= [1,-1,-1]
        base,faces = subdivide(transform(pv,T),pf,a.subdivide)
        priors[state] = (base,faces,source,T)
        protect = fragile_vertices(base,faces)
        protected_union = protect if protected_union is None else protected_union|protect
    for state in ('closed','open'):
        info = report['outputs'][state]
        ref_id = info['reference_frame_id']
        frame_row = next(x for x in report['frames'] if x['id']==ref_id)
        T = np.array(frame_row['T_reference_from_camera'])
        # Coarse prior only: never stack the old optional 3mm deviation.
        base,faces,source,T = priors[state]
        target = np.load(a.a3/state/'support_refined.npz',allow_pickle=False)
        target_v = transform(target['vertices_m'],T)
        target_f = target['faces']
        target_valid = np.zeros(common['depth'].shape,bool)
        pixels = target['common_pixels_yx']
        supported = target['observation_count']>=2
        target_valid[pixels[supported,0],pixels[supported,1]] = True
        fitted,fit = fit_surface(base,faces,target_v,target_f,target_valid,common['K'],
            common['depth'].shape,np.linalg.inv(T),iterations=a.iterations,
            fixed_mask=shared_mask if state=='open' else None,
            fixed_delta=closed_field if state=='open' else None,protected_mask=protected_union)
        if state=='closed':
            closed_field = fit['delta'].copy()
            uv = (base[:,1]/base[:,2]*common['K'][1]+common['K'][3])*common['stride']
            shared_mask = fit['active']&(uv<common_reference['upper_limit_y_px'])
        count,bits,residuals = observation_support(fitted,faces,frames,transforms)
        out_v = transform(fitted,np.linalg.inv(T))
        original_v = transform(base,np.linalg.inv(T))
        category = np.zeros(len(out_v),np.uint8)
        category[fit['active']] = 1
        category[fit['active']&fit['direct']&(count>=2)] = 2
        folder = a.output/state
        folder.mkdir()
        path = folder/'head_combined_m.obj'
        write_grouped_obj(path,out_v,faces,category)
        write_grouped_obj(folder/'head_prior_m.obj',original_v,faces,np.zeros(len(out_v),np.uint8))
        # The grouped OBJ reorders faces only; source arrays retain original topology.
        loaded,loaded_f = read_obj(path)
        np.testing.assert_allclose(loaded,out_v,atol=5.1e-11,rtol=0)
        assert len(loaded_f)==len(faces)
        assert set(map(tuple,loaded_f))==set(map(tuple,faces))
        assert topology(loaded,loaded_f)['nonmanifold_edges']==0
        np.savez_compressed(folder/'provenance.npz',vertices_m=out_v,faces=faces,
            prior_vertices_m=original_v,common_displacement_m=fit['delta'],
            category=category,active=fit['active'],data_constraint=fit['direct'],
            observation_count=count,observation_bits=bits,signed_axial_residual_m=residuals,
            shared_upper_correction=shared_mask,frame_ids=np.array(active_ids))
        labels = dict(category_0='prior_only',category_1='regularized_transition',category_2='data_constrained_and_depth_supported')
        dump(folder/'provenance.json',dict(unit='m',coordinates='X_right_Y_down_Z_forward',
            reference_frame_id=ref_id,T_reference_from_camera=T.tolist(),labels=labels,
            source=str(source),source_sha256=sha(source),model_sha256=MODEL_SHA,
            restrictions='local noncommercial research only; no redistribution; not a full-head scan'))
        own_ref = next(f for f in frames if f['row']['id']==ref_id)
        dump(folder/'landmarks_combined_head.json',landmarks_sidecar(own_ref,out_v,T,np.linalg.inv(T),None,'combined_head'))
        summary = dict(reference_frame_id=ref_id,mesh_sha256=sha(path),topology=topology(out_v,faces),
            fit_guard=fit['guard'],history=fit['history'],
            frozen_vertices=fit['frozen_count'],data_constrained_vertices=int((category==2).sum()),
            protected_fragile_vertices=int(protected_union.sum()),
            regularized_vertices=int((category==1).sum()),prior_only_vertices=int((category==0).sum()),
            observed_at_least_two_vertices=int((count>=2).sum()),
            shared_upper_correction_vertices=int(shared_mask.sum()),
            shared_correction_error_m=0. if state=='closed' else float(np.max(np.abs(fit['delta'][shared_mask]-closed_field[shared_mask]))),
            frozen_region_max_change_m=float(np.max(np.linalg.norm(fit['delta'][~fit['active']],axis=1))),
            metric_scale=1.,unit='m',rear_and_scalp='unobserved model prior, accuracy unverified')
        summaries[state] = summary
        meshes[state] = (fitted,faces)
        for f,tf in zip(frames,transforms):
            # Fixed historical representative set; no extra acquired viewpoints.
            if f['row']['id'] not in ('closed_front_1','closed_left_0','closed_right_3','open_front_2') or f['row']['state']!=state:
                continue
            inv = np.linalg.inv(tf)
            comparisons.append(dict(frame_id=f['row']['id'],state=state,
                prior=diagnostic(transform(base,inv),faces,f),
                combined=diagnostic(transform(fitted,inv),faces,f),
                A3_face=diagnostic(transform(target_v,inv),target_f,f),
                independent_surface_error=None,independent_coverage=None))
        print(state,json.dumps({k:v for k,v in summary.items() if k!='history'}),flush=True)
    result = dict(method='existing project full-head prior + independent A3 bounded surface adapter',
        head_source=a.head_source,legacy_baseline_commit=baseline['baseline_commit'],
        model_sha256=MODEL_SHA,authorization_evidence='user confirmed official academic license; local noncommercial research only',
        authorization_file_sha256=sha(a.authorization),A3_report_sha256=sha(a.a3/'report.json'),
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('head_adapter.py')]},
        source_model_weights_loaded=False,TUFA_or_segmentation_executed=False,
        budget=dict(subdivision=a.subdivide,iterations=a.iterations,max_total_new_offset_m=.003,
                    max_step_m=.00075,shared_upper_correction=True,neck_or_mouth_capping=False),
        states=summaries,diagnostics=comparisons,elapsed_s=time.perf_counter()-started,device='CPU',cuda_used=False,
        full_head_shape_present=True,full_head_scan_accuracy_verified=False,
        limitations=['unseen rear/scalp/ears are model prior',
            'retains original topological boundary loops; not watertight',
            'shared identity correction and separate reference-state lower face; no new neck model',
            'same-data diagnostics, no independent scan/contours/extrinsics',
            'commercial and redistribution rights not granted'],publication=dict(meshes=False,models=False,raw_data=False))
    dump(a.output/'report.json',result)
    if a.preview:
        preview_meshes(meshes,a.output)
    print('Completed local research head combination; no full-head accuracy claim',flush=True)


if __name__=='__main__':
    main()
