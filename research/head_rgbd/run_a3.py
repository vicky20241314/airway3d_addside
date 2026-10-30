"""Explicit A3 entry: raw aligned RGB-D + local manual ROIs/landmarks -> metre OBJ.

Research prototype inspired by multi-view RGB-D geometry, NOT an author-method
reproduction. No models/weights and no imports from the original application.
"""
import argparse
import ctypes
import json
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import scipy

from a3_geometry import (sha,dump,transform,load_frame,register,fuse,refine_depth,
                         mesh_from_depth,write_obj,observation_support,diagnostic,principal_component)


def peak_working_set():
    if sys.platform != 'win32':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)] + [
            (n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    p = Counters()
    p.cb = ctypes.sizeof(p)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),ctypes.byref(p),p.cb):
        return None
    return p.PeakWorkingSetSize/2**20


def landmarks_sidecar(frame, vertices, reference_from_camera, common_to_output, pixels, stage):
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    rows = []
    for name,uv in frame['row']['landmarks_px'].items():
        xyz = frame['landmarks'][name]
        if xyz is None:
            rows.append(dict(name=name,uv_px=uv,observed_xyz_m=None,mesh_xyz_m=None,
                             valid=False,reason='invalid original landmark depth'))
            continue
        expected = transform(transform(xyz[None],reference_from_camera),common_to_output)[0]
        dist,ix = tree.query(expected)
        good = dist<.006
        rows.append(dict(name=name,uv_px=uv,observed_xyz_m=xyz.tolist(),
            mesh_xyz_m=vertices[ix].tolist() if good else None,
            mesh_vertex_index=int(ix) if good else None,valid=bool(good),
            mesh_distance_to_observation_mm=float(dist*1000),
            reason=None if good else 'no supported mesh vertex within fixed 6mm'))
    return dict(frame_id=frame['row']['id'],source=str(frame['path']),
        source_frame=frame['row']['source_frame'],timestamp_s=frame['timestamp_s'],
        stage=stage,unit='m',coordinates='X_right_Y_down_Z_forward',
        annotation_origin=frame['row'].get('annotation_origin','manual_seed')+'; ~3px seed uncertainty; not independent truth',
        schema='named_manual_points.v1; NOT 68/86 model topology or clinical metrics',landmarks=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--icp-iterations',type=int,default=30,help='per each of three fixed gates')
    p.add_argument('--front-only',action='store_true')
    p.add_argument('--skip-diagnostics',action='store_true')
    a = p.parse_args()
    if a.stride<1 or a.icp_iterations<1:
        p.error('positive stride and iteration budget required')
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    if manifest.get('schema')!='head_rgbd.a3_manual.v1' or manifest.get('depth_unit')!='mm':
        raise ValueError('explicit raw-millimetre input contract required')
    if manifest.get('rgb_depth_alignment')!='capture_claim_not_independently_calibrated':
        raise ValueError('declare alignment evidence explicitly')
    if manifest.get('data_rights',{}).get('local_processing_authorized') is not True:
        raise ValueError('local data authorization declaration missing')
    rows = manifest['frames']
    if a.front_only:
        rows = [r for r in rows if r['view']=='front']
    if not 1<=len(rows)<=32 or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('need 1..32 unique explicit frames')
    physical_keys = [(str(Path(r['source']).resolve()),int(r['source_frame'])) for r in rows]
    if len(set(physical_keys))!=len(physical_keys):
        raise ValueError('one physical frame cannot impersonate multiple views')
    if any(r['view'] not in ('front','left','right','up') or r['state'] not in ('closed','open') for r in rows):
        raise ValueError('unknown explicit view/state role')
    common_id = manifest['references']['closed']
    rows = sorted(rows,key=lambda r:r['id']!=common_id)
    if rows[0]['id']!=common_id:
        raise ValueError('closed reference must be an actual selected frame')
    a.output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    frames = [load_frame(r,a.stride,a.manifest.parent) for r in rows]
    ref = frames[0]
    transforms,poses = [np.eye(4)],[dict(accepted=True,scale=1.,reason='fixed real reference',history=[])]
    for frame in frames[1:]:
        try:
            T,info = register(frame,ref,manifest['rigid_landmark_names'],a.icp_iterations)
        except (ValueError,np.linalg.LinAlgError) as error:
            T = np.eye(4)  # placeholder is excluded; NEVER exported as an estimated pose
            info = dict(accepted=False,scale=1.,reason=str(error),history=[])
        transforms.append(T)
        poses.append(info)
        print(frame['row']['id'],json.dumps({k:v for k,v in info.items() if k not in ('history','initial_T')}),flush=True)
    selected = [(f,T,info) for f,T,info in zip(frames,transforms,poses) if info['accepted']]
    # Record failures before excluding. Reference frames cannot silently disappear.
    all_frame_records = []
    hashes = {str(path):sha(path) for path in {f['path'] for f in frames}}
    for f,T,info in zip(frames,transforms,poses):
        expression = {}
        for name,v in f['landmarks'].items():
            rv = ref['landmarks'].get(name)
            if info['accepted'] and v is not None and rv is not None:
                expression[name] = (transform(v[None],T)[0]-rv).tolist()
        all_frame_records.append(dict(id=f['row']['id'],view=f['row']['view'],state=f['row']['state'],
            source=str(f['path']),source_sha256=hashes[str(f['path'])],source_frame=f['row']['source_frame'],
            source_frame_count=f['source_frame_count'],
            timestamp_s=f['timestamp_s'],source_shape=f['source_shape'],fit_shape=list(f['depth'].shape),
            intrinsics_original=f['raw_K'].tolist(),intrinsics_used=f['K'].tolist(),
            invalid_depth_pixels=int((~((f['depth']>.1)&(f['depth']<1.5))).sum()),
            roi_valid_pixels=int(f['valid'].sum()),T_reference_from_camera=T.tolist() if info['accepted'] else None,
            final_scale=1. if info['accepted'] else None,rotation_determinant=float(np.linalg.det(T[:3,:3])) if info['accepted'] else None,fit=info,
            landmark_residual_vectors_in_reference_m=expression,
            annotation_origin=f['row'].get('annotation_origin','manual_seed'),
            tracking=f['row'].get('tracking'),
            expression_policy='shared upper identity; lower-face reference-anchored same-burst depths only, fixed 1.5mm consistency; other state/view burst excluded',
            selection_reason='real short-burst RGB tracking from manually annotated representative; no synthetic views'))
    frames,transforms,_ = map(list,zip(*selected))
    ids = [f['row']['id'] for f in frames]
    missing_refs = set(manifest['references'].values())-set(ids)
    if missing_refs:
        dump(a.output/'failed_poses.json',all_frame_records)
        raise ValueError('reference registration failed: '+str(missing_refs))
    outputs = {}
    diagnostics = []
    for state,reference_id in manifest['references'].items():
        rindex = ids.index(reference_id)
        local_ref = frames[rindex]
        state_dir = a.output/state
        state_dir.mkdir()
        coarse,count,bits,layers,upper = fuse(frames,transforms,ref,reference_id,state)
        fine = refine_depth(coarse,count,upper)
        np.savez_compressed(state_dir/'depth_and_validity.npz',coarse_depth_m=coarse,
            refined_depth_m=fine,depth_valid=np.isfinite(coarse),fusion_contribution_count=count,
            fusion_contributor_bits=bits,frame_depth_layers_m=layers,shared_upper_mask=upper,
            intrinsics_common=ref['K'],T_output_from_common=np.linalg.inv(transforms[rindex]))
        state_stats = {}
        for stage,depth in [('coarse',coarse),('refined',fine)]:
            vertices,faces,pixels = mesh_from_depth(depth,ref['K'])
            counts,support_bits,residuals = observation_support(vertices,faces,frames,transforms)
            # Isolated unsupported vertices are not made into plausible completion.
            keep = (counts[faces]>0).all(1)
            faces = faces[keep]
            used = np.unique(faces)
            remap = np.full(len(vertices),-1,int)
            remap[used] = np.arange(len(used))
            vertices,faces,pixels = vertices[used],remap[faces],pixels[used]
            counts,support_bits,residuals = counts[used],support_bits[used],residuals[:,used]
            selected,faces,component_stats = principal_component(vertices,faces)
            vertices,pixels = vertices[selected],pixels[selected]
            counts,support_bits,residuals = counts[selected],support_bits[selected],residuals[:,selected]
            if len(faces)<100:
                raise ValueError('insufficient supported mesh')
            to_output = np.linalg.inv(transforms[rindex])
            vout = transform(vertices,to_output)
            mesh_path = state_dir/('face_'+stage+'_m.obj')
            write_obj(mesh_path,vout,faces)
            np.savez_compressed(state_dir/('support_'+stage+'.npz'),vertices_m=vout,faces=faces,
                common_pixels_yx=pixels,observation_count=counts,observation_bits=support_bits,
                signed_axial_residual_m=residuals,
                fusion_contributor_count=count[pixels[:,0],pixels[:,1]],
                fusion_contributor_bits=bits[pixels[:,0],pixels[:,1]],
                shared_identity_upper=upper[pixels[:,0],pixels[:,1]],frame_ids=np.asarray(ids))
            dump(state_dir/('landmarks_'+stage+'.json'),landmarks_sidecar(local_ref,vout,
                transforms[rindex],to_output,pixels,stage))
            state_stats[stage] = dict(vertices=len(vout),triangles=len(faces),mesh_sha256=sha(mesh_path),
                finite=bool(np.isfinite(vout).all()),supported_at_least_one_fraction=float(np.mean(counts>0)),
                supported_at_least_two_fraction=float(np.mean(counts>=2)),
                actual_multiframe_fusion_fraction=float(np.mean(count[pixels[:,0],pixels[:,1]]>=2)),
                extent_xyz_m=np.ptp(vout,axis=0).tolist(),unobserved_completion_vertices=0,
                connectivity_filter=component_stats)
            if not a.skip_diagnostics and stage=='refined':
                for frame,T in zip(frames,transforms):
                    if frame['row']['state']!=state:
                        continue
                    posed = transform(vertices,np.linalg.inv(T))
                    diagnostics.append(dict(state=state,frame_id=frame['row']['id'],
                        stage=stage,metrics=diagnostic(posed,faces,frame),
                        independence='training-data diagnostic; poses fitted on same data',
                        evaluation_stride_original_px=a.stride*2))
        outputs[state] = dict(reference_frame_id=reference_id,source_frame=local_ref['row']['source_frame'],
            source=str(local_ref['path']),timestamp_s=local_ref['timestamp_s'],
            T_output_from_common=np.linalg.inv(transforms[rindex]).tolist(),
            geometry=state_stats,unit='m',coordinates='X_right_Y_down_Z_forward',
            missing_views=sorted({'front','left','right','up'}-{f['row']['view'] for f in frames if f['row']['state']==state}))
    report = dict(schema='head_rgbd.a3_result.v1',method='A3 independent model-free observed-face prototype',
        author_method_reproduction=False,manifest_sha256=sha(a.manifest),
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('a3_geometry.py')]},
        common_reference_frame=common_id,frames=all_frame_records,active_frame_ids=ids,outputs=outputs,
        source_frame_selection=[dict(source=str(f['path']),available=f['source_frame_count'],
            used=[g['row']['source_frame'] for g in frames if g['path']==f['path']],
            excluded=[dict(source_frame=i,reason='not manually annotated; one real representative per burst, no temporal-model claim')
                for i in range(f['source_frame_count']) if i not in [g['row']['source_frame'] for g in frames if g['path']==f['path']]])
                for f in {str(f['path']):f for f in frames}.values()],
        manifest_excluded_frames=[dict(id=r['id'],reason='front-only control excludes genuine oblique observations'
            if a.front_only and r['view']!='front' else 'registration guard failed; see frame fit record')
            for r in manifest['frames'] if r['id'] not in [f['row']['id'] for f in frames]],
        budget=dict(icp_iterations_per_gate=a.icp_iterations,gates_m=[.015,.008,.005],
                    stride=a.stride,front_only=a.front_only,cpu_diagnostics=not a.skip_diagnostics,
                    refine_passes=2,support_tolerance_m=.003,lower_fusion_gate_m=.0015,max_triangle_edge_m=.006),
        calibration_chain=dict(raw_encoding='uint16 millimetres',raw_to_m=1/1000,
            geometric_scale_optimized=False,all_frame_scales=1.,
            pose_convention='p_common = R @ p_camera_metres + t_metres; column homogeneous T',
            intrinsic_resize='fx fy cx cy divided by pixel stride; actual [::stride] pixels, no transpose',
            rgb_depth_alignment='capture claims physical alignment; not independently verified'),
        diagnostics=diagnostics,elapsed_s=time.perf_counter()-started,
        performance=dict(device='CPU',cuda_used=False,gpu_memory='not applicable, no CUDA context',
            peak_process_working_set_mib=peak_working_set(),
            timing_scope='raw loading + manual-landmark lifting + SE3 fit + fusion + export + optional CPU diagnostics; annotation time excluded'),
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        limitations=['manual per-frame ROI/landmarks required; not fully automatic',
            '2.5D common-camera surface; unseen/overhanging head regions not recovered',
            'no learned shape/expression model; lower-face sharing restricted to reference-compatible same-burst measurements',
            'identity upper shared; expression reference-specific; no continuous temporal inference across gaps',
            'no independent calibration, scan, silhouette truth or external poses',
            'RGB-D alignment and sensor metric calibration not independently measured'],
        acceptance={k:'unverified' for k in ['independent_front_surface_error','independent_side_surface_error',
            'manual_contour_error','independent_surface_coverage','front_non_degradation','jaw_accuracy','facial_thickness_accuracy']},
        publication=dict(raw_data=False,landmarks=False,meshes=False,models=False,
            reason='no explicit biometric-data publication authorization; results stay local'))
    dump(a.output/'report.json',report)
    print(json.dumps(dict(output=str(a.output),elapsed_s=report['elapsed_s'],outputs=outputs),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
