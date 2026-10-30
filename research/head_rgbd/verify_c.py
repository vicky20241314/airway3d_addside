"""Validate exported geometry/reference/evidence contracts against actual data."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from audit import dump,sha
from evaluate import raster_depth


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=json.loads((a.fit/'report.json').read_text(encoding='utf-8'))
    refs=json.loads((a.fit/'reference_geometry.json').read_text(encoding='utf-8'))
    params=np.load(a.fit/'parameters.npz',allow_pickle=False);support=np.load(a.fit/'support.npz',allow_pickle=False)
    rows={r['id']:r for r in report['frames']};sequence=json.loads(a.sequence.read_text(encoding='utf-8'))
    assert sha(a.sequence)==report['sequence_sha256']
    assert np.isfinite(params['refined_vertices']).all()
    assert np.max(np.abs(params['deviation_m']))<=.00300001
    assert np.max(np.abs(params['effective_deviation_m']))<=.00300001
    for v,r in zip(params['posed_vertices'],params['refined_vertices']):
        t=v[params['faces']];u=r[params['faces']]
        old=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);new=np.cross(u[:,1]-u[:,0],u[:,2]-u[:,0])
        area2=(old**2).sum(1)
        assert np.all((old*new).sum(1)[area2>1e-24]>=.04999*area2[area2>1e-24])
    assert np.all(params['deviation_m'][params['mouth_identity_excluded']]==0)
    assert all(report['frames'][i]['state']=='closed' for i in params['identity_selected'])
    assert support['observed_frame_mask'].shape==(len(report['frames']),params['posed_vertices'].shape[1])
    np.testing.assert_array_equal(support['observed_frames'],support['observed_frame_mask'].sum(0))
    for before,after in report['temporal_edges']:
        assert rows[before]['segment']==rows[after]['segment']
        assert 0<rows[after]['timestamp_s']-rows[before]['timestamp_s']<.2
    result={}
    guard_checks=[]
    if report.get('front_anchor'):
        anchor_path=Path(report['front_anchor']['path']);ar=json.loads((anchor_path/'report.json').read_text(encoding='utf-8'))
        ap=np.load(anchor_path/'parameters.npz',allow_pickle=False);ai={r['id']:i for i,r in enumerate(ar['frames'])}
        np.testing.assert_array_equal(params['shape'],ap['shape'])
        for i,row in enumerate(report['frames']):
            if row['id'] not in ai:continue
            j=ai[row['id']];source=next(r for r in sequence['frames'] if r['id']==row['id'])
            with np.load(a.sequence.parent/source['file'],allow_pickle=False) as d:
                current=raster_depth(params['refined_vertices'][i],params['faces'],d['intrinsics'],d['depth'].shape,4)
                previous=raster_depth(ap['refined_vertices'][j],ap['faces'],d['intrinsics'],d['depth'].shape,4)
                valid=d['face_mask'][::4,::4]&(d['depth'][::4,::4]>100)&(d['depth'][::4,::4]<1500)
                old=valid&np.isfinite(previous);new=valid&np.isfinite(current)
                np.testing.assert_array_equal(old,new)
                error=float(np.max(np.abs(current[old]-previous[old]))*1000)
                assert error<.0001,'front protected geometry changed beyond numerical precision'
                guard_checks.append(dict(frame_id=row['id'],maximum_depth_change_mm=error,coverage_identical=True))
    for state,reference in refs['references'].items():
        fid=reference['frame_id'];i=next(i for i,r in enumerate(report['frames']) if r['id']==fid)
        row=rows[fid];source=next(r for r in sequence['frames'] if r['id']==fid)
        with np.load(a.sequence.parent/source['file'],allow_pickle=False) as data:
            for j,valid in enumerate(data['valid']):
                if not valid:assert reference['observed_landmarks_m'][j] is None
            if not data['valid'][[62,66]].all():assert reference['observed_inner_lip_gap_mm'] is None
        for stage in ['coarse','refined']:
            mesh=trimesh.load(a.fit/reference[stage+'_mesh'],process=False)
            assert np.isfinite(mesh.vertices).all()
            assert .1<max(mesh.extents)<.5,'implausible metre scale'
            assert len(mesh.vertices)==params['posed_vertices'].shape[1]
            tri=np.asarray(mesh.faces)[np.asarray(reference['landmark_triangle_indices'])]
            lm=(np.asarray(mesh.vertices)[tri]*np.asarray(reference['landmark_barycentric'])[:,:,None]).sum(1)
            np.testing.assert_allclose(lm,reference[stage+'_landmarks_m'],atol=2e-7)
        result[state]=dict(frame_id=fid,source_frame=row['source_frame'],
            coarse_gap_mm=reference['coarse_inner_lip_gap_mm'],refined_gap_mm=reference['refined_inner_lip_gap_mm'],
            raw_depth_gap_mm=reference['observed_inner_lip_gap_mm'],invalid_depth_serialized_as_null=True,
            landmark_mesh_stage_and_coordinates_consistent=True)
    output=dict(status='passed',references=result,identity_frames=[report['frames'][i]['id'] for i in params['identity_selected']],
        motion_only_frames=[r['id'] for r in report['frames'] if not r['identity_keyframe']],
        temporal_edges=len(report['temporal_edges']),no_cross_gap_constraint=True,
        finite_meshes=True,metre_scale=True,maximum_normal_refinement_mm=float(np.abs(params['deviation_m']).max()*1000),
        support_label_counts=np.bincount(support['region_label'],minlength=3),
        front_guard_checks=guard_checks,
        refined_triangle_foldovers=0,
        metric_scope='export and modelling contracts only; no independent accuracy acceptance')
    dump(a.output,output);print(json.dumps(output,default=lambda x:x.tolist(),indent=2))


if __name__=='__main__':main()
