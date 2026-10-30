"""Saved solver overlays and a readable participation summary (not evaluation)."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from core.multiview_geometry import project


def save_fit_diagnostics(problem,theta,folder):
    folder=Path(folder);vertices,landmarks=problem.forward(theta)
    lines=['# Reconstruction solver diagnostics','',
        'All residuals below use dynamic fitting correspondences; they are not held-out accuracy.',
        'Orange: projected model landmarks. Cyan: reliable observations. Green: supported observed profile.',
        '', '| View | Sampled / observed | Kept correspondences | Landmarks | Profile equations |',
        '|---|---:|---:|---:|---:|']
    for b,(o,diag) in enumerate(zip(problem.observations,problem.view_diagnostics)):
        image=Image.fromarray(o.rgb.astype(np.uint8));draw=ImageDraw.Draw(image)
        uv,_=project(landmarks[b].cpu().numpy(),o.camera)
        for i in diag['landmark_ids']:
            for xy,color in ((uv[i],'orange'),(o.uv[i],'cyan')):
                x,y=xy
                if np.isfinite(xy).all():draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        if problem.profiles and problem.profiles[b] is not None:
            for x,y in problem.profiles[b]['uv']:draw.ellipse((x-1,y-1,x+1,y+1),fill='lime')
        draw.rectangle((0,0,min(image.width,640),24),fill='black')
        draw.text((6,6),f'Stage2 | {o.view_id} | IN FIT, dynamic correspondences',fill='white')
        image.save(folder/f'stage2_overlay_{o.view_id}.png')
        lines.append(f"| {o.view_id} | {diag['sampled_points']} / {diag['observed_points']} | {diag['valid_points']} | {len(diag['landmark_ids'])} | {diag.get('profile_equations',0)} |")
    lines+=['','Relative pose drift from registration (rotation degrees / translation mm):','',
        '```json',__import__('json').dumps(problem.iteration_diagnostics(theta,2)['relative_pose_from_registration'],indent=2),'```']
    (folder/'solver_diagnostics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def save_geometry_changes(folder):
    """Rebuild local FLAME geometry from saved coefficients; no refitting.

    Rigid camera drift is excluded; state expression/joint changes are included.
    All maps use an explicit 0..6 mm color scale, independent of their maximum.
    """
    import json
    import torch
    import trimesh
    from core.fit_options import FitOptions
    from core.integrated_fitting import make_flame
    from core.pipeline import write_json
    folder=Path(folder);report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if report['status']!='complete':raise ValueError('Geometry maps require a completed reconstruction')
    options=FitOptions(**report['fit_configuration'])
    with np.load(folder/'flame_parameters.npz') as a:parameters={k:a[k] for k in a.files}
    states=parameters['states'];flame=make_flame(options,'cpu',batch=len(states),dtype=torch.float64)
    with np.load(folder/'front_initialization.npz') as a:
        shapes=np.stack([a[f'{s}_shape'] for s in states]);expressions=np.stack([a[f'{s}_expression'] for s in states]);poses=np.stack([a[f'{s}_pose'] for s in states])
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float64)
    with torch.no_grad():
        initial,_=flame(shape_params=tensor(shapes),expression_params=tensor(expressions),pose_params=tensor(poses))
        final,_=flame(shape_params=tensor(np.tile(parameters['shared_shape'],(len(states),1))),
            expression_params=tensor(parameters['expression_by_view']),pose_params=tensor(parameters['pose_by_view']))
    maps=[]
    def save(state,kind,vertices,change):
        distance=np.linalg.norm(change,axis=1)*1000;fraction=np.clip(distance/6.,0,1)
        color=np.column_stack((255*fraction,100*(1-fraction),255*(1-fraction),np.full(len(distance),255))).astype(np.uint8)
        path=folder/state/f'{kind}_displacement_0_to_6mm.ply'
        trimesh.Trimesh(vertices,flame.faces_tensor.numpy(),vertex_colors=color,process=False).export(path)
        maps.append(dict(state=state,kind=kind,path=str(path.relative_to(folder)),max_mm=float(distance.max()),mean_mm=float(distance.mean())))
    for state in ('closed','open'):
        i=int(np.flatnonzero(states==state)[0]);v=final[i].numpy()
        save(state,'stage2_from_frontal_initialization',v,v-initial[i].numpy())
    if (folder/'canonical_shape.npz').exists():
        with np.load(folder/'canonical_shape.npz') as a:offset=a['canonical_offset']
        with torch.no_grad():
            corrected,_=flame(shape_params=tensor(np.tile(parameters['shared_shape'],(len(states),1))),
                expression_params=tensor(parameters['expression_by_view']),pose_params=tensor(parameters['pose_by_view']),canonical_offset=tensor(offset))
        for state in ('closed','open'):
            i=int(np.flatnonzero(states==state)[0]);save(state,'shape_corrected_from_stage2',corrected[i].numpy(),corrected[i].numpy()-final[i].numpy())
    write_json(folder/'geometry_change_maps.json',dict(color_scale_mm=[0,6],geometry_coordinates='FLAME local posed coordinates; rigid external camera alignment removed',
        stage2_change_includes='identity, state expression and joint pose',independent_accuracy_validation=False,maps=maps))
