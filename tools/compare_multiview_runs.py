"""Fixed-camera diagnostics and renders for saved reconstructions, no fitting."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from core.multiview_data import load_session
from core.multiview_geometry import RigidCamera
from core.multiview_evaluation import evaluate_view,summarize
from core.pipeline import write_json


def render_camera(mesh,transform,observation,renderer):
    import pyrender
    v=transform.to_view(mesh.vertices)
    surface=trimesh.Trimesh(v,mesh.faces,process=False)
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(.68,.73,.8,1.),metallicFactor=0.,roughnessFactor=.9)
    scene=pyrender.Scene(bg_color=[1.,1.,1.,1.],ambient_light=[.35,.35,.35])
    scene.add(pyrender.Mesh.from_trimesh(surface,material=material,smooth=True))
    c=observation.camera
    scene.add(pyrender.IntrinsicsCamera(c.fx,c.fy,c.cx,c.cy,znear=.01,zfar=3.),pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1.,1.,1.],intensity=2.),pose=np.eye(4))
    return renderer.render(scene)[0][:,:,:3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True,help='six-view run with initial side-to-front registration JSONs')
    p.add_argument('--run',nargs=3,action='append',metavar=('LABEL','FOLDER','GEOMETRY'),required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output')
    p.add_argument('--max-points',type=int,default=4000)
    a=p.parse_args();observations=load_session(a.reference_run/'prepared/session.json')
    transforms={};skipped=[]
    for o in observations:
        if o.view=='front':transforms[o.view_id]=RigidCamera(np.eye(3),np.zeros(3))
        else:
            path=a.reference_run/f'registration_{o.view_id}.json'
            if not path.exists():path=a.reference_run/f'registration_{o.view_id}_failed.json'
            if not path.exists():
                skipped.append(o.view_id);continue
            row=json.loads(path.read_text(encoding='utf-8'))
            transforms[o.view_id]=RigidCamera(row['R_side_to_front'],row['t_side_to_front']).inverse()
    observations=[o for o in observations if o.view_id in transforms]
    models=[]
    for label,folder,geometry in a.run:
        folder=Path(folder)
        if geometry not in ('stage2','stage3'):p.error('GEOMETRY must be stage2 or stage3')
        status=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if status['status']!='complete':raise ValueError('Compare completed results only')
        if geometry=='stage3' and not status['stage3_enabled']:raise ValueError('Requested refinement is absent')
        meshes={s:trimesh.load(folder/s/(f'mesh_{s}_stage2.obj' if geometry=='stage2' else f'mesh_{s}.obj'),process=False,force='mesh') for s in ('closed','open')}
        fitted=status if 'view_ids' in status else json.loads((Path(status['source'])/'report.json').read_text(encoding='utf-8'))
        models.append((label,folder,geometry,meshes,set(fitted['view_ids'])))
    out=a.output/('comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',reference_session=str((a.reference_run/'prepared/session.json').resolve()),
        camera_policy='same initial face-ICP side-to-front transforms for ALL results; never refit to each mesh',
        independent_accuracy_validation=False,reviewed_contours_available=False,max_points=a.max_points,runs=[],
        missing_registration_view_ids=skipped)
    write_json(out/'comparison.json',report)
    import pyrender
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    except OSError:font=ImageFont.load_default()
    try:
        for label,folder,geometry,meshes,fitted_ids in models:
            rows=[]
            for o in observations:
                row=evaluate_view(meshes[o.state],o,transforms[o.view_id],max_points=a.max_points)
                row['role']='in_fit'
                row['input_use']='geometry_fitted' if o.view_id in fitted_ids else 'camera_registration_only'
                rows.append(row)
            summary={k.replace('in_fit_','observed_'):v for k,v in summarize(rows).items()}
            for row in rows:row['role']='shared_observation_diagnostic'
            report['runs'].append(dict(label=label,folder=str(folder.resolve()),geometry=geometry,views=rows,summary=summary))
            print(label,json.dumps(summary),flush=True)
        for state in ('closed','open'):
            views=sorted([o for o in observations if o.state==state],key=lambda o:('front','left','right').index(o.view))
            cell_w,cell_h=300,340;canvas=Image.new('RGB',((len(models)+1)*cell_w,len(views)*(cell_h+30)+35),'white')
            draw=ImageDraw.Draw(canvas)
            for col,name in enumerate(['Observed RGB']+[m[0] for m in models]):draw.text((col*cell_w+10,7),name,fill='black',font=font)
            for r,o in enumerate(views):
                finite=o.uv[np.isfinite(o.uv).all(1)];lo=finite.min(0);hi=finite.max(0);extent=hi-lo
                h,w=o.depth.shape
                box=(max(0,int(lo[0]-.3*extent[0])),max(0,int(lo[1]-.4*extent[1])),
                     min(w,int(hi[0]+.3*extent[0])),min(h,int(hi[1]+.5*extent[1])))
                renderer=pyrender.OffscreenRenderer(w,h)
                try:images=[o.rgb]+[render_camera(m[3][state],transforms[o.view_id],o,renderer) for m in models]
                finally:renderer.delete()
                for col,rgb in enumerate(images):
                    tile=Image.fromarray(np.asarray(rgb,dtype=np.uint8)).crop(box)
                    tile.thumbnail((cell_w,cell_h),Image.Resampling.LANCZOS)
                    x=col*cell_w+(cell_w-tile.width)//2;y=35+r*(cell_h+30)
                    canvas.paste(tile,(x,y));draw.text((col*cell_w+10,y+cell_h+3),o.view_id,fill='black',font=font)
            canvas.save(out/f'{state}_comparison.png')
        report['status']='complete';write_json(out/'comparison.json',report);print(out,flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc));write_json(out/'comparison.json',report);raise


if __name__=='__main__':main()
