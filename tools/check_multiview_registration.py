"""Fast saved-session registration check, without FLAME fitting or refinement."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.multiview_data import load_session,validate_session
from core.multiview_registration import register_to_front,save_registration_diagnostics
from core.pipeline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output')
    a=p.parse_args();observations=validate_session(load_session(a.session))
    folder=a.output/('registration_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    fronts={o.state:o for o in observations if o.view=='front'};rows=[]
    for o in observations:
        if o.view=='front':continue
        try:
            T,diag,source,target=register_to_front(fronts[o.state],o)
            save_registration_diagnostics(folder,o.view_id,T,diag,source,target)
            rows.append(dict(view_id=o.view_id,status='passed',**diag))
            print(f"{o.view_id}: face RMSE={diag['rmse_m']*1000:.3f}mm; neck={diag['neck_check']}",flush=True)
        except Exception as exc:
            diag=getattr(exc,'diagnostics',{})
            if hasattr(exc,'transform'):
                save_registration_diagnostics(folder,o.view_id,exc.transform,diag,exc.source,exc.target,failed=True)
            rows.append(dict(view_id=o.view_id,status='failed',error=str(exc),diagnostics=diag))
            print(f'{o.view_id}: {exc}',flush=True)
    failed=any(row['status']=='failed' for row in rows)
    write_json(folder/'report.json',dict(status='failed' if failed else 'passed',stage='registration_only',
        session=str(a.session.resolve()),version=(ROOT/'VERSION').read_text().strip(),views=rows,
        real_side_effect_validated=False))
    print(folder,flush=True)
    return 1 if failed else 0


if __name__=='__main__':sys.exit(main())
