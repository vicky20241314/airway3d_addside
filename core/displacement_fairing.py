"""Bounded displacement-only fairing; never smooth baseline mesh positions."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def fair_displacement(base, current, ids, laplacian, protected, guard, *,
                      strength=1., max_change_m=.0001, max_total_m=.002):
    """Minimize ||X||² + strength*||L(D+X)||² with frozen protected support.

    L contains every mesh row but only the active columns. X is a correction
    to the already accepted student D. A 0.1 mm per-vertex correction also
    bounds motion of each barycentric surface point by 0.1 mm (same topology).
    """
    if (guard is None or not np.isfinite([strength,max_change_m,max_total_m]).all()
            or min(strength,max_change_m,max_total_m)<=0):
        raise ValueError('Fairing requires a guard and positive finite bounds')
    delta=current[ids]-base[ids]
    free=np.flatnonzero(~np.asarray(protected,bool))
    report=dict(strength=strength,max_change_m=max_change_m,n_free=len(free),
                n_protected=int(np.count_nonzero(protected)),status='no_free_vertices')
    if not len(free):return current.copy(),report
    L=laplacian;F=L[:,free];ld=L@delta
    x=spsolve((sparse.eye(len(free))+strength*(F.T@F)).tocsc(),-strength*(F.T@ld))
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite displacement fairing')
    x*=np.minimum(1.,max_change_m/np.maximum(np.linalg.norm(x,axis=1),1e-12))[:,None]
    direction=np.zeros_like(base);direction[ids[free]]=x
    for _ in range(20):
        trial=current+direction
        bad=guard.inspect(trial)[0]
        over=np.flatnonzero(np.linalg.norm(trial-base,axis=1)>max_total_m+1e-12)
        blocked=np.union1d(bad,over)
        if not len(blocked):break
        direction[blocked]*=.5
    before=float(np.sum(ld**2));energy0=strength*before
    result=current.copy();alpha=1.;accepted=False
    for _ in range(20):
        change=alpha*direction
        candidate=current+change
        ld_new=L@(candidate[ids]-base[ids])
        energy=float(np.sum(change**2)+strength*np.sum(ld_new**2))
        if (energy < energy0-1e-20 and not len(guard.inspect(candidate)[0])
                and np.linalg.norm(candidate-base,axis=1).max()<=max_total_m+1e-12):
            result=candidate;accepted=True;break
        alpha*=.5
    assert np.array_equal(result[ids[np.asarray(protected,bool)]],current[ids[np.asarray(protected,bool)]])
    report.update(status='accepted' if accepted else 'kept_student_result',
                  correction_max_mm=float(np.linalg.norm(result-current,axis=1).max()*1000),
                  displacement_laplacian_norm_before_m=float(np.sqrt(before)),
                  displacement_laplacian_norm_after_m=float(np.linalg.norm(L@(result[ids]-base[ids]))),
                  step_scale=alpha if accepted else 0.)
    return result,report
