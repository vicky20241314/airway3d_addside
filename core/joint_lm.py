"""Shared-identity FLAME fitting. No Adam or DECA.

Analytic fixed-pose LBS derivatives follow final_project/jacobians.py via
release's lm_jacobians. Pose and global rotation use central differences.
Legacy geometric/landmark losses have individual Jacobian rows.
Surface correspondences and normals are frozen inside each LM solve.
"""
from dataclasses import dataclass
import math
import torch
from smplx.lbs import batch_rodrigues

from core.initial_surface import surface_correspondence, matched_points
from core.lm_jacobians import dV_shape_expr
from core.landmark_policy import ALIGN_START, ALIGN_STOP, ALIGN_EXCLUDED


@dataclass
class Layout:
    shape_n: int
    expr_n: int
    pose_indices: tuple
    frames: int = 2

    def __post_init__(self):
        self.shape = slice(0, self.shape_n)
        self.expr = []; self.pose = []; self.rot = []; self.trans = []
        cursor = self.shape_n
        for _ in range(self.frames):
            for slots, size in ((self.expr, self.expr_n), (self.pose, len(self.pose_indices)),
                                (self.rot, 3), (self.trans, 3)):
                slots.append(slice(cursor, cursor + size)); cursor += size
        self.size = cursor

    def active(self, stage):
        if stage == 2:
            return list(range(self.size))
        return [i for b in range(self.frames) for s in
                (self.pose[b], self.rot[b], self.trans[b]) for i in range(s.start, s.stop)]


def rigid_align(source, target, valid):
    """Column-vector rigid transform; never estimate scale from noisy depth."""
    finite = valid & torch.isfinite(target).all(-1) & torch.isfinite(source).all(-1)
    candidates = torch.zeros_like(finite)
    candidates[ALIGN_START:ALIGN_STOP] = True
    candidates &= finite
    good = candidates.clone()
    good[list(ALIGN_EXCLUDED)] = False
    # Match old_version's low-level fallback. The application/fit entry requires
    # 8 preferred alignment points, so normal captures never need this fallback.
    if int(good.sum()) < 3:
        good = candidates
    if int(good.sum()) < 3:
        raise ValueError('有效配准三维关键点少于 3 个，无法完成初始对齐')
    x, y = source[good], target[good]
    xc, yc = x.mean(0), y.mean(0)
    u, singular, vh = torch.linalg.svd((x-xc).T @ (y-yc))
    if float(singular[1]) < 1e-10:
        raise ValueError('三维关键点退化，无法确定头部姿态')
    correction = torch.eye(3, dtype=x.dtype, device=x.device)
    correction[-1, -1] = torch.linalg.det(vh.T @ u.T)
    r = vh.T @ correction @ u.T
    return r, yc - r @ xc


def project(points, camera, derivative=False):
    x, y, z = points.unbind(-1)
    z = torch.where(z.abs() < 1e-6, -torch.ones_like(z)*1e-6, z)
    uv = torch.stack((-camera.fx*x/z+camera.cx, camera.fy*y/z+camera.cy), -1)
    if not derivative:
        return uv
    jac = points.new_zeros((*points.shape[:-1], 2, 3))
    jac[..., 0, 0] = -camera.fx/z
    jac[..., 0, 2] = camera.fx*x/z.square()
    jac[..., 1, 1] = camera.fy/z
    jac[..., 1, 2] = -camera.fy*y/z.square()
    return uv, jac


def vertex_normals(vertices, faces):
    tri = vertices[faces]
    cross = torch.linalg.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals = torch.zeros_like(vertices)
    for k in range(3):
        normals.index_add_(0, faces[:, k], cross)
    return torch.nn.functional.normalize(normals, dim=-1)


class JointProblem:
    def __init__(self, flame, camera, points, lmk3d, lmk2d, masks, parsing, options, cameras=None):
        self.flame, self.camera, self.options = flame, camera, options
        self.cameras = cameras or [camera]*len(points)
        self.layout = Layout(options.shape_components, options.expression_components,
                             options.pose_indices, len(points))
        self.faces = flame.faces_tensor
        self.device, self.dtype = flame.v_template.device, flame.v_template.dtype
        self.points = points
        self.lmk3d, self.lmk2d, self.masks = lmk3d, lmk2d, masks
        self.theta = flame.v_template.new_zeros(self.layout.size)
        self.rotations = torch.eye(3, device=self.device, dtype=self.dtype).repeat(len(points), 1, 1)
        _, landmarks = self.forward(self.theta)
        for b in range(len(points)):
            r, t = rigid_align(landmarks[b], lmk3d[b], masks[b].clone())
            self.rotations[b] = r
            self.theta[self.layout.trans[b]] = t
        self.jaw=[]
        self.jaw_initialized=False
        if options.jaw_contour_weight>0 or options.jaw_region_balance:
            from core.jaw_constraints import prepare_jaw
            for b in range(len(points)):
                data=prepare_jaw(parsing[b],lmk2d[b].cpu().numpy(),lmk3d[b].cpu().numpy(),
                    masks[b].cpu().numpy(),self.cameras[b],points[b].cpu().numpy())
                data['ids']=torch.as_tensor(data['ids'],device=self.device,dtype=torch.long)
                data['normals']=torch.as_tensor(data['normals'],device=self.device,dtype=self.dtype)
                data['weights']=torch.as_tensor(data['weights'],device=self.device,dtype=self.dtype)
                self.jaw.append(data)
        self.semantic_ids = []
        self.mouth_pairs = {1: [], 2: []}
        for b in range(len(points)):
            ids = torch.tensor([8,21,22,33,50,51,52,56,57,58], device=self.device)
            valid = masks[b][ids] & torch.isfinite(lmk3d[b][ids]).all(-1)
            self.semantic_ids.append(ids[valid])
            for stage, pairs in ((1, ((50,58),(51,57),(52,56))),
                                 (2, ((61,67),(62,66),(63,65)))):
                target = lmk3d[b] if stage == 1 else lmk2d[b]
                self.mouth_pairs[stage].append([
                    (i,k) for i,k in pairs if bool(torch.isfinite(target[[i,k]]).all())
                    and (stage == 2 or bool(masks[b][i] & masks[b][k]))])
        # old_version averages semantic errors over all coordinates in both
        # frames, and mouth errors over all valid pairs in both frames.
        self.semantic_count = sum(len(ids) for ids in self.semantic_ids)
        self.mouth_counts = {s: sum(map(len, pairs)) for s,pairs in self.mouth_pairs.items()}
        self.context = []

    def parameters(self, theta):
        l = self.layout
        shape = theta[l.shape].unsqueeze(0).expand(l.frames, -1)
        expr = torch.stack([theta[s] for s in l.expr])
        pose = theta.new_zeros((l.frames, 6))
        for b, s in enumerate(l.pose):
            pose[b, list(l.pose_indices)] = theta[s]
        rot = batch_rodrigues(torch.stack([theta[s] for s in l.rot])) @ self.rotations
        trans = torch.stack([theta[s] for s in l.trans])
        return shape, expr, pose, rot, trans

    @torch.no_grad()
    def forward(self, theta):
        shape, expr, pose, rot, trans = self.parameters(theta)
        verts, landmarks = self.flame(shape_params=shape, expression_params=expr,
                                     pose_params=pose, transl=torch.zeros_like(trans))
        return (torch.einsum('bij,bvj->bvi', rot, verts)+trans[:, None],
                torch.einsum('bij,bkj->bki', rot, landmarks)+trans[:, None])

    @torch.no_grad()
    def vertex_jacobians(self, theta, stage):
        l = self.layout
        shape, expr, pose, rot, trans = self.parameters(theta)
        active = l.active(stage)
        columns = {i: j for j, i in enumerate(active)}
        jac = theta.new_zeros((l.frames, len(self.flame.v_template), 3, len(active)))
        if stage == 2:
            full_pose = torch.cat((self.flame.head_pose, pose, self.flame.eye_pose), 1)
            local = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(l.frames, -1, 3, 3)
            global_r = local.clone()
            for k in range(1, local.shape[1]):
                global_r[:, k] = global_r[:, int(self.flame.parents[k])] @ local[:, k]
            for b in range(l.frames):
                js, je = dV_shape_expr(self.flame, global_r[b], l.shape_n, l.expr_n)
                jac[b, :, :, l.shape] = torch.einsum('ij,vjp->vip', rot[b], js)
                jac[b, :, :, l.expr[b]] = torch.einsum('ij,vjp->vip', rot[b], je)
        eps = 1e-4 if theta.dtype == torch.float32 else 1e-5
        for b in range(l.frames):
            for s in (l.pose[b], l.rot[b]):
                for i in range(s.start, s.stop):
                    plus, minus = theta.clone(), theta.clone()
                    plus[i] += eps; minus[i] -= eps
                    jac[b, :, :, columns[i]] = (self.forward(plus)[0][b]-self.forward(minus)[0][b])/(2*eps)
            for k, i in enumerate(range(l.trans[b].start, l.trans[b].stop)):
                jac[b, :, k, columns[i]] = 1
        return jac

    @torch.no_grad()
    def landmark_embedding(self, theta, vertices):
        """The same full 68-point embedding as FLAME.forward, including chin 8.

        Contour selection is discrete, with a piecewise-constant derivative.
        The default neck/jaw flexion does not change the contour yaw bin.
        """
        pose = self.parameters(theta)[2]
        full_pose = torch.cat((self.flame.head_pose, pose, self.flame.eye_pose), 1)
        dyn_faces, dyn_bary = self.flame._find_dynamic_lmk_idx_and_bcoords(
            vertices, full_pose, self.flame.dynamic_lmk_faces_idx,
            self.flame.dynamic_lmk_bary_coords, self.flame.neck_kin_chain,
            dtype=self.dtype)
        static_faces = self.flame.lmk_faces_idx[None].expand(self.layout.frames, -1)
        static_bary = self.flame.lmk_bary_coords[None].expand(self.layout.frames, -1, -1)
        return (self.faces[torch.cat((dyn_faces, static_faces), 1)],
                torch.cat((dyn_bary, static_bary), 1))

    @torch.no_grad()
    def refresh(self, theta, stage):
        verts, landmarks = self.forward(theta)
        if stage==2 and self.jaw and not self.jaw_initialized:
            rotation=self.parameters(theta)[3]
            for b,data in enumerate(self.jaw):
                yaw=math.degrees(math.atan2(float(rotation[b,0,2]),float(rotation[b,2,2])))
                data['initial_stage2_yaw_deg']=yaw
                if abs(yaw)>self.options.jaw_max_yaw_deg:
                    data['ids']=data['ids'][:0];data['normals']=data['normals'][:0]
                    data['weights']=torch.full_like(data['weights'],1/len(data['weights']))
                    data['reason']='oblique_pose_fallback';data['group_budgets']='uniform_fallback'
            self.jaw_initialized=True
        self.context = []
        for b, points in enumerate(self.points):
            corr = surface_correspondence(verts[b], self.faces, points)
            # Keep surface correspondences untrimmed. Optional fixed regional
            # weights apply in residual(); contour IRLS is frozen per outer step.
            self.context.append(dict(points=points, corr=corr,
                normals=matched_points(vertex_normals(verts[b], self.faces), corr, normals=True)))
            if stage==2 and self.jaw:
                data=self.jaw[b];ids=data['ids']
                residual=((project(landmarks[b][ids],self.cameras[b])-self.lmk2d[b][ids])*data['normals']).sum(1)*data['pixel_to_m']
                self.context[-1]['jaw_irls']=torch.clamp(self.options.jaw_contour_huber_m/residual.abs().clamp_min(1e-12),max=1.)

    @torch.no_grad()
    def residual(self, theta, stage, need_jac=True):
        verts, landmarks = self.forward(theta)
        dj = self.vertex_jacobians(theta, stage) if need_jac else None
        landmark_tri, landmark_bary = self.landmark_embedding(theta, verts)
        active = self.layout.active(stage)
        mapping = {v: i for i,v in enumerate(active)}
        rs, js = [], []

        def add(r, j, scale=1.):
            rs.append((r*scale).reshape(-1))
            if need_jac:
                js.append((j*scale).reshape(r.numel(), -1))

        for b, ctx in enumerate(self.context):
            tri, bary = ctx['corr']
            match = (verts[b][tri]*bary[..., None]).sum(1)
            jm = (dj[b][tri]*bary[..., None, None]).sum(1) if need_jac else None
            diff = match-ctx['points']
            balanced=stage==2 and self.jaw and self.options.jaw_region_balance
            if balanced:
                weights=self.jaw[b]['weights'].sqrt()
                add(diff*weights[:,None],jm*weights[:,None,None] if need_jac else None,math.sqrt(.1))
            else:add(diff, jm, math.sqrt(.1/len(diff)))
            plane = (diff*ctx['normals']).sum(-1)
            jp = torch.einsum('ni,nip->np', ctx['normals'], jm) if need_jac else None
            if balanced:
                add(plane*weights,jp*weights[:,None] if need_jac else None,math.sqrt(.9))
            else:add(plane, jp, math.sqrt(.9/len(diff)))

            tri, bary = landmark_tri[b], landmark_bary[b]
            lm = landmarks[b]
            jl = (dj[b][tri]*bary[..., None, None]).sum(1) if need_jac else None
            if need_jac:
                # Embeddings were stored in float32 and may not sum to exactly
                # one. FLAME applies translation AFTER barycentric interpolation.
                for k, idx in enumerate(range(self.layout.trans[b].start, self.layout.trans[b].stop)):
                    jl[:, k, mapping[idx]] += 1-bary.sum(-1)
            if stage == 2:
                ids = self.semantic_ids[b]
                if len(ids):
                    add(lm[ids]-self.lmk3d[b][ids], jl[ids] if need_jac else None,
                        math.sqrt(2./(3*self.semantic_count)))
                if need_jac:
                    uv, proj_j = project(lm, self.cameras[b], True)
                    juv = proj_j @ jl
                else:
                    uv, juv = project(lm, self.cameras[b]), None
                if self.jaw and self.options.jaw_contour_weight>0:
                    data=self.jaw[b];jids=data['ids']
                    if len(jids):
                        normals=data['normals'];unit=data['pixel_to_m']
                        residual=((uv[jids]-self.lmk2d[b][jids])*normals).sum(1)*unit
                        derivative=torch.einsum('ni,nip->np',normals,juv[jids])*unit if need_jac else None
                        weight=(ctx['jaw_irls']*self.options.jaw_contour_weight/(len(jids)*self.layout.frames)).sqrt()
                        add(residual*weight,derivative*weight[:,None] if need_jac else None)

            for i, k in self.mouth_pairs[stage][b]:
                target = self.lmk3d[b] if stage == 1 else self.lmk2d[b]
                values, deriv = (lm, jl) if stage == 1 else (uv, juv)
                delta = values[i]-values[k]
                distance = delta.norm().clamp_min(1e-12)
                residual = distance-(target[i]-target[k]).norm()
                jac = (delta/distance) @ (deriv[i]-deriv[k]) if need_jac else None
                add(residual, jac, math.sqrt((10. if stage == 1 else 1e-5)/self.mouth_counts[stage]))

        regs = [(self.layout.shape, self.options.shape_reg)] if stage == 2 else []
        regs += [(s, self.options.expression_reg) for s in self.layout.expr] if stage == 2 else []
        for s, weight in regs:
            j = theta.new_zeros((s.stop-s.start, len(active))) if need_jac else None
            if need_jac:
                for row, idx in enumerate(range(s.start, s.stop)):
                    j[row, mapping[idx]] = 1
            add(theta[s], j, math.sqrt(weight))
        return torch.cat(rs), torch.cat(js) if need_jac else None


def solve_stage(problem, theta, stage, progress=lambda _: None, basis=None):
    options = problem.options
    count = options.stage1_outer if stage == 1 else options.stage2_outer
    full_active = problem.layout.active(stage)
    active = list(full_active)
    if stage == 2 and basis is not None:
        shape_n, expr_n = basis
        layout = problem.layout
        active = [i for i in active if
                  not (layout.shape.start+shape_n <= i < layout.shape.stop)
                  and not any(s.start+expr_n <= i < s.stop for s in layout.expr)]
    damping = options.damping
    history = []
    reason = 'max_outer'
    previous_loss = None
    plateau = 0
    for outer in range(count):
        label = f' shape={basis[0]}, expression={basis[1]}' if basis else ''
        progress(f'第 {stage} 阶段 LM{label}：{outer+1}/{count}')
        problem.refresh(theta, stage)
        before = theta.clone()
        r = jac = None
        accepted = rejected = 0
        inner_reason = 'max_inner'
        for step in range(options.inner_steps):
            if r is None:
                r, jac = problem.residual(theta, stage)
                if active != full_active:
                    jac = jac[:, [full_active.index(i) for i in active]]
                if not torch.isfinite(r).all() or not torch.isfinite(jac).all():
                    raise ValueError('LM 残差或雅可比出现非有限值')
                # Accumulate/solve the small normal equations in double precision.
                j64, r64 = jac.double(), r.double()
                h, g = j64.T@j64, j64.T@r64
                cost = .5*float(r64@r64)
                diagonal = h.diag().clamp_min(1e-10)
            if float(g.abs().max()) < options.gradient_tol:
                inner_reason = 'gradient'; break
            try:
                delta = torch.linalg.solve(h+damping*torch.diag(diagonal), -g).to(theta.dtype)
            except RuntimeError:
                damping = min(damping*10, 1e12)
                rejected += 1
                if rejected >= options.max_rejections:
                    inner_reason = 'linear_solve'; break
                continue
            candidate = theta.clone(); candidate[active] += delta
            candidate_verts, _ = problem.forward(candidate)
            valid = torch.isfinite(candidate_verts).all() and bool((candidate_verts[:,:,2] < -1e-4).all())
            candidate_r = problem.residual(candidate, stage, False)[0] if valid else None
            candidate_cost = .5*float(candidate_r.double().square().sum()) if candidate_r is not None else float('inf')
            if math.isfinite(candidate_cost) and candidate_cost < cost:
                theta = candidate
                damping = max(damping*.5, 1e-10)
                accepted += 1; rejected = 0
                r = jac = None
                if float(delta.norm()) < options.parameter_step_tol:
                    inner_reason = 'step'; break
            else:
                damping = min(damping*2, 1e12)
                rejected += 1
                if rejected >= options.max_rejections:
                    inner_reason = 'rejected'; break
        final_r = problem.residual(theta, stage, False)[0]
        final_loss = .5*float(final_r.double().square().sum())
        movement = float((problem.forward(theta)[0]-problem.forward(before)[0]).norm(dim=-1).max())
        relative = abs(final_loss-previous_loss)/max(abs(previous_loss),1e-12) if previous_loss is not None else float('inf')
        settled = accepted > 0 or inner_reason == 'gradient'
        plateau = plateau+1 if (settled and relative < options.loss_relative_tol
                                and movement < options.plateau_vertex_tol_m) else 0
        history.append(dict(outer=outer+1, loss=final_loss, accepted=accepted,
                            inner_reason=inner_reason, damping=damping, max_move_m=movement,
                            points=[len(c['points']) for c in problem.context]))
        if hasattr(problem,'iteration_diagnostics'):
            history[-1]['diagnostics']=problem.iteration_diagnostics(theta,stage)
        previous_loss = final_loss
        # Don't stop after an entirely rejected inner loop: refresh and retry.
        if outer+1 >= options.min_outer and settled and movement < options.vertex_step_tol_m:
            reason = 'vertex_step'; break
        if outer+1 >= options.min_outer and plateau >= options.plateau_patience:
            reason = 'loss_plateau'; break
    return theta, dict(solver='LM', stopped=reason, history=history,
                      basis=basis, active_parameter_count=len(active))


def solve_progressive(problem, theta, progress=lambda _: None):
    """Warm-start each larger prefix; higher coefficients remain zero until opened."""
    passes = []
    for basis in problem.options.schedule():
        theta, report = solve_stage(problem, theta, 2, progress, basis=basis)
        passes.append(report)
    return theta, dict(solver='progressive_LM', passes=passes)
