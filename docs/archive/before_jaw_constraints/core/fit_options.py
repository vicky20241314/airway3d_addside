"""Tunable integration defaults; distances are metres, depth images millimetres."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FitOptions:
    shape_components: int = 100
    expression_components: int = 30
    # Prefixes are unlocked cumulatively; the last pass always opens all bases.
    basis_levels: tuple = ((20, 5), (50, 10), (100, 30))
    # Match the old application's neck/jaw flexion degrees of freedom.
    pose_indices: tuple = (0, 3)
    stage1_outer: int = 10
    stage2_outer: int = 20  # per basis level
    inner_steps: int = 12
    max_points: int = 4000
    damping: float = 1e-2
    shape_reg: float = 1e-6
    expression_reg: float = 1e-6
    gradient_tol: float = 1e-10
    parameter_step_tol: float = 1e-7
    vertex_step_tol_m: float = 1e-6
    loss_relative_tol: float = 1e-4
    plateau_vertex_tol_m: float = 2e-6
    plateau_patience: int = 4
    min_outer: int = 6
    max_rejections: int = 6
    seed: int = 1234

    def schedule(self):
        levels = []
        for shape, expr in (*self.basis_levels, (self.shape_components, self.expression_components)):
            pair = (min(shape, self.shape_components), min(expr, self.expression_components))
            if min(pair) < 1 or (levels and (pair[0] < levels[-1][0] or pair[1] < levels[-1][1])):
                raise ValueError('Basis levels must be positive and nondecreasing')
            if not levels or pair != levels[-1]:
                levels.append(pair)
        return tuple(levels)

