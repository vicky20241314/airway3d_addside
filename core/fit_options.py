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
    # Observation-supported lower face, validated on paired captures.
    jaw_contour_weight: float = .5
    jaw_contour_huber_m: float = .003
    jaw_region_balance: bool = True
    jaw_max_yaw_deg: float = 20.
    # Explicit multiview ablations. joint/legacy reproduce the previous losses.
    registration_fallback: bool = True
    side_expression_policy: str = 'upper_face'
    side_landmark_mode: str = 'legacy'
    side_landmark_huber_m: float = .004
    side_landmark_gross_m: float = .020
    pose_strategy: str = 'prior'
    pose_prior_rotation_std_deg: float = 6.
    pose_prior_translation_std_m: float = .015
    pose_prior_weight: float = 1e-6
    side_contour_weight: float = 0.
    side_contour_noise_px: float = 2.
    side_contour_huber_px: float = 3.
    canonical_shape_enabled: bool = False
    canonical_shape_modes: int = 12
    canonical_shape_max_m: float = .003
    canonical_shape_reg: float = .05
    refinement_source: str = 'stage2'
    front_initialization_cache: str | None = None

    def algorithm_version(self, stage3_algorithm):
        return 'cross-v4-jaw-supported' if (self.jaw_contour_weight>0 or self.jaw_region_balance) else stage3_algorithm

    def schedule(self):
        if self.pose_strategy not in ('joint','prior','fixed'):
            raise ValueError('pose_strategy must be joint, prior or fixed')
        if self.side_landmark_mode not in ('legacy','soft') or self.side_expression_policy not in ('strict','upper_face'):
            raise ValueError('Invalid side landmark/expression policy')
        import math
        positive=(self.pose_prior_rotation_std_deg,self.pose_prior_translation_std_m,
            self.side_landmark_huber_m,self.side_landmark_gross_m,self.side_contour_noise_px,
            self.side_contour_huber_px,self.canonical_shape_max_m,self.canonical_shape_reg)
        if any(not math.isfinite(x) or x<=0 for x in positive):raise ValueError('Invalid multiview noise/guard scale')
        if any(not math.isfinite(x) or x<0 for x in (self.pose_prior_weight,self.side_contour_weight)):
            raise ValueError('Invalid multiview constraint weight')
        if not 1<=self.canonical_shape_modes<=40 or self.canonical_shape_max_m>.003:
            raise ValueError('Canonical correction supports 1..40 modes and at most 3 mm')
        if self.refinement_source not in ('stage2','shape_corrected'):
            raise ValueError('Unknown refinement source')
        if self.refinement_source=='shape_corrected' and not self.canonical_shape_enabled:
            raise ValueError('shape_corrected refinement requires canonical_shape_enabled')
        levels = []
        for shape, expr in (*self.basis_levels, (self.shape_components, self.expression_components)):
            pair = (min(shape, self.shape_components), min(expr, self.expression_components))
            if min(pair) < 1 or (levels and (pair[0] < levels[-1][0] or pair[1] < levels[-1][1])):
                raise ValueError('Basis levels must be positive and nondecreasing')
            if not levels or pair != levels[-1]:
                levels.append(pair)
        return tuple(levels)

