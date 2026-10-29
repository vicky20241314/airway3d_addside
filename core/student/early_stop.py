"""Early-stop helpers for iterative FLAME Adam (Stage2) and student vertex ICP."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdamEarlyStop:
    """Relative plateau stop for inner Adam steps and successive outer ICP refreshes."""

    inner_rel_tol: float = 3e-4
    inner_patience: int = 20
    inner_min_steps: int = 40
    # Correspondence refresh keeps shaving ~1–3% per outer; 0.2% vs best never fired.
    outer_rel_tol: float = 5e-2
    outer_patience: int = 2
    outer_min_steps: int = 4


STAGE2_EARLY_STOP = AdamEarlyStop()


@dataclass(frozen=True)
class StudentEarlyStop:
    """Stop wasted student ICP iters once the sampled residual plateaus."""

    step_tol_m: float = 1e-4
    rmse_rel_tol: float = 1e-2
    rmse_patience: int = 2
    rmse_min_iters: int = 3


STUDENT_EARLY_STOP = StudentEarlyStop()


class ConsecutiveRelativeStop:
    """Stop when successive values change by less than rel_tol for `patience` steps."""

    def __init__(self, rel_tol: float, patience: int, min_steps: int):
        if rel_tol <= 0 or patience < 1 or min_steps < 1:
            raise ValueError("rel_tol, patience, and min_steps must be positive")
        self.rel_tol = float(rel_tol)
        self.patience = int(patience)
        self.min_steps = int(min_steps)
        self.prev = None
        self.streak = 0
        self.n = 0

    def step(self, value: float) -> bool:
        value = float(value)
        self.n += 1
        if self.prev is None:
            self.prev = value
            return False
        denom = max(abs(self.prev), abs(value), 1e-12)
        rel = abs(value - self.prev) / denom
        self.prev = value
        if rel < self.rel_tol:
            self.streak += 1
        else:
            self.streak = 0
        return self.n >= self.min_steps and self.streak >= self.patience


class BestPatienceStop:
    """Stop when `value` fails to beat the best by rel_tol for `patience` updates."""

    def __init__(self, rel_tol: float, patience: int, min_steps: int):
        if rel_tol <= 0 or patience < 1 or min_steps < 1:
            raise ValueError("rel_tol, patience, and min_steps must be positive")
        self.rel_tol = float(rel_tol)
        self.patience = int(patience)
        self.min_steps = int(min_steps)
        self.best = None
        self.bad = 0
        self.n = 0

    def step(self, value: float) -> bool:
        value = float(value)
        self.n += 1
        if self.best is None or value < self.best * (1.0 - self.rel_tol):
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.n >= self.min_steps and self.bad >= self.patience


def inner_stop_from(cfg: AdamEarlyStop) -> ConsecutiveRelativeStop:
    return ConsecutiveRelativeStop(cfg.inner_rel_tol, cfg.inner_patience, cfg.inner_min_steps)


def outer_stop_from(cfg: AdamEarlyStop) -> ConsecutiveRelativeStop:
    return ConsecutiveRelativeStop(cfg.outer_rel_tol, cfg.outer_patience, cfg.outer_min_steps)


def as_report(cfg, **extra) -> dict:
    report = asdict(cfg)
    report.update(extra)
    return report
