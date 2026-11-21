"""Select source hypotheses while preserving visibility of discarded probability."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceSelection:
    """Flattened cell indices, normalized mixture weights and retained original mass."""

    indices: np.ndarray
    weights: np.ndarray
    retained_mass: float
    target_reached: bool | None


def select_source_cells(
    probability: np.ndarray,
    walls: np.ndarray,
    max_sources: int,
    target_mass: float | None = None,
) -> SourceSelection:
    """Select up to ``max_sources`` open cells, optionally targeting original probability mass.

    The target and reported retained mass include any invalid mass on walls, so
    excluding such mass cannot silently appear to achieve the target. O(N log N).
    """
    prob = np.asarray(probability, dtype=float)
    walls = np.asarray(walls, dtype=bool)
    if prob.shape != walls.shape or prob.ndim != 2:
        raise ValueError("source probabilities must match the two-dimensional grid")
    if np.any(~np.isfinite(prob)) or np.any(prob < 0):
        raise ValueError("source probabilities must be finite and nonnegative")
    if not isinstance(max_sources, (int, np.integer)) or max_sources < 1:
        raise ValueError("max_sources must be a positive integer")
    if target_mass is not None and (not np.isfinite(target_mass) or not 0 < target_mass <= 1):
        raise ValueError("target_mass must be in (0, 1]")
    total = float(prob.sum())
    valid = np.where(walls, 0.0, prob).ravel()
    if not np.isfinite(total) or total <= 0 or valid.sum() <= 0:
        raise ValueError("source needs positive finite mass on open cells")
    order = np.argsort(valid)[::-1]
    order = order[valid[order] > 0][:max_sources]
    if target_mass is not None:
        desired = max(
            1,
            int(np.searchsorted(np.cumsum(valid[order]), target_mass * total - 1e-12 * total) + 1),
        )
        order = order[:desired]
    selected = valid[order]
    retained = float(selected.sum() / total)
    reached = None if target_mass is None else bool(retained + 1e-12 >= target_mass)
    return SourceSelection(order, selected / selected.sum(), retained, reached)
