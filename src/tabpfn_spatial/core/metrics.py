"""Scores for probabilistic predictions: the numbers we pitch.

All functions are O(n q) for n rows and q quantiles (or k classes).
"""

import numpy as np

from tabpfn_spatial.core.dist import QuantileDist


def crps(dist: QuantileDist, y: np.ndarray) -> float:
    """Mean CRPS, approximated as twice the mean pinball loss over ``dist.levels``.

    Lower is better; rewards forecasts that are both sharp and correct.
    """
    y = np.asarray(y, dtype=float)[:, None]
    diff = y - dist.values
    pinball = np.maximum(dist.levels * diff, (dist.levels - 1) * diff)
    return float(2 * pinball.mean())


def coverage(dist: QuantileDist, y: np.ndarray, level: float = 0.9) -> float:
    """Fraction of ``y`` inside the central ``level`` interval (calibrated: ~= level)."""
    lo, hi = dist.interval(level)
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-class probability (0 is perfect)."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)
