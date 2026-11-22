"""Score a simulated field against a real experiment.

Concentrations are in different units (relative PID vs per-unit-release), so every metric is
scale-free:
- ``rho``: Spearman rank correlation over all observation points;
- ``auc_hot``: ROC AUC of the model ranking the real hottest 10 % of points ("does the model
  know where the gas is?"; 0.5 = chance);
- ``profile_r``: Pearson correlation of the normalised vertical profiles downwind of the
  leak, within 0.3 m of its axis (the question Steven asked: does height come out right?);
- ``width_ratio``: model / real crosswind spread (standard deviation of the
  concentration-weighted crosswind position, above 5 % of each field's peak) downwind;
  1 = right, < 1 the model plume is too narrow.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _downwind(xyz: np.ndarray, src: np.ndarray, direction: tuple[float, float],
              half_width: float = 0.3) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    d = np.asarray(direction)
    rel = xyz[:, :2] - src[:2]
    s, n = rel @ d, rel @ np.array([-d[1], d[0]])
    return (s > 0.1) & (np.abs(n) < half_width), n


def vertical_profile(xyz: np.ndarray, c: np.ndarray, src: np.ndarray,
                     direction: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Heights and mean concentration per height downwind, normalised to a peak of 1."""
    m, _ = _downwind(xyz, src, direction)
    z = np.round(xyz[m, 2], 3)
    zs = np.unique(z)
    p = np.array([np.nanmean(np.clip(c[m][z == v], 0, None)) for v in zs])
    return zs, p / max(np.nanmax(p), 1e-30)


def crosswind_spread(xyz: np.ndarray, c: np.ndarray, src: np.ndarray,
                     direction: tuple[float, float]) -> float:  # fmt: skip
    """Concentration-weighted standard deviation of crosswind position downwind (m).

    Weights are the concentration above 5 % of the field's own peak, so the measurement
    noise floor spread over the whole domain does not count as plume.
    """
    m, n = _downwind(xyz, src, direction, half_width=np.inf)
    w = np.clip(c[m] - 0.05 * np.nanmax(c[m]), 0, None)
    if w.sum() <= 0:
        return np.nan
    mu = np.average(n[m], weights=w)
    return float(np.sqrt(np.average((n[m] - mu) ** 2, weights=w)))


def score(xyz: np.ndarray, real: np.ndarray, model: np.ndarray, src: np.ndarray,
          direction: tuple[float, float]) -> dict[str, float]:  # fmt: skip
    """All metrics of one model field against one real experiment."""
    hot = real >= np.nanpercentile(real, 90)
    _, pr = vertical_profile(xyz, real, src, direction)
    _, pm = vertical_profile(xyz, model, src, direction)
    return {
        "rho": float(spearmanr(model, real).statistic),
        "auc_hot": float(roc_auc_score(hot, model)),
        "profile_r": float(np.corrcoef(pr, pm)[0, 1]),
        "width_ratio": crosswind_spread(xyz, model, src, direction)
        / crosswind_spread(xyz, real, src, direction),
    }
