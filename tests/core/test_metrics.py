import numpy as np
import pytest

from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.core.metrics import coverage, crps, ece

LEVELS = np.linspace(0.05, 0.95, 19)


def _gauss_dist(mu: np.ndarray, sd: float) -> QuantileDist:
    from statistics import NormalDist

    z = np.array([NormalDist().inv_cdf(p) for p in LEVELS])
    return QuantileDist(LEVELS, mu[:, None] + sd * z[None], mean=mu)


def test_crps_prefers_sharp_correct_forecast() -> None:
    y = np.zeros(200)
    sharp = crps(_gauss_dist(np.zeros(200), 0.1), y)
    wide = crps(_gauss_dist(np.zeros(200), 2.0), y)
    wrong = crps(_gauss_dist(np.ones(200), 0.1), y)
    assert sharp < wide and sharp < wrong


def test_coverage_of_calibrated_gaussian() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    assert coverage(_gauss_dist(np.zeros(5000), 1.0), y, 0.9) == pytest.approx(0.9, abs=0.02)


def test_ece_perfect_and_bad() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    probs = np.stack([1 - p, p], axis=1)
    assert ece(probs, y) < 0.02
    assert ece(np.stack([p * 0 + 0.01, p * 0 + 0.99], axis=1), y) > 0.4
