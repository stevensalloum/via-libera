import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tabpfn_spatial.core.dist import DEFAULT_LEVELS, QuantileDist


def _dist(n: int = 3) -> QuantileDist:
    base = np.linspace(-1, 1, len(DEFAULT_LEVELS))
    values = np.arange(n)[:, None] + base[None]
    return QuantileDist(np.asarray(DEFAULT_LEVELS), values, mean=np.arange(n, dtype=float))


def test_shapes_and_median() -> None:
    d = _dist(4)
    assert len(d) == 4
    np.testing.assert_allclose(d.median, np.arange(4))


def test_interval_uses_symmetric_levels() -> None:
    d = _dist(1)
    lo, hi = d.interval(0.9)
    assert lo[0] < d.median[0] < hi[0]
    np.testing.assert_allclose(hi - lo, 2 * np.ones(1))  # 0.05 and 0.95 are the extremes
    np.testing.assert_allclose(d.width(0.9), hi - lo)


def test_interval_rejects_missing_level() -> None:
    with pytest.raises(ValueError, match="level"):
        _dist(1).interval(0.33)


def test_validation() -> None:
    with pytest.raises(ValueError):
        QuantileDist(np.array([0.5, 0.1]), np.zeros((2, 2)), mean=np.zeros(2))
    with pytest.raises(ValueError):
        QuantileDist(np.array([0.1, 0.5]), np.zeros((2, 3)), mean=np.zeros(2))


@settings(max_examples=30, deadline=None)
@given(st.lists(st.floats(-1e3, 1e3), min_size=len(DEFAULT_LEVELS), max_size=len(DEFAULT_LEVELS)))
def test_crossing_quantiles_are_sorted(raw: list[float]) -> None:
    d = QuantileDist(np.asarray(DEFAULT_LEVELS), np.asarray([raw]), mean=np.zeros(1))
    assert np.all(np.diff(d.values, axis=1) >= 0)


def test_from_main_output() -> None:
    levels = [0.1, 0.5, 0.9]
    out = {
        "mean": np.array([1.0]),
        "mode": np.array([0.5]),
        "quantiles": [np.array([v]) for v in (0, 1, 2)],
    }
    d = QuantileDist.from_main(out, levels)
    np.testing.assert_allclose(d.values, [[0, 1, 2]])
    np.testing.assert_allclose(d.mode, [0.5])
