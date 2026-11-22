import sys

import numpy as np
import pytest

from tabpfn_spatial.core import models
from tabpfn_spatial.core.budget import BudgetExceeded, CostGuard
from tabpfn_spatial.core.models import (
    BaselineClassifier,
    BaselineRegressor,
    TabPFNClassifierAdapter,
    TabPFNRegressorAdapter,
    make_classifier,
    make_regressor,
    pick_backend,
)
from tabpfn_spatial.env import TOKEN_VAR


def _line(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    return X, 3 * X[:, 0] + rng.normal(scale=0.1, size=n)


def test_baseline_regressor_learns_and_quantifies() -> None:
    X, y = _line()
    d = BaselineRegressor().fit(X, y).predict_dist(X[:10])
    assert d.values.shape == (10, len(d.levels))
    assert np.mean(np.abs(d.median - y[:10])) < 0.5
    assert np.all(d.width(0.9) > 0)


def test_baseline_classifier_probabilities() -> None:
    X, y = _line()
    labels = (y > 0).astype(int)
    m = BaselineClassifier().fit(X, labels)
    p = m.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert list(m.classes_) == [0, 1]
    assert np.mean(m.predict(X) == labels) > 0.85


class _FakeTabPFNRegressor:
    """Mimics tabpfn / tabpfn_client TabPFNRegressor for output_type='main'."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FakeTabPFNRegressor":
        self.mu = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray, output_type: str, quantiles: list[float]) -> dict:
        assert output_type == "main"
        n = len(X)
        return {
            "mean": np.full(n, self.mu),
            "median": np.full(n, self.mu),
            "mode": np.full(n, self.mu),
            "quantiles": [np.full(n, self.mu + q - 0.5) for q in quantiles],
        }


class _FakeTabPFNClassifier:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FakeTabPFNClassifier":
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


def test_regressor_adapter_wraps_main_output() -> None:
    X, y = _line()
    m = TabPFNRegressorAdapter(_FakeTabPFNRegressor, n_estimators=2).fit(X, y)
    d = m.predict_dist(X[:5])
    assert d.values.shape == (5, len(d.levels))
    assert m.estimator.kwargs == {"n_estimators": 2}


def test_classifier_adapter() -> None:
    X, y = _line()
    m = TabPFNClassifierAdapter(_FakeTabPFNClassifier).fit(X, (y > 0).astype(int))
    assert m.predict_proba(X[:3]).shape == (3, 2)
    assert m.predict(X[:3]).shape == (3,)


def test_cost_guard_blocks_big_calls() -> None:
    guard = CostGuard(max_tokens_per_call=50_000, quote=lambda tr, te, c: 10_000 + te)
    assert guard.check(100, 1_000, 3) == 11_000
    assert guard.spent == 11_000 and guard.calls == 1
    with pytest.raises(BudgetExceeded):
        guard.check(100, 100_000, 3)
    assert guard.calls == 1  # refused calls are not counted


def test_adapter_consults_guard_before_predict() -> None:
    X, y = _line()
    guard = CostGuard(max_tokens_per_call=5, quote=lambda tr, te, c: 10)
    m = TabPFNRegressorAdapter(_FakeTabPFNRegressor, guard=guard).fit(X, y)
    with pytest.raises(BudgetExceeded):
        m.predict_dist(X)


def test_pick_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pick_backend("cloud") == "cloud"
    monkeypatch.setattr(models, "_local_available", lambda: False)
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    assert pick_backend("auto", env_file=None) == "baseline"
    monkeypatch.setenv(TOKEN_VAR, "x")
    assert pick_backend("auto", env_file=None) == "cloud"
    monkeypatch.setattr(models, "_local_available", lambda: True)
    assert pick_backend("auto", env_file=None) == "local"
    with pytest.raises(ValueError):
        pick_backend("gpu")


def test_factories_return_baselines() -> None:
    assert isinstance(make_regressor("baseline"), BaselineRegressor)
    assert isinstance(make_classifier("baseline"), BaselineClassifier)


def test_cloud_factory_uses_tabpfn_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = type(sys)("tabpfn_client")
    fake.TabPFNRegressor = _FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake)
    m = make_regressor("cloud", n_estimators=4, guard=None)
    assert isinstance(m, TabPFNRegressorAdapter)
    assert m.estimator.kwargs == {"n_estimators": 4}


def test_rate_limited_calls_are_retried(monkeypatch) -> None:
    import tabpfn_spatial.core.models as m

    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError(
                "Fail to call predict: [HTTP 429] Rate limit exceeded. Retry in 10s."
            )
        return "ok"

    assert m.with_retry(flaky) == "ok" and len(calls) == 3


def test_other_errors_are_not_retried() -> None:
    import pytest

    import tabpfn_spatial.core.models as m

    calls = []

    def broken() -> None:
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        m.with_retry(broken)
    assert len(calls) == 1
