import numpy as np
import pytest

from tabpfn_spatial.core.models import BaselineClassifier
from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.source import TabPFNLocator, localisation_error
from tabpfn_spatial.source.features import alarm_time

WS = (0.0, 5.0, 10.0, 20.0)


class _CountingClassifier(BaselineClassifier):
    """k-NN baseline that counts predict_proba calls (stands in for the cloud)."""

    calls = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        type(self).calls += 1
        return super().predict_proba(X)


@pytest.fixture(scope="module")
def factory() -> Factory:
    return Factory.default()


@pytest.fixture(scope="module")
def fitted(factory: Factory) -> TabPFNLocator:
    """One shared fit (k-NN stand-in, no network)."""
    _CountingClassifier.calls = 0
    return TabPFNLocator(classifier=_CountingClassifier(), n_train=250, ws=WS).fit(factory)


def _sites(f: Factory, n: int, seed: int) -> np.ndarray:
    """``n`` distinct leak-site indices drawn at random (the count depends on the plant)."""
    return np.random.default_rng(seed).choice(len(f.leak_sites), size=n, replace=False)


def _after_alarm(f: Factory, site: int, seed: int, w: float = 10.0):
    """Scenario and its readings ``w`` seconds after the first alarm."""
    sc = Scenario(f.leak_sites[site], 1.5, np.array([0.5, 0.2]))
    full = readings(f, sc, 60.0, np.random.default_rng(seed))
    t_a = alarm_time(full.times, full.values)
    assert t_a is not None
    return sc, readings(f, sc, t_a + w, np.random.default_rng(seed))


def _valid(f: Factory, prob: np.ndarray) -> None:
    assert prob.shape == (f.ny, f.nx)
    assert np.isclose(prob.sum(), 1.0) and np.all(prob >= 0)
    assert np.all(prob[f.leak_prior == 0] == 0)  # only on leak sites (the pipes)


def test_locate_returns_a_valid_map_on_the_pipes(factory: Factory, fitted: TabPFNLocator) -> None:
    _valid(
        factory,
        fitted.locate(factory, _after_alarm(factory, int(_sites(factory, 1, 5)[0]), 5)[1]).prob,
    )


def test_before_any_alarm_the_map_is_the_prior(factory: Factory, fitted: TabPFNLocator) -> None:
    sc = Scenario(factory.leak_sites[0], 1.0, np.array([0.5, 0.0]))
    r = readings(factory, sc, 0.0, np.random.default_rng(0))  # only the t = 0 sample
    np.testing.assert_allclose(fitted.locate(factory, r).prob, factory.leak_prior)


def test_locates_leaks_better_than_the_prior(factory: Factory, fitted: TabPFNLocator) -> None:
    pairs = [_after_alarm(factory, int(i), 100 + int(i), 20.0) for i in _sites(factory, 6, 11)]
    ests = fitted.locate_many(factory, [r for _, r in pairs])
    err = np.mean(
        [localisation_error(factory, e, sc.source) for (sc, _), e in zip(pairs, ests, strict=True)]
    )
    prior_cell = np.unravel_index(factory.leak_prior.argmax(), factory.leak_prior.shape)
    prior_err = np.mean(
        [np.linalg.norm(factory.centre(*prior_cell) - sc.source) for sc, _ in pairs]
    )
    assert err < prior_err


def test_locate_many_is_one_batched_call(factory: Factory, fitted: TabPFNLocator) -> None:
    rs = [
        _after_alarm(factory, int(i), int(i), w)[1]
        for i, w in zip(_sites(factory, 3, 1), (0, 5, 20), strict=True)
    ]
    before = _CountingClassifier.calls
    many = fitted.locate_many(factory, rs)
    assert _CountingClassifier.calls == before + 1
    for r, est in zip(rs, many, strict=True):
        np.testing.assert_allclose(est.prob, fitted.locate(factory, r).prob)


def test_works_on_a_random_plant() -> None:
    f = Factory.random(np.random.default_rng(1))
    loc = TabPFNLocator(backend="baseline", n_train=80, ws=WS).fit(f)
    _valid(f, loc.locate(f, _after_alarm(f, int(_sites(f, 1, 0)[0]), 0, 5.0)[1]).prob)


def test_dead_sensor_does_not_crash(factory: Factory, fitted: TabPFNLocator) -> None:
    _, r = _after_alarm(factory, int(_sites(factory, 1, 9)[0]), 9)
    values = r.values.copy()
    values[3] = np.nan
    dead = type(r)(r.positions, r.times, values, r.wind)
    _valid(factory, fitted.locate(factory, dead).prob)
