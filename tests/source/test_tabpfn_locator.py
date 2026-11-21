import numpy as np
import pytest

from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.core.models import BaselineClassifier, BaselineRegressor
from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.source import TabPFNLocator, localisation_error
from tabpfn_spatial.source.features import alarm_time
from tabpfn_spatial.source.tabpfn_locator import prior_rate_quantiles

WS = (0.0, 5.0, 10.0, 20.0)


class _CountingClassifier(BaselineClassifier):
    """k-NN baseline that counts predict_proba calls (stands in for the cloud)."""

    calls = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        type(self).calls += 1
        return super().predict_proba(X)


class _CountingRegressor(BaselineRegressor):
    """k-NN baseline regressor that counts predict_dist calls."""

    calls = 0

    def predict_dist(self, X: np.ndarray) -> QuantileDist:
        type(self).calls += 1
        return super().predict_dist(X)


@pytest.fixture(scope="module")
def factory() -> Factory:
    return Factory.default()


@pytest.fixture(scope="module")
def fitted(factory: Factory) -> TabPFNLocator:
    """One shared fit (k-NN stand-ins, no network)."""
    _CountingClassifier.calls = _CountingRegressor.calls = 0
    return TabPFNLocator(
        classifier=_CountingClassifier(), regressor=_CountingRegressor(), n_train=250, ws=WS
    ).fit(factory)


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
    before = _CountingClassifier.calls, _CountingRegressor.calls
    many = fitted.locate_many(factory, rs)
    assert (_CountingClassifier.calls, _CountingRegressor.calls) == (before[0] + 1, before[1] + 1)
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


# ---------------------------------------------------------------- leak size (rate) --
def _valid_rate(est) -> None:
    lv, q = np.asarray(est.rate_levels), np.asarray(est.rate)
    assert lv.shape == q.shape and len(lv) >= 3
    assert np.all(np.diff(lv) > 0) and lv[0] > 0 and lv[-1] < 1
    assert np.all(q > 0) and np.all(np.diff(q) >= 0)


def test_locate_also_gives_leak_size_quantiles(factory: Factory, fitted: TabPFNLocator) -> None:
    _valid_rate(fitted.locate(factory, _after_alarm(factory, int(_sites(factory, 1, 4)[0]), 4)[1]))


def test_before_any_alarm_the_size_is_the_prior(factory: Factory, fitted: TabPFNLocator) -> None:
    sc = Scenario(factory.leak_sites[0], 1.0, np.array([0.5, 0.0]))
    est = fitted.locate(factory, readings(factory, sc, 0.0, np.random.default_rng(0)))
    lv, q = prior_rate_quantiles(est.rate_levels)
    np.testing.assert_allclose(est.rate, q)
    assert np.isclose(np.interp(0.5, lv, q), 1.0)  # log-uniform 0.5-2 has median 1


def test_bigger_leak_gives_bigger_size_estimate(factory: Factory, fitted: TabPFNLocator) -> None:
    site = factory.leak_sites[int(_sites(factory, 1, 8)[0])]
    med = []
    for rate in (0.55, 1.9):
        sc = Scenario(site, rate, np.array([0.4, 0.3]))
        full = readings(factory, sc, 60.0, np.random.default_rng(3))
        r = readings(
            factory, sc, alarm_time(full.times, full.values) + 20.0, np.random.default_rng(3)
        )
        est = fitted.locate(factory, r)
        med.append(np.interp(0.5, est.rate_levels, est.rate))
    assert med[1] > med[0]


def test_size_can_be_switched_off(factory: Factory) -> None:
    loc = TabPFNLocator(backend="baseline", n_train=60, ws=WS, estimate_rate=False).fit(factory)
    est = loc.locate(factory, _after_alarm(factory, int(_sites(factory, 1, 2)[0]), 2)[1])
    assert est.rate is None and est.rate_levels is None


# ------------------------------------------------------- live: map first, size later --
class _SlowRegressor(BaselineRegressor):
    """Size model that takes ``delay`` seconds (stands in for a slow cloud call)."""

    delay = 0.6

    def predict_dist(self, X: np.ndarray) -> QuantileDist:
        import time

        time.sleep(self.delay)
        return super().predict_dist(X)


@pytest.fixture(scope="module")
def live(factory: Factory) -> TabPFNLocator:
    return TabPFNLocator(
        backend="baseline", regressor=_SlowRegressor(), n_train=120, ws=WS, live=True
    ).fit(factory)


def test_live_map_does_not_wait_for_the_size(factory: Factory, live: TabPFNLocator) -> None:
    import time

    _, r = _after_alarm(factory, int(_sites(factory, 1, 21)[0]), 21, 0.0)
    t0 = time.perf_counter()
    est = live.locate(factory, r)
    assert time.perf_counter() - t0 < _SlowRegressor.delay / 2
    _valid(factory, est.prob)
    _valid_rate(est)  # first update of an emergency: the prior size
    np.testing.assert_allclose(est.rate, prior_rate_quantiles()[1])


def test_live_size_arrives_by_the_next_update(factory: Factory, live: TabPFNLocator) -> None:
    import time

    site = int(_sites(factory, 1, 22)[0])
    _, r5 = _after_alarm(factory, site, 22, 5.0)
    keep = r5.times <= r5.times[-1] - 5.0  # the same stream, 5 s earlier
    r0 = type(r5)(r5.positions, r5.times[keep], r5.values[:, keep], r5.wind)
    live.locate(factory, r0)
    time.sleep(_SlowRegressor.delay + 0.3)  # the 5 s between updates, shortened
    est = live.locate(factory, r5)
    assert not np.allclose(est.rate, prior_rate_quantiles()[1])  # a real estimate now
    _valid_rate(est)


def test_live_size_is_not_shared_between_emergencies(factory: Factory, live: TabPFNLocator) -> None:
    import time

    a, b = _sites(factory, 2, 23)
    live.locate(factory, _after_alarm(factory, int(a), 23, 0.0)[1])
    time.sleep(_SlowRegressor.delay + 0.3)
    est = live.locate(factory, _after_alarm(factory, int(b), 24, 0.0)[1])  # new emergency
    np.testing.assert_allclose(est.rate, prior_rate_quantiles()[1])
