import numpy as np
import pytest

from tabpfn_spatial.core.models import BaselineClassifier
from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.source import TabPFNLocator, localisation_error
from tabpfn_spatial.source.library import leak_zones

N_ZONES = 60


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
    """One shared fit (k-NN stand-in, no network): ~1.5 s for 400 simulated leaks."""
    _CountingClassifier.calls = 0
    clf = _CountingClassifier()
    return TabPFNLocator(classifier=clf, n_train=400, n_zones=N_ZONES).fit(factory)


def _sites(f: Factory, n: int, seed: int) -> np.ndarray:
    """``n`` distinct leak-site indices, drawn at random: the site count depends on the plant."""
    return np.random.default_rng(seed).choice(len(f.leak_sites), size=n, replace=False)


def _read(f: Factory, site: int, seed: int, wind: tuple[float, float] = (0.5, 0.0)):
    sc = Scenario(f.leak_sites[site], 1.5, np.array(wind))
    return sc, readings(f, sc, 60.0, np.random.default_rng(seed))


def test_locate_returns_a_valid_map_on_leak_sites(factory: Factory, fitted: TabPFNLocator) -> None:
    f = factory
    est = fitted.locate(f, _read(f, int(_sites(f, 1, 5)[0]), 5, (0.8, 0.3))[1])
    assert est.prob.shape == (f.ny, f.nx)
    assert np.isclose(est.prob.sum(), 1.0)
    assert np.all(est.prob[f.walls] == 0)
    sites = np.zeros((f.ny, f.nx), dtype=bool)
    for xy in f.leak_sites:
        sites[f.cell_of(xy)] = True
    assert np.all(est.prob[~sites] == 0)  # leaks only start on leak sites (the pipes)


def test_locates_clear_leaks_better_than_chance(factory: Factory, fitted: TabPFNLocator) -> None:
    pairs = [_read(factory, int(i), 11 + int(i)) for i in _sites(factory, 4, 11)]
    ests = fitted.locate_many(factory, [r for _, r in pairs])
    errors = [
        localisation_error(factory, e, sc.source) for (sc, _), e in zip(pairs, ests, strict=True)
    ]
    assert np.mean(errors) < 10.0  # a random leak site would be ~25 m off


def test_locate_many_uses_one_batched_call_and_matches_locate(
    factory: Factory, fitted: TabPFNLocator
) -> None:
    rs = [_read(factory, int(i), int(i), (0.3, 0.3))[1] for i in _sites(factory, 3, 1)]
    before = _CountingClassifier.calls
    many = fitted.locate_many(factory, rs)
    assert _CountingClassifier.calls == before + 1
    for r, est in zip(rs, many, strict=True):
        np.testing.assert_allclose(est.prob, fitted.locate(factory, r).prob)


def test_zone_proba_columns_are_zones(factory: Factory, fitted: TabPFNLocator) -> None:
    site = int(_sites(factory, 1, 3)[0])
    p = fitted.zone_proba(factory, [_read(factory, site, 0, (0.2, 0.0))[1]])
    assert p.shape == (1, leak_zones(factory, N_ZONES).n_zones)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_locate_fits_lazily_on_first_use(factory: Factory) -> None:
    loc = TabPFNLocator(backend="baseline", n_train=60, n_zones=N_ZONES)
    assert np.isclose(loc.locate(factory, _read(factory, 0, 0, (0.0, 0.5))[1]).prob.sum(), 1.0)
