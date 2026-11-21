import numpy as np

from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.source.features import moment_row, n_features
from tabpfn_spatial.source.library import (
    build_library,
    leak_zones,
    library_from_scenarios,
)


def _site(f: Factory, seed: int) -> np.ndarray:
    return f.leak_sites[np.random.default_rng(seed).choice(len(f.leak_sites))]


def test_zones_follow_the_plant() -> None:
    f = Factory.default()
    z = leak_zones(f, 100)
    assert z.n_zones == len(f.leak_sites) == 24  # pipes: no merging needed
    np.testing.assert_array_equal(z.zone_of(f.leak_sites), z.site_zone)
    small = leak_zones(f, 10)
    assert small.n_zones == 10 and set(np.unique(small.site_zone)) == set(range(10))


def test_library_has_one_row_per_leak_and_moment() -> None:
    f = Factory.default()
    lib = build_library(f, 30, seed=3, ws=(0.0, 5.0, 10.0))
    assert lib.X.shape[1] == n_features(len(f.sensors))
    assert len(lib.X) == len(lib.zone) == len(lib.w) == len(lib.rate) == len(lib.source)
    assert set(np.unique(lib.w)) <= {0.0, 5.0, 10.0}
    np.testing.assert_array_equal(lib.X[:, -1], lib.w)  # last column = seconds since alarm
    # every leak that alarmed contributes each moment once
    assert len(lib.X) % 3 == 0 and len(lib.X) >= 3 * 25


def test_library_is_seeded() -> None:
    f = Factory.default()
    a = build_library(f, 20, seed=5, ws=(0.0, 10.0))
    b = build_library(f, 20, seed=5, ws=(0.0, 10.0))
    np.testing.assert_array_equal(a.X, b.X)
    c = build_library(f, 20, seed=6, ws=(0.0, 10.0))
    assert a.X.shape != c.X.shape or not np.allclose(a.X, c.X, equal_nan=True)


def test_noise_free_library_row_equals_row_from_sim_readings() -> None:
    f = Factory.default()
    sc = Scenario(_site(f, 7), 1.3, np.array([0.6, -0.4]))
    quiet = {"noise": 0.0, "floor": 0.0, "wind_noise": 0.0}
    lib = library_from_scenarios(f, [sc], np.random.default_rng(0), ws=(0.0, 10.0), **quiet)
    for row, w in zip(lib.X, lib.w, strict=True):
        t_obs = lib.t_alarm[0] + w
        r = readings(f, sc, t_obs, np.random.default_rng(0), **quiet)
        np.testing.assert_allclose(row, moment_row(r.times, r.values, r.wind, lib.t_alarm[0]))


def test_rate_and_source_are_the_truth() -> None:
    f = Factory.default()
    sc = Scenario(_site(f, 2), 1.7, np.array([0.5, 0.5]))
    lib = library_from_scenarios(f, [sc], np.random.default_rng(1), ws=(0.0,))
    np.testing.assert_allclose(lib.rate, [1.7])
    np.testing.assert_allclose(lib.source[0], sc.source)
