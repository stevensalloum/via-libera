import numpy as np

from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.source.features import leak_features, n_features
from tabpfn_spatial.source.library import (
    leak_zones,
    library_from_scenarios,
    make_library,
    noisy_series,
)


def test_zones_cover_every_leak_site_deterministically() -> None:
    f = Factory.default()
    z = leak_zones(f, 100)
    assert len(z.site_zone) == len(f.leak_sites)
    assert z.n_zones <= 100
    assert set(np.unique(z.site_zone)) == set(range(z.n_zones))
    np.testing.assert_array_equal(z.site_zone, leak_zones(f, 100).site_zone)
    # zones are local: every site lies within 4 m of its zone centre
    d = np.linalg.norm(f.leak_sites - z.centres[z.site_zone], axis=1)
    assert d.max() <= 4.0


def test_zone_of_maps_sources_to_their_site_zone() -> None:
    f = Factory.default()
    z = leak_zones(f, 40)
    np.testing.assert_array_equal(z.zone_of(f.leak_sites), z.site_zone)


def test_make_library_shapes_and_determinism() -> None:
    f = Factory.default()
    X, zone = make_library(f, 40, seed=3, n_zones=50)
    assert X.shape == (40, n_features(len(f.sensors)))
    assert zone.shape == (40,) and zone.min() >= 0 and zone.max() < 50
    X2, zone2 = make_library(f, 40, seed=3, n_zones=50)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(zone, zone2)
    X3, _ = make_library(f, 40, seed=4, n_zones=50)
    assert not np.allclose(X, X3)


def test_noise_free_library_row_equals_features_of_sim_readings() -> None:
    f = Factory.default()
    sc = Scenario(
        f.leak_sites[np.random.default_rng(7).choice(len(f.leak_sites))], 1.3, np.array([0.6, -0.4])
    )
    rng = np.random.default_rng(0)
    quiet = {"noise": 0.0, "floor": 0.0, "wind_noise": 0.0}
    X, _ = library_from_scenarios(f, [sc], rng, leak_zones(f), **quiet)
    r = readings(f, sc, 60.0, rng, **quiet)
    np.testing.assert_allclose(X[0], leak_features(r))


def test_noisy_series_matches_readings_layout() -> None:
    f = Factory.default()
    sc = Scenario(
        f.leak_sites[np.random.default_rng(0).choice(len(f.leak_sites))], 1.0, np.array([1.0, 0.0])
    )
    times, values, wind = noisy_series(f, [sc, sc], np.random.default_rng(1))
    r = readings(f, sc, 60.0, np.random.default_rng(1))
    np.testing.assert_array_equal(times, r.times)
    assert values.shape == (2, *r.values.shape) and wind.shape == (2, 2)
    assert np.all(values >= 0)
