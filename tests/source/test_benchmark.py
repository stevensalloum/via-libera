import numpy as np

from tabpfn_spatial.contracts import SourceEstimate
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.source.benchmark import held_out, map_to_zones, source_metrics
from tabpfn_spatial.source.library import leak_zones


def _point_map(f: Factory, xy: np.ndarray) -> SourceEstimate:
    prob = np.zeros((f.ny, f.nx))
    prob[f.cell_of(xy)] = 1.0
    return SourceEstimate(prob)


def test_map_to_zones_sums_cell_probability_per_zone() -> None:
    f = Factory.default()
    z = leak_zones(f, 50)
    a, b = np.random.default_rng(0).choice(len(f.leak_sites), size=2, replace=False)
    p = map_to_zones(f, z, [_point_map(f, f.leak_sites[i]) for i in (a, b)])
    assert p.shape == (2, z.n_zones)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0, z.site_zone[a]] == 1.0 and p[1, z.site_zone[b]] == 1.0


def test_perfect_maps_score_perfectly() -> None:
    f = Factory.default()
    z = leak_zones(f, 50)
    sources = f.leak_sites[np.random.default_rng(1).choice(len(f.leak_sites), 3, replace=False)]
    m = source_metrics(f, z, [_point_map(f, s) for s in sources], sources)
    assert m["error_m"] == 0.0 and m["p_true_zone"] == 1.0
    assert m["top1"] == 1.0 and m["top3"] == 1.0 and m["ece"] == 0.0


def test_held_out_gives_readings_at_each_moment_after_the_alarm() -> None:
    f = Factory.default()
    h = held_out(f, 4, seed=9, ws=(0.0, 10.0))
    assert len(h.readings) == len(h.sources) == len(h.w)
    for r, w, t_a in zip(h.readings, h.w, h.t_alarm, strict=True):
        assert r.times[-1] == t_a + w
    h2 = held_out(f, 4, seed=9, ws=(0.0, 10.0))
    np.testing.assert_array_equal(h.readings[0].values, h2.readings[0].values)
