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
    p = map_to_zones(f, z, [_point_map(f, f.leak_sites[i]) for i in (0, 9)])
    assert p.shape == (2, 50)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0, z.site_zone[0]] == 1.0 and p[1, z.site_zone[9]] == 1.0


def test_perfect_maps_score_perfectly() -> None:
    f = Factory.default()
    z = leak_zones(f, 50)
    sources = f.leak_sites[[3, 20, 60]]
    m = source_metrics(f, z, [_point_map(f, s) for s in sources], sources)
    assert m["error_m"] == 0.0 and m["p_true_zone"] == 1.0
    assert m["top1"] == 1.0 and m["top3"] == 1.0 and m["ece"] == 0.0


def test_held_out_is_seeded() -> None:
    f = Factory.default()
    s1, r1 = held_out(f, 3, seed=9)
    s2, r2 = held_out(f, 3, seed=9)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(r1[0].values, r2[0].values)
    assert len(r1) == 3 and r1[0].values.shape == (len(f.sensors), 13)
