import dataclasses

import numpy as np

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.plant3d import MOUNT_Z, Plant3D
from tabpfn_spatial.source.features import n_features
from tabpfn_spatial.source.vertical3d import (
    clean_readings,
    height_rows,
    layouts3d,
    sample_pipe_leaks,
    vertical_sites,
)


def _plant(gas: str = "methane") -> Plant3D:
    return Plant3D(dataclasses.replace(Factory.default(), gas=gas))


def test_vertical_sites_are_the_plants_sites_on_vertical_pipes() -> None:
    plant = _plant()
    sites, pipe = vertical_sites(plant)
    all_sites, _ = plant.leak_sites
    assert len(sites) == len(pipe) > 0
    assert {tuple(s) for s in sites} <= {tuple(s) for s in all_sites}
    for s, p in zip(sites, pipe, strict=True):
        v = plant.vertical_pipes[p]
        assert np.allclose(s[:2], v.xy) and v.z0 <= s[2] < v.z1


def test_layouts_follow_the_mast_rule() -> None:
    plant = _plant("propane")
    k = len(plant.f.sensors)
    lay = layouts3d(plant)
    assert np.all(lay["mounted"][:, 2] == MOUNT_Z["heavy"]) and len(lay["mounted"]) == k
    np.testing.assert_array_equal(lay["mast_rule"], plant.detectors[0])
    assert len(lay["three_heights"]) == 3 * k
    assert set(lay["three_heights"][:, 2]) == set(MOUNT_Z.values())
    # same masts, 1 m beside the pipe in open air, same heights
    moved, on_pipe = lay["mast_1m"], lay["mast_rule"]
    gap = np.linalg.norm(moved[:, :2] - on_pipe[:, :2], axis=1)
    assert np.allclose(gap[:k], 0.0) and np.allclose(gap[k:], 1.0)
    np.testing.assert_array_equal(moved[:, 2], on_pipe[:, 2])
    idx = plant.grid.index(moved)
    assert not plant.solid[idx[:, 0], idx[:, 1], idx[:, 2]].any()


def test_leaks_are_seeded_and_on_vertical_pipes() -> None:
    plant = _plant()
    a = sample_pipe_leaks(plant, 20, np.random.default_rng(0))
    b = sample_pipe_leaks(plant, 20, np.random.default_rng(0))
    assert a == b
    sites, _ = vertical_sites(plant)
    assert all(np.any(np.all(np.isclose(sites, lk.xyz), axis=1)) for lk in a)
    assert all(0.5 <= lk.rate <= 2.0 and 0.2 <= np.hypot(*lk.wind) <= 1.5 for lk in a)


def test_rows_from_the_3d_simulator() -> None:
    plant = _plant()
    leaks = sample_pipe_leaks(plant, 3, np.random.default_rng(1))
    lay = layouts3d(plant)
    times, clean = clean_readings(plant, leaks, lay, t_end=40.0)
    assert set(clean) == set(lay)
    assert clean["mast_rule"].shape == (3, len(lay["mast_rule"]), len(times))
    assert np.all(clean["mast_rule"] >= 0) and clean["mast_rule"].max() > 0.03
    rows = height_rows(times, clean["mast_rule"], leaks, np.random.default_rng(2), ws=(0.0, 5.0))
    assert rows.X.shape[1] == n_features(len(lay["mast_rule"]))
    np.testing.assert_allclose(rows.z, [leaks[i].xyz[2] for i in rows.leak])
    again = height_rows(times, clean["mast_rule"], leaks, np.random.default_rng(2), ws=(0.0, 5.0))
    np.testing.assert_array_equal(rows.X, again.X)
