from collections import deque

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tabpfn_spatial.sim import Factory, Scenario, readings, sample_scenarios, simulate, wind_at

SRC = np.array([34.0, 20.0])


def _sc(drift: float = 0.0, seed: int = 3) -> Scenario:
    return Scenario(SRC, 1.0, np.array([1.0, 0.0]), drift=drift, seed=seed)


# ------------------------------------------------------------------ shifting wind --
def test_wind_without_drift_is_constant() -> None:
    sc = _sc()
    np.testing.assert_allclose(wind_at(sc, np.array([0.0, 30.0, 90.0])), [[1, 0]] * 3)


def test_drifting_wind_turns_keeps_speed_and_is_reproducible() -> None:
    times = np.arange(0.0, 120.0, 10.0)
    w = wind_at(_sc(drift=0.1), times)
    np.testing.assert_allclose(np.linalg.norm(w, axis=1), 1.0)
    assert np.ptp(np.arctan2(w[:, 1], w[:, 0])) > 0.1
    np.testing.assert_array_equal(w, wind_at(_sc(drift=0.1), times))
    assert not np.allclose(w, wind_at(_sc(drift=0.1, seed=4), times))


def test_drift_changes_the_plume_deterministically() -> None:
    f = Factory.default()
    times = np.array([60.0])
    calm = simulate(f, [_sc()], times)
    drift = simulate(f, [_sc(drift=0.15)], times)
    again = simulate(f, [_sc(drift=0.15)], times)
    np.testing.assert_array_equal(drift, again)
    assert not np.allclose(calm, drift)


def test_sampled_scenarios_carry_drift_and_distinct_seeds() -> None:
    scs = sample_scenarios(Factory.default(), 20, np.random.default_rng(0), drift=0.05)
    assert all(s.drift == 0.05 for s in scs)
    assert len({s.seed for s in scs}) > 15


# ---------------------------------------------------------------- sensor faults --
def test_measured_wind_is_the_mean_over_the_observation_window() -> None:
    f, sc = Factory.default(), _sc(drift=0.1)
    r = readings(f, sc, 60.0, np.random.default_rng(0), wind_noise=0.0)
    np.testing.assert_allclose(r.wind, wind_at(sc, r.times).mean(axis=0))


def test_dropouts_are_whole_sensors_of_nan() -> None:
    f = Factory.default()
    r = readings(f, _sc(), 60.0, np.random.default_rng(1), dropout=0.4)
    dead = np.isnan(r.values).all(axis=1)
    alive = np.isfinite(r.values).all(axis=1)
    assert dead.any() and alive.any() and np.all(dead | alive)
    assert np.isnan(readings(f, _sc(), 60.0, np.random.default_rng(1), dropout=1.0).values).all()


def test_stuck_sensors_repeat_their_first_value() -> None:
    f = Factory.default()
    r = readings(f, _sc(), 60.0, np.random.default_rng(2), stuck=1.0)
    np.testing.assert_array_equal(r.values, np.repeat(r.values[:, :1], r.values.shape[1], axis=1))


def test_defaults_add_no_faults() -> None:
    r = readings(Factory.default(), _sc(), 60.0, np.random.default_rng(0))
    assert np.isfinite(r.values).all()


# --------------------------------------------------------------- random layouts --
def _reachable_from_exits(f: Factory) -> np.ndarray:
    seen = np.zeros_like(f.walls)
    queue = deque(f.cell_of(e) for e in f.exits)
    for c in queue:
        seen[c] = True
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (y + dy, x + dx)
            if 0 <= n[0] < f.ny and 0 <= n[1] < f.nx and not f.walls[n] and not seen[n]:
                seen[n] = True
                queue.append(n)
    return seen


@settings(max_examples=10, deadline=None)
@given(st.integers(0, 10_000))
def test_random_factories_are_valid_and_connected(seed: int) -> None:
    f = Factory.random(np.random.default_rng(seed))
    assert any(m.hot for m in f.machines) and len(f.leak_sites) > 0
    assert len(f.exits) >= 3 and len(f.sensors) >= 10
    for xy in np.vstack([f.sensors, f.exits]):
        assert not f.walls[f.cell_of(xy)]
    np.testing.assert_array_equal(_reachable_from_exits(f), ~f.walls)


def test_random_factory_is_reproducible_and_varies() -> None:
    a = Factory.random(np.random.default_rng(7))
    b = Factory.random(np.random.default_rng(7))
    c = Factory.random(np.random.default_rng(8))
    np.testing.assert_array_equal(a.walls, b.walls)
    assert not np.array_equal(a.walls, c.walls)


def test_simulation_runs_on_a_random_factory() -> None:
    f = Factory.random(np.random.default_rng(5))
    scs = sample_scenarios(f, 3, np.random.default_rng(5), drift=0.05)
    fields = simulate(f, scs, np.array([30.0]))
    assert fields.shape == (3, 1, f.ny, f.nx) and np.all(fields[..., f.walls] == 0)


def test_drift_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        wind_at(_sc(drift=-0.1), np.array([1.0]))
