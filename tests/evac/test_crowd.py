"""Crowd simulator: cell-to-cell walking with door capacity, density-dependent speed."""

import itertools

import numpy as np
import pytest

from tabpfn_spatial.evac.crowd import Crowd, CrowdParams, sample_workers, weidmann_speed
from tabpfn_spatial.evac.fields import exit_fields, follow
from tabpfn_spatial.sim import Factory

F = Factory.default()
P = CrowdParams()


def _walk_all(crowd: Crowd, t_max: float = 400.0) -> None:
    t = 0.0
    while t < t_max and crowd.active.any():
        crowd.step(t)
        t += 1.0


def _send_to_nearest(crowd: Crowd) -> None:
    fields = exit_fields(F, np.zeros((F.ny, F.nx)), 0.0)
    for i, c in enumerate(map(tuple, crowd.cells)):
        e = int(np.argmin([fe.length[c] for fe in fields]))
        crowd.set_route(i, follow(fields[e], c), e)


def test_weidmann_speed_is_free_when_empty_and_slows_with_density() -> None:
    assert weidmann_speed(0.05, P) == pytest.approx(P.speed, rel=0.01)
    v = [weidmann_speed(r, P) for r in (0.5, 1.0, 2.0, 4.0, 5.3)]
    assert all(a > b for a, b in itertools.pairwise(v))
    assert min(v) >= P.min_speed


def test_one_person_walks_at_free_speed() -> None:
    crowd = Crowd(F, np.array([[5.0, 15.0]]), P)
    _send_to_nearest(crowd)
    length = np.linalg.norm(np.diff(crowd.routes[0], axis=0), axis=1).sum() * F.cell
    _walk_all(crowd)
    assert crowd.out_at[0] == pytest.approx(length / P.speed, abs=1.5)


def test_an_exit_door_passes_its_capacity_per_second() -> None:
    door = F.cell_of(F.exits[0])
    n = 40
    crowd = Crowd(F, np.repeat([F.centre(*door)], n, axis=0), P)
    for i in range(n):
        crowd.set_route(i, np.array([door]), 0)
    _walk_all(crowd)
    rate = P.specific_flow * P.door_m
    assert crowd.out_at.max() == pytest.approx(n / rate, abs=2.0)


def test_cells_never_hold_more_than_their_capacity() -> None:
    rng = np.random.default_rng(0)
    crowd = Crowd(F, sample_workers(F, 400, rng), P)
    _send_to_nearest(crowd)
    t, peak = 0.0, 0
    while crowd.active.any() and t < 600:
        crowd.step(t)
        peak = max(peak, int(crowd.occupancy.max()))
        t += 1.0
    assert not crowd.active.any()
    assert peak <= crowd.max_occupancy


def test_exposure_counts_every_second_until_out() -> None:
    crowd = Crowd(F, np.array([[5.0, 15.0], [41.0, 25.0]]), P)
    _send_to_nearest(crowd)
    gas = np.ones((F.ny, F.nx))
    t = 0.0
    while crowd.active.any():
        crowd.expose(gas * 0.2, gas.astype(bool))
        crowd.step(t)
        t += 1.0
    np.testing.assert_allclose(crowd.harm_s, np.ceil(crowd.out_at), atol=1.0)
    np.testing.assert_allclose(crowd.dose, 0.2 * crowd.harm_s)


def test_workers_stand_on_open_floor_mostly_near_machines() -> None:
    xy = sample_workers(F, 500, np.random.default_rng(1))
    cells = [F.cell_of(p) for p in xy]
    assert not any(F.walls[c] for c in cells)
    c = F.centres()
    near = np.zeros(F.walls.shape, dtype=bool)
    for m in F.machines:
        near |= (
            (c[..., 0] > m.x0 - 6) & (c[..., 0] < m.x1 + 6)
            & (c[..., 1] > m.y0 - 6) & (c[..., 1] < m.y1 + 6)
        )  # fmt: skip
    assert np.mean([near[c] for c in cells]) > 0.6


def test_crowd_is_deterministic() -> None:
    runs = []
    for _ in range(2):
        crowd = Crowd(F, sample_workers(F, 120, np.random.default_rng(3)), P)
        _send_to_nearest(crowd)
        _walk_all(crowd)
        runs.append(crowd.out_at.copy())
    np.testing.assert_array_equal(*runs)
