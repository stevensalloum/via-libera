"""Cost-to-exit fields: one reverse search per exit serves every person."""

import itertools

import numpy as np
import pytest

from tabpfn_spatial.contracts import HazardMap
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.evac.fields import danger_window, exit_fields, follow
from tabpfn_spatial.sim import Factory

F = Factory.default()
ZERO = np.zeros((F.ny, F.nx))


def test_zero_risk_lengths_match_the_one_person_planner() -> None:
    fields = exit_fields(F, ZERO, risk_weight=0.0)
    z = np.zeros((1, F.ny, F.nx))
    calm = HazardMap(np.array([0.0]), z, z)
    for xy in ([5.0, 15.0], [41.0, 25.0], [27.0, 37.0]):
        rt = RiskAwarePlanner().plan(F, np.array([xy]), calm, 0.0).routes[0]
        length = np.linalg.norm(np.diff(rt.cells, axis=0), axis=1).sum() * F.cell
        best = min(fields[e].length[F.cell_of(np.array(xy))] for e in range(len(F.exits)))
        assert best == pytest.approx(length, abs=1e-6)


def test_following_a_field_reaches_its_exit_through_open_cells_without_corner_cuts() -> None:
    fields = exit_fields(F, ZERO, risk_weight=0.0)
    for e, fe in enumerate(fields):
        for start in map(tuple, np.argwhere(~F.walls)[::53]):
            path = follow(fe, start)
            assert tuple(path[-1]) == F.cell_of(F.exits[e])
            assert not F.walls[path[:, 0], path[:, 1]].any()
            for a, b in itertools.pairwise(path):
                dy, dx = b - a
                assert max(abs(dy), abs(dx)) == 1
                if dy and dx:
                    assert not F.walls[a[0] + dy, a[1]] and not F.walls[a[0], a[1] + dx]


def test_solid_cells_are_unreachable() -> None:
    fe = exit_fields(F, ZERO, risk_weight=0.0)[0]
    assert np.isinf(fe.cost[F.walls]).all()
    assert np.isfinite(fe.cost[~F.walls]).all()


def test_risk_makes_the_field_detour_around_danger() -> None:
    start = F.cell_of(np.array([5.0, 15.0]))
    calm = follow(exit_fields(F, ZERO, 0.0)[0], start)
    danger = ZERO.copy()
    danger[calm[1:-1, 0], calm[1:-1, 1]] = 1.0
    risky = follow(exit_fields(F, danger, 200.0)[0], start)
    assert danger[risky[:, 0], risky[:, 1]].sum() < danger[calm[:, 0], calm[:, 1]].sum()


def test_danger_window_is_the_worst_forecast_in_the_window() -> None:
    times = np.array([10.0, 20.0, 30.0, 90.0])
    pt = np.zeros((4, F.ny, F.nx))
    pt[1, 3, 3], pt[3, 4, 4] = 0.6, 0.9
    hz = HazardMap(times, pt, np.zeros_like(pt))
    w = danger_window(hz, t=10.0, ahead=60.0)
    assert w[3, 3] == pytest.approx(0.6) and w[4, 4] == 0.0
