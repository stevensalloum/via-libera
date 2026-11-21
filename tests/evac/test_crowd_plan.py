"""Crowd planners (nearest, risk, capacity-aware risk) and the crowd run loop."""

import numpy as np
import pytest

from tabpfn_spatial.evac.crowd import Crowd, CrowdParams
from tabpfn_spatial.evac.crowd_plan import PLANNERS, assign
from tabpfn_spatial.evac.crowd_run import prepare, run_crowd
from tabpfn_spatial.evac.fields import exit_fields
from tabpfn_spatial.sim import Factory, Scenario

F = Factory.default()
P = CrowdParams()
ZERO = np.zeros((F.ny, F.nx))


def _walk(crowd: Crowd, name: str, danger: np.ndarray = ZERO) -> Crowd:
    assign(name, F, crowd, danger, 0.0)
    t = 0.0
    while crowd.active.any() and t < 900:
        crowd.step(t)
        t += 1.0
    return crowd


def test_every_planner_gives_everyone_a_route_from_their_cell_to_their_exit() -> None:
    xy = np.array([[5.0, 15.0], [41.0, 25.0], [27.0, 37.0]])
    for name in PLANNERS:
        crowd = Crowd(F, xy, P)
        assign(name, F, crowd, ZERO, 0.0)
        for i, route in enumerate(crowd.routes):
            assert tuple(route[0]) == tuple(crowd.cells[i])
            assert tuple(route[-1]) == F.cell_of(F.exits[crowd.exit_index[i]])


def test_nearest_picks_the_shortest_walk() -> None:
    crowd = Crowd(F, np.array([[5.0, 15.0], [55.0, 15.0]]), P)
    assign("nearest", F, crowd, ZERO, 0.0)
    fields = exit_fields(F, ZERO, 0.0)
    for i, c in enumerate(map(tuple, crowd.cells)):
        assert crowd.exit_index[i] == int(np.argmin([fe.length[c] for fe in fields]))


def test_risk_planners_avoid_a_gassed_exit_nearest_does_not() -> None:
    danger = ZERO.copy()
    e0 = F.cell_of(F.exits[0])
    danger[e0[0] - 3 : e0[0] + 4, : e0[1] + 4] = 1.0  # around exit E0
    for name, expect_e0 in (("nearest", True), ("risk", False), ("ccrp", False)):
        crowd = Crowd(F, np.array([[9.0, 19.0]]), P)
        assign(name, F, crowd, danger, 0.0)
        assert bool(crowd.exit_index[0] == 0) is expect_e0, name


def test_capacity_aware_planning_spreads_a_crowd_over_exits_and_empties_faster() -> None:
    xy = np.repeat([[9.0, 19.0]], 150, axis=0) + np.random.default_rng(0).uniform(-3, 3, (150, 2))
    near = _walk(Crowd(F, xy, P), "nearest")
    ccrp = _walk(Crowd(F, xy, P), "ccrp")
    assert len(set(near.exit_index)) == 1
    assert len(set(ccrp.exit_index)) >= 2
    assert np.nanmax(ccrp.out_at) < 0.8 * np.nanmax(near.out_at)


def test_prepare_replans_from_the_detector_alarm_at_the_update_rate() -> None:
    sc = Scenario(np.array([11.0, 19.0]), 1.5, np.array([-0.8, 0.0]))
    prep = prepare(F, sc, "perfect", t_max=120.0, update_s=15.0, seed=0)
    assert prep.t_alarm is not None and prep.t_alarm % 5 == 0
    assert prep.replan_times[0] == prep.t_alarm
    assert np.allclose(np.diff(prep.replan_times), 15.0)
    assert len(prep.danger) == len(prep.replan_times)


def test_run_crowd_scores_everyone_on_the_true_gas_and_is_deterministic() -> None:
    sc = Scenario(np.array([11.0, 19.0]), 1.5, np.array([-0.8, 0.0]))
    prep = prepare(F, sc, "perfect", t_max=200.0, update_s=15.0, seed=0)
    xy = np.random.default_rng(1).uniform([2, 2], [58, 38], (60, 2))
    xy = xy[[not F.walls[F.cell_of(p)] for p in xy]]
    a = run_crowd(F, prep, xy, "ccrp", P)
    b = run_crowd(F, prep, xy, "ccrp", P)
    assert a.metrics == b.metrics
    m = a.metrics
    assert m["people"] == len(xy) and m["still_inside"] == 0
    assert m["t_alarm"] <= m["out_95_s"] <= m["out_all_s"]
    assert 0 <= m["exposed_pct"] <= 100
    near = run_crowd(F, prep, xy, "nearest", P).metrics
    assert m["harm_s_mean"] <= near["harm_s_mean"] + 1e-9
    with pytest.raises(ValueError, match="planner"):
        run_crowd(F, prep, xy, "fastest", P)


def test_detour_weight_makes_ccrp_divert_fewer_people() -> None:
    xy = np.repeat([[9.0, 19.0]], 150, axis=0) + np.random.default_rng(0).uniform(-3, 3, (150, 2))
    moved = []
    for w in (1.0, 2.0, 50.0):
        crowd = Crowd(F, xy, P)
        assign("ccrp", F, crowd, ZERO, 0.0, detour_weight=w)
        moved.append(int((crowd.exit_index != 0).sum()))
    assert moved[0] > moved[1] > moved[2] >= 0


def test_prepare_runs_the_teams_tabpfn_models_and_logs_each_update() -> None:
    from tabpfn_spatial.evac.crowd_run import make_models

    sc = Scenario(F.leak_sites[3], 1.4, np.array([0.7, 0.2]))
    fitted = make_models(F, sc, "tabpfn", backend="baseline", n_train=150)
    prep = prepare(F, sc, "tabpfn", t_max=30.0, update_s=60.0, seed=0, fitted=fitted)
    assert len(prep.log) == len(prep.replan_times) == len(prep.danger)
    first = prep.log[0]
    assert first["t"] == prep.t_alarm
    assert first["locate_s"] >= 0 and first["predict_s"] >= 0
    assert first["source_error_m"] >= 0 and first["rate_median"] > 0
    assert all(d.min() >= 0 and d.max() <= 1 for d in prep.danger)
    with pytest.raises(ValueError, match="models"):
        prepare(F, sc, "magic", t_max=30.0)


def test_planners_commit_to_an_exit_under_a_diffuse_hazard_map() -> None:
    """A smooth, uncertain map (every exit somewhat risky) must not make people flip."""
    xy = np.random.default_rng(2).uniform([2, 2], [58, 38], (200, 2))
    xy = xy[[not F.walls[F.cell_of(p)] for p in xy]]
    danger = np.full((F.ny, F.nx), 0.3)
    for name in ("risk", "ccrp"):
        crowd = Crowd(F, xy, P)
        switches, last, t = 0, None, 0.0
        while crowd.active.any() and t < 600:
            if t % 15 == 0:
                assign(name, F, crowd, danger, t)
                if last is not None:
                    switches += int(((crowd.exit_index != last) & crowd.active).sum())
                last = crowd.exit_index.copy()
            crowd.step(t)
            t += 1.0
        assert not crowd.active.any(), name  # everyone gets out
        assert switches <= 0.1 * len(xy), (name, switches)
