"""One person, replanned every few seconds while the leak grows."""

import numpy as np
import pytest

from tabpfn_spatial.contracts import HazardMap, SensorReadings, SourceEstimate
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.evac.live import POLICIES, Walker, run_live
from tabpfn_spatial.evac.standins import OracleHazard, OracleLocator
from tabpfn_spatial.sim import Factory, Scenario

F = Factory.default()


class _NoHazard:
    def predict(self, factory, readings, source, times) -> HazardMap:
        z = np.zeros((len(times), factory.ny, factory.nx))
        return HazardMap(np.asarray(times, float), z, z)


# ----------------------------------------------------------------------- walker --
def test_walker_moves_its_speed_along_the_path_and_stops_at_the_end() -> None:
    w = Walker(np.array([0.0, 0.0]), speed=1.0)
    w.follow(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]), exit_index=2)
    w.step(2.0, t=0.0)
    np.testing.assert_allclose(w.pos, [2.0, 0.0])
    w.step(2.0, t=2.0)
    np.testing.assert_allclose(w.pos, [3.0, 1.0])
    w.step(10.0, t=4.0)
    np.testing.assert_allclose(w.pos, [3.0, 4.0])
    assert w.out_at == pytest.approx(7.0)  # 7 m at 1 m/s
    assert w.walked == pytest.approx(7.0)


# ---------------------------------------------------------------------- planner --
def test_routes_never_cut_a_corner_between_solid_cells() -> None:
    zero = _NoHazard().predict(F, None, None, np.array([0.0]))
    starts = np.array([[17.0, 13.0], [27.0, 13.0], [37.0, 11.0], [47.0, 27.0], [17.0, 33.0]])
    for route in RiskAwarePlanner().plan(F, starts, zero, 0.0).routes:
        for a, b in zip(route.cells[:-1], route.cells[1:], strict=True):
            dy, dx = b - a
            if dy and dx:
                assert not F.walls[a[0] + dy, a[1]] and not F.walls[a[0], a[1] + dx]


# ---------------------------------------------------------------------- stand-ins --
def test_oracle_locator_is_a_map_on_the_pipes_near_the_truth() -> None:
    sc = Scenario(F.leak_sites[5], 1.0, np.array([1.0, 0.0]))
    loc = OracleLocator(sc, sigma=2.0, rng=np.random.default_rng(0))
    est = loc.locate(F, None)
    assert est.prob.sum() == pytest.approx(1.0)
    assert (est.prob[F.leak_prior == 0] == 0).all()
    iy, ix = est.top_cells(1)[0]
    assert np.linalg.norm(F.centre(iy, ix) - sc.source) < 8.0


def test_exact_oracle_locator_points_at_the_true_cell_even_next_to_a_fitting() -> None:
    fit = {F.cell_of(s) for s in F.leak_sites if F.fittings[F.cell_of(s)]}
    for xy in F.leak_sites:  # every site, including plain pipe next to a fitting
        sc = Scenario(xy, 1.0, np.array([1.0, 0.0]))
        iy, ix = OracleLocator(sc, sigma=0.0).locate(F, None).top_cells(1)[0]
        assert (iy, ix) == F.cell_of(xy), (xy, fit)


def test_oracle_hazard_is_the_true_future() -> None:
    sc = Scenario(F.leak_sites[5], 1.0, np.array([1.0, 0.0]))
    hz = OracleHazard(sc).predict(F, None, None, np.array([30.0, 90.0]))
    assert hz.p_toxic.shape == (2, F.ny, F.nx)
    assert set(np.unique(hz.p_toxic)) <= {0.0, 1.0}
    assert hz.p_toxic[1].sum() > hz.p_toxic[0].sum() > 0


# ---------------------------------------------------------------------- the loop --
def _run(**kw):
    sc = Scenario(np.array([11.0, 19.0]), 1.5, np.array([-0.8, 0.0]))  # blows west to E0
    args = dict(
        locator=OracleLocator(sc, sigma=0.0, rng=np.random.default_rng(0)),
        predictor=OracleHazard(sc),
        t_alarm=20.0,
        update_s=10.0,
        t_max=120.0,
        rng=np.random.default_rng(0),
    )
    args.update(kw)
    return list(run_live(F, sc, np.array([5.0, 15.0]), **args))


def test_frames_every_second_and_replans_at_the_update_rate() -> None:
    frames = _run(predictor=_NoHazard())
    assert [fr.t for fr in frames] == list(np.arange(0.0, 121.0, 1.0))
    replans = [fr.t for fr in frames if fr.replanned]
    assert replans[0] == 20.0
    assert np.allclose(np.diff(replans), 10.0)
    assert set(frames[-1].walkers) == set(POLICIES)
    before = frames[10].walkers["live"]
    np.testing.assert_allclose(before.pos, [5.0, 15.0])  # nobody moves before the alarm


def test_without_hazard_every_policy_takes_the_nearest_exit() -> None:
    last = _run(predictor=_NoHazard())[-1].walkers
    assert {w.exit_index for w in last.values()} == {0}
    assert all(w.out_at is not None for w in last.values())


def test_live_replanning_keeps_the_person_out_of_the_gas() -> None:
    frames = _run()
    last = frames[-1].walkers
    assert last["nearest"].exit_index == 0
    assert last["live"].exit_index != 0
    assert last["live"].harmful_s < last["nearest"].harmful_s
    assert last["live"].dose < last["nearest"].dose
    first = next(fr for fr in frames if fr.replanned)
    assert sorted(o.exit_index for o in first.options) == list(range(len(F.exits)))
    best = min(first.options, key=lambda o: o.cost)
    assert best.exit_index == first.walkers["live"].exit_index


def test_live_loop_accepts_the_real_baselines() -> None:
    from tabpfn_spatial.hazard import ForwardSimPredictor
    from tabpfn_spatial.source import WeightedCentroidLocator

    frames = _run(locator=WeightedCentroidLocator(), predictor=ForwardSimPredictor(), t_max=60.0)
    fr = next(fr for fr in frames if fr.replanned)
    assert isinstance(fr.source, SourceEstimate) and isinstance(fr.hazard, HazardMap)
    assert isinstance(fr.readings, SensorReadings)


# ----------------------------------------------------------------------- latency --
def test_with_latency_people_walk_to_the_nearest_exit_until_the_plan_arrives() -> None:
    at = {fr.t: fr.walkers for fr in _run(latency=8.0)}
    assert at[21.0]["live"].exit_index == at[21.0]["nearest"].exit_index == 0
    assert at[21.0]["live"].pos.tolist() != [5.0, 15.0]  # already walking
    assert at[27.0]["once"].exit_index == 0  # plan not in yet


def test_with_latency_people_can_wait_for_the_plan_instead() -> None:
    frames = _run(latency=8.0, before_plan="wait")
    at = {fr.t: fr.walkers for fr in frames}
    np.testing.assert_allclose(at[27.0]["live"].pos, [5.0, 15.0])  # waiting
    assert at[27.0]["nearest"].pos.tolist() != [5.0, 15.0]  # the conventional plan walks
    assert at[29.0]["live"].exit_index != 0  # the plan from 20 s data landed at 28 s
    assert at[29.0]["once"].exit_index == at[29.0]["live"].exit_index
    with pytest.raises(ValueError, match="before_plan"):
        _run(before_plan="run")


def test_latency_costs_exposure_on_a_leak_where_the_nearest_exit_is_gassed() -> None:
    fast = _run()[-1].walkers["live"]
    slow = _run(latency=20.0)[-1].walkers["live"]
    assert slow.dose >= fast.dose


# ---------------------------------------------------------------- detector alarm --
def test_detector_alarm_starts_the_evacuation_at_the_first_sensor_alarm() -> None:
    frames = _run(t_alarm=None)
    t_alarm = frames[0].t_alarm
    assert t_alarm is not None and t_alarm % 5.0 == 0
    assert all(fr.t_alarm == t_alarm for fr in frames)
    assert next(fr.t for fr in frames if fr.alarm) == t_alarm
    assert next(fr.t for fr in frames if fr.replanned) == t_alarm
    before = [fr for fr in frames if fr.t < t_alarm]
    assert all(fr.walkers["live"].pos.tolist() == [5.0, 15.0] for fr in before)


def test_no_detector_alarm_means_nobody_is_told_to_move() -> None:
    sc = Scenario(np.array([11.0, 19.0]), 1e-6, np.array([-0.8, 0.0]))  # tiny leak
    frames = list(run_live(F, sc, np.array([5.0, 15.0]), OracleLocator(sc), OracleHazard(sc),
                           t_alarm=None, t_max=60.0))  # fmt: skip
    assert frames[-1].t_alarm is None and not any(fr.alarm for fr in frames)
    assert all(w.out_at is None for w in frames[-1].walkers.values())
