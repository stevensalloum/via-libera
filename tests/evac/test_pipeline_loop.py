"""Event-driven loop: locator, then hazard, then route, back to back, no fixed period."""

import itertools

import numpy as np
import pytest

from tabpfn_spatial.contracts import HazardMap, SourceEstimate
from tabpfn_spatial.evac.live import run_pipeline
from tabpfn_spatial.evac.standins import OracleHazard, OracleLocator
from tabpfn_spatial.sim import Factory, Scenario

F = Factory.default()
SC = Scenario(np.array([11.0, 19.0]), 1.5, np.array([-0.8, 0.0]))  # blows west to E0
START = np.array([5.0, 15.0])


class CountingLocator:
    def __init__(self) -> None:
        self.samples = []

    def locate(self, factory, readings) -> SourceEstimate:
        self.samples.append(len(readings.times))
        return SourceEstimate(factory.leak_prior)


class NoHazard:
    def predict(self, factory, readings, source, times) -> HazardMap:
        z = np.zeros((len(times), factory.ny, factory.nx))
        return HazardMap(np.asarray(times, float), z, z)


def _run(start=START, **kw):
    args = dict(
        locator=OracleLocator(SC, 0.0),
        predictor=OracleHazard(SC),
        t_alarm=10.0,
        t_max=90.0,
        max_runs=6,
        fixed_latency={"locate": 1.7, "predict": 3.9},
    )
    args.update(kw)
    return list(run_pipeline(F, SC, start, **args))  # fmt: skip


def test_cycles_run_back_to_back_from_the_alarm_with_the_models_latency() -> None:
    cycles = _run()[-1].cycles
    assert cycles[0].t_start == 10.0
    for c in cycles:
        assert c.t_land == c.t_start + np.ceil(1.7 + 3.9)  # 6 s: locate + hazard, rounded up
    for a, b in itertools.pairwise(cycles):
        assert b.t_start == a.t_land  # the next run starts as soon as the last one lands


def test_runs_stop_at_the_limit_and_use_more_readings_each_time() -> None:
    loc = CountingLocator()
    frames = _run(np.array([41.0, 25.0]), locator=loc, predictor=NoHazard(), max_runs=4,
                  fixed_latency={"locate": 1.0, "predict": 2.0})  # fmt: skip
    assert len(frames[-1].cycles) == 4
    assert loc.samples == sorted(loc.samples) and loc.samples[-1] > loc.samples[0]


def test_frames_say_which_model_is_working() -> None:
    frames = {fr.t: fr for fr in _run()}
    assert "alarm" not in frames[5.0].phase.lower() or "no alarm" in frames[5.0].phase.lower()
    assert "locat" in frames[10.0].phase.lower()  # Rayan's model from 10 s to 11.7 s
    assert "hazard" in frames[13.0].phase.lower()  # Ahmad's from 11.7 s to 15.6 s
    assert frames[16.0].cycles and frames[16.0].cycles[0].t_land == 16.0


def test_people_head_for_the_nearest_exit_until_the_first_route_lands() -> None:
    frames = {fr.t: fr for fr in _run()}
    assert frames[12.0].walkers["live"].exit_index == frames[12.0].walkers["nearest"].exit_index
    last = _run()[-1].walkers
    assert last["live"].harmful_s <= last["nearest"].harmful_s


def test_measured_latency_is_used_when_none_is_fixed() -> None:
    cycles = _run(fixed_latency=None, max_runs=2)[-1].cycles
    for c in cycles:
        assert c.locate_s >= 0 and c.predict_s >= 0
        assert c.t_land >= c.t_start + 1  # at least one simulated second


def test_unknown_options_fail_loudly() -> None:
    with pytest.raises(ValueError, match="max_runs"):
        _run(max_runs=0)
