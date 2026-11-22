"""Checks for the integrated timing experiment; no cloud calls."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _study():
    path = Path(__file__).with_name("integrated_timing.py")
    spec = importlib.util.spec_from_file_location("integrated_timing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cycle_measures_source_hazard_and_route_and_passes_same_inputs(monkeypatch):
    study = _study()
    ticks = iter([100.0, 101.0, 104.0, 104.2])
    monkeypatch.setattr(study.time, "perf_counter", lambda: next(ticks))
    factory, obs, source, hazard = object(), SimpleNamespace(t_obs=10.0), object(), object()
    positions = np.array([[1.0, 2.0]])

    class Locator:
        def locate(self, received_factory, received_obs):
            assert received_factory is factory and received_obs is obs
            return source

    class Predictor:
        def predict(self, received_factory, received_obs, received_source, times):
            assert received_factory is factory and received_obs is obs and received_source is source
            np.testing.assert_array_equal(times, np.arange(10.0, 131.0, 10.0))
            return hazard

    class Planner:
        def plan(self, received_factory, received_people, received_hazard, start):
            assert received_factory is factory and received_hazard is hazard and start == 10.0
            np.testing.assert_array_equal(received_people, positions)
            return SimpleNamespace(routes=[SimpleNamespace(exit_index=2)])

    result, measured_source, measured_hazard = study.measure_cycle(
        factory, obs, positions, Locator(), Predictor(), Planner(), 10.0
    )
    assert measured_source is source and measured_hazard is hazard
    assert result["source_s"] == 1.0 and result["hazard_s"] == 3.0
    assert abs(result["route_s"] - 0.2) < 1e-9
    assert abs(result["total_s"] - 4.2) < 1e-9 and result["exit_index"] == 2


def test_next_decision_advances_by_measured_time_without_fixed_update_interval():
    study = _study()
    assert study.next_decision(10.0, 4.2) == 15.0
    assert study.next_decision(10.0, 8.0) == 18.0
    assert study.next_decision(10.0, 0.0) == 11.0
