"""The detector and model calls must use the same simulated sensor history."""

import numpy as np
import pytest

import tabpfn_spatial.evac.crowd_run as crowd_run
import tabpfn_spatial.evac.live as live
from tabpfn_spatial.contracts import HazardMap, SensorReadings, SourceEstimate
from tabpfn_spatial.sim import Factory, Scenario
from tabpfn_spatial.source.features import alarm_time

F = Factory.default()
SC = Scenario(F.leak_sites[3], 1.4, np.array([0.7, 0.2]))
START = np.array([41.0, 25.0])


class RecordingLocator:
    """Record the readings actually delivered to the source model."""

    def __init__(self):
        self.readings = []

    def locate(self, factory, readings):
        self.readings.append(readings)
        return SourceEstimate(factory.leak_prior)


class NoHazard:
    """Return an empty hazard map without spending model credits."""

    def predict(self, factory, readings, source, times):
        z = np.zeros((len(times), factory.ny, factory.nx))
        return HazardMap(np.asarray(times, float), z, z)


def _history(trigger_time):
    times = np.arange(0.0, 31.0, 5.0)
    values = np.zeros((len(F.sensors), len(times)))
    if trigger_time is not None:
        values[0, times >= trigger_time] = 0.04

    def at(t):
        count = np.searchsorted(times, t, side="right")
        return SensorReadings(F.sensors, times[:count], values[:, :count], np.array([0.7, 0.2]))

    return at


@pytest.mark.parametrize("trigger_time", [0.0, 10.0, None])
@pytest.mark.parametrize("workflow", ["live", "pipeline", "crowd"])
def test_alarm_and_locator_share_sensor_history(monkeypatch, workflow, trigger_time):
    history = _history(trigger_time)
    locator = RecordingLocator()
    monkeypatch.setattr(
        live if workflow != "crowd" else crowd_run, "sensor_history", lambda *a, **k: history
    )
    if workflow == "crowd":
        result = crowd_run.prepare(F, SC, "baseline", t_max=30.0, fitted=(locator, NoHazard()))
        actual_alarm = result.t_alarm
    else:
        fn = live.run_live if workflow == "live" else live.run_pipeline
        kwargs = {"max_runs": 2, "fixed_latency": {"locate": 0.0, "predict": 0.0}}
        if workflow == "live":
            kwargs = {"update_s": 5.0}
        frames = list(fn(F, SC, START, locator, NoHazard(), t_max=30.0, **kwargs))
        actual_alarm = frames[0].t_alarm
        assert next((f.t for f in frames if f.alarm), None) == trigger_time
    assert actual_alarm == trigger_time
    if trigger_time is not None:
        assert locator.readings
        assert alarm_time(locator.readings[0].times, locator.readings[0].values) == actual_alarm
        for reading in locator.readings:
            np.testing.assert_array_equal(reading.values, history(float(reading.times[-1])).values)
    else:
        assert not locator.readings


def test_crowd_does_not_initialize_models_without_an_alarm(monkeypatch):
    monkeypatch.setattr(crowd_run, "sensor_history", lambda *args, **kwargs: _history(None))

    def unexpected_models(*args, **kwargs):
        raise AssertionError("models must not initialize without a sensor alarm")

    monkeypatch.setattr(crowd_run, "make_models", unexpected_models)
    prepared = crowd_run.prepare(F, SC, "tabpfn", t_max=30.0)
    assert prepared.t_alarm is None and not prepared.danger and not prepared.log
