"""Ahmad's stress conditions, defined once in the simulator."""

import numpy as np
import pytest

from tabpfn_spatial.sim import CONDITIONS, Factory, Scenario, readings, sensor_history


def test_the_conditions_match_the_hazard_study() -> None:
    assert list(CONDITIONS) == ["standard", "wind_drift", "sensor_faults", "slower_mixing",
                                "faster_mixing", "combined_stress"]  # fmt: skip
    c = CONDITIONS["combined_stress"]
    assert (c.diffusivity, c.drift, c.dropout, c.stuck) == (0.4, 0.05, 0.15, 0.05)
    assert CONDITIONS["standard"].diffusivity == 0.8


def test_mixing_changes_what_sensors_read() -> None:
    f = Factory.default()
    sc = Scenario(f.leak_sites[3], 1.0, np.array([0.5, 0.0]))
    slow = readings(f, sc, 40.0, np.random.default_rng(0), noise=0, floor=0, diffusivity=0.4)
    fast = readings(f, sc, 40.0, np.random.default_rng(0), noise=0, floor=0, diffusivity=1.6)
    assert not np.allclose(slow.values, fast.values)


def test_sensor_faults_reach_the_history() -> None:
    f = Factory.default()
    sc = Scenario(f.leak_sites[3], 1.0, np.array([0.5, 0.0]))
    at = sensor_history(f, sc, 30.0, np.random.default_rng(1), dropout=0.5)
    assert np.isnan(at(30.0).values).all(axis=1).any()


def test_unknown_condition_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        CONDITIONS["hurricane"]
