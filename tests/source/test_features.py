import numpy as np

from tabpfn_spatial.sim import ALARM_LEVEL
from tabpfn_spatial.source.features import (
    ARRIVAL_LEVEL,
    alarm_time,
    feature_names,
    moment_row,
    n_features,
)

T = np.arange(0.0, 31.0, 5.0)  # samples 0, 5, ..., 30 s


def _ramp(start: float, slope: float) -> np.ndarray:
    """One sensor's series: zero until ``start``, then rising by ``slope`` per second."""
    return np.clip((T - start) * slope, 0.0, None)


def test_alarm_is_first_sample_above_the_alarm_level() -> None:
    values = np.vstack([_ramp(5, 0.004), _ramp(0, 0.002)])  # 0.03 first exceeded at 15 s
    assert alarm_time(T, values) == 15.0
    assert alarm_time(T, np.zeros((2, len(T)))) is None
    assert ALARM_LEVEL == 0.03


def test_dead_sensors_are_ignored_for_the_alarm() -> None:
    values = np.vstack([np.full(len(T), np.nan), _ramp(0, 0.01)])
    assert alarm_time(T, values) == 5.0


def test_row_has_31_columns_for_14_sensors() -> None:
    assert n_features(14) == 31
    names = feature_names(14)
    assert len(names) == 31 and names[-1] == "seconds_since_alarm"


def test_row_is_relative_to_the_alarm() -> None:
    # sensor 0 reaches 0.01 at 10 s, sensor 1 at 20 s, sensor 2 never; alarm (0.03) at 15 s
    values = np.vstack([_ramp(5, 0.004), _ramp(15, 0.003), np.zeros(len(T))])
    row = moment_row(T, values, np.array([0.4, -0.1]), t_alarm=15.0)
    k = 3
    np.testing.assert_allclose(row[:k], np.log1p(values[:, -1]))  # latest reading
    arrival = row[k : 2 * k]
    assert arrival[0] == 10.0 - 15.0  # arrived 5 s before the alarm
    assert arrival[1] == 20.0 - 15.0
    assert arrival[2] == 30.0 - 15.0 + 5.0  # not yet: one step after "now"
    np.testing.assert_allclose(row[-3:], [0.4, -0.1, 15.0])  # wind u, v, seconds since alarm
    assert ARRIVAL_LEVEL < ALARM_LEVEL


def test_same_leak_shifted_in_time_gives_the_same_row() -> None:
    # a real plant does not know when the leak started: only alarm-relative times matter
    a = np.vstack([_ramp(5, 0.004), _ramp(10, 0.003)])
    late = np.hstack([np.zeros((2, 2)), a])[:, : len(T)]  # same leak, started 10 s later
    t_a, t_b = alarm_time(T, a), alarm_time(T, late)
    assert t_b == t_a + 10.0
    ra = moment_row(T[: len(T) - 2], a[:, : len(T) - 2], np.zeros(2), t_a)
    rb = moment_row(T, late, np.zeros(2), t_b)
    np.testing.assert_allclose(ra, rb)


def test_dead_sensor_columns_are_nan() -> None:
    values = np.vstack([_ramp(0, 0.01), np.full(len(T), np.nan)])
    row = moment_row(T, values, np.zeros(2), t_alarm=5.0)
    assert np.isnan(row[1]) and np.isnan(row[3])
    assert np.isfinite(row[0]) and np.isfinite(row[2])


def test_other_sensor_periods_work() -> None:
    t2 = np.arange(0.0, 21.0, 2.0)
    values = np.vstack([np.clip((t2 - 4) * 0.01, 0, None)])
    t_a = alarm_time(t2, values)
    assert t_a == 8.0  # 0.04 at 8 s
    row = moment_row(t2, values, np.zeros(2), t_a)
    assert row[1] == 6.0 - 8.0 and row[-1] == 12.0
