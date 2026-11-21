"""Gas detectors raise the alarm when a sampled reading crosses the alarm level."""

import numpy as np

from tabpfn_spatial.sim import Factory


def test_first_alarm_is_the_first_sample_where_a_sensor_reads_above_the_level() -> None:
    from tabpfn_spatial.sim import ALARM_LEVEL, Scenario, first_alarm, sensor_values, simulate

    f = Factory.default()
    sc = Scenario(f.leak_sites[3], 1.0, np.array([0.8, 0.0]))
    times = np.arange(0.0, 121.0, 1.0)
    fields = simulate(f, [sc], times)[0]
    t = first_alarm(f, times, fields, every=5.0)
    assert t is not None and t % 5.0 == 0
    seen = sensor_values(f, fields).max(axis=1)  # (T,)
    assert seen[int(t)] > ALARM_LEVEL
    assert all(seen[int(s)] <= ALARM_LEVEL for s in np.arange(0.0, t, 5.0))
    assert first_alarm(f, times, fields, every=1.0) <= t  # faster sampling, earlier or same
    assert first_alarm(f, times, fields, level=1e9) is None
