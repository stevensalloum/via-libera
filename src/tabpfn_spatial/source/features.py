"""One feature row per moment, on a clock that starts at the first alarm.

A real plant does not know when the leak started, only when a detector first alarmed.
So every time here is relative to that alarm. Per sensor the row holds two numbers:

- ``log1p`` of its latest reading (how much gas is there now),
- its arrival time: first sample above ``ARRIVAL_LEVEL``, in seconds relative to the
  alarm (negative = before the alarm; "not yet" = one sample step after now),

then the measured wind u, v and the seconds since the alarm. With k sensors that is
2k + 3 columns (31 for the default plant's 14). The arrival time carries the history, so
one row per update is enough. Dead sensors (NaN rows) give NaN columns.
Time O(k * t) per row.
"""

import numpy as np

from tabpfn_spatial.sim import ALARM_LEVEL

ARRIVAL_LEVEL = 0.01  # "gas has reached this sensor" (a third of the alarm level)


def n_features(n_sensors: int) -> int:
    """Number of columns of ``moment_row``."""
    return 2 * n_sensors + 3


def feature_names(n_sensors: int) -> list[str]:
    """Column names, in row order."""
    return (
        [f"log1p_now_s{i}" for i in range(n_sensors)]
        + [f"arrival_s{i}" for i in range(n_sensors)]
        + ["wind_u", "wind_v", "seconds_since_alarm"]
    )


def _first_above(times: np.ndarray, values: np.ndarray, level: float) -> np.ndarray:
    """(k,) time of each sensor's first sample above ``level``; inf if none, NaN if dead."""
    above = np.nan_to_num(values, nan=-np.inf) > level
    first = np.where(above.any(axis=1), times[above.argmax(axis=1)], np.inf)
    return np.where(np.isnan(values).all(axis=1), np.nan, first)


def alarm_time(times: np.ndarray, values: np.ndarray, level: float = ALARM_LEVEL) -> float | None:
    """First sample time where any working sensor reads above ``level``; None if none yet."""
    first = _first_above(np.asarray(times, dtype=float), np.asarray(values, dtype=float), level)
    t = np.nanmin(first) if np.isfinite(first).any() else np.inf
    return float(t) if np.isfinite(t) else None


def moment_row(
    times: np.ndarray, values: np.ndarray, wind: np.ndarray, t_alarm: float
) -> np.ndarray:
    """(2k + 3,) row from readings ``values`` (k, t) up to now (``times[-1]``)."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    now = times[-1] - t_alarm
    step = times[1] - times[0] if len(times) > 1 else 5.0
    arrival = _first_above(times, values, ARRIVAL_LEVEL) - t_alarm
    arrival = np.where(np.isinf(arrival), now + step, arrival)  # not yet: just after now
    latest = np.log1p(values[:, -1])
    return np.concatenate([latest, arrival, np.asarray(wind, dtype=float), [now]])
