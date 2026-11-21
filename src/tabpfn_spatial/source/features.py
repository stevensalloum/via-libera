"""Feature rows for leak localisation: one row per leak, from the sensor time series.

Per sensor: ``log1p`` of the reading at a few times (default 20, 40, 60 s) and the
arrival time (first sample above ``level``); plus the measured wind u, v. With 14 sensors
and 3 times that is 14 * 3 + 14 + 2 = 58 columns. Time O(B * k * t).
"""

from collections.abc import Sequence

import numpy as np

from tabpfn_spatial.contracts import SensorReadings

DEFAULT_TIMES: tuple[float, ...] = (20.0, 40.0, 60.0)
ARRIVAL_LEVEL = 0.01  # concentration that counts as "gas has arrived" (10 % of toxic)


def n_features(n_sensors: int, at: Sequence[float] = DEFAULT_TIMES) -> int:
    """Number of columns ``feature_matrix`` produces."""
    return n_sensors * (len(at) + 1) + 2


def feature_matrix(
    times: np.ndarray,
    values: np.ndarray,
    wind: np.ndarray,
    at: Sequence[float] = DEFAULT_TIMES,
    level: float = ARRIVAL_LEVEL,
) -> np.ndarray:
    """(B, n_features) rows from ``values`` (B, k, t) sampled at ``times`` and ``wind`` (B, 2).

    The reading "at" a time is the latest sample not after it. A sensor that never
    exceeds ``level`` gets arrival time one sample step after the last sample.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    idx = np.clip(np.searchsorted(times, np.asarray(at, dtype=float), side="right") - 1, 0, None)
    logs = np.log1p(values[:, :, idx])  # (B, k, len(at))
    step = times[1] - times[0] if len(times) > 1 else 1.0
    above = values > level
    first = np.where(above.any(axis=2), times[above.argmax(axis=2)], times[-1] + step)
    b = len(values)
    return np.hstack([logs.reshape(b, -1), first, np.asarray(wind, dtype=float).reshape(b, 2)])


def leak_features(
    readings: SensorReadings, at: Sequence[float] = DEFAULT_TIMES, level: float = ARRIVAL_LEVEL
) -> np.ndarray:
    """(n_features,) row for one emergency's readings."""
    return feature_matrix(readings.times, readings.values[None], readings.wind[None], at, level)[0]
