"""Compact, causal sensor summaries for optional hazard features."""

import numpy as np

from tabpfn_spatial.contracts import SensorReadings


def time_context_features(t_obs: float, forecast_time: float, n_cells: int) -> np.ndarray:
    """Observation time and forecast horizon per cell; O(n_cells)."""
    if not np.isfinite(t_obs) or not np.isfinite(forecast_time) or forecast_time < t_obs:
        raise ValueError("forecast time must be finite and not precede the observation")
    return np.tile([t_obs, forecast_time - t_obs], (n_cells, 1))


def _latest(readings: SensorReadings, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sensor values, trends and availability using samples up to cutoff."""
    times = np.asarray(readings.times, dtype=float)
    positions = np.asarray(readings.positions, dtype=float)
    values = np.asarray(readings.values, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("sensor times must be a nonempty finite vector")
    if np.any(np.diff(times) <= 0) or not np.isfinite(cutoff):
        raise ValueError("sensor times must be strictly increasing and cutoff finite")
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ValueError("sensor positions must be finite (n, 2) coordinates")
    if values.shape != (len(positions), len(times)):
        raise ValueError("sensor values must match positions and times")
    result = np.zeros((len(positions), 4))  # value, slope, value available, trend available
    for i, series in enumerate(values):
        valid = np.flatnonzero((times <= cutoff) & np.isfinite(series))
        if len(valid):
            result[i, 0], result[i, 2] = series[valid[-1]], 1.0
        if len(valid) >= 2:
            prev, last = valid[-2:]
            with np.errstate(over="ignore", invalid="ignore"):
                slope = (series[last] - series[prev]) / (times[last] - times[prev])
            if np.isfinite(slope):
                result[i, 1], result[i, 3] = slope, 1.0
    return positions, result


def _nearest(
    positions: np.ndarray, sensor_data: np.ndarray, points: np.ndarray, k: int
) -> np.ndarray:
    """Distance and sensor summaries for the k geometrically nearest sensors, padded with zeros."""
    count = len(points)
    output = np.zeros((count, k, 5))
    if len(positions):
        distance = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
        indices = np.argsort(distance, axis=1, kind="stable")[:, :k]
        selected = indices.shape[1]
        output[:, :selected, 0] = np.take_along_axis(distance, indices, axis=1)
        output[:, :selected, 1:] = sensor_data[indices]
    return output.reshape(count, k * 5)


def sensor_history_features(
    readings: SensorReadings,
    source_xy: np.ndarray,
    cells: np.ndarray,
    observation_time: float,
) -> np.ndarray:
    """Describe 3 cell-nearest and 2 source-nearest sensors plus global context.

    Each sensor contributes distance, latest finite value, rate of change, and
    value/trend availability indicators. Last two finite readings determine the
    trend; missing values and unavailable trends are zero with separate flags.
    Global columns are max, mean and valid-sensor fraction. Only measurements at
    or before observation_time are used. Cost: O(cells * sensors log sensors).
    """
    cells = np.asarray(cells, dtype=float).reshape(-1, 2)
    source = np.asarray(source_xy, dtype=float).reshape(1, 2)
    if not np.isfinite(cells).all() or not np.isfinite(source).all():
        raise ValueError("source and cells must be finite")
    positions, data = _latest(readings, observation_time)
    near_cells = _nearest(positions, data, cells, 3)
    near_source = np.repeat(_nearest(positions, data, source, 2), len(cells), axis=0)
    valid = data[:, 2] > 0
    available = data[valid, 0]
    summary = (
        [float(available.max()), float(available.mean()), float(valid.mean())]
        if len(available)
        else [0.0, 0.0, 0.0]
    )
    global_context = np.tile(summary, (len(cells), 1))
    return np.column_stack([near_cells, near_source, global_context])
