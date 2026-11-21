import runpy
from pathlib import Path

import numpy as np
import pytest

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.hazard.sensor_history import sensor_history_features, time_context_features
from tabpfn_spatial.hazard.tabpfn_predictor import TabPFNHazardPredictor
from tabpfn_spatial.sim import Factory


def observations():
    positions = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 6.0], [10.0, 0.0]])
    times = np.array([0.0, 5.0, 12.0])
    values = np.array([[1.0, 3.0, 100.0], [np.nan, 2.0, 99.0], [0.0, 5.0, 98.0], [2.0, 4.0, 97.0]])
    return SensorReadings(positions, times, values, np.array([1.0, 0.0]))


def test_history_uses_nearest_sensors_real_time_steps_and_no_future_values():
    obs = observations()
    cells = np.array([[0.0, 0.0], [10.0, 0.0]])
    x = sensor_history_features(obs, np.array([10.0, 0.0]), cells, 5.0)
    assert x.shape == (2, 28)
    # Per sensor: distance, latest value, slope, latest available, trend available.
    np.testing.assert_allclose(x[0, :5], [0.0, 3.0, 0.4, 1.0, 1.0])
    np.testing.assert_allclose(x[0, 5:10], [4.0, 2.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(x[1, 15:20], [0.0, 4.0, 0.4, 1.0, 1.0])
    np.testing.assert_allclose(x[0, -3:], [5.0, 3.5, 1.0])  # maximum, mean, valid fraction
    truncated = SensorReadings(obs.positions, obs.times[:2], obs.values[:, :2], obs.wind)
    np.testing.assert_array_equal(x, sensor_history_features(truncated, [10, 0], cells, 5.0))


def test_missing_single_and_fewer_sensors_are_distinguishable_from_zero_trend():
    obs = SensorReadings(
        np.array([[0.0, 0.0], [3.0, 0.0]]),
        np.array([0.0]),
        np.array([[0.0], [np.nan]]),
        np.zeros(2),
    )
    x = sensor_history_features(obs, [0, 0], [[0, 0]], 0.0)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x[0, :5], [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(x[0, 5:10], [3, 0, 0, 0, 0])
    np.testing.assert_array_equal(x[0, 10:15], np.zeros(5))
    np.testing.assert_array_equal(x[0, -3:], [0, 0, 0.5])


def test_time_context_uses_latest_available_observation_and_nonnegative_horizon():
    np.testing.assert_array_equal(time_context_features(5.0, 15.0, 2), [[5, 10], [5, 10]])
    with pytest.raises(ValueError, match="forecast"):
        time_context_features(5.0, 4.0, 1)


@pytest.mark.parametrize("geometry", [False, True])
def test_paired_contexts_have_identical_legacy_columns_targets_and_examples(geometry):
    f = Factory.default()
    args = dict(
        n_scenarios=4,
        cells_per_scenario=3,
        seed=1001,
        include_geometry=geometry,
        observation_times=np.array([0.0, 5.0, 10.0]),
    )
    a, ya = training_rows(f, **args)
    b, yb = training_rows(f, **args, include_time_context=True)
    c, yc = training_rows(f, **args, include_sensor_history=True)
    common = 9 if geometry else 7
    assert a.shape == (48, common)
    assert b.shape == (48, common + 2)
    assert c.shape == (48, common + 30)
    np.testing.assert_array_equal(a, b[:, :common])
    np.testing.assert_array_equal(a, c[:, :common])
    np.testing.assert_array_equal(b[:, common:], c[:, common : common + 2])
    np.testing.assert_array_equal(ya, yb)
    np.testing.assert_array_equal(ya, yc)


def test_predictor_handles_multiple_hypotheses_and_one_batched_call():
    f = Factory.default()
    prob = np.zeros_like(f.walls, dtype=float)
    first, second = [f.cell_of(p) for p in f.leak_sites[:2]]
    prob[first], prob[second] = 0.7, 0.3
    obs = SensorReadings(f.sensors, np.array([0.0]), np.zeros((len(f.sensors), 1)), np.zeros(2))

    class Capture:
        def predict_dist(self, x):
            self.rows = x
            values = np.zeros((len(x), 2))
            return QuantileDist(np.array([0.1, 0.9]), values, values[:, 0])

    model = Capture()
    predictor = TabPFNHazardPredictor(model=model, top_k=2, include_sensor_history=True)
    result = predictor.predict(f, obs, SourceEstimate(prob), np.array([0.0, 10.0]))
    assert result.p_toxic.shape == (2, f.ny, f.nx)
    assert model.rows.shape == (2 * 2 * f.ny * f.nx, 37)
    assert np.isfinite(model.rows).all()
    # Query-relative sensor columns agree; source-relative sensor distances vary.
    n = 2 * f.ny * f.nx
    np.testing.assert_array_equal(model.rows[0, 9:24], model.rows[n, 9:24])
    assert not np.array_equal(model.rows[0, 24:34], model.rows[n, 24:34])


def test_default_fit_and_prediction_do_not_add_columns():
    f = Factory.default()
    model = TabPFNHazardPredictor(backend="baseline").fit(f, n_scenarios=2, cells_per_scenario=2)
    assert model.model._X.shape[1] == 7


def test_training_and_inference_produce_the_same_feature_row(monkeypatch):
    from tabpfn_spatial.hazard import library

    f = Factory.default()
    source_xy = f.leak_sites[0]
    prob = np.zeros_like(f.walls, dtype=float)
    prob[f.cell_of(source_xy)] = 1.0
    obs = SensorReadings(
        f.sensors, np.array([0.0, 5.0]), np.full((len(f.sensors), 2), 0.05), np.array([1.0, 0.0])
    )

    class Locator:
        def locate(self, factory, readings):
            return SourceEstimate(prob)

    observed_cells = []
    actual_features = library.cell_features

    def record_cells(factory, source, wind, rate, t, cells):
        observed_cells.append(cells.copy())
        return actual_features(factory, source, wind, rate, t, cells)

    monkeypatch.setattr(library, "readings", lambda *args: obs)
    monkeypatch.setattr(library, "cell_features", record_cells)
    train, _ = training_rows(
        f,
        1,
        1,
        times=np.array([10.0]),
        locator=Locator(),
        include_sensor_history=True,
        observation_times=np.array([5.0]),
    )

    class Capture:
        def predict_dist(self, x):
            self.rows = x
            values = np.zeros((len(x), 2))
            return QuantileDist(np.array([0.1, 0.9]), values, values[:, 0])

    model = Capture()
    predictor = TabPFNHazardPredictor(model=model, top_k=1, include_sensor_history=True)
    predictor.predict(f, obs, SourceEstimate(prob), np.array([10.0]))
    iy, ix = f.cell_of(observed_cells[0][0])
    np.testing.assert_allclose(train[0], model.rows[iy * f.nx + ix])


def test_experimental_predictor_rejects_forecasts_before_the_latest_reading():
    f = Factory.default()
    prob = np.zeros_like(f.walls, dtype=float)
    prob[f.cell_of(f.leak_sites[0])] = 1.0
    obs = SensorReadings(f.sensors, np.array([10.0]), np.zeros((len(f.sensors), 1)), np.zeros(2))
    predictor = TabPFNHazardPredictor(model=object(), top_k=1, include_time_context=True)
    with pytest.raises(ValueError, match="forecast"):
        predictor.predict(f, obs, SourceEstimate(prob), np.array([5.0]))


def test_pilot_threshold_metrics_count_misses_and_false_alarms():
    script = Path(__file__).with_name("compare_sensor_history.py")
    metric = runpy.run_path(str(script))["threshold_metrics"]
    result = metric([0.8, 0.2, 0.7, 0.1], [1, 1, 0, 0])
    assert result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["precision"] == 0.5
