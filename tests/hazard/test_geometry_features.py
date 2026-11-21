from dataclasses import replace

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.hazard.geometry import path_features
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.hazard.tabpfn_predictor import TabPFNHazardPredictor
from tabpfn_spatial.sim import Factory


def test_path_features_distinguish_doors_above_and_below_wall() -> None:
    walls = np.zeros((5, 7), dtype=bool)
    walls[:, 3] = True
    walls[0, 3] = False
    factory = Factory(14.0, 10.0, 2.0, walls, (), np.empty((0, 2)), np.empty((0, 2)))
    far_walls = walls.copy()
    far_walls[0, 3] = True
    far_walls[4, 3] = False
    source = np.array([3.0, 3.0])
    target = np.array([[11.0, 3.0]])
    near = path_features(factory, source, target)
    far = path_features(replace(factory, walls=far_walls), source, target)
    np.testing.assert_array_equal(near, [[12.0, 1.0]])
    np.testing.assert_array_equal(far, [[20.0, 1.0]])


def test_unreachable_or_wall_targets_are_marked_with_finite_values() -> None:
    walls = np.zeros((3, 3), dtype=bool)
    walls[:, 1] = True
    factory = Factory(6.0, 6.0, 2.0, walls, (), np.empty((0, 2)), np.empty((0, 2)))
    features = path_features(factory, np.array([1.0, 1.0]), np.array([[5.0, 1.0], [3.0, 1.0]]))
    assert np.all(np.isfinite(features))
    np.testing.assert_array_equal(features[:, 1], [0.0, 0.0])


def test_optional_geometry_extends_same_context_examples() -> None:
    factory = Factory.default()
    old_x, old_y = training_rows(factory, 2, 3, np.array([70.0]), 1001)
    new_x, new_y = training_rows(factory, 2, 3, np.array([70.0]), 1001, include_geometry=True)
    assert old_x.shape == (6, 7) and new_x.shape == (6, 9)
    np.testing.assert_array_equal(old_x, new_x[:, :7])
    np.testing.assert_array_equal(old_y, new_y)


def test_default_context_seed_uses_proposed_training_range() -> None:
    factory = Factory.default()
    times = np.array([70.0])
    default_x, default_y = training_rows(factory, 2, 3, times)
    expected_x, expected_y = training_rows(factory, 2, 3, times, 1001)
    np.testing.assert_array_equal(default_x, expected_x)
    np.testing.assert_array_equal(default_y, expected_y)


def test_predictor_uses_optional_geometry_features() -> None:
    factory = Factory.default()
    prob = np.zeros_like(factory.walls, dtype=float)
    prob[1, 1] = 1.0
    obs = SensorReadings(
        factory.sensors, np.array([60.0]), np.zeros((len(factory.sensors), 1)), np.array([1.0, 0.0])
    )

    class Fake:
        def predict_dist(self, x):
            assert x.shape == (factory.nx * factory.ny, 9)
            values = np.zeros((len(x), 2))
            return QuantileDist(np.array([0.1, 0.9]), values, values[:, 0])

    predictor = TabPFNHazardPredictor(model=Fake(), top_k=1, include_geometry=True)
    result = predictor.predict(factory, obs, SourceEstimate(prob), np.array([90.0]))
    assert result.p_toxic.shape == (1, factory.ny, factory.nx)


def test_optional_geometry_fits_and_predicts_offline() -> None:
    factory = Factory.default()
    predictor = TabPFNHazardPredictor(backend="baseline", top_k=1, include_geometry=True)
    predictor.fit(factory, n_scenarios=3, cells_per_scenario=5, times=np.array([70.0]), seed=1001)
    prob = np.zeros_like(factory.walls, dtype=float)
    prob[1, 1] = 1.0
    obs = SensorReadings(
        factory.sensors,
        np.array([60.0]),
        np.zeros((len(factory.sensors), 1)),
        np.array([1.0, 0.0]),
    )
    result = predictor.predict(factory, obs, SourceEstimate(prob), np.array([70.0]))
    assert np.all((result.p_toxic >= 0) & (result.p_toxic <= 1))
