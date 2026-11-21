import numpy as np
import pytest

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.hazard.tabpfn_predictor import TabPFNHazardPredictor, _cdf
from tabpfn_spatial.sim import Factory


def test_cdf_interpolates_and_handles_point_masses() -> None:
    values = np.array([[0.0, 1.0], [0.2, 0.2], [0.0, 0.0]])
    result = _cdf(values, np.array([0.1, 0.9]), 0.2)
    np.testing.assert_allclose(result, [0.26, 1.0, 1.0])


def test_features_wind_coordinates_and_obstructions() -> None:
    f = Factory.default()
    source = np.array([3.0, 3.0])
    cells = np.array([[7.0, 3.0], [3.0, 7.0], [11.0, 11.0]])
    x = cell_features(f, source, np.array([1.0, 0.0]), 1.5, 90.0, cells)
    np.testing.assert_allclose(x[:2, :3], [[4, 0, 4], [0, 4, 4]])
    assert x[2, 6] < 1  # the line passes through a machine
    assert np.all(np.isfinite(cell_features(f, source, np.zeros(2), 1, 90, cells)))


def test_training_rows_are_seeded_and_have_concentrations() -> None:
    f = Factory.default()
    x, y = training_rows(
        f, n_scenarios=3, cells_per_scenario=5, times=np.array([70.0, 90.0]), seed=2
    )
    other_x, other_y = training_rows(
        f, n_scenarios=3, cells_per_scenario=5, times=np.array([70.0, 90.0]), seed=2
    )
    assert x.shape == (30, 7) and y.shape == (30,)
    assert np.all(y >= 0)
    np.testing.assert_array_equal(x, other_x)
    np.testing.assert_array_equal(y, other_y)


def test_training_uses_estimated_source_and_measured_wind_without_future_readings(monkeypatch):
    from tabpfn_spatial.hazard import library

    factory = Factory.default()
    estimated = factory.leak_sites[0]
    seen = []
    actual_features = library.cell_features
    actual_readings = library.readings

    def noisy_readings(f, scenario, t_obs, rng):
        obs = actual_readings(f, scenario, t_obs, rng)
        return SensorReadings(obs.positions, obs.times, obs.values, np.array([0.0, 2.0]))

    class Locator:
        def locate(self, f, obs):
            prob = np.zeros_like(f.walls, dtype=float)
            prob[f.cell_of(estimated)] = 1.0
            seen.append(float(obs.times[-1]))
            return SourceEstimate(prob)

    def features(f, source, wind, rate, t, cells):
        np.testing.assert_allclose(source, estimated)
        np.testing.assert_allclose(wind, [0.0, 2.0])
        assert rate == 1.0
        assert t >= seen[-1]
        return actual_features(f, source, wind, rate, t, cells)

    monkeypatch.setattr(library, "readings", noisy_readings)
    monkeypatch.setattr(library, "cell_features", features)
    x, _ = training_rows(factory, 4, 2, np.array([10.0, 20.0]), 1001, locator=Locator())
    assert x.shape == (16, 7)
    assert len(seen) == 4


def test_default_training_covers_early_and_late_forecasts():
    x, _ = training_rows(Factory.default(), 40, 1, seed=1001)
    assert x[:, 3].min() <= 5.0
    assert x[:, 3].max() >= 130.0


def test_fit_accepts_a_source_locator_to_match_the_live_pipeline():
    factory = Factory.default()

    class Locator:
        calls = 0

        def locate(self, f, obs):
            self.calls += 1
            prob = np.zeros_like(f.walls, dtype=float)
            prob[f.cell_of(f.leak_sites[0])] = 1.0
            return SourceEstimate(prob)

    locator = Locator()
    TabPFNHazardPredictor(backend="baseline").fit(
        factory, n_scenarios=2, cells_per_scenario=2, source_locator=locator
    )
    assert locator.calls == 2


def test_predict_batches_sources_and_mixes_probabilities() -> None:
    f = Factory.default()
    prob = np.zeros_like(f.walls, dtype=float)
    prob[1, 1], prob[1, 2] = 0.75, 0.25
    readings = SensorReadings(
        f.sensors, np.array([60.0]), np.zeros((len(f.sensors), 1)), np.array([1.0, 0.0])
    )

    class Fake:
        calls = 0

        def predict_dist(self, x):
            self.calls += 1
            # Distinct deterministic distributions for the two source blocks.
            n = len(x) // 2
            values = np.repeat(np.r_[np.full(n, 0.2), np.zeros(n)][:, None], 2, axis=1)
            return QuantileDist(np.array([0.1, 0.9]), values, values[:, 0])

    model = Fake()
    predictor = TabPFNHazardPredictor(model=model, top_k=2)
    hz = predictor.predict(f, readings, SourceEstimate(prob), np.array([90.0]))
    assert model.calls == 1
    assert hz.p_toxic[0, 1, 1] == pytest.approx(0.75)
    assert np.all(hz.p_toxic[:, f.walls] == 0)
    assert np.all(hz.p_fire[:, ~f.ignition] == 0)


@pytest.mark.parametrize("include_geometry", [False, True])
def test_open_cells_only_skips_wall_rows_and_preserves_output_contract(include_geometry) -> None:
    f = Factory.default()
    prob = np.zeros_like(f.walls, dtype=float)
    prob[1, 1], prob[1, 2] = 0.6, 0.4
    obs = SensorReadings(f.sensors, np.array([10.0]), np.zeros((len(f.sensors), 1)), np.zeros(2))

    class Fake:
        rows = 0

        def predict_dist(self, x):
            self.rows = len(x)
            values = np.repeat(np.where(x[:, 2] < 10.0, 0.2, 0.0)[:, None], 2, axis=1)
            return QuantileDist(np.array([0.1, 0.9]), values, values[:, 0])

    times = np.array([10.0, 20.0])
    full_model, open_model = Fake(), Fake()
    full = TabPFNHazardPredictor(
        model=full_model, top_k=2, include_geometry=include_geometry
    ).predict(f, obs, SourceEstimate(prob), times)
    compact = TabPFNHazardPredictor(
        model=open_model, top_k=2, open_cells_only=True, include_geometry=include_geometry
    ).predict(f, obs, SourceEstimate(prob), times)
    assert full_model.rows == 2 * len(times) * f.ny * f.nx
    assert open_model.rows == 2 * len(times) * np.count_nonzero(~f.walls)
    np.testing.assert_array_equal(compact.p_toxic, full.p_toxic)
    np.testing.assert_array_equal(compact.p_fire, full.p_fire)


def test_offline_predictor_fit_and_predict() -> None:
    f = Factory.default()
    model = TabPFNHazardPredictor(backend="baseline", top_k=1)
    model.fit(f, n_scenarios=3, cells_per_scenario=5, times=np.array([70.0, 90.0]), seed=1)
    prob = np.zeros_like(f.walls, dtype=float)
    prob[1, 1] = 1
    readings = SensorReadings(
        f.sensors, np.array([60.0]), np.zeros((len(f.sensors), 1)), np.zeros(2)
    )
    hz = model.predict(f, readings, SourceEstimate(prob), np.array([90.0]))
    assert hz.p_toxic.shape == (1, f.ny, f.nx)
    assert np.all((hz.p_toxic >= 0) & (hz.p_toxic <= 1))
