import runpy
from pathlib import Path

import numpy as np
import pytest

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.hazard import hazard_from_fields, library
from tabpfn_spatial.hazard.tabpfn_predictor import TabPFNHazardPredictor
from tabpfn_spatial.sim import Factory, readings, sample_scenarios, simulate
from tabpfn_spatial.sim.layouts import template
from tabpfn_spatial.source import TabPFNLocator


def test_varying_diffusivity_pairs_observations_and_targets_without_extra_features(monkeypatch):
    factory = Factory.default()
    observations, outcomes = [], []
    real_readings, real_simulate = library.readings, library.simulate

    def readings(*args, **kwargs):
        observations.append(kwargs["diffusivity"])
        return real_readings(*args, **kwargs)

    def simulate(*args, **kwargs):
        outcomes.append(kwargs["d"])
        return real_simulate(*args, **kwargs)

    monkeypatch.setattr(library, "readings", readings)
    monkeypatch.setattr(library, "simulate", simulate)
    x, y = library.training_rows(factory, 8, 3, seed=1001, diffusivity_range=(0.4, 1.6))
    assert x.shape == (96, 7) and y.shape == (96,)
    np.testing.assert_array_equal(observations, outcomes)
    assert all(0.4 <= d <= 1.6 for d in observations)
    assert len(set(observations)) == 8
    repeat_x, repeat_y = library.training_rows(
        factory, 8, 3, seed=1001, diffusivity_range=(0.4, 1.6)
    )
    np.testing.assert_array_equal(x, repeat_x)
    np.testing.assert_array_equal(y, repeat_y)


def test_fixed_diffusivity_preserves_default_context():
    factory = Factory.default()
    x, y = library.training_rows(factory, 5, 3, seed=1001)
    fixed_x, fixed_y = library.training_rows(factory, 5, 3, seed=1001, diffusivity_range=(0.8, 0.8))
    np.testing.assert_array_equal(x, fixed_x)
    np.testing.assert_array_equal(y, fixed_y)


@pytest.mark.parametrize("bounds", [(0, 1), (-1, 1), (2, 1), (np.nan, 1), (1, np.inf)])
def test_invalid_diffusivity_range_is_rejected(bounds):
    with pytest.raises(ValueError, match="diffusivity"):
        library.training_rows(Factory.default(), 1, 1, diffusivity_range=bounds)


def test_predictor_passes_diffusivity_range_to_training(monkeypatch):
    seen = {}

    def rows(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((2, 7)), np.zeros(2)

    monkeypatch.setattr("tabpfn_spatial.hazard.tabpfn_predictor.training_rows", rows)
    TabPFNHazardPredictor(backend="baseline").fit(Factory.default(), diffusivity_range=(0.4, 1.6))
    assert seen["diffusivity_range"] == (0.4, 1.6)


def test_varied_context_predicts_valid_probabilities_offline():
    factory = Factory.default()
    predictor = TabPFNHazardPredictor(backend="baseline", top_k=1).fit(
        factory, n_scenarios=3, cells_per_scenario=3, diffusivity_range=(0.4, 1.6)
    )
    probabilities = np.zeros_like(factory.walls, dtype=float)
    probabilities[factory.cell_of(factory.leak_sites[0])] = 1.0
    obs = SensorReadings(
        factory.sensors, np.array([10.0]), np.zeros((len(factory.sensors), 1)), np.ones(2)
    )
    result = predictor.predict(factory, obs, SourceEstimate(probabilities), np.array([10.0]))
    assert result.p_toxic.shape == (1, factory.ny, factory.nx)
    assert np.all((result.p_toxic >= 0) & (result.p_toxic <= 1))


@pytest.mark.parametrize("plant", ["L-plant", "U-plant", "T-plant", "campus"])
def test_varied_context_accepts_shaped_plant_and_team_source(plant):
    factory = template(plant)
    locator = TabPFNLocator(backend="baseline", n_train=8).fit(factory)
    predictor = TabPFNHazardPredictor(backend="baseline", top_k=1).fit(
        factory, n_scenarios=2, cells_per_scenario=2, diffusivity_range=(0.4, 1.6)
    )
    values = np.zeros((len(factory.sensors), 1))
    values[0, 0] = 0.04  # an alarmed measurement, to exercise the source model
    obs = SensorReadings(factory.sensors, np.array([10.0]), values, np.array([0.7, 0.2]))
    source = locator.locate(factory, obs)
    result = predictor.predict(factory, obs, source, np.array([10.0]))
    assert result.p_toxic.shape == (1, factory.ny, factory.nx)
    assert np.isfinite(result.p_toxic).all()
    assert np.all((result.p_toxic >= 0) & (result.p_toxic <= 1))
    assert np.all(result.p_toxic[:, factory.walls] == 0)


def test_evaluation_uses_held_out_scenarios_and_consistent_physics():
    experiment = runpy.run_path(str(Path(__file__).with_name("compare_diffusivity.py")))
    factory, seed = Factory.default(), experiment["TEST_SEED"]
    with pytest.raises(ValueError, match="test seed"):
        experiment["make_cases"](factory, 2, experiment["CONTEXT_SEED"])
    with pytest.raises(ValueError, match="test seed"):
        experiment["make_cases"](factory, 2, experiment["SOURCE_SEED"])
    cases = experiment["make_cases"](factory, 2, seed)
    scenarios = sample_scenarios(factory, 2, np.random.default_rng(seed))
    for case in cases:
        i, diffusion = case["index"], case["diffusion"]
        expected = readings(
            factory,
            scenarios[i],
            case["obs"].t_obs,
            np.random.default_rng(seed + 10000 + i),
            diffusivity=diffusion,
        )
        np.testing.assert_array_equal(case["obs"].values, expected.values)
        fields = simulate(factory, [scenarios[i]], case["times"], d=diffusion)[0]
        toxic, _ = hazard_from_fields(factory, fields)
        np.testing.assert_array_equal(case["truth"], toxic)


def test_paired_bootstrap_rejects_incomplete_layout_groups():
    analysis = runpy.run_path(str(Path(__file__).with_name("summarise_diffusivity.py")))
    scores = {"fixed": {"brier": 0.1}, "varied": {"brier": 0.2}}
    rows = [
        {"layout": "a", "episode": 0, **scores},
        {"layout": "b", "episode": 0, **scores},
        {"layout": "a", "episode": 1, **scores},
    ]
    with pytest.raises(ValueError, match="one result per layout"):
        analysis["paired_interval"](rows)
