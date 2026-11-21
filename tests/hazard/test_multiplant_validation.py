"""Tests for the paired validation harness; never contact the cloud."""

import runpy
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.hazard import TabPFNHazardPredictor
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.sim import Factory, readings, sample_scenarios


def load_harness():
    """Load the research script without executing its CLI."""
    return runpy.run_path(str(Path(__file__).with_name("compare_multiplant.py")))


class FakeRegressor:
    """Return deterministic quantiles with a counted prediction call."""

    def __init__(self):
        self.calls = 0

    def predict_dist(self, x):
        self.calls += 1
        mean = 0.12 + 0.004 * np.tanh(x[:, 0]) + 0.001 * x[:, 3]
        return QuantileDist(np.array([0.1, 0.9]), np.column_stack([mean - 0.1, mean + 0.1]), mean)


def test_batched_forecasts_match_the_real_predictor_with_one_model_call():
    harness = load_harness()
    f = Factory.default()
    obs = SensorReadings(f.sensors, np.array([10.0]), np.zeros((len(f.sensors), 1)), [1.0, 0.0])
    prob = np.zeros_like(f.walls, dtype=float)
    for pos, p in zip(f.leak_sites[:3], [0.4, 0.35, 0.25], strict=True):
        prob[f.cell_of(pos)] = p
    source = SourceEstimate(prob, np.array([0.7, 1.2, 1.8]), np.array([0.1, 0.5, 0.9]))
    times = [np.array([10.0, 20.0]), np.array([20.0, 30.0])]
    expected_model = FakeRegressor()
    single = TabPFNHazardPredictor(model=expected_model, top_k=3, open_cells_only=True)
    expected = [single.predict(f, obs, source, ts) for ts in times]
    batched_model = FakeRegressor()
    actual = harness["batched_hazards"](f, [(obs, source, ts) for ts in times], batched_model)
    assert expected_model.calls == 2
    assert batched_model.calls == 1
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a.p_toxic, b.p_toxic)
        np.testing.assert_allclose(a.p_fire, b.p_fire)


def test_variant_readings_match_original_at_default_diffusivity():
    harness = load_harness()
    f = Factory.default()
    sc = sample_scenarios(f, 1, np.random.default_rng(44))[0]
    a = readings(f, sc, 20.0, np.random.default_rng(3), dropout=0.1, stuck=0.05)
    b = harness["variant_readings"](f, sc, 20.0, np.random.default_rng(3), 0.8, 0.1, 0.05)
    for key in ("positions", "times", "values", "wind"):
        np.testing.assert_allclose(getattr(a, key), getattr(b, key), equal_nan=True)


def test_batched_source_contexts_replay_the_same_simulated_targets():
    harness = load_harness()
    f = Factory.default()
    record = harness["RecordingLocator"]()
    args = dict(n_scenarios=4, cells_per_scenario=3, seed=1001)
    x1, y1 = training_rows(f, **args, locator=record)
    replay = harness["ReplayLocator"](
        record.obs, [SourceEstimate(f.leak_prior.copy()) for _ in record.obs]
    )
    x2, y2 = training_rows(f, **args, locator=replay)
    assert replay.index == len(record.obs)
    np.testing.assert_array_equal(y1, y2)
    assert x1.shape == x2.shape
    assert not np.array_equal(x1, x2)


def test_recorded_figure_can_be_redrawn_without_case_data(tmp_path):
    script = Path(__file__).with_name("summarise_multiplant.py")
    harness = runpy.run_path(str(script))
    output = tmp_path / "recorded.png"
    harness["make_recorded_figure"](output)
    assert output.is_file()
    assert output.stat().st_size > 0
