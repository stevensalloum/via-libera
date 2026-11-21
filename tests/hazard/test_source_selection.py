import numpy as np
import pytest

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.hazard.source_selection import select_source_cells
from tabpfn_spatial.hazard.tabpfn_predictor import TabPFNHazardPredictor
from tabpfn_spatial.sim import Factory


def test_selects_minimum_cells_needed_for_target_mass() -> None:
    prob = np.array([[0.4, 0.3, 0.2, 0.1]])
    selection = select_source_cells(prob, np.zeros_like(prob, dtype=bool), 4, 0.8)
    np.testing.assert_array_equal(selection.indices, [0, 1, 2])
    np.testing.assert_allclose(selection.weights, np.array([0.4, 0.3, 0.2]) / 0.9)
    assert selection.retained_mass == pytest.approx(0.9)
    assert selection.target_reached is True


def test_cap_reports_when_target_cannot_be_reached() -> None:
    prob = np.array([[0.4, 0.3, 0.2, 0.1]])
    selection = select_source_cells(prob, np.zeros_like(prob, dtype=bool), 2, 0.9)
    assert selection.retained_mass == pytest.approx(0.7)
    assert selection.target_reached is False


def test_invalid_wall_mass_is_not_hidden_by_renormalization() -> None:
    prob = np.array([[0.6, 0.4]])
    selection = select_source_cells(prob, np.array([[True, False]]), 2, 0.9)
    assert selection.retained_mass == pytest.approx(0.4)
    assert selection.target_reached is False
    np.testing.assert_array_equal(selection.indices, [1])


@pytest.mark.parametrize("target", [0.0, -0.1, 1.1, np.nan])
def test_rejects_invalid_mass_target(target) -> None:
    with pytest.raises(ValueError):
        select_source_cells(np.array([[1.0]]), np.array([[False]]), 1, target)
    with pytest.raises(ValueError):
        TabPFNHazardPredictor(source_mass=target)


def test_predictor_exposes_retained_mass_when_cap_binds() -> None:
    factory = Factory.default()
    prob = np.zeros_like(factory.walls, dtype=float)
    prob[1, :4] = [0.4, 0.3, 0.2, 0.1]
    obs = SensorReadings(
        factory.sensors, np.array([60.0]), np.zeros((len(factory.sensors), 1)), np.array([1.0, 0.0])
    )

    class Fake:
        def predict_dist(self, x):
            values = np.zeros((len(x), 2))
            return QuantileDist(np.array([0.1, 0.9]), values, values[:, 0])

    predictor = TabPFNHazardPredictor(model=Fake(), top_k=2, source_mass=0.9)
    with pytest.warns(RuntimeWarning, match="source probability"):
        predictor.predict(factory, obs, SourceEstimate(prob), np.array([90.0]))
    assert predictor.last_selection.retained_mass == pytest.approx(0.7)
    assert predictor.last_selection.target_reached is False
