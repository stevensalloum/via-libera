"""Focused checks for the source-estimate and factory 3D handover."""

import numpy as np
import pytest

from tabpfn_spatial.contracts import SourceEstimate
from tabpfn_spatial.hazard.plant3d_adapter import plant_grid, plant_training_rows, source_hypotheses
from tabpfn_spatial.hazard.volume3d_predictor import TabPFNVolumePredictor
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.plant3d import Plant3D


def test_plant_grid_preserves_solids_and_hot_zones() -> None:
    plant = Plant3D(Factory.default())
    grid = plant_grid(plant, ignition_radius=1.5)
    np.testing.assert_array_equal(grid.solid, plant.to_zyx(plant.solid))
    assert grid.ignition.any()
    assert not (grid.ignition & grid.solid).any()


def test_source_hypotheses_keep_height_uncertainty_and_mass() -> None:
    plant = Plant3D(Factory.default())
    grid = plant_grid(plant, ignition_radius=1.5)
    sites, _ = plant.leak_sites
    x, y, _ = sites[0]
    iy, ix = plant.f.cell_of((x, y))
    probability = np.zeros_like(plant.f.leak_prior)
    probability[iy, ix] = 1.0
    est = SourceEstimate(
        probability, np.array([1.0]), np.array([0.5]), np.array([1.5, 3.5]), np.array([0.25, 0.75])
    )
    xyz, weights, kept = source_hypotheses(plant, grid, est)
    assert np.allclose(np.sort(xyz[:, 2]), [1.5, 3.5])
    np.testing.assert_allclose(weights, [0.5, 0.5])
    assert kept == pytest.approx(1.0)
    assert np.all(~grid.solid[tuple(grid.indices(xyz).T)])


def test_source_hypotheses_require_real_height_and_valid_probabilities() -> None:
    plant = Plant3D(Factory.default())
    grid = plant_grid(plant, ignition_radius=1.5)
    with pytest.raises(ValueError, match="height"):
        source_hypotheses(plant, grid, SourceEstimate(plant.f.leak_prior))
    est = SourceEstimate(
        plant.f.leak_prior, height=np.array([1.0, 2.0]), height_levels=np.array([0.9, 0.1])
    )
    with pytest.raises(ValueError, match="height"):
        source_hypotheses(plant, grid, est)


def test_training_rows_use_the_shared_3d_simulator() -> None:
    plant = Plant3D(Factory.default())
    grid = plant_grid(plant, ignition_radius=1.5)
    x, y = plant_training_rows(
        plant, grid, n_scenarios=1, cells_per_scenario=5, times=np.array([5.0]), seed=123
    )
    assert x.shape == (5, 9) and y.shape == (5,)
    assert np.isfinite(x).all() and np.isfinite(y).all() and (y >= 0).all()
    model = TabPFNVolumePredictor(backend="baseline").fit_examples(grid, x, y)
    assert model.model is not None
