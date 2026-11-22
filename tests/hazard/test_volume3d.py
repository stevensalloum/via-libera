"""Height-aware hazard prototype: physics, inputs and probability volumes."""

import numpy as np
import pytest

from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.hazard.volume3d import HazardVolume, MachineBox3D, VolumeGrid, simulate_volume
from tabpfn_spatial.hazard.volume3d_predictor import (
    ForwardVolumePredictor,
    TabPFNVolumePredictor,
    features_3d,
    training_rows_3d,
)


def _grid():
    solid = np.zeros((4, 5, 6), bool)  # z, y, x
    solid[:, 2, 3] = True
    ignition = np.zeros_like(solid)
    ignition[1, 1, 2] = True
    return VolumeGrid(solid=solid, ignition=ignition, cell=1.0, dz=1.0)


def test_transient_volume_varies_by_height_and_spreads_over_time():
    grid = _grid()
    fields = simulate_volume(grid, np.array([1.5, 1.5, 1.5]), 1.0, np.array([0.3, 0]),
                             np.array([0., 1., 3.]), diffusivity=0.2)  # fmt: skip
    assert fields.shape == (3, 4, 5, 6)
    assert np.all(fields >= 0) and np.all(fields[:, grid.solid] == 0)
    assert fields[0].sum() == 0
    assert fields[-1, 1, 1, 1] > fields[-1, 3, 1, 1] > 0
    assert fields[-1].sum() > fields[1].sum()


def test_diffusion_is_symmetric_and_conserves_mass_away_from_boundaries():
    solid = np.zeros((7, 7, 7), dtype=bool)
    grid = VolumeGrid(solid, np.zeros_like(solid), cell=1.0, dz=1.0)
    field = simulate_volume(grid, np.array([3.5, 3.5, 3.5]), 1.0, np.zeros(2),
                            np.array([3.]), diffusivity=0.1)[0]  # fmt: skip
    neighbours = (
        field[2, 3, 3],
        field[4, 3, 3],
        field[3, 2, 3],
        field[3, 4, 3],
        field[3, 3, 2],
        field[3, 3, 4],
    )
    assert field.sum() == pytest.approx(3.0)
    np.testing.assert_allclose(neighbours, neighbours[0])


def test_3d_source_is_required_and_cannot_be_in_a_solid_cell():
    grid = _grid()
    for source in (np.array([1.5, 1.5]), np.array([3.5, 2.5, 1.5]),
                   np.array([1.5, 1.5, 4.5])):  # fmt: skip
        with pytest.raises(ValueError):
            simulate_volume(grid, source, 1.0, np.array([1.0, 0]), np.array([1.0]))


def test_solid_floor_blocks_vertical_diffusion():
    solid = np.zeros((4, 5, 6), bool)
    solid[2] = True
    grid = VolumeGrid(solid, np.zeros_like(solid), cell=1.0, dz=1.0)
    fields = simulate_volume(grid, np.array([1.5, 1.5, 0.5]), 1.0, np.array([0., 0.]),
                             np.array([5.]), diffusivity=0.5)  # fmt: skip
    assert fields[0, 1].sum() > 0
    assert fields[0, 3].sum() == 0


def test_3d_machine_dimensions_control_obstacles_and_ignition():
    walls = np.zeros((4, 5, 6), dtype=bool)
    machine = MachineBox3D((2.0, 1.0, 0.0), (4.0, 3.0, 1.5), hot=True)
    grid = VolumeGrid.from_layout(walls, [machine], cell=1.0, dz=1.0, ignition_radius=1.1)
    assert grid.solid[0, 1, 2] and grid.solid[1, 1, 2]
    assert not grid.solid[2, 1, 2]  # gas can pass over the machine
    assert grid.ignition[2, 1, 2]  # hot machine's influence has a specified radius
    assert not grid.ignition[3, 1, 2]
    assert not np.any(grid.ignition & grid.solid)


def test_3d_machine_layout_rejects_incomplete_or_invalid_dimensions():
    walls = np.zeros((4, 5, 6), dtype=bool)
    with pytest.raises(ValueError):
        MachineBox3D((1.0, 1.0), (2.0, 2.0))
    with pytest.raises(ValueError):
        VolumeGrid.from_layout(walls, [MachineBox3D((0., 0., 0.), (1., 1., 5.))],
                               cell=1.0, dz=1.0, ignition_radius=1.0)  # fmt: skip
    with pytest.raises(ValueError):
        VolumeGrid.from_layout(walls, [MachineBox3D((0., 0., 0.), (1., 1., 1.), hot=True)],
                               cell=1.0, dz=1.0, ignition_radius=0.0)  # fmt: skip


def test_gas_can_travel_over_a_short_machine_but_not_a_full_height_barrier():
    walls = np.zeros((4, 5, 6), dtype=bool)
    short = VolumeGrid.from_layout(
        walls,
        [MachineBox3D((2.0, 0.0, 0.0), (3.0, 5.0, 2.0))],
        cell=1.0,
        dz=1.0,
        ignition_radius=0.0,
    )
    tall = VolumeGrid.from_layout(
        walls,
        [MachineBox3D((2.0, 0.0, 0.0), (3.0, 5.0, 4.0))],
        cell=1.0,
        dz=1.0,
        ignition_radius=0.0,
    )
    args = (np.array([1.5, 2.5, 0.5]), 1.0, np.array([0.5, 0.0]), np.array([12.0]))
    short_field = simulate_volume(short, *args, diffusivity=0.5)
    tall_field = simulate_volume(tall, *args, diffusivity=0.5)
    assert short_field[..., 3:].sum() > 0
    assert tall_field[..., 3:].sum() == 0
    source, target = np.array([1.5, 2.5, 2.5]), np.array([[4.5, 2.5, 2.5]])
    short_features = features_3d(short, source, np.array([1.0, 0.0]), 1.0, 12.0, target)
    tall_features = features_3d(tall, source, np.array([1.0, 0.0]), 1.0, 12.0, target)
    assert short_features[0, -1] == 1.0
    assert tall_features[0, -1] < 1.0


def test_3d_training_is_reproducible_and_uses_real_3d_sources():
    grid = _grid()
    candidates = np.zeros_like(grid.solid)
    candidates[1, 1, 1] = candidates[2, 3, 4] = True
    args = dict(n_scenarios=5, cells_per_scenario=4, seed=1001, times=np.array([1.0, 3.0]))
    x, y = training_rows_3d(grid, candidates, **args)
    x2, y2 = training_rows_3d(grid, candidates, **args)
    assert x.shape[0] == len(y) == 40
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    assert np.ptp(x[:, 3]) > 0  # height is an actual model feature
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_models_return_3d_hazard_and_an_explicit_2d_slice():
    grid = _grid()
    candidates = ~grid.solid
    predictor = TabPFNVolumePredictor(backend="baseline", top_k=2).fit(
        grid, candidates, n_scenarios=6, cells_per_scenario=5, times=np.array([1.0, 3.0])
    )
    sources = np.array([[1.5, 1.5, 1.5], [4.5, 3.5, 2.5]])
    weights = np.array([0.7, 0.3])
    times = np.array([1.0, 3.0])
    for model in (predictor, ForwardVolumePredictor(diffusivity=0.8)):
        result = model.predict(grid, sources, weights, np.array([0.3, 0.0]), 1.0, times)
        assert result.p_toxic.shape == result.p_fire.shape == (2, 4, 5, 6)
        assert np.isfinite(result.p_toxic).all()
        assert np.all((result.p_toxic >= 0) & (result.p_toxic <= 1))
        assert np.all(result.p_toxic[:, grid.solid] == 0)
        assert np.all(result.p_fire[:, ~grid.ignition] == 0)
        slice_2d = result.at_height(1)
        np.testing.assert_array_equal(slice_2d.p_toxic, result.p_toxic[:, 1])
        assert slice_2d.p_toxic.shape == (2, 5, 6)


def test_hazard_rejects_missing_3d_source_and_invalid_weights():
    grid = _grid()
    model = ForwardVolumePredictor()
    for positions, weights in (([[1.5, 1.5]], [1.0]), ([[1.5, 1.5, 1.5]], [0.0])):
        with pytest.raises(ValueError):
            model.predict(grid, np.array(positions), np.array(weights), np.array([1., 0.]),
                          1.0, np.array([1.]))  # fmt: skip


def test_hazard_volume_rejects_incompatible_shapes_and_invalid_probabilities():
    toxic = np.zeros((2, 3, 4, 5))
    with pytest.raises(ValueError, match="match"):
        HazardVolume(np.array([1.0]), toxic, toxic)
    invalid = toxic.copy()
    invalid[0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="probabilities"):
        HazardVolume(np.array([1.0, 2.0]), invalid, toxic)
    result = HazardVolume(np.array([1.0, 2.0]), toxic, toxic)
    with pytest.raises(ValueError, match="height"):
        result.at_height(3)


def test_source_hypotheses_are_weighted_and_query_batches_keep_order():
    grid = _grid()

    class Model:
        def predict_dist(self, rows):
            mean = (rows[:, 0] > -2).astype(float)
            return QuantileDist(np.array([0.05, 0.95]), np.column_stack([mean, mean]), mean)

    predictor = TabPFNVolumePredictor(backend="baseline", top_k=2, batch_size=1)
    predictor.model = Model()
    predictor._training_grid = grid
    sources = np.array([[1.5, 1.5, 1.5], [4.5, 3.5, 2.5]])
    result = predictor.predict(grid, sources, np.array([0.7, 0.3]), np.array([1., 0.]),
                               1.0, np.array([1.]))  # fmt: skip
    assert result.p_toxic[0, 0, 0, 0] == pytest.approx(0.7)
    assert predictor.last_retained_mass == pytest.approx(1.0)
    predictor.top_k = 1
    result = predictor.predict(grid, sources, np.array([0.7, 0.3]), np.array([1., 0.]),
                               1.0, np.array([1.]))  # fmt: skip
    assert result.p_toxic[0, 0, 0, 0] == 1.0
    assert predictor.last_retained_mass == pytest.approx(0.7)


def test_predictor_rejects_a_layout_different_from_its_training_layout():
    grid = _grid()
    predictor = TabPFNVolumePredictor(backend="baseline").fit(
        grid, ~grid.solid, n_scenarios=2, cells_per_scenario=2, times=np.array([1.0])
    )
    new_solid = grid.solid.copy()
    new_solid[3, 4, 5] = True
    other = VolumeGrid(new_solid, grid.ignition, grid.cell, grid.dz)
    with pytest.raises(ValueError, match="layout"):
        predictor.predict(other, np.array([[1.5, 1.5, 1.5]]), np.array([1.]),
                          np.array([1., 0.]), 1.0, np.array([1.]))  # fmt: skip


def test_feature_construction_and_model_queries_are_bounded_by_batch_size(monkeypatch):
    import tabpfn_spatial.hazard.volume3d_predictor as module

    grid = _grid()
    real_features = module.features_3d
    feature_sizes, model_sizes = [], []

    def measured_features(grid, source, wind, rate, t, xyz):
        feature_sizes.append(len(xyz))
        return real_features(grid, source, wind, rate, t, xyz)

    class Model:
        def predict_dist(self, rows):
            model_sizes.append(len(rows))
            mean = np.zeros(len(rows))
            return QuantileDist(np.array([0.05, 0.95]), np.column_stack([mean, mean]), mean)

    monkeypatch.setattr(module, "features_3d", measured_features)
    predictor = TabPFNVolumePredictor(backend="baseline", top_k=2, batch_size=7)
    predictor.model, predictor._training_grid = Model(), grid
    predictor.predict(grid, np.array([[1.5, 1.5, 1.5], [4.5, 3.5, 2.5]]), np.array([0.7, 0.3]),
                      np.array([1., 0.]), 1.0, np.array([1., 3.]))  # fmt: skip
    assert max(feature_sizes) <= 7 and max(model_sizes) <= 7
    assert sum(model_sizes) == 2 * 2 * int((~grid.solid).sum())
