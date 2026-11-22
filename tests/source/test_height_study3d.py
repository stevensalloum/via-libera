import numpy as np

from tabpfn_spatial.source.height_study3d import plant_for, prior_error, site_heights, voxel_limit


def test_site_heights_lie_on_the_drop_pipes() -> None:
    h = site_heights(plant_for("methane"))
    assert len(h) >= 6 and h.min() >= 1.5 and h.max() < 4.2


def test_prior_and_voxel_limit() -> None:
    plant = plant_for("propane")
    h = site_heights(plant)
    middle = (h.min() + h.max()) / 2
    assert np.isclose(prior_error(np.array([middle + 1.0, middle - 1.0]), plant), 1.0)
    # sites in one 1 m voxel give the same gas: the best guess is their mean
    same = h[np.floor(h) == 2.0]
    assert len(same) >= 2
    assert np.isclose(voxel_limit(same, plant), np.mean(np.abs(same - same.mean())))
    assert voxel_limit(h, plant) < prior_error(h, plant)
