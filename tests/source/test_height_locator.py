import dataclasses

import numpy as np
import pytest

from tabpfn_spatial.contracts import SourceEstimate
from tabpfn_spatial.core.dist import DEFAULT_LEVELS
from tabpfn_spatial.core.models import BaselineRegressor
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, simulate
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.source.height_locator import HeightLocator


@pytest.fixture(scope="module")
def fitted() -> tuple[Plant3D, HeightLocator]:
    plant = Plant3D(dataclasses.replace(Factory.default(), gas="propane"))
    loc = HeightLocator(n_train=6, ws=(0.0, 10.0), regressor=BaselineRegressor(k=3))
    return plant, loc.fit(plant)


def _readings(plant: Plant3D, z: float, t_end: float):
    v = plant.vertical_pipes[0]
    run = simulate(plant, Leak3D((*v.xy, z), 1.0, plant.f.gas), Ventilation(0.6, 1.0), t_end)
    return run.readings(np.random.default_rng(0))


def test_height_quantiles_after_the_alarm(fitted) -> None:
    plant, loc = fitted
    q = loc.height([_readings(plant, 3.06, 30.0)])
    assert q.shape == (1, len(DEFAULT_LEVELS))
    assert np.all(np.diff(q[0]) >= 0)
    assert loc.z_range[0] <= q[0, 0] and q[0, -1] <= loc.z_range[1]


def test_prior_before_any_alarm_and_estimate_gets_its_height(fitted) -> None:
    plant, loc = fitted
    quiet = _readings(plant, 2.0, 0.0)  # only t = 0: nothing has alarmed
    np.testing.assert_allclose(loc.height([quiet])[0], loc.prior())
    est = SourceEstimate(np.full((plant.f.ny, plant.f.nx), 1.0 / (plant.f.ny * plant.f.nx)))
    out = loc.attach(est, quiet)
    np.testing.assert_allclose(out.height, loc.prior())
    np.testing.assert_allclose(out.height_levels, DEFAULT_LEVELS)


def test_rejects_readings_from_other_detectors(fitted) -> None:
    plant, loc = fitted
    r = _readings(plant, 2.0, 10.0)
    with pytest.raises(ValueError, match="detectors"):
        loc.height([dataclasses.replace(r, values=r.values[:5], positions=r.positions[:5])])
