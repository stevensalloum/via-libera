"""Check that the horizontal locator uses every 3D detector at the alarm."""

import dataclasses

import numpy as np
import pytest

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, simulate
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.source.locator3d import TabPFNLocator3D


def test_locator3d_accepts_mast_alarm_and_returns_location_and_rate() -> None:
    plant = Plant3D(dataclasses.replace(Factory.default(), gas="propane"))
    loc = TabPFNLocator3D(backend="baseline", n_train=8, seed=100, ws=(0.0, 10.0))
    loc.fit(plant)
    v = plant.vertical_pipes[0]
    run = simulate(plant, Leak3D((*v.xy, 3.06), 1.0, plant.f.gas), Ventilation(0.6, 1.0), 30)
    r = run.readings(np.random.default_rng(899))
    alarm = run.alarm(r)
    assert alarm is not None
    keep = r.times <= alarm
    r = dataclasses.replace(r, times=r.times[keep], values=r.values[:, keep])
    assert len(r.values) > len(plant.f.sensors)
    assert np.any(r.values > 0.03)
    assert not np.any(r.values[: len(plant.f.sensors)] > 0.03)
    est = loc.locate(plant.f, r)
    assert est.prob.shape == plant.f.walls.shape and np.isclose(est.prob.sum(), 1.0)
    assert not np.allclose(est.prob, plant.f.leak_prior)
    assert est.rate is not None and np.all(np.isfinite(est.rate))
    with pytest.raises(ValueError, match="detector"):
        loc.locate(
            plant.f,
            dataclasses.replace(
                r, values=r.values[:-1], positions=r.positions[:-1], heights=r.heights[:-1]
            ),
        )
