"""3D plant and time-stepped gas: geometry rules and physics, checked against laws."""

import numpy as np
import pytest

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, _wind, simulate
from tabpfn_spatial.sim.layouts import template
from tabpfn_spatial.sim.plant3d import MAST_MIN_M, MOUNT_Z, PIPE_HEIGHT, Plant3D
from tabpfn_spatial.study3d import physics as ph


@pytest.fixture(scope="module")
def plant() -> Plant3D:
    return Plant3D(Factory.default())


def test_machines_have_height_and_walls_reach_the_roof(plant: Plant3D) -> None:
    g, s = plant.grid, plant.solid
    for m in plant.machines:
        x0, y0, _, x1, y1, h = m["box"]
        i = g.index(np.array([[(x0 + x1) / 2, (y0 + y1) / 2, h - 0.5]]))[0]
        j = g.index(np.array([[(x0 + x1) / 2, (y0 + y1) / 2, h + 0.6]]))[0]
        assert s[tuple(i)] and (h + 0.6 >= 9 or not s[tuple(j)])


def test_mast_rule_and_mounting_heights() -> None:
    p = Plant3D(template("U-plant"))  # propane: heavy, detectors low
    det, why = p.detectors
    assert np.allclose(det[np.array(why) == "mounted", 2], MOUNT_Z["heavy"])
    long = [v for v in p.vertical_pipes if v.z1 - v.z0 >= MAST_MIN_M]
    assert why.count("mast") == 3 * len(long)


def test_leak_sites_cover_the_vertical_pipes(plant: Plant3D) -> None:
    sites, w = plant.leak_sites
    assert (sites[:, 2] == PIPE_HEIGHT).any() and (sites[:, 2] < PIPE_HEIGHT - 1).any()
    assert len(sites) == len(w) and (w > 0).all()


def test_stepping_converges_to_the_steady_solve(plant: Plant3D) -> None:
    vent = Ventilation(speed=0.5, direction=0.3, diffusivity=0.8)
    leak = Leak3D((40.0, 16.0, 1.6), 1.0, "CO")
    run = simulate(plant, leak, vent, t_end=900, every=300)
    flow, w = _wind(plant, vent, vent.direction)
    steady = ph.transport(
        plant.grid,
        plant.solid,
        flow,
        w,
        np.array(leak.xyz),
        rate=19.4,
        k_field=np.full(plant.solid.shape, vent.diffusivity),
    )
    near = steady > 0.05 * steady.max()
    assert np.median(run.volumes[-1][near] / steady[near]) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(("gas", "up"), [("hydrogen", True), ("chlorine", False)])
def test_light_rises_heavy_sinks_near_the_leak(plant: Plant3D, gas: str, up: bool) -> None:
    leak = Leak3D((40.0, 16.0, 3.0), 2.0, gas)
    run = simulate(plant, leak, Ventilation(speed=0.2), t_end=60, every=60)
    g = plant.grid
    zs = g.axis(2)
    i = g.index(np.array([leak.xyz]))[0]
    col = run.volumes[-1][i[0] - 2 : i[0] + 3, i[1] - 2 : i[1] + 3].mean(axis=(0, 1))
    above, below = col[zs > 4.0].mean(), col[zs < 2.0].mean()
    assert (above > below) == up


def test_readings_alarm_and_arrivals(plant: Plant3D) -> None:
    run = simulate(plant, Leak3D((40.0, 16.0, 1.6), 1.5, "CO"), Ventilation(0.6), t_end=120)
    r = run.readings(np.random.default_rng(0))
    assert r.values.shape == (len(r.positions), len(run.times)) and r.heights is not None
    arr = run.arrivals()
    assert np.isfinite(arr).any()
    alarm = run.alarm(r)
    assert alarm is not None and alarm <= np.nanmin(arr) + 10
    assert run.volumes_zyx().shape[1:] == plant.solid.shape[::-1]
    assert run.breathing_map().shape[1:] == plant.f.walls.shape
