import numpy as np

from tabpfn_spatial.evac.crowd import sample_workers
from tabpfn_spatial.evac.crowd_run import run_crowd
from tabpfn_spatial.evac.prepare3d import breathing_mask, leak_for, prepare3d, to_cells
from tabpfn_spatial.sim import Factory, sample_scenarios
from tabpfn_spatial.sim.plant3d import Plant3D


def _leak():
    f = Factory.default()
    return f, sample_scenarios(f, 1, np.random.default_rng(500))[0]


def test_leak_is_a_3d_pipe_site_and_maps_fit_the_planner_grid():
    f, sc = _leak()
    plant = Plant3D(f)
    leak = leak_for(f, sc, 0)
    sites, _ = plant.leak_sites
    assert np.isclose(np.linalg.norm(sites - leak.xyz, axis=1).min(), 0.0)
    assert leak.rate == sc.rate
    mask = breathing_mask(plant)
    assert mask.shape == plant.to_zyx(plant.solid).shape and 0 < mask.mean() < 0.5
    vox = np.zeros(plant.to_zyx(plant.solid).shape[1:])
    vox[3, 5] = 1.0
    cells = to_cells(f, vox, plant.grid.cell)
    assert cells.shape == f.walls.shape and cells.sum() == 1.0


def test_prepare3d_perfect_models_drive_every_policy():
    f, sc = _leak()
    prep, run = prepare3d(f, sc, "perfect", t_max=90.0, seed=0)
    assert prep.fields.shape == (91, *f.walls.shape)
    assert prep.t_alarm is not None and len(prep.danger) == len(prep.replan_times)
    assert np.allclose(prep.scenario.source, run.leak.xyz[:2])
    people = sample_workers(f, 20, np.random.default_rng(1))
    live = run_crowd(f, prep, people, "ccrp", record_routes=True)
    once = run_crowd(f, prep, people, "ccrp", record_routes=True, once=True)
    assert len(once.routes) == 1 <= len(live.routes) <= len(prep.replan_times)
    assert np.isfinite(live.metrics["harm_s_mean"]) and np.isfinite(once.metrics["harm_s_mean"])
