"""The evacuation loop on the 3D plant with the team's 3D models (the final pipeline).

Same output as ``crowd_run.prepare`` (a ``Prepared``), so every planner, the crowd and the
3D viewer run unchanged; only the physics and the models are 3D:

- **gas:** ``sim.gas3d`` on ``Plant3D(f)``, a leak at a 3D pipe site (rack or vertical
  section, by the plant's leak-site weights); people breathe the worst value between
  0.5 and 2 m (``Run3D.breathing_map``), interpolated to 1 s;
- **alarm:** the first noisy 3D detector reading above the alarm level (masts included);
- **models** (``models="tabpfn"``): Rayan's ``TabPFNLocator3D`` (where + leak size, from
  every detector) and ``HeightLocator`` (leak height), Ahmad's ``TabPFNVolumePredictor``
  (P(toxic), P(fire) in 3D), all fitted on 3D simulations of this plant. The planner's
  danger map is the worst P(harm) at breathing height over the next ``AHEAD`` s, so
  Ahmad's model is only asked for those voxels (``keep``), about a quarter of the hall.
- ``models="perfect"``: the true 3D gas as the danger map (oracle).
"""

import dataclasses
import time
from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.evac.crowd_run import AHEAD, Prepared
from tabpfn_spatial.hazard.baseline import hazard_from_fields
from tabpfn_spatial.sim.factory import Factory
from tabpfn_spatial.sim.gas3d import BREATHING, Leak3D, Run3D, Ventilation, simulate
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.sim.plume import Scenario

FORECAST = np.arange(0.0, AHEAD + 1e-9, 20.0)  # s ahead asked of the hazard model
IGNITION_M = 1.5  # m around hot machines counted as ignition (Ahmad's adapter)


@dataclass
class Models3D:
    """The team's fitted 3D models for one plant (reuse across leaks of that plant)."""

    plant: Plant3D
    locator: object
    heights: object
    hazard: object
    grid: object


def fit_models3d(f: Factory, backend: str = "cloud", n_train: int = 200, n_hazard: int = 24,
                 seed: int = 0) -> Models3D:  # fmt: skip
    """Fit Rayan's 3D locator + height model and Ahmad's 3D hazard model on ``f`` in 3D."""
    from tabpfn_spatial.hazard.plant3d_adapter import plant_grid, plant_training_rows
    from tabpfn_spatial.hazard.volume3d_predictor import TabPFNVolumePredictor
    from tabpfn_spatial.source.height_locator import HeightLocator
    from tabpfn_spatial.source.locator3d import TabPFNLocator3D

    plant = Plant3D(f)
    grid = plant_grid(plant, ignition_radius=IGNITION_M)
    loc = TabPFNLocator3D(backend=backend, n_train=n_train, seed=seed + 100).fit(plant)
    hts = HeightLocator(backend=backend, n_train=n_train, seed=seed + 111, workers=4).fit(plant)
    x, y = plant_training_rows(plant, grid, n_scenarios=n_hazard, cells_per_scenario=80,
                               times=np.arange(5.0, 121.0, 15.0), seed=seed + 222)  # fmt: skip
    hz = TabPFNVolumePredictor(backend=backend, top_k=2).fit_examples(grid, x, y)
    return Models3D(plant, loc, hts, hz, grid)


def breathing_mask(plant: Plant3D) -> np.ndarray:
    """(z, y, x) voxels between the breathing heights."""
    zs = plant.grid.axis(2)
    layer = (zs >= BREATHING[0]) & (zs <= BREATHING[1])
    return np.broadcast_to(layer[:, None, None], plant.to_zyx(plant.solid).shape)


def to_cells(f: Factory, vox_yx: np.ndarray, cell: float) -> np.ndarray:
    """(..., ny_vox, nx_vox) -> (..., ny, nx) on the factory grid (max per 2D cell)."""
    r = round(f.cell / cell)
    a = vox_yx[..., : f.ny * r, : f.nx * r]
    return a.reshape(*a.shape[:-2], f.ny, r, f.nx, r).max(axis=(-3, -1))


def _per_second(run: Run3D, times: np.ndarray) -> np.ndarray:
    """Breathing-height map at 1 s (linear between the 5 s frames)."""
    frames = run.breathing_map()
    pos = np.clip(times / (run.times[1] - run.times[0]), 0, len(frames) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(frames) - 1)
    w = (pos - lo)[:, None, None]
    return (1 - w) * frames[lo] + w * frames[hi]


def leak_for(f: Factory, sc: Scenario, seed: int) -> Leak3D:
    """A 3D leak at one of the plant's pipe sites (rack or vertical), with ``sc``'s size."""
    plant = Plant3D(f)
    sites, w = plant.leak_sites
    idx = plant.grid.index(sites)
    ok = ~plant.solid[tuple(idx.T)]
    sites, w = sites[ok], w[ok]
    i = np.random.default_rng(seed).choice(len(sites), p=w / w.sum())
    return Leak3D(tuple(float(v) for v in sites[i]), float(sc.rate), f.gas)


def prepare3d(f: Factory, sc: Scenario, models: str, t_max: float = 300.0,
              update_s: float = 15.0, seed: int = 0, fitted: Models3D | None = None,
              backend: str = "cloud") -> tuple[Prepared, Run3D]:  # fmt: skip
    """Truth, sensor-triggered alarm and 3D-model danger maps for one leak."""
    plant = fitted.plant if fitted else Plant3D(f)
    leak = leak_for(f, sc, seed)
    sc = dataclasses.replace(sc, source=np.array(leak.xyz[:2]))
    run = simulate(plant, leak, Ventilation.from_scenario(sc), t_end=t_max, every=5.0)
    times = np.arange(0.0, t_max + 0.5, 1.0)
    fields = _per_second(run, times)
    toxic, fire = hazard_from_fields(f, fields)
    readings = run.readings(np.random.default_rng(seed))
    t_alarm = run.alarm(readings)
    replan = np.arange(t_alarm, t_max + 1e-9, update_s) if t_alarm is not None else np.array([])
    danger, log = [], []
    if t_alarm is None:
        return Prepared(sc, times, fields, toxic | fire, None, replan, danger, log), run
    m = fitted if models == "tabpfn" else None
    if models == "tabpfn" and m is None:
        m = fit_models3d(f, backend, seed=seed)
    keep = breathing_mask(plant)
    for t in replan:
        k = int(np.searchsorted(times, t + AHEAD, side="right"))
        if models == "perfect":
            danger.append((toxic | fire)[int(t) : k].max(axis=0).astype(float))
            log.append({"t": float(t), "locate_s": 0.0, "predict_s": 0.0, "source_top": [],
                        "source_error_m": 0.0, "rate_median": sc.rate,
                        "rate_true": sc.rate})  # fmt: skip
            continue
        now = readings.times <= t + 1e-9
        cur = dataclasses.replace(
            readings, times=readings.times[now], values=readings.values[:, now]
        )
        t0 = time.perf_counter()
        est = m.heights.attach(m.locator.locate(f, cur), cur)
        t1 = time.perf_counter()
        from tabpfn_spatial.hazard.plant3d_adapter import source_hypotheses

        src, w, _ = source_hypotheses(plant, m.grid, est)
        rate = float(np.interp(0.5, est.rate_levels, est.rate)) if est.rate is not None else 1.0
        hv = m.hazard.predict(m.grid, src, w, cur.wind, rate, FORECAST, keep=keep)
        t2 = time.perf_counter()
        p = 1 - (1 - hv.p_toxic) * (1 - hv.p_fire)  # (T, z, y, x)
        danger.append(to_cells(f, p.max(axis=(0, 1)), plant.grid.cell))
        iy, ix = est.top_cells(1)[0]
        z50 = float(np.interp(0.5, est.height_levels, est.height))
        top = [{"xy": f.centre(int(a), int(b)).tolist(), "p": float(est.prob[a, b])}
               for a, b in est.top_cells(3)]  # fmt: skip
        err = np.linalg.norm([*(f.centre(int(iy), int(ix)) - sc.source), z50 - leak.xyz[2]])
        log.append({"t": float(t), "locate_s": t1 - t0, "predict_s": t2 - t1, "source_top": top,
                    "source_error_m": float(err), "height_median": z50,
                    "height_true": leak.xyz[2], "rate_median": rate,
                    "rate_true": float(sc.rate)})  # fmt: skip
    return Prepared(sc, times, fields, toxic | fire, t_alarm, replan, danger, log), run
