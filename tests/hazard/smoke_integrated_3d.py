"""One 3D source-to-hazard smoke case; requires the source-height integration PRs.

Run from the repository root. Pass ``--backend cloud`` to spend TabPFN credits.
This is a functionality check, not a representative validation or safety study.
"""

import argparse
import dataclasses
import json
import time

import numpy as np
from tabpfn_spatial.source.height_locator import HeightLocator
from tabpfn_spatial.source.locator3d import TabPFNLocator3D

from tabpfn_spatial.contracts import TOXIC_LIMIT
from tabpfn_spatial.env import load_env
from tabpfn_spatial.hazard.plant3d_adapter import plant_grid, plant_training_rows, source_hypotheses
from tabpfn_spatial.hazard.volume3d_predictor import ForwardVolumePredictor, TabPFNVolumePredictor
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, simulate
from tabpfn_spatial.sim.plant3d import Plant3D


def main() -> None:
    """Compare one simulated leak using one weighted 3D source hypothesis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    backend = parser.parse_args().backend
    load_env()
    plant = Plant3D(dataclasses.replace(Factory.default(), gas="propane"))
    grid = plant_grid(plant, ignition_radius=1.5)
    pipe = plant.vertical_pipes[0]
    leak = Leak3D((*pipe.xy, 3.06), 1.0, plant.f.gas)
    run = simulate(plant, leak, Ventilation(0.6, 1.0), t_end=40)
    readings = run.readings(np.random.default_rng(899))
    alarm = run.alarm(readings)
    if alarm is None:
        raise RuntimeError("the selected simulated leak did not trigger the alarm")
    mask = readings.times <= alarm
    current = dataclasses.replace(
        readings, times=readings.times[mask], values=readings.values[:, mask]
    )
    locator = TabPFNLocator3D(backend=backend, n_train=8, seed=100, ws=(0, 10)).fit(plant)
    xy_estimate = locator.locate(plant.f, current)
    heights = HeightLocator(backend=backend, n_train=8, seed=111, ws=(0, 10)).fit(plant)
    estimate = heights.attach(xy_estimate, current)
    sources, weights, retained = source_hypotheses(plant, grid, estimate)
    rate = float(np.interp(0.5, estimate.rate_levels, estimate.rate))
    times = np.array([alarm + 5])
    x, y = plant_training_rows(
        plant,
        grid,
        n_scenarios=8,
        cells_per_scenario=80,
        times=np.array([5.0, 10.0, 20.0, 30.0, 40.0]),
        seed=222,
    )
    hazard = TabPFNVolumePredictor(backend=backend, top_k=1).fit_examples(grid, x, y)
    truth = run.volumes_zyx()[int(times[0] / 5)][~grid.solid] > TOXIC_LIMIT
    results = {}
    for name, model in (("forward", ForwardVolumePredictor()), ("hazard", hazard)):
        start = time.perf_counter()
        result = model.predict(grid, sources, weights, current.wind, rate, times)
        elapsed = time.perf_counter() - start
        p = result.p_toxic[0, ~grid.solid]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("invalid hazard probabilities")
        results[name] = {
            "brier": float(np.mean((p - truth) ** 2)),
            "recall_0_5": float(np.mean(p[truth] >= 0.5)) if truth.any() else None,
            "seconds": elapsed,
            "shape": list(result.p_toxic.shape),
        }
    guards = (locator.model, locator.rate_model, heights.model, hazard.model)
    quotes = [getattr(getattr(model, "guard", None), "spent", 0) for model in guards]
    print(
        json.dumps(
            {
                "backend": backend,
                "alarm": alarm,
                "forecast_time": times.tolist(),
                "height_median": float(np.interp(0.5, estimate.height_levels, estimate.height)),
                "true_xyz": leak.xyz,
                "rate_estimate": rate,
                "mass_kept": retained,
                "hazard_top1_mass": hazard.last_retained_mass,
                "train_leaks_per_model": 8,
                "hazard_train_seed": 222,
                "source_train_seed": 100,
                "height_train_seed": 111,
                "eval_sensor_seed": 899,
                "training_rows": len(y),
                "truth_toxic_voxels": int(truth.sum()),
                "results": results,
                "quoted_credits_source_xy_rate_height_hazard": quotes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
