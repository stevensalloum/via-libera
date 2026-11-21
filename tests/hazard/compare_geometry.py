"""Compare original versus optional path features on the same simulated cases.

Requires generated data from compare_source_maps.py and explicit --run-cloud.
These tests use the same simplified simulator for context and evaluation.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import HazardMap, SensorReadings, SourceEstimate
from tabpfn_spatial.evac import RiskAwarePlanner, route_exposed
from tabpfn_spatial.hazard import ForwardSimPredictor, TabPFNHazardPredictor, hazard_from_fields
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.sim import Factory, Scenario, simulate


def _exposed(factory, hazard, people, truth_times, truth):
    """Count people whose planned path reaches a truly hazardous cell."""
    plan = RiskAwarePlanner().plan(factory, people, hazard, 60.0)
    return sum(route_exposed(factory, route, truth_times, truth, 60.0) for route in plan.routes)


def _interval(values):
    """Exploratory paired episode-bootstrap interval for a mean difference."""
    values = np.asarray(values)
    rng = np.random.default_rng(777)
    means = values[rng.integers(len(values), size=(5000, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    """Run the paired comparison; two cloud predictions per episode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-cloud", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_cloud:
        parser.error("cloud calls spend credits; pass --run-cloud to opt in")
    data = np.load(args.input)
    factory = Factory.default()
    times = np.arange(70.0, 181.0, 10.0)
    truth_times = np.r_[60.0, times]
    mask = ~factory.walls
    models = {"forward": ForwardSimPredictor()}
    for name, geometry in (("original", False), ("path_features", True)):
        start = time.perf_counter()
        models[name] = TabPFNHazardPredictor(
            backend="cloud", top_k=3, include_geometry=geometry
        ).fit(factory, seed=1001)
        print(name, "context+fit seconds", round(time.perf_counter() - start, 2), flush=True)
    metrics = {name: [] for name in models}
    pooled = {name: [] for name in models}
    harm_briers = {name: [] for name in models}
    exposures = {name: [] for name in models}
    timing = {name: [] for name in models}
    truth_all, standard_exposures = [], []
    for i, prob in enumerate(data["rayan_maps"]):
        obs = SensorReadings(
            factory.sensors,
            data["sensor_times"],
            data["sensor_values"][i],
            data["measured_winds"][i],
        )
        scenario = Scenario(data["sources"][i], float(data["rates"][i]), data["true_winds"][i])
        toxic, fire = hazard_from_fields(factory, simulate(factory, [scenario], truth_times)[0])
        truth = toxic | fire
        target = toxic[1:, :, :][:, mask]
        truth_all.append(target.ravel())
        empty = np.zeros((len(times), factory.ny, factory.nx))
        standard = HazardMap(times, empty, empty)
        standard_exposures.append(
            _exposed(factory, standard, data["people"][i], truth_times, truth)
        )
        for name, model in models.items():
            start = time.perf_counter()
            hazard = model.predict(factory, obs, SourceEstimate(prob), times)
            timing[name].append(time.perf_counter() - start)
            p = hazard.p_toxic[:, mask]
            pooled[name].append(p.ravel())
            metrics[name].append(float(np.mean((p - target) ** 2)))
            harm = 1 - (1 - hazard.p_toxic) * (1 - hazard.p_fire)
            harm_briers[name].append(
                float(np.mean((harm[:, mask] - truth[1:, :, :][:, mask]) ** 2))
            )
            exposures[name].append(_exposed(factory, hazard, data["people"][i], truth_times, truth))
        print(
            "episode",
            i + 1,
            "toxic Brier",
            {key: round(val[-1], 5) for key, val in metrics.items()},
            flush=True,
        )
    y = np.concatenate(truth_all)
    for name in models:
        print(
            name,
            "toxic",
            binary_metrics(np.concatenate(pooled[name]), y),
            "harm Brier",
            float(np.mean(harm_briers[name])),
            "mean exposed",
            float(np.mean(exposures[name])),
            "hazard secs median/p95",
            np.quantile(timing[name], [0.5, 0.95]).tolist(),
            flush=True,
        )
    delta = np.asarray(metrics["path_features"]) - np.asarray(metrics["original"])
    print(
        "path minus original paired mean Brier",
        float(delta.mean()),
        "bootstrap 95% interval",
        _interval(delta),
        flush=True,
    )
    print("mean exposed standard plan", float(np.mean(standard_exposures)))
    for name in ("original", "path_features"):
        guard = models[name].model.guard
        print(name, "quoted credits", guard.spent, "calls", guard.calls, flush=True)


if __name__ == "__main__":
    main()
