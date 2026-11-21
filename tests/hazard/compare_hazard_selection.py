"""Compare source selections from one cloud hazard batch per simulated episode.

Requires synthetic scenario data from compare_source_maps.py and --run-cloud.
The selected maps use Rayan's locator; the forward reference uses its top cell.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import TOXIC_LIMIT, SensorReadings, SourceEstimate
from tabpfn_spatial.hazard import ForwardSimPredictor, TabPFNHazardPredictor
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.hazard.tabpfn_predictor import _cdf
from tabpfn_spatial.sim import Factory, Scenario, simulate


def main():
    """Benchmark source selections on paired scenarios after explicit cloud opt-in."""
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
    xy = factory.centres().reshape(-1, 2)
    model = TabPFNHazardPredictor(backend="cloud").fit(factory, seed=1001).model
    scores = {key: [] for key in ("forward", "top3", "top10", "adaptive90")}
    predictions = {key: [] for key in scores}
    truths, timings, cap_count = [], [], 0
    baseline = ForwardSimPredictor()
    for i, prob in enumerate(data["rayan_maps"]):
        obs = SensorReadings(
            factory.sensors,
            data["sensor_times"],
            data["sensor_values"][i],
            data["measured_winds"][i],
        )
        scenario = Scenario(data["sources"][i], float(data["rates"][i]), data["true_winds"][i])
        truth = simulate(factory, [scenario], truth_times)[0, 1:] > TOXIC_LIMIT
        flat = prob.ravel()
        order = np.argsort(flat)[::-1][:10]
        desired = int(np.searchsorted(np.cumsum(flat[np.argsort(flat)[::-1]]), 0.9) + 1)
        k90 = min(desired, 10)
        cap_count += int(desired > 10)
        start = time.perf_counter()
        rows = np.concatenate(
            [
                cell_features(factory, xy[index], obs.wind, 1.0, t, xy)
                for index in order
                for t in times
            ]
        )
        dist = model.predict_dist(rows)
        conditionals = (1 - _cdf(dist.values, dist.levels, TOXIC_LIMIT)).reshape(
            10, len(times), factory.ny, factory.nx
        )
        timings.append(time.perf_counter() - start)
        maps = {"forward": baseline.predict(factory, obs, SourceEstimate(prob), times).p_toxic}
        for name, n in (("top3", 3), ("top10", 10), ("adaptive90", k90)):
            weights = flat[order[:n]]
            weights = weights / weights.sum()
            maps[name] = np.tensordot(weights, conditionals[:n], axes=1) * mask
        target = truth[:, mask]
        truths.append(target.ravel())
        for name, prediction in maps.items():
            p = prediction[:, mask]
            predictions[name].append(p.ravel())
            scores[name].append(float(np.mean((p - target) ** 2)))
        print(
            "episode",
            i + 1,
            "brier",
            {name: round(v[-1], 5) for name, v in scores.items()},
            "seconds",
            round(timings[-1], 2),
            flush=True,
        )
    y = np.concatenate(truths)
    for name in scores:
        print(
            name, "pooled toxic", binary_metrics(np.concatenate(predictions[name]), y), flush=True
        )
    for name in ("top10", "adaptive90"):
        delta = np.asarray(scores[name]) - np.asarray(scores["top3"])
        rng = np.random.default_rng(777)
        boot = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(axis=1)
        print(
            name,
            "minus top3 paired mean Brier",
            float(delta.mean()),
            "bootstrap 95% interval",
            np.quantile(boot, [0.025, 0.975]).tolist(),
            flush=True,
        )
    print("adaptive 90% capped at 10 in", cap_count, "episodes")
    print(
        "hazard call seconds min median max",
        float(np.min(timings)),
        float(np.median(timings)),
        float(np.max(timings)),
    )
    print("hazard quoted credits", model.guard.spent, "calls", model.guard.calls, flush=True)


if __name__ == "__main__":
    main()
