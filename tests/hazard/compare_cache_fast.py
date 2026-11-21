"""Measure TabPFN variant and cache settings on paired simulated emergencies.

Requires generated data from compare_source_maps.py and explicit --run-cloud.
Cloud calls cost credits; quotes are estimates and actual usage must be checked.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import TOXIC_LIMIT, SensorReadings, SourceEstimate
from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.hazard import ForwardSimPredictor, TabPFNHazardPredictor
from tabpfn_spatial.hazard.batching import ChunkedRegressor
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.sim import Factory, Scenario, simulate


def main():
    """Compare two pinned model versions with and without explicit KV caching."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-cloud", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_cloud:
        parser.error("cloud calls spend credits; pass --run-cloud to opt in")
    data = np.load(args.input)
    factory = Factory.default()
    times = np.arange(70.0, 181.0, 10.0)
    mask = ~factory.walls
    x, y = training_rows(factory, seed=1001)
    models, adapters = {}, {}
    configs = [
        (version, mode)
        for version in ("v3.5", "v3.5-fast")
        for mode in ("fit_preprocessors", "fit_with_cache")
    ]
    for version, mode in configs:
        name = f"{version}/{mode}"
        try:
            start = time.perf_counter()
            model = make_regressor(
                "cloud", model_path=f"{version}_default", fit_mode=mode, n_estimators=8
            ).fit(x, y)
            adapters[name] = model
            prediction_model = (
                ChunkedRegressor(model, max_rows=10_000) if mode == "fit_with_cache" else model
            )
            models[name] = TabPFNHazardPredictor(model=prediction_model, top_k=3)
            print(name, "fit seconds", round(time.perf_counter() - start, 2), flush=True)
        except Exception as exc:
            print(name, "fit unavailable", type(exc).__name__, flush=True)
    if not models:
        raise SystemExit("no requested model configuration could be fitted")

    predictions = {name: [] for name in models}
    timings = {name: [] for name in models}
    observations = {name: [] for name in models}
    failed = set()
    truth_rows, forward_rows = [], []
    baseline = ForwardSimPredictor()
    for i in range(5):
        prob = data["rayan_maps"][i]
        obs = SensorReadings(
            factory.sensors,
            data["sensor_times"],
            data["sensor_values"][i],
            data["measured_winds"][i],
        )
        scenario = Scenario(data["sources"][i], float(data["rates"][i]), data["true_winds"][i])
        truth = simulate(factory, [scenario], np.r_[60.0, times])[0, 1:] > TOXIC_LIMIT
        truth_rows.append(truth[:, mask].ravel())
        forward_rows.append(
            baseline.predict(factory, obs, SourceEstimate(prob), times).p_toxic[:, mask].ravel()
        )
        order = [name for name in models if name not in failed]
        if not order:
            break
        order = order[i % len(order) :] + order[: i % len(order)]
        for name in order:
            model = models[name]
            start = time.perf_counter()
            try:
                hazard = model.predict(factory, obs, SourceEstimate(prob), times)
            except Exception as exc:
                print(name, "prediction unavailable", type(exc).__name__, flush=True)
                failed.add(name)
                continue
            timings[name].append(time.perf_counter() - start)
            predictions[name].append(hazard.p_toxic[:, mask].ravel())
            estimator = adapters[name].estimator
            metadata = getattr(estimator, "_last_meta", None)
            if isinstance(metadata, dict):
                execution = metadata.get("execution_mode")
                cache_outcome = metadata.get("cache_outcome")
            else:
                execution = getattr(metadata, "execution_mode", None)
                cache_outcome = getattr(metadata, "cache_outcome", None)
            observations[name].append(
                {
                    "execution_mode": execution,
                    "cache_outcome": cache_outcome,
                }
            )
        print("episode", i + 1, "complete", flush=True)
    if len(truth_rows) < 5:
        raise SystemExit("all model configurations failed before completing five cases")
    truth = np.concatenate(truth_rows)
    print("forward", binary_metrics(np.concatenate(forward_rows), truth), flush=True)
    for name in models:
        if len(predictions[name]) != 5:
            print(
                name,
                "incomplete cases",
                len(predictions[name]),
                "quoted credits",
                adapters[name].guard.spent,
                flush=True,
            )
            continue
        print(
            name,
            "metrics",
            binary_metrics(np.concatenate(predictions[name]), truth),
            "hazard seconds",
            np.round(timings[name], 3).tolist(),
            "metadata",
            observations[name],
            "quoted credits",
            adapters[name].guard.spent,
            flush=True,
        )
    for version in ("v3.5", "v3.5-fast"):
        uncached, cached = f"{version}/fit_preprocessors", f"{version}/fit_with_cache"
        if (
            uncached in predictions
            and cached in predictions
            and all(len(predictions[name]) == 5 for name in (uncached, cached))
        ):
            diff = np.max(
                np.abs(np.concatenate(predictions[uncached]) - np.concatenate(predictions[cached]))
            )
            print(
                version,
                "maximum absolute probability difference cache vs default",
                float(diff),
                flush=True,
            )


if __name__ == "__main__":
    main()
