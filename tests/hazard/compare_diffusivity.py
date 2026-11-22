"""Compare fixed and varied training spread on predeclared, held-out simulations.

The five evaluation values and training range are fixed before evaluation. Each
layout has its own simulated training context; leaks, noise and outcomes use
separate seeds from the source and hazard contexts. The same held-out cases and
source estimates are used for all three methods. No real-world validation is
implied. Outputs are per-case aggregates, not training or sensor data.
"""

import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.core.budget import cloud_quote
from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.env import load_env
from tabpfn_spatial.hazard import ForwardSimPredictor, hazard_from_fields
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.sim import Factory, readings, sample_scenarios, simulate
from tabpfn_spatial.source import TabPFNLocator
from tabpfn_spatial.source.features import alarm_time

HELPERS = runpy.run_path(str(Path(__file__).with_name("compare_multiplant.py")))
RecordingLocator = HELPERS["RecordingLocator"]
ReplayLocator = HELPERS["ReplayLocator"]
batched_hazards = HELPERS["batched_hazards"]
TRAIN_RANGE = (0.4, 1.6)  # exploratory simulator parameters, not real-world calibration
TEST_VALUES = (0.2, 0.4, 0.8, 1.6, 3.2)  # first/last are outside the training range
HORIZONS = np.array([0.0, 10.0, 30.0, 60.0, 120.0])
CONTEXT_SEED = 1001
SOURCE_SEED = 12345
TEST_SEED = 20261129


def make_cases(factory, count, seed):
    """Generate paired conditions from held-out leak and sensor-noise streams."""
    if seed in (CONTEXT_SEED, SOURCE_SEED):
        raise ValueError("test seed must differ from both training seeds")
    scenarios = sample_scenarios(factory, count, np.random.default_rng(seed))
    cases = []
    for diffusion in TEST_VALUES:
        for i, scenario in enumerate(scenarios):
            t_obs = float((10, 20, 30)[i % 3])
            obs = readings(
                factory,
                scenario,
                t_obs,
                np.random.default_rng(seed + 10000 + i),
                diffusivity=diffusion,
            )
            times = t_obs + HORIZONS
            fields = simulate(factory, [scenario], times, d=diffusion)[0]
            toxic, _ = hazard_from_fields(factory, fields)
            cases.append(dict(diffusion=diffusion, index=i, obs=obs, times=times, truth=toxic))
    return cases


def make_contexts(factory, locator, backend, scenarios, cells):
    """Make each context with its own observed sources; never use evaluation outcomes."""
    records, tables = {}, {}
    for name, bounds in (("fixed", None), ("varied", TRAIN_RANGE)):
        recorder = RecordingLocator()
        args = dict(
            n_scenarios=scenarios,
            cells_per_scenario=cells,
            seed=CONTEXT_SEED,
            diffusivity_range=bounds,
        )
        training_rows(factory, locator=recorder, **args)
        records[name] = (recorder, args)
    all_obs = records["fixed"][0].obs + records["varied"][0].obs
    estimates = locator.locate_many(factory, all_obs)
    for j, name in enumerate(("fixed", "varied")):
        recorder, args = records[name]
        replay = ReplayLocator(recorder.obs, estimates[j * scenarios : (j + 1) * scenarios])
        x, y = training_rows(factory, locator=replay, **args)
        if replay.index != scenarios:
            raise ValueError("incomplete source context replay")
        tables[name] = make_regressor(backend).fit(x, y)
    return tables


def score(factory, case, hazard):
    """Calculate case Brier and toxic counts at a fixed probability threshold."""
    floor = ~factory.walls
    p = hazard.p_toxic[:, floor].ravel()
    y = case["truth"][:, floor].ravel().astype(bool)
    pred = p >= 0.5
    metrics = binary_metrics(p, y)
    return dict(
        brier=metrics["brier"],
        ece=metrics["ece"],
        auc=metrics["auc"],
        tp=int((pred & y).sum()),
        fp=int((pred & ~y).sum()),
        positive=int(y.sum()),
        negative=int((~y).sum()),
    )


def quote(layouts, count, train_scenarios, cells, batch):
    """Print a conservative shape-only cost estimate before any inference."""
    load_env()
    total = 0
    for name, factory in layouts:
        source_cols = 2 * len(factory.sensors) + 3
        source = 2 * (
            cloud_quote(8400, 2 * train_scenarios, source_cols)
            + cloud_quote(8400, len(TEST_VALUES) * count, source_cols)
        )
        query = batch * int((~factory.walls).sum()) * 3 * len(HORIZONS)
        calls = int(np.ceil(len(TEST_VALUES) * count / batch))
        hazard = 2 * calls * cloud_quote(train_scenarios * cells * 4, query, 7)
        print(name, "source upper", source, "hazard upper", hazard, "query rows", query)
        total += source + hazard
    print("quoted upper total", total, "credits; actual charges may differ")


def main():
    """Run paired cases and save per-case metrics to a new JSONL file."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    p.add_argument("--run-cloud", action="store_true")
    p.add_argument("--quote-only", action="store_true")
    p.add_argument("--episodes", type=int, default=10, help="base leaks per layout")
    p.add_argument("--train-scenarios", type=int, default=100)
    p.add_argument("--cells", type=int, default=20)
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--layouts", nargs="+", default=["default", "random7"])
    p.add_argument("--output", type=Path, default=Path("outputs/diffusivity-study.jsonl"))
    args = p.parse_args()
    if min(args.episodes, args.train_scenarios, args.cells, args.batch) < 1:
        p.error("counts must be positive")
    layouts = [
        (
            name,
            Factory.default()
            if name == "default"
            else Factory.random(np.random.default_rng(int(name.removeprefix("random")))),
        )
        for name in args.layouts
    ]
    if args.quote_only:
        quote(layouts, args.episodes, args.train_scenarios, args.cells, args.batch)
        return
    if args.backend == "cloud" and not args.run_cloud:
        p.error("cloud calls require --run-cloud")
    if args.backend == "baseline":
        import tabpfn_spatial.core.models as models

        models._CHUNK = 32
    else:
        load_env()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for name, factory in layouts:
            start = time.perf_counter()
            locator = TabPFNLocator(backend=args.backend, seed=SOURCE_SEED).fit(factory)
            contexts = make_contexts(
                factory, locator, args.backend, args.train_scenarios, args.cells
            )
            cases = make_cases(factory, args.episodes, TEST_SEED)
            estimates = locator.locate_many(factory, [c["obs"] for c in cases])
            print(
                name,
                "prepared",
                len(cases),
                "cases in",
                round(time.perf_counter() - start, 2),
                "s",
                flush=True,
            )
            for first in range(0, len(cases), args.batch):
                group = cases[first : first + args.batch]
                sources = estimates[first : first + args.batch]
                requests = [(c["obs"], s, c["times"]) for c, s in zip(group, sources, strict=True)]
                predictions = {
                    key: batched_hazards(factory, requests, model)
                    for key, model in contexts.items()
                }
                for j, (case, source) in enumerate(zip(group, sources, strict=True)):
                    forward = ForwardSimPredictor().predict(
                        factory, case["obs"], source, case["times"]
                    )
                    row = dict(
                        layout=name,
                        episode=case["index"],
                        diffusion=case["diffusion"],
                        observation_time=case["obs"].t_obs,
                        source_alarmed=alarm_time(case["obs"].times, case["obs"].values)
                        is not None,
                        forward=score(factory, case, forward),
                    )
                    for key in contexts:
                        row[key] = score(factory, case, predictions[key][j])
                    output.write(json.dumps(row, allow_nan=True) + "\n")
                output.flush()
                print(name, "finished", first + len(group), "/", len(cases), flush=True)
            for key, model in contexts.items():
                print(
                    name,
                    key,
                    "hazard quoted credits",
                    getattr(model.guard, "spent", 0) if hasattr(model, "guard") else 0,
                    flush=True,
                )
            for key in ("model", "rate_model"):
                model = getattr(locator, key, None)
                print(
                    name,
                    key,
                    "source quoted credits",
                    getattr(getattr(model, "guard", None), "spent", 0),
                    flush=True,
                )


if __name__ == "__main__":
    main()
