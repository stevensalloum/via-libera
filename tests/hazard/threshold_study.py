"""Choose and independently test hazard thresholds on alarm-triggered simulations.

Run validation, freeze its selected thresholds, then collect an independent test
split. Thresholds classify toxic cell/time probabilities; they do not change the
predictor or planner. The source alarm and forecast use the same noisy readings.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from tabpfn_spatial.core.budget import cloud_quote
from tabpfn_spatial.env import load_env
from tabpfn_spatial.hazard import ForwardSimPredictor, hazard_from_fields
from tabpfn_spatial.sim import Factory, first_alarm, sample_scenarios, sensor_history, simulate
from tabpfn_spatial.source import TabPFNLocator
from tabpfn_spatial.source.features import alarm_time

PREVIOUS = runpy.run_path(str(Path(__file__).with_name("compare_diffusivity.py")))
make_contexts = PREVIOUS["make_contexts"]
batched_hazards = PREVIOUS["batched_hazards"]
VALUES = PREVIOUS["TEST_VALUES"]
HORIZONS = PREVIOUS["HORIZONS"]
THRESHOLDS = np.arange(101, dtype=float) / 100
SPLIT_SEEDS = {"validation": 20261201, "test": 20261202}
TRAIN_SEEDS = (1001, 12345, 20261129)
MODELS = ("forward", "fixed", "varied")
MAX_ALARM = 120.0


def study_settings(args):
    """Return all parameters that must match between validation and test."""
    return {
        key: getattr(args, key)
        for key in ("backend", "layouts", "episodes", "train_scenarios", "cells", "batch")
    }


def threshold_counts(prob, truth, thresholds=THRESHOLDS):
    """Calculate binary counts for all thresholds, including exact ties; O(n log k)."""
    p, y = np.asarray(prob).ravel(), np.asarray(truth, dtype=bool).ravel()
    thresholds = np.asarray(thresholds, float)
    if len(p) != len(y) or not np.all(np.isfinite(p)) or np.any((p < -1e-12) | (p > 1 + 1e-12)):
        raise ValueError("probabilities and labels must be matching and valid")
    p = np.clip(p, 0.0, 1.0)
    if not len(thresholds) or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be nonempty and increasing")
    bins = np.searchsorted(thresholds, p, side="right")

    def cumulative(mask):
        hist = np.bincount(bins[mask], minlength=len(thresholds) + 1)
        return np.cumsum(hist[::-1])[::-1][1:].astype(int).tolist()

    return dict(
        tp=cumulative(y), fp=cumulative(~y), positive=int(y.sum()), negative=int((~y).sum())
    )


def choose_threshold(rows, thresholds, target):
    """Select the highest validation threshold with pooled recall at least target."""
    if not 0 < target <= 1 or not rows:
        raise ValueError("target and validation rows must be valid")
    positives = sum(row["positive"] for row in rows)
    if not positives:
        raise ValueError("validation needs positive labels")
    tp = np.sum([row["tp"] for row in rows], axis=0)
    valid = np.flatnonzero(tp / positives >= target)
    if not len(valid):
        raise ValueError("no threshold achieves target on validation data")
    return float(thresholds[int(valid[-1])])


def make_cases(factory, n, seed):
    """Make a fixed sensor history and forecast at its first noisy alarm, if any."""
    if seed in TRAIN_SEEDS:
        raise ValueError("split seed overlaps a previous training or test seed")
    scenarios = sample_scenarios(factory, n, np.random.default_rng(seed))
    cases = []
    for diffusion in VALUES:
        for i, sc in enumerate(scenarios):
            history = sensor_history(
                factory,
                sc,
                MAX_ALARM,
                np.random.default_rng(seed + 10000 + i),
                diffusivity=diffusion,
            )
            full = history(MAX_ALARM)
            noisy_alarm = alarm_time(full.times, full.values)
            fields = simulate(factory, [sc], full.times, d=diffusion)[0]
            clean_alarm = first_alarm(factory, full.times, fields)
            prior_at_clean = (
                clean_alarm is not None
                and alarm_time(history(clean_alarm).times, history(clean_alarm).values) is None
            )
            case = dict(
                index=i,
                diffusion=diffusion,
                alarm=noisy_alarm,
                clean_alarm=clean_alarm,
                prior_at_clean=prior_at_clean,
            )
            if noisy_alarm is not None:
                obs = history(noisy_alarm)
                times = noisy_alarm + HORIZONS
                truth = hazard_from_fields(factory, simulate(factory, [sc], times, d=diffusion)[0])[
                    0
                ]
                case.update(obs=obs, times=times, truth=truth)
            cases.append(case)
    return cases


def score(factory, case, hazard):
    """Counts over open cells, separately for each forecast horizon."""
    floor = ~factory.walls
    return [
        threshold_counts(hazard.p_toxic[i, floor], case["truth"][i, floor])
        for i in range(len(HORIZONS))
    ]


def merge_counts(parts):
    """Combine per-horizon confusion counts without retaining predictions."""
    return {
        "tp": np.sum([p["tp"] for p in parts], axis=0).tolist(),
        "fp": np.sum([p["fp"] for p in parts], axis=0).tolist(),
        "positive": sum(p["positive"] for p in parts),
        "negative": sum(p["negative"] for p in parts),
    }


def collect(args, layouts):
    """Simulate an isolated split and record per-case threshold counts."""
    if args.backend == "cloud" and not args.run_cloud:
        raise ValueError("cloud calls require --run-cloud")
    if args.backend == "baseline":
        import tabpfn_spatial.core.models as models

        models._CHUNK = 32
    else:
        load_env()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "split": args.split,
        "seed": SPLIT_SEEDS[args.split],
        "settings": study_settings(args),
    }
    completed = set()
    if args.resume:
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("resume settings do not match the original run")
        old_rows = [json.loads(line) for line in args.output.read_text().splitlines() if line]
        for row in old_rows:
            if row["split"] != args.split:
                raise ValueError("cannot resume a different split")
            key = (row["layout"], row["episode"], row["diffusion"])
            if key in completed:
                raise ValueError("duplicate case in existing output")
            completed.add(key)
        print("resuming", len(completed), "completed case variations", flush=True)
    else:
        if args.output.exists() or manifest.exists():
            raise FileExistsError("output or its metadata already exists")
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with args.output.open("a" if args.resume else "x", encoding="utf-8") as output:
        for label, factory in layouts:
            planned = {(label, i, d) for i in range(args.episodes) for d in VALUES}
            if planned <= completed:
                print(label, args.split, "already complete", flush=True)
                continue
            locator = TabPFNLocator(backend=args.backend, seed=12345).fit(factory)
            contexts = make_contexts(
                factory, locator, args.backend, args.train_scenarios, args.cells
            )
            cases = make_cases(factory, args.episodes, SPLIT_SEEDS[args.split])
            cases = [c for c in cases if (label, c["index"], c["diffusion"]) not in completed]
            triggered = [c for c in cases if c["alarm"] is not None]
            sources = locator.locate_many(factory, [c["obs"] for c in triggered])
            print(label, args.split, "detected", len(triggered), "/", len(cases), flush=True)
            for case in cases:
                if case["alarm"] is None:
                    output.write(
                        json.dumps(
                            dict(
                                layout=label,
                                split=args.split,
                                episode=case["index"],
                                diffusion=case["diffusion"],
                                alarm=None,
                                clean_alarm=case["clean_alarm"],
                                prior_at_clean=case["prior_at_clean"],
                            )
                        )
                        + "\n"
                    )
            for first in range(0, len(triggered), args.batch):
                group, estimates = (
                    triggered[first : first + args.batch],
                    sources[first : first + args.batch],
                )
                requests = [
                    (c["obs"], s, c["times"]) for c, s in zip(group, estimates, strict=True)
                ]
                predictions = {
                    k: batched_hazards(factory, requests, model) for k, model in contexts.items()
                }
                for j, (case, source) in enumerate(zip(group, estimates, strict=True)):
                    maps = {k: predictions[k][j] for k in contexts}
                    maps["forward"] = ForwardSimPredictor().predict(
                        factory, case["obs"], source, case["times"]
                    )
                    row = dict(
                        layout=label,
                        split=args.split,
                        episode=case["index"],
                        diffusion=case["diffusion"],
                        alarm=case["alarm"],
                        clean_alarm=case["clean_alarm"],
                        prior_at_clean=case["prior_at_clean"],
                    )
                    for key in MODELS:
                        parts = score(factory, case, maps[key])
                        row[key] = merge_counts(parts)
                        row[key]["horizons"] = parts
                    output.write(json.dumps(row) + "\n")
                output.flush()
                print(
                    label,
                    args.split,
                    "finished",
                    first + len(group),
                    "/",
                    len(triggered),
                    flush=True,
                )
            for key, model in contexts.items():
                print(
                    label,
                    key,
                    "quoted credits",
                    getattr(getattr(model, "guard", None), "spent", 0),
                    flush=True,
                )
            for key in ("model", "rate_model"):
                print(
                    label,
                    key,
                    "quoted credits",
                    getattr(getattr(getattr(locator, key, None), "guard", None), "spent", 0),
                    flush=True,
                )


def quote(layouts, args):
    """Print a conservative shape-only quote for one split before any calls."""
    load_env()
    total = 0
    for name, factory in layouts:
        source_cols = 2 * len(factory.sensors) + 3
        num_cases = len(VALUES) * args.episodes
        source = 2 * (
            cloud_quote(8400, 2 * args.train_scenarios, source_cols)
            + cloud_quote(8400, num_cases, source_cols)
        )
        query = args.batch * int((~factory.walls).sum()) * 3 * len(HORIZONS)
        calls = int(np.ceil(num_cases / args.batch))
        hazard = 2 * calls * cloud_quote(args.train_scenarios * args.cells * 4, query, 7)
        print(name, "source", source, "hazard upper", hazard)
        total += source + hazard
    print("one split quoted upper", total, "credits; actual charges may differ")


def require_complete(rows, layouts, episodes):
    """Reject missing or duplicated case variations before selecting a threshold."""
    expected = {
        (layout, i, diffusion)
        for layout in layouts
        for i in range(episodes)
        for diffusion in VALUES
    }
    actual = [(r["layout"], r["episode"], r["diffusion"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("validation results must contain each complete case exactly once")


def select(input_path, output_path, target, settings):
    """Freeze thresholds from validation results without opening any test results."""
    data = input_path.read_bytes()
    manifest = input_path.with_suffix(input_path.suffix + ".meta.json")
    expected_metadata = {
        "split": "validation",
        "seed": SPLIT_SEEDS["validation"],
        "settings": settings,
    }
    if json.loads(manifest.read_text()) != expected_metadata:
        raise ValueError("validation settings do not match recorded metadata")
    rows = [json.loads(line) for line in data.splitlines() if line]
    if not rows or any(row["split"] != "validation" for row in rows):
        raise ValueError("selection needs validation rows only")
    require_complete(rows, settings["layouts"], settings["episodes"])
    seen = [row for row in rows if row["alarm"] is not None]
    choices = {
        key: choose_threshold([r[key] for r in seen], THRESHOLDS, target)
        for key in ("fixed", "varied")
    }
    result = dict(
        validation_seed=SPLIT_SEEDS["validation"],
        target=target,
        validation_sha256=hashlib.sha256(data).hexdigest(),
        thresholds=choices,
        settings=settings,
    )
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


def main():
    """Quote, collect, or select thresholds; test collection requires frozen choices."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    p.add_argument("--run-cloud", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume a matching partial output")
    p.add_argument("--quote-only", action="store_true")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--train-scenarios", type=int, default=100)
    p.add_argument("--cells", type=int, default=20)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--layouts", nargs="+", default=["default", "random7"])
    p.add_argument("--output", type=Path)
    p.add_argument("--select", type=Path, help="select from this validation JSONL")
    p.add_argument("--selection", type=Path, help="frozen validation choices, required for test")
    args = p.parse_args()
    if min(args.episodes, args.train_scenarios, args.cells, args.batch) < 1 or not args.output:
        p.error("counts and --output must be provided")
    settings = study_settings(args)
    if args.select:
        select(args.select, args.output, 0.9, settings)
        return
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
        quote(layouts, args)
        return
    if args.split == "test":
        if not args.selection:
            p.error("freeze validation thresholds before the test run")
        selection = json.loads(args.selection.read_text())
        if (
            selection["validation_seed"] != SPLIT_SEEDS["validation"]
            or selection["settings"] != settings
        ):
            p.error("selection seed and experiment settings must match the validation run")
    collect(args, layouts)


if __name__ == "__main__":
    main()
