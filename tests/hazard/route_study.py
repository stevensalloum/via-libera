"""Replay paired alarm-time evacuation routes with and without hazard forecasts.

The warning overlays are observational: they never change the planner's input.
This is a single-route 2D simulator study, not a crowd or live-latency test.
"""

import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import LEL, TOXIC_LIMIT, UEL, HazardMap
from tabpfn_spatial.core.budget import cloud_quote
from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.env import load_env
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.evac.live import Walker
from tabpfn_spatial.hazard import ForwardSimPredictor
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.sim import Factory, sample_people, sample_scenarios, sensor_history, simulate
from tabpfn_spatial.source import TabPFNLocator
from tabpfn_spatial.source.features import alarm_time

PREVIOUS = runpy.run_path(str(Path(__file__).with_name("compare_diffusivity.py")))
RecordingLocator = PREVIOUS["RecordingLocator"]
ReplayLocator = PREVIOUS["ReplayLocator"]
batched_hazards = PREVIOUS["batched_hazards"]
DIFFUSIVITIES = (0.2, 0.4, 0.8, 1.6, 3.2)
HORIZONS = np.arange(0.0, 121.0, 10.0)
SEED = 20261203
CONTEXT_SEED = 1001
SOURCE_SEED = 12345
OVERLAYS = (0.19, 0.50)
DELAY = 5  # exploratory assumption, not a measurement


def _follow(factory, walker, planner, hazard, t):
    """Ask Steven's existing planner for a route and assign its waypoints."""
    if hazard is None:
        zero = np.zeros((1, factory.ny, factory.nx))
        hazard = HazardMap(np.array([t], float), zero, zero)
    route = planner.plan(factory, walker.pos[None], hazard, float(t)).routes[0]
    walker.follow((route.cells[:, ::-1] + 0.5) * factory.cell, route.exit_index)


def replay(factory, start, alarm, fields, hazard, delay=0):
    """Replay one route from the alarm; track toxicity, distance and overlay flags.

    The 1-second samples are correlated. Before a delayed forecast arrives, the
    person follows the nearest route. The overlay is never supplied to the planner.
    """
    planner = RiskAwarePlanner()
    walker = Walker(np.asarray(start, float).copy(), planner.speed)
    overlay = {
        f"{p:.2f}": dict(
            safe_s=0, toxic_s=0, true_positive_s=0, false_positive_s=0, false_negative_s=0
        )
        for p in OVERLAYS
    }
    toxic_s = harmful_s = beyond_s = 0
    for t in range(int(alarm), min(len(fields), int(alarm) + 181)):
        if walker.out_at is not None:
            break
        if t == alarm:
            _follow(factory, walker, planner, hazard if delay == 0 else None, t)
        if delay > 0 and t == alarm + delay and hazard is not None:
            _follow(factory, walker, planner, hazard, t)
        iy, ix = factory.cell_of(walker.pos)
        c = float(fields[t, iy, ix])
        toxic = c > TOXIC_LIMIT
        fire = LEL <= c <= UEL and bool(factory.ignition[iy, ix])
        toxic_s += int(toxic)
        harmful_s += int(toxic or fire)
        if hazard is not None and t >= alarm + delay:
            k = int(np.argmin(np.abs(hazard.times - t)))
            beyond_s += int(t > hazard.times[-1])
            p = float(hazard.p_toxic[k, iy, ix])
            for threshold in OVERLAYS:
                row = overlay[f"{threshold:.2f}"]
                flagged = p >= threshold
                row["safe_s"] += int(not toxic)
                row["toxic_s"] += int(toxic)
                row["true_positive_s"] += int(flagged and toxic)
                row["false_positive_s"] += int(flagged and not toxic)
                row["false_negative_s"] += int(not flagged and toxic)
        walker.step(1.0, float(t))
    return dict(
        exposed=harmful_s > 0,
        toxic_s=toxic_s,
        harmful_s=harmful_s,
        walked_m=walker.walked,
        completed=walker.out_at is not None,
        out_at=walker.out_at,
        beyond_horizon_s=beyond_s,
        overlay=overlay,
    )


def make_cases(factory, count, people):
    """Make paired leak variations; the same noisy alarm starts each workflow."""
    scenarios = sample_scenarios(factory, count, np.random.default_rng(SEED))
    cases = []
    for d in DIFFUSIVITIES:
        for i, sc in enumerate(scenarios):
            history = sensor_history(
                factory, sc, 120.0, np.random.default_rng(SEED + 10000 + i), diffusivity=d
            )
            full = history(120.0)
            alarm = alarm_time(full.times, full.values)
            starts = sample_people(factory, people, np.random.default_rng(SEED + 20000 + i))
            case = dict(index=i, diffusion=d, scenario=sc, alarm=alarm, starts=starts)
            if alarm is not None:
                case.update(obs=history(alarm), times=alarm + HORIZONS)
            cases.append(case)
    return cases


def context(factory, locator, backend, scenarios, cells):
    """Use simulated training readings and batched source outputs, with fixed diffusivity."""
    args = dict(n_scenarios=scenarios, cells_per_scenario=cells, seed=CONTEXT_SEED)
    recorder = RecordingLocator()
    training_rows(factory, locator=recorder, **args)
    estimates = locator.locate_many(factory, recorder.obs)
    replay_locator = ReplayLocator(recorder.obs, estimates)
    x, y = training_rows(factory, locator=replay_locator, **args)
    if replay_locator.index != scenarios:
        raise ValueError("incomplete training source replay")
    return make_regressor(backend).fit(x, y)


def map_score(factory, hazard, fields, times):
    """Whole-open-floor Brier and toxicity counts at the candidate 19% threshold."""
    floor = ~factory.walls
    truth = fields[np.asarray(times, dtype=int)][:, floor] > TOXIC_LIMIT
    p = hazard.p_toxic[:, floor]
    score = binary_metrics(p, truth)
    return dict(
        brier=score["brier"],
        tp=int(((p >= 0.19) & truth).sum()),
        fp=int(((p >= 0.19) & ~truth).sum()),
        positive=int(truth.sum()),
        negative=int((~truth).sum()),
    )


def evaluate_case(factory, case, source, fixed):
    """Pair all route policies with the same leak, starts, readings and true field."""
    scenario, alarm, d = case["scenario"], case["alarm"], case["diffusion"]
    fields = simulate(factory, [scenario], np.arange(0.0, alarm + 182.0), d=d)[0]
    forward = ForwardSimPredictor().predict(factory, case["obs"], source, case["times"])
    methods = (
        ("nearest", None, 0),
        ("forward", forward, 0),
        ("forward_delay5", forward, DELAY),
        ("fixed", fixed, 0),
        ("fixed_delay5", fixed, DELAY),
    )
    return dict(
        map_fixed=map_score(factory, fixed, fields, case["times"]),
        map_forward=map_score(factory, forward, fields, case["times"]),
        routes={
            name: [replay(factory, start, alarm, fields, hz, delay) for start in case["starts"]]
            for name, hz, delay in methods
        },
    )


def settings(args):
    """Parameters that must match when resuming an interrupted collection."""
    return {
        k: getattr(args, k)
        for k in ("backend", "layouts", "episodes", "people", "train_scenarios", "cells", "batch")
    }


def collect(args, layouts):
    """Write per-case outcome aggregates, preserving completed work on resume."""
    if args.backend == "cloud" and not args.run_cloud:
        raise ValueError("cloud calls require --run-cloud")
    if args.backend == "baseline":
        import tabpfn_spatial.core.models as models

        models._CHUNK = 32
    else:
        load_env()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = args.output.with_suffix(args.output.suffix + ".meta.json")
    manifest = {"seed": SEED, "settings": settings(args)}
    done = set()
    if args.resume:
        if json.loads(meta.read_text()) != manifest:
            raise ValueError("resume settings do not match")
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            key = (row["layout"], row["episode"], row["diffusion"])
            if key in done:
                raise ValueError("duplicate case")
            done.add(key)
        print("resuming", len(done), "completed cases", flush=True)
    else:
        if args.output.exists() or meta.exists():
            raise FileExistsError("output or metadata already exists")
        meta.write_text(json.dumps(manifest, indent=2) + "\n")
    with args.output.open("a" if args.resume else "x", encoding="utf-8") as output:
        for label, factory in layouts:
            planned = {(label, i, d) for i in range(args.episodes) for d in DIFFUSIVITIES}
            if planned <= done:
                print(label, "already complete", flush=True)
                continue
            started = time.perf_counter()
            locator = TabPFNLocator(backend=args.backend, seed=SOURCE_SEED).fit(factory)
            model = context(factory, locator, args.backend, args.train_scenarios, args.cells)
            cases = [
                c
                for c in make_cases(factory, args.episodes, args.people)
                if (label, c["index"], c["diffusion"]) not in done
            ]
            detected = [c for c in cases if c["alarm"] is not None]
            estimates = locator.locate_many(factory, [c["obs"] for c in detected])
            print(
                label,
                "detected",
                len(detected),
                "/",
                len(cases),
                "setup_s",
                round(time.perf_counter() - started, 2),
                flush=True,
            )
            for case in cases:
                if case["alarm"] is None:
                    output.write(
                        json.dumps(
                            dict(
                                layout=label,
                                episode=case["index"],
                                diffusion=case["diffusion"],
                                alarm=None,
                            )
                        )
                        + "\n"
                    )
            output.flush()
            for first in range(0, len(detected), args.batch):
                group = detected[first : first + args.batch]
                sources = estimates[first : first + args.batch]
                requests = [(c["obs"], s, c["times"]) for c, s in zip(group, sources, strict=True)]
                hazards = batched_hazards(factory, requests, model)
                for case, source, hazard in zip(group, sources, hazards, strict=True):
                    row = dict(
                        layout=label,
                        episode=case["index"],
                        diffusion=case["diffusion"],
                        alarm=case["alarm"],
                    )
                    row.update(evaluate_case(factory, case, source, hazard))
                    output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                print(label, "completed", first + len(group), "/", len(detected), flush=True)
            for key, m in (
                ("hazard", model),
                ("source", locator.model),
                ("source_rate", locator.rate_model),
            ):
                print(
                    label,
                    key,
                    "quoted credits",
                    getattr(getattr(m, "guard", None), "spent", 0),
                    flush=True,
                )


def quote(args, layouts):
    """Quote an upper bound for source and hazard cloud calls from array shapes."""
    load_env()
    total = 0
    for label, f in layouts:
        cases = args.episodes * len(DIFFUSIVITIES)
        cols = 2 * len(f.sensors) + 3
        source = 2 * (
            cloud_quote(8400, args.train_scenarios, cols) + cloud_quote(8400, cases, cols)
        )
        queries = args.batch * int((~f.walls).sum()) * 3 * len(HORIZONS)
        hazard = int(np.ceil(cases / args.batch)) * cloud_quote(
            args.train_scenarios * args.cells * 4, queries, 7
        )
        print(label, "source upper", source, "hazard upper", hazard, "rows", queries)
        total += source + hazard
    print("quoted upper total", total, "credits; actual charges may differ")


def main():
    """Run the paired route study or obtain a free shape-only cost quote."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    p.add_argument("--run-cloud", action="store_true")
    p.add_argument("--quote-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--people", type=int, default=8)
    p.add_argument("--train-scenarios", type=int, default=100)
    p.add_argument("--cells", type=int, default=20)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--layouts", nargs="+", default=["default", "random7"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.episodes, args.people, args.train_scenarios, args.cells, args.batch) < 1:
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
        quote(args, layouts)
        return
    collect(args, layouts)


if __name__ == "__main__":
    main()
