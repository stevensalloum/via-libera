"""Small, opt-in alarm-to-route latency probe using the existing team models.

Uses one default factory, one fixed person per leak and simulated sensor histories.
The second decision begins after the rounded measured first decision; this is not
a real-time deployment or a moving-person/crowd replay. Cloud calls cost credits.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.env import load_env
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.evac.pipeline_view import models_for
from tabpfn_spatial.hazard import ForwardSimPredictor
from tabpfn_spatial.sim import Factory, sample_people, sample_scenarios, sensor_history
from tabpfn_spatial.source.features import alarm_time

SEED = 20261204
HORIZONS = np.arange(0.0, 121.0, 10.0)


def measure_cycle(factory, obs, positions, locator, predictor, planner, decision):
    """Time the sequential source, hazard and one-person route calls in seconds."""
    start = time.perf_counter()
    source = locator.locate(factory, obs)
    after_source = time.perf_counter()
    hazard = predictor.predict(factory, obs, source, decision + HORIZONS)
    after_hazard = time.perf_counter()
    plan = planner.plan(factory, positions, hazard, decision)
    after_route = time.perf_counter()
    return (
        dict(
            source_s=after_source - start,
            hazard_s=after_hazard - after_source,
            route_s=after_route - after_hazard,
            total_s=after_route - start,
            exit_index=int(plan.routes[0].exit_index),
        ),
        source,
        hazard,
    )


def next_decision(t: float, elapsed: float) -> float:
    """Next one-second simulation step after the measured route lands."""
    return t + max(1, math.ceil(elapsed))


def run(backend: str, output: Path, count: int) -> None:
    """Collect measurements for alarmed scenarios; save after every decision."""
    if output.exists():
        raise FileExistsError(output)
    load_env()
    if backend == "baseline":
        import tabpfn_spatial.core.models as models

        models._CHUNK = 32
    factory = Factory.default()
    rng = np.random.default_rng(SEED)
    scenarios = sample_scenarios(factory, count, rng)
    starts = sample_people(factory, count, rng)
    locator, predictor = models_for("team", scenarios[0], factory, backend)
    planner = RiskAwarePlanner()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for index, (scenario, position) in enumerate(zip(scenarios, starts, strict=True)):
            history = sensor_history(
                factory, scenario, 120.0, np.random.default_rng(SEED + 100 + index)
            )
            full = history(120.0)
            alarm = alarm_time(full.times, full.values)
            if alarm is None:
                record = dict(case=index, alarm=None)
                print(json.dumps(record), file=stream, flush=True)
                print("case", index, "no alarm", flush=True)
                continue
            decision = alarm
            for cycle in range(2):
                if decision > 120.0:
                    print("case", index, "next decision beyond sensor history", flush=True)
                    break
                obs = history(decision)
                measured, source, _ = measure_cycle(
                    factory, obs, position[None], locator, predictor, planner, decision
                )
                start = time.perf_counter()
                forward = ForwardSimPredictor().predict(factory, obs, source, decision + HORIZONS)
                forward_prediction_s = time.perf_counter() - start
                start = time.perf_counter()
                forward_plan = planner.plan(factory, position[None], forward, decision)
                forward_route_s = time.perf_counter() - start
                iy, ix = source.top_cells(1)[0]
                error = np.linalg.norm(factory.centre(int(iy), int(ix)) - scenario.source)
                record = dict(
                    case=index,
                    cycle=cycle + 1,
                    alarm=alarm,
                    decision_time=decision,
                    last_sample=float(obs.t_obs),
                    samples=len(obs.times),
                    source_error_m=float(error),
                    true_rate=float(scenario.rate),
                    rate_median=float(np.interp(0.5, source.rate_levels, source.rate)),
                    forward_hazard_s=forward_prediction_s,
                    forward_route_s=forward_route_s,
                    forward_exit=int(forward_plan.routes[0].exit_index),
                    **measured,
                )
                print(json.dumps(record), file=stream, flush=True)
                print(json.dumps(record), flush=True)
                decision = next_decision(decision, measured["total_s"])
            if locator._pending is not None:
                # Complete the last background call before the next leak.
                locator._pending[1].result()


def main() -> None:
    """Run only when explicitly invoked; the default is the offline stand-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    parser.add_argument("--run-cloud", action="store_true")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "cloud" and not args.run_cloud:
        parser.error("cloud predictions spend credits; pass --run-cloud")
    if args.count < 1:
        parser.error("count must be positive")
    run(args.backend, args.output, args.count)


if __name__ == "__main__":
    main()
