"""Run Steven's simulated live loop with team models and time planner calls.

One person walks, while two source/hazard cycles run; this is not live hardware.
The alarm and model readings use the same noisy sensor history.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.env import load_env
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.evac.live import run_pipeline
from tabpfn_spatial.evac.pipeline_view import models_for
from tabpfn_spatial.sim import Factory, sample_scenarios
from tabpfn_spatial.source.features import alarm_time

SEED = 20261204


class TimedPlanner(RiskAwarePlanner):
    """Record every real planner call made by the simulated loop."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def plan(self, factory, people, hazard, t_start):
        """Time the original planner without changing its algorithm."""
        started = time.perf_counter()
        result = super().plan(factory, people, hazard, t_start)
        self.calls.append(dict(t=float(t_start), seconds=time.perf_counter() - started))
        return result


def run(backend: str, output: Path) -> None:
    """Record alarm, landed cycles and planner overhead for one simulation."""
    if output.exists():
        raise FileExistsError(output)
    load_env()
    if backend == "baseline":
        import tabpfn_spatial.core.models as models

        models._CHUNK = 32
    factory = Factory.default()
    scenario = sample_scenarios(factory, 1, np.random.default_rng(SEED))[0]
    cells = factory.centres()[~factory.walls]
    nearest = np.linalg.norm(cells[:, None, :] - factory.exits[None, :, :], axis=2).min(axis=1)
    start = cells[nearest.argmax()]
    locator, predictor = models_for("team", scenario, factory, backend)
    planner = TimedPlanner()
    landed = []
    observations = []
    for frame in run_pipeline(
        factory,
        scenario,
        start,
        locator,
        predictor,
        planner=planner,
        max_runs=2,
        t_max=120.0,
        rng=np.random.default_rng(SEED + 100),
    ):
        if frame.running is not None and len(observations) <= frame.running.index:
            observations.append(
                dict(
                    cycle=frame.running.index + 1,
                    system_alarm=frame.t_alarm,
                    noisy_alarm=alarm_time(frame.readings.times, frame.readings.values),
                    last_reading=float(frame.readings.t_obs),
                    samples=len(frame.readings.times),
                )
            )
        if len(frame.cycles) > len(landed):
            cycle = frame.cycles[-1]
            walker = frame.walkers["live"]
            result = dict(
                cycle=cycle.index + 1,
                start=cycle.t_start,
                lands=cycle.t_land,
                source_s=cycle.locate_s,
                hazard_s=cycle.predict_s,
                source_error_m=cycle.source_error_m,
                rate=cycle.rate_est,
                exit_index=walker.exit_index,
                position=walker.pos.tolist(),
            )
            landed.append(result)
            print(json.dumps(result), flush=True)
    if locator._pending is not None:
        locator._pending[1].result()
    result = dict(
        seed=SEED,
        backend=backend,
        alarm=frame.t_alarm,
        start=start.tolist(),
        true_rate=scenario.rate,
        observations=observations,
        landed=landed,
        planner_calls=planner.calls,
        live_out_at=frame.walkers["live"].out_at,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", output, flush=True)


def main() -> None:
    """Require an explicit flag for credit-spending cloud calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    parser.add_argument("--run-cloud", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "cloud" and not args.run_cloud:
        parser.error("cloud predictions spend credits; pass --run-cloud")
    run(args.backend, args.output)


if __name__ == "__main__":
    main()
