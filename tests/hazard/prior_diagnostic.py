"""Extend the original pre-alarm snapshots to study a possible later source update.

The original run recorded only snapshots. The new future sensor noise is a
reproducible hypothetical continuation, not a reconstruction of actual readings.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.budget import cloud_quote
from tabpfn_spatial.env import load_env
from tabpfn_spatial.sim import Factory, sample_scenarios, sensor_values, simulate
from tabpfn_spatial.source import TabPFNLocator
from tabpfn_spatial.source.features import alarm_time

PREVIOUS = runpy.run_path(str(Path(__file__).with_name("compare_diffusivity.py")))


def continue_readings(factory, scenario, case, last_time, seed):
    """Preserve original readings and add newly sampled future noise; O(time * grid)."""
    old = case["obs"]
    future = np.arange(old.t_obs + 5.0, last_time + 0.1, 5.0)
    fields = simulate(factory, [scenario], future, d=case["diffusion"])[0]
    clean = sensor_values(factory, fields).T
    rng = np.random.default_rng(seed)
    values = np.maximum(
        clean * (1 + 0.1 * rng.standard_normal(clean.shape))
        + 0.001 * rng.standard_normal(clean.shape),
        0.0,
    )
    return SensorReadings(
        old.positions,
        np.concatenate([old.times, future]),
        np.concatenate([old.values, values], axis=1),
        old.wind,
    )


def truncate(obs, t):
    """Keep the historical measurements up to and including t."""
    count = int(np.searchsorted(obs.times, t, side="right"))
    return SensorReadings(obs.positions, obs.times[:count], obs.values[:, :count], obs.wind)


def error(factory, probability, scenario):
    """Distance from the highest-probability cell to the simulated leak."""
    iy, ix = SourceEstimate(probability).top_cells(1)[0]
    return float(np.linalg.norm(factory.centre(int(iy), int(ix)) - scenario.source))


def run(backend, output):
    """Evaluate source predictions at the future first alarm of original prior cases."""
    if backend == "cloud":
        load_env()
    elif backend == "baseline":
        import tabpfn_spatial.core.models as models

        models._CHUNK = 32
    layouts = [
        ("default", Factory.default()),
        ("random7", Factory.random(np.random.default_rng(7))),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as dest:
        for label, factory in layouts:
            locator = TabPFNLocator(backend=backend, seed=12345).fit(factory)
            cases = PREVIOUS["make_cases"](factory, 20, PREVIOUS["TEST_SEED"])
            scenarios = sample_scenarios(factory, 20, np.random.default_rng(PREVIOUS["TEST_SEED"]))
            pending = []
            for case in cases:
                if alarm_time(case["obs"].times, case["obs"].values) is not None:
                    continue
                sc = scenarios[case["index"]]
                extended = continue_readings(factory, sc, case, 120.0, 307000 + case["index"])
                alarm = alarm_time(extended.times, extended.values)
                pending.append(
                    (case, sc, alarm, truncate(extended, alarm) if alarm is not None else None)
                )
            detected = [r for r in pending if r[2] is not None]
            estimates = locator.locate_many(factory, [r[3] for r in detected]) if detected else []
            by_id = {id(r[0]): est for r, est in zip(detected, estimates, strict=True)}
            for case, sc, alarm, _ in pending:
                row = dict(
                    layout=label,
                    episode=case["index"],
                    diffusion=case["diffusion"],
                    observation_time=case["obs"].t_obs,
                    later_alarm=alarm,
                    prior_error_m=error(factory, factory.leak_prior, sc),
                )
                if alarm is not None:
                    row["updated_error_m"] = error(factory, by_id[id(case)].prob, sc)
                dest.write(json.dumps(row) + "\n")
            dest.flush()
            print(
                label,
                "prior snapshots",
                len(pending),
                "later alarm",
                len(detected),
                "source quoted credits",
                sum(
                    getattr(getattr(getattr(locator, name, None), "guard", None), "spent", 0)
                    for name in ("model", "rate_model")
                ),
                flush=True,
            )


def main():
    """Quote or run the hypothetical continuation; writes a new JSONL file."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cloud", "baseline"), default="baseline")
    p.add_argument("--run-cloud", action="store_true")
    p.add_argument("--quote-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.quote_only:
        load_env()
        total = 0
        for label, factory in (
            ("default", Factory.default()),
            ("random7", Factory.random(np.random.default_rng(7))),
        ):
            cost = 2 * cloud_quote(8400, 100, 2 * len(factory.sensors) + 3)
            total += cost
            print(label, "source upper", cost)
        print("quoted upper", total, "credits")
        return
    if args.backend == "cloud" and not args.run_cloud:
        p.error("cloud calls require --run-cloud")
    run(args.backend, args.output)


if __name__ == "__main__":
    main()
