"""Compare source-map coverage for paired simulated cases using Rayan's locator.

Requires the source locator module, an explicit cloud opt-in, and an output path
outside the repository. The saved arrays contain only regenerated simulation data.
"""

import argparse
from pathlib import Path

import numpy as np

from tabpfn_spatial.sim import Factory, readings, sample_people, sample_scenarios
from tabpfn_spatial.source import WeightedCentroidLocator


def describe(name, maps, scenarios, factory):
    """Print paired source selection coverage; this is not a hazard score."""
    print(name, flush=True)
    for label, limit in (("top3", 3), ("top10", 10), ("top25", 25), ("top50", 50), ("90%", None)):
        mass, count, hits = [], [], 0
        for prob, scenario in zip(maps, scenarios, strict=True):
            flat = prob.ravel()
            order = np.argsort(flat)[::-1]
            n = (
                limit
                if limit is not None
                else int(np.searchsorted(np.cumsum(flat[order]), 0.9) + 1)
            )
            selected = order[:n]
            mass.append(float(flat[selected].sum()))
            count.append(n)
            actual = np.ravel_multi_index(factory.cell_of(scenario.source), factory.walls.shape)
            hits += int(actual in selected)
        print(
            label,
            "mean mass",
            round(float(np.mean(mass)), 4),
            "mean cells",
            round(float(np.mean(count)), 2),
            "source hits",
            hits,
            "/",
            len(scenarios),
            flush=True,
        )


def main():
    """Run the paired source-map diagnostic after explicit cloud opt-in."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-cloud", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_cloud:
        parser.error("pass --run-cloud to authorize the cloud source prediction")
    if args.output.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error("save generated data outside the repository")
    try:
        from tabpfn_spatial.source.tabpfn_locator import TabPFNLocator
    except ImportError as exc:
        raise SystemExit("Rayan's source locator is not present in this checkout") from exc

    factory = Factory.default()
    rng = np.random.default_rng(0)
    scenarios = sample_scenarios(factory, 30, rng)
    people, observations = [], []
    for scenario in scenarios:
        people.append(sample_people(factory, 30, rng))
        observations.append(readings(factory, scenario, 60.0, rng))
    baseline = WeightedCentroidLocator()
    baseline_maps = [baseline.locate(factory, obs).prob for obs in observations]
    print("Fitting source locator on simulated examples", flush=True)
    locator = TabPFNLocator(backend="cloud", n_train=3000, seed=12345).fit(factory)
    print("Predicting 30 source maps in one batch", flush=True)
    estimates = locator.locate_many(factory, observations)
    rayan_maps = [estimate.prob for estimate in estimates]
    describe("weighted centroid", baseline_maps, scenarios, factory)
    describe("TabPFN source locator", rayan_maps, scenarios, factory)
    guard = locator.model.guard
    print("source quoted credits", guard.spent, "calls", guard.calls, flush=True)
    np.savez_compressed(
        args.output,
        sources=np.array([s.source for s in scenarios]),
        rates=np.array([s.rate for s in scenarios]),
        true_winds=np.array([s.wind for s in scenarios]),
        measured_winds=np.array([o.wind for o in observations]),
        sensor_values=np.array([o.values for o in observations]),
        sensor_times=observations[0].times,
        people=np.array(people),
        baseline_maps=np.array(baseline_maps),
        rayan_maps=np.array(rayan_maps),
    )
    print("saved temporary simulated inputs:", args.output, flush=True)


if __name__ == "__main__":
    main()
