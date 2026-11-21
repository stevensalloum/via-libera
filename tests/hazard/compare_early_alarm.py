"""Compare alarm-time hazard forecasts and delayed first routes on the pipe simulator.

Run from a checkout containing Steven's pipe/live branches, with ``--run-cloud``.
The cloud model receives simulated examples and observations, and spends credits.
The comparison uses the same baseline source estimate and readings for both models.
"""

import argparse
import math
import time

import numpy as np

from tabpfn_spatial.contracts import HazardMap
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.hazard import ForwardSimPredictor, TabPFNHazardPredictor, hazard_from_fields
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.sim import Factory, readings, sample_people, sample_scenarios, simulate
from tabpfn_spatial.source import WeightedCentroidLocator


class FixedLocator:
    """Replay the already measured estimate without extra model calls."""

    def __init__(self, estimate):
        self.estimate = estimate

    def locate(self, factory, obs):
        return self.estimate


class FixedPredictor:
    """Replay one forecast for the one-decision route experiment."""

    def __init__(self, forecast):
        self.forecast = forecast

    def predict(self, factory, obs, source, times):
        np.testing.assert_allclose(times, self.forecast.times)
        return self.forecast


def route_outcomes(factory, scenario, start, source, forecast, alarm, latency):
    """Replay one first route, walking to the nearest exit until it arrives."""
    from tabpfn_spatial.evac.live import run_live

    final = None
    for frame in run_live(
        factory,
        scenario,
        start,
        FixedLocator(source),
        FixedPredictor(forecast),
        t_alarm=alarm,
        update_s=300.0,
        t_max=180.0,
        latency=math.ceil(latency),
        before_plan="nearest",
        rng=np.random.default_rng(9000),
    ):
        final = frame
    return final.walkers


def hybrid_outcome(factory, fields, person, alarm, baseline, baseline_s, cloud, cloud_s):
    """Nearest exit at alarm, then baseline and cloud routes at their measured delays.

    Delays are rounded up to whole seconds; forecasts are replayed with no extra
    calls. Both models are assumed to start together on the same readings.
    """
    from tabpfn_spatial.evac.live import Walker

    planner = RiskAwarePlanner()
    walker = Walker(np.asarray(person, dtype=float).copy(), planner.speed)
    empty = np.zeros((1, factory.ny, factory.nx))
    toxic, fire = hazard_from_fields(factory, fields)
    events = {
        int(alarm): HazardMap(np.array([alarm]), empty, empty),
        int(alarm + math.ceil(baseline_s)): baseline,
        int(alarm + math.ceil(cloud_s)): cloud,
    }
    for k, field in enumerate(fields):
        if k in events and walker.out_at is None:
            route = planner.plan(factory, walker.pos[None], events[k], float(k)).routes[0]
            walker.follow((route.cells[:, ::-1] + 0.5) * factory.cell, route.exit_index)
        if walker.out_at is None:
            iy, ix = factory.cell_of(walker.pos)
            walker.dose += float(field[iy, ix])
            walker.harmful_s += float(toxic[k, iy, ix] or fire[k, iy, ix])
            if k >= alarm:
                walker.step(1.0, float(k))
    return walker


def main():
    """Run the paired simulated comparison with explicit cloud opt-in."""
    from tabpfn_spatial.sim import first_alarm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-cloud", action="store_true")
    parser.add_argument("--compare-open", action="store_true")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.run_cloud:
        parser.error("cloud calls spend credits; pass --run-cloud to opt in")
    if args.episodes < 1:
        parser.error("episodes must be positive")

    factory = Factory.default()
    rng = np.random.default_rng(args.seed)
    scenarios = sample_scenarios(factory, args.episodes, rng)
    starts = sample_people(factory, args.episodes, rng)
    locator, planner = WeightedCentroidLocator(), RiskAwarePlanner()
    model = TabPFNHazardPredictor(backend="cloud", top_k=3)
    start_fit = time.perf_counter()
    model.fit(factory, seed=1001)
    print("context+fit seconds", round(time.perf_counter() - start_fit, 3), flush=True)
    models = {"forward": ForwardSimPredictor(), "tabpfn": model}
    if args.compare_open:
        models["tabpfn_open"] = TabPFNHazardPredictor(
            model=model.model, top_k=3, open_cells_only=True
        )
    preds, early_preds, timings, routes = ({k: [] for k in models} for _ in range(4))
    truth, early_truth, alarms, nearest, hybrid, differences = [], [], [], [], [], []
    skipped = 0
    floor = ~factory.walls
    for i, (scenario, person) in enumerate(zip(scenarios, starts, strict=True)):
        timeline = np.arange(0.0, 181.0)
        fields = simulate(factory, [scenario], timeline)[0]
        alarm = first_alarm(factory, timeline, fields)
        if alarm is None:
            skipped += 1
            continue
        obs = readings(factory, scenario, alarm, np.random.default_rng(2000 + i))
        t0 = time.perf_counter()
        source = locator.locate(factory, obs)
        source_s = time.perf_counter() - t0
        times = alarm + np.arange(0.0, 121.0, 10.0)
        target, _ = hazard_from_fields(factory, simulate(factory, [scenario], times)[0])
        truth.append(target[:, floor].ravel())
        early_truth.append(target[:4, floor].ravel())
        alarms.append(alarm)
        forecasts = {}
        for name, predictor in models.items():
            t0 = time.perf_counter()
            hazard = predictor.predict(factory, obs, source, times)
            hazard_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            planner.plan(factory, person[None], hazard, alarm)
            planning_s = time.perf_counter() - t0
            latency = source_s + hazard_s + planning_s
            forecasts[name] = hazard
            timings[name].append([source_s, hazard_s, planning_s, latency])
            preds[name].append(hazard.p_toxic[:, floor].ravel())
            early_preds[name].append(hazard.p_toxic[:4, floor].ravel())
            walkers = route_outcomes(factory, scenario, person, source, hazard, alarm, latency)
            routes[name].append([walkers["live"].harmful_s > 0, walkers["live"].harmful_s])
            if name == "forward":
                nearest.append([walkers["nearest"].harmful_s > 0, walkers["nearest"].harmful_s])
        walker = hybrid_outcome(
            factory,
            fields,
            person,
            alarm,
            forecasts["forward"],
            timings["forward"][-1][-1],
            forecasts["tabpfn"],
            timings["tabpfn"][-1][-1],
        )
        hybrid.append([walker.harmful_s > 0, walker.harmful_s])
        if args.compare_open:
            differences.append(
                np.max(np.abs(forecasts["tabpfn"].p_toxic - forecasts["tabpfn_open"].p_toxic))
            )
        print("episode", i + 1, "alarm", alarm, "seconds", flush=True)
    if not truth:
        raise RuntimeError("no alarms in the selected episodes")
    target = np.concatenate(truth)
    print(
        "completed", len(truth), "skipped without alarm", skipped, "alarm median", np.median(alarms)
    )
    for name in models:
        print(name, "toxic", binary_metrics(np.concatenate(preds[name]), target))
        print(
            name,
            "toxic first 30s",
            binary_metrics(np.concatenate(early_preds[name]), np.concatenate(early_truth)),
        )
        print(name, "source/hazard/plan/total median", np.median(timings[name], axis=0))
        print(name, "source/hazard/plan/total p95", np.quantile(timings[name], 0.95, axis=0))
        print(name, "route exposed fraction / harmful seconds", np.mean(routes[name], axis=0))
    print("nearest exposed fraction / harmful seconds", np.mean(nearest, axis=0))
    print("hybrid exposed fraction / harmful seconds", np.mean(hybrid, axis=0))
    if differences:
        print("maximum per-cell toxic probability difference", max(differences))
    print("tabpfn quoted credits", model.model.guard.spent, "calls", model.model.guard.calls)


if __name__ == "__main__":
    main()
