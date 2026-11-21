"""Paired A/B/C sensor-history pilot, offline by default.

The simulator starts the leak at the assumed alarm (t=0); this is an artificial
alarm event, not the simulator's detector-triggered alarm. The default source is
the centroid baseline. Cloud inference requires explicit --run-cloud; use
--quote-only to obtain the free cost quote before requesting approval.
"""

import argparse
import time

import numpy as np

from tabpfn_spatial.core import models as core_models
from tabpfn_spatial.core.budget import cloud_quote
from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.env import load_env
from tabpfn_spatial.evac import RiskAwarePlanner, route_exposed
from tabpfn_spatial.hazard import ForwardSimPredictor, TabPFNHazardPredictor, hazard_from_fields
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.sim import Factory, readings, sample_people, sample_scenarios, simulate
from tabpfn_spatial.source import WeightedCentroidLocator


def threshold_metrics(prob: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> dict:
    """Return recall, false-positive rate and precision; undefined ratios are NaN."""
    pred = np.asarray(prob).ravel() >= threshold
    target = np.asarray(truth, dtype=bool).ravel()
    tp = int(np.sum(pred & target))
    fp = int(np.sum(pred & ~target))
    positives, negatives = int(target.sum()), int((~target).sum())
    return {
        "positives": positives,
        "negatives": negatives,
        "recall": tp / positives if positives else float("nan"),
        "false_positive_rate": fp / negatives if negatives else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }


def quote(factory: Factory, train_rows: int, episodes: int) -> None:
    """Print free per-call quotes for three configurations and the intended call count."""
    load_env()
    n_query = int((~factory.walls).sum()) * 3 * 4
    costs = {name: cloud_quote(train_rows, n_query, n) for name, n in CONFIGS.items()}
    print("train rows", train_rows, "query rows per call", n_query)
    print("quoted credits per call", costs)
    print("hazard calls", 3 * episodes, "estimated total", episodes * sum(costs.values()))
    print("Centroid source uses no cloud calls; actual charges and model setup may vary.")


CONFIGS = {"A_spatial": 7, "B_time": 9, "C_sensor": 37}


def main() -> None:
    """Run paired configurations on identical simulated examples and observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    parser.add_argument("--run-cloud", action="store_true")
    parser.add_argument("--quote-only", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-scenarios", type=int, default=20)
    parser.add_argument("--cells-per-scenario", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.episodes, args.train_scenarios, args.cells_per_scenario) < 1:
        parser.error("episodes, training scenarios and cells per scenario must be positive")
    factory = Factory.default()
    train_count = args.train_scenarios * args.cells_per_scenario * 4
    if args.quote_only:
        quote(factory, train_count, args.episodes)
        return
    if args.backend == "cloud" and not args.run_cloud:
        parser.error(
            "cloud inference spends credits: quote first and pass --run-cloud after approval"
        )
    if args.backend == "baseline":
        core_models._CHUNK = 64  # reduce peak k-NN memory; reported timings include this setting

    observations = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    train_x, train_y = training_rows(
        factory,
        args.train_scenarios,
        args.cells_per_scenario,
        seed=1001,
        observation_times=observations,
        include_sensor_history=True,
    )
    # All three contexts are column subsets of exactly the same generated rows.
    models = {"forward": ForwardSimPredictor()}
    for name, n in CONFIGS.items():
        fitted = make_regressor(args.backend).fit(train_x[:, :n], train_y)
        models[name] = TabPFNHazardPredictor(
            backend=args.backend,
            model=fitted,
            top_k=3,
            open_cells_only=True,
            include_time_context=name != "A_spatial",
            include_sensor_history=name == "C_sensor",
        )
    rng = np.random.default_rng(args.seed)
    scenarios = sample_scenarios(factory, args.episodes, rng)
    people = sample_people(factory, args.episodes, rng)
    locator, planner = WeightedCentroidLocator(), RiskAwarePlanner()
    forecasts = {name: [] for name in models}
    timings = {name: [] for name in models}
    exposure = {name: [] for name in models}
    episode_scores = {name: [] for name in models}
    truth_all, near_all, retained = [], [], []
    floor = ~factory.walls
    centres = factory.centres()[floor]
    nearest_distance = np.linalg.norm(
        centres[:, None, :] - factory.sensors[None, :, :], axis=2
    ).min(axis=1)
    near = nearest_distance <= 6.0
    for i, (scenario, person) in enumerate(zip(scenarios, people, strict=True)):
        t_obs = float(observations[i % len(observations)])
        obs = readings(factory, scenario, t_obs, np.random.default_rng(2000 + i))
        source = locator.locate(factory, obs)
        times = t_obs + np.arange(0.0, 31.0, 10.0)
        target, _ = hazard_from_fields(factory, simulate(factory, [scenario], times)[0])
        y = target[:, floor]
        truth_all.append(y.ravel())
        near_all.append(np.broadcast_to(near, y.shape).ravel())
        truth_times = np.arange(t_obs, t_obs + 121.0)
        toxic, fire = hazard_from_fields(factory, simulate(factory, [scenario], truth_times)[0])
        names = list(models)
        names = names[i % len(names) :] + names[: i % len(names)]
        for name in names:
            start = time.perf_counter()
            hazard = models[name].predict(factory, obs, source, times)
            timings[name].append(time.perf_counter() - start)
            p = hazard.p_toxic[:, floor]
            forecasts[name].append(p.ravel())
            episode_scores[name].append(float(np.mean((p - y) ** 2)))
            route = planner.plan(factory, person[None], hazard, t_obs).routes[0]
            exposure[name].append(route_exposed(factory, route, truth_times, toxic | fire, t_obs))
            if name == "A_spatial":
                retained.append(models[name].last_selection.retained_mass)
        print("completed", i + 1, "of", args.episodes, "t_obs", t_obs, flush=True)

    target = np.concatenate(truth_all)
    is_near = np.concatenate(near_all)
    print("settings: simulated alarm=leak start; centroid source; evaluation seed", args.seed)
    print("context seed 1001, rows", len(train_y), "query times 0/10/20/30 s after observation")
    print(
        "positive fraction",
        float(target.mean()),
        "source top-3 retained mass",
        float(np.mean(retained)),
    )
    for name, model in models.items():
        prediction = np.concatenate(forecasts[name])
        print(name, "toxic", binary_metrics(prediction, target))
        print(name, "threshold=0.5", threshold_metrics(prediction, target))
        print(
            name,
            "Brier near/far sensors",
            float(np.mean((prediction[is_near] - target[is_near]) ** 2)),
            float(np.mean((prediction[~is_near] - target[~is_near]) ** 2)),
        )
        print(
            name,
            "hazard seconds median/range",
            float(np.median(timings[name])),
            float(np.min(timings[name])),
            float(np.max(timings[name])),
        )
        print(
            name,
            "exposed routes",
            sum(exposure[name]),
            "/",
            args.episodes,
            "episode Brier",
            [round(x, 5) for x in episode_scores[name]],
        )
        guard = getattr(getattr(model, "model", None), "guard", None)
        print(
            name, "quoted credits", getattr(guard, "spent", 0), "calls", getattr(guard, "calls", 0)
        )
    print(
        "Route comparison excludes prediction delay; "
        "the planner's toxic/fire combination is provisional."
    )


if __name__ == "__main__":
    main()
