"""Paired hazard validation on varied layouts and simulator perturbations.

The research script batches source and hazard calls, stores one JSON line per
case, and defaults to a cheap offline smoke test. Cloud requires --run-cloud;
--quote-only obtains a free shape-only estimate. It uses the same simulator for
context and outcomes, with controlled shifts at evaluation time.
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import (
    LEL,
    TOXIC_LIMIT,
    UEL,
    HazardMap,
    SensorReadings,
    SourceEstimate,
)
from tabpfn_spatial.core.budget import cloud_quote
from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.env import load_env
from tabpfn_spatial.evac import RiskAwarePlanner, route_exposed
from tabpfn_spatial.hazard import ForwardSimPredictor, hazard_from_fields
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.hazard.source_selection import select_source_cells
from tabpfn_spatial.hazard.tabpfn_predictor import _cdf
from tabpfn_spatial.sim import (
    Factory,
    sample_people,
    sample_scenarios,
    sensor_values,
    simulate,
    wind_at,
)
from tabpfn_spatial.source import TabPFNLocator, WeightedCentroidLocator
from tabpfn_spatial.source.features import alarm_time

HORIZONS = np.array([0.0, 10.0, 30.0, 60.0, 120.0])
# (name, diffusion m^2/s, wind drift rad/sqrt(s), dead-sensor prob, stuck-sensor prob)
CONDITIONS = (
    ("standard", 0.8, 0.0, 0.0, 0.0),
    ("wind_drift", 0.8, 0.05, 0.0, 0.0),
    ("sensor_faults", 0.8, 0.0, 0.15, 0.05),
    ("slower_mixing", 0.4, 0.0, 0.0, 0.0),
    ("faster_mixing", 1.6, 0.0, 0.0, 0.0),
    ("combined_stress", 0.4, 0.05, 0.15, 0.05),
)


@dataclass
class Case:
    """One paired episode and its independent simulated outcome."""

    condition: str
    index: int
    scenario: object
    obs: SensorReadings
    person: np.ndarray
    times: np.ndarray
    truth_times: np.ndarray
    truth_toxic: np.ndarray
    truth_harm: np.ndarray
    source: SourceEstimate | None = None


class RecordingLocator:
    """Capture context readings while using the default centroid source."""

    def __init__(self):
        self.obs = []
        self.centroid = WeightedCentroidLocator()

    def locate(self, factory, readings):
        """Record observations and return the standard proxy estimate."""
        self.obs.append(readings)
        return self.centroid.locate(factory, readings)


class ReplayLocator:
    """Pair batched source results with exactly their recorded observations."""

    def __init__(self, obs, estimates):
        if len(obs) != len(estimates):
            raise ValueError("missing source estimates")
        self.obs, self.estimates, self.index = obs, estimates, 0

    def locate(self, factory, obs):
        """Verify each observation to avoid mismatching RNG streams."""
        if self.index >= len(self.obs):
            raise ValueError("too many training observations")
        old = self.obs[self.index]
        if any(
            not np.array_equal(getattr(old, k), getattr(obs, k), equal_nan=True)
            for k in ("positions", "times", "values", "wind")
        ):
            raise ValueError("training replay observations do not match")
        result = self.estimates[self.index]
        self.index += 1
        return result


def variant_readings(factory, scenario, t_obs, rng, diffusivity, dropout=0.0, stuck=0.0):
    """Reproduce simulator readings with a chosen diffusion coefficient."""
    times = np.arange(0.0, t_obs + 1e-9, 5.0)
    clean = sensor_values(factory, simulate(factory, [scenario], times, d=diffusivity)[0]).T
    noisy = clean * (1 + 0.1 * rng.standard_normal(clean.shape))
    noisy += 0.001 * rng.standard_normal(clean.shape)
    values = np.maximum(noisy, 0.0)
    k = len(values)
    is_stuck = rng.uniform(size=k) < stuck
    values[is_stuck] = values[is_stuck, :1]
    values[rng.uniform(size=k) < dropout] = np.nan
    wind = wind_at(scenario, times).mean(axis=0) + 0.1 * rng.standard_normal(2)
    return SensorReadings(factory.sensors.copy(), times, values, wind)


def batched_hazards(factory, requests, model):
    """Compute exactly the normal top-three hazard forecasts in one batch.

    One batch has sum(cases * sources * horizons * open cells) query rows;
    its memory grows linearly with that number. Tests compare it to the actual
    predictor, including source weighting and ignition masking.
    """
    mask = (~factory.walls).ravel()
    all_xy = factory.centres().reshape(-1, 2)
    xy = all_xy[mask]
    rows, layouts = [], []
    for obs, source, times in requests:
        selected = select_source_cells(source.prob, factory.walls, 3)
        rate = (
            1.0
            if source.rate is None or source.rate_levels is None
            else float(np.interp(0.5, source.rate_levels, source.rate))
        )
        for idx in selected.indices:
            for t in times:
                rows.append(cell_features(factory, all_xy[idx], obs.wind, rate, float(t), xy))
        layouts.append((selected, times, len(selected.indices) * len(times) * len(xy)))
    dist = model.predict_dist(np.concatenate(rows))
    toxic = 1 - _cdf(dist.values, dist.levels, TOXIC_LIMIT)
    fire = _cdf(dist.values, dist.levels, UEL) - _cdf(dist.values, dist.levels, LEL)
    result, offset = [], 0
    for selected, times, count in layouts:
        shape = (len(selected.indices), len(times), len(xy))
        pt = np.zeros((len(times), len(all_xy)))
        pf = np.zeros_like(pt)
        pt[:, mask] = np.tensordot(
            selected.weights, toxic[offset : offset + count].reshape(shape), axes=1
        )
        pf[:, mask] = np.tensordot(
            selected.weights, fire[offset : offset + count].reshape(shape), axes=1
        )
        pt = pt.reshape(-1, factory.ny, factory.nx) * ~factory.walls
        pf = pf.reshape(-1, factory.ny, factory.nx) * factory.ignition
        result.append(HazardMap(times, pt, pf))
        offset += count
    return result


def make_contexts(factory, locator, backend, scenarios=100, cells=20):
    """Build paired centroid and batched Rayan-source context tables."""
    args = dict(n_scenarios=scenarios, cells_per_scenario=cells, seed=1001)
    recorder = RecordingLocator()
    xa, ya = training_rows(factory, **args, locator=recorder)
    estimates = locator.locate_many(factory, recorder.obs)
    replay = ReplayLocator(recorder.obs, estimates)
    xb, yb = training_rows(factory, **args, locator=replay)
    if replay.index != len(recorder.obs) or not np.array_equal(ya, yb):
        raise ValueError("paired context replay did not preserve examples and targets")
    return {
        "centroid_context": make_regressor(backend).fit(xa, ya),
        "rayan_context": make_regressor(backend).fit(xb, yb),
    }


def make_cases(factory, n, start, replicate=0):
    """Make paired conditions from the same leaks, observation times and people."""
    rng = np.random.default_rng(20261121 + replicate)
    scenarios = sample_scenarios(factory, start + n, rng)
    people = sample_people(factory, start + n, rng)
    cases = []
    for name, diffusion, drift, dropout, stuck in CONDITIONS:
        for i in range(start, start + n):
            sc = replace(scenarios[i], drift=drift)
            t_obs = float((10, 20, 30)[i % 3])
            obs = variant_readings(
                factory,
                sc,
                t_obs,
                np.random.default_rng(30_000 + i + 10_000 * replicate),
                diffusion,
                dropout,
                stuck,
            )
            times = t_obs + HORIZONS
            truth_times = np.arange(0.0, times[-1] + 1.0)
            fields = simulate(factory, [sc], truth_times, d=diffusion)[0]
            toxic, fire = hazard_from_fields(factory, fields)
            target = toxic[times.astype(int)]
            cases.append(
                Case(name, i, sc, obs, people[i], times, truth_times, target, toxic | fire)
            )
    return cases


def score(factory, case, hazard, planner):
    """Score toxic probabilities and a single immediate route on the same truth."""
    floor = ~factory.walls
    p, y = hazard.p_toxic[:, floor].ravel(), case.truth_toxic[:, floor].ravel()
    m = binary_metrics(p, y)
    pred = p >= 0.5
    positive = y.astype(bool)
    tp, fp = int(np.sum(pred & positive)), int(np.sum(pred & ~positive))
    route = planner.plan(factory, case.person[None], hazard, case.obs.t_obs).routes[0]
    exposed = bool(route_exposed(factory, route, case.truth_times, case.truth_harm, case.obs.t_obs))
    return {
        "brier": m["brier"],
        "ece": m["ece"],
        "auc": m["auc"],
        "tp": tp,
        "fp": fp,
        "positive": int(positive.sum()),
        "negative": int((~positive).sum()),
        "exposed": exposed,
    }


def quote(layouts, n, scenarios, cells, batch):
    """Quote an upper bound before cloud inference, using only table shapes."""
    load_env()
    total, cases = 0, len(CONDITIONS) * n
    for label, f in layouts:
        cols = 2 * len(f.sensors) + 3
        source_cost = 2 * (cloud_quote(8400, scenarios, cols) + cloud_quote(8400, cases, cols))
        qrows = min(batch, cases) * int((~f.walls).sum()) * 3 * len(HORIZONS)
        hazard_cost = 2 * int(np.ceil(cases / batch)) * cloud_quote(scenarios * cells * 4, qrows, 7)
        print(
            label,
            "source upper",
            source_cost,
            "hazard upper",
            hazard_cost,
            "query rows per full batch",
            qrows,
            flush=True,
        )
        total += source_cost + hazard_cost
    print("quoted upper total", total, "credits; actual charges can vary", flush=True)


def main():
    """Run reproducible cases, writing aggregate-ready JSONL as they finish."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    p.add_argument("--run-cloud", action="store_true")
    p.add_argument("--quote-only", action="store_true")
    p.add_argument("--episodes", type=int, default=10, help="leaks per layout and condition")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--replicate", type=int, default=0, help="independent scenario/noise seed")
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--train-scenarios", type=int, default=100)
    p.add_argument("--cells-per-scenario", type=int, default=20)
    p.add_argument("--layouts", nargs="+", default=["default", "random7", "random19", "random31"])
    p.add_argument("--output", type=Path, default=Path("outputs/multiplant.jsonl"))
    args = p.parse_args()
    if (
        min(args.episodes, args.batch, args.train_scenarios, args.cells_per_scenario) < 1
        or args.start < 0
        or args.replicate < 0
    ):
        p.error("episode counts, batch and context sizes must be positive")
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
        quote(layouts, args.episodes, args.train_scenarios, args.cells_per_scenario, args.batch)
        return
    if args.backend == "cloud" and not args.run_cloud:
        p.error("cloud calls require --run-cloud")
    if args.backend == "baseline":
        import tabpfn_spatial.core.models as models

        models._CHUNK = 32
    else:
        load_env()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for label, factory in layouts:
            started = time.perf_counter()
            locator = TabPFNLocator(backend=args.backend).fit(factory)
            models = make_contexts(
                factory, locator, args.backend, args.train_scenarios, args.cells_per_scenario
            )
            cases = make_cases(factory, args.episodes, args.start, args.replicate)
            estimates = locator.locate_many(factory, [c.obs for c in cases])
            for case, estimate in zip(cases, estimates, strict=True):
                case.source = estimate
            _, detected = locator.rows([c.obs for c in cases])
            print(
                label,
                "cases",
                len(cases),
                "source activated",
                len(detected),
                "context seconds",
                round(time.perf_counter() - started, 2),
                flush=True,
            )
            planner = RiskAwarePlanner()
            forward = ForwardSimPredictor()
            for start in range(0, len(cases), args.batch):
                group = cases[start : start + args.batch]
                requests = [(c.obs, c.source, c.times) for c in group]
                forecasts = {
                    name: batched_hazards(factory, requests, model)
                    for name, model in models.items()
                }
                for j, case in enumerate(group):
                    fwd = forward.predict(factory, case.obs, case.source, case.times)
                    selection = select_source_cells(case.source.prob, factory.walls, 3)
                    iy, ix = case.source.top_cells(1)[0]
                    sy, sx = factory.cell_of(case.scenario.source)
                    zero = np.zeros_like(fwd.p_toxic)
                    nearest = HazardMap(case.times, zero, zero)
                    row = {
                        "layout": label,
                        "condition": case.condition,
                        "episode": case.index,
                        "t_obs": case.obs.t_obs,
                        "seed": 20261121 + args.replicate,
                        "replicate": args.replicate,
                        "source_error_m": float(
                            np.linalg.norm(factory.centre(int(iy), int(ix)) - case.scenario.source)
                        ),
                        "source_top3_mass": selection.retained_mass,
                        "source_top3_contains_truth": bool(
                            sy * factory.nx + sx in selection.indices
                        ),
                        "source_activated": alarm_time(case.obs.times, case.obs.values) is not None,
                        "true_rate": case.scenario.rate,
                        "estimated_rate": float(
                            np.interp(0.5, case.source.rate_levels, case.source.rate)
                        ),
                        "forward": score(factory, case, fwd, planner),
                        "nearest_exposed": score(factory, case, nearest, planner)["exposed"],
                    }
                    for name in models:
                        row[name] = score(factory, case, forecasts[name][j], planner)
                    output.write(json.dumps(row, allow_nan=True) + "\n")
                output.flush()
                print(
                    label,
                    "completed",
                    min(start + args.batch, len(cases)),
                    "/",
                    len(cases),
                    "elapsed",
                    round(time.perf_counter() - started, 1),
                    flush=True,
                )
            guards = [
                getattr(x, "guard", None)
                for x in (locator.model, locator.rate_model, *models.values())
            ]
            print(
                label,
                "quoted credits",
                sum(getattr(g, "spent", 0) for g in guards),
                "calls",
                sum(getattr(g, "calls", 0) for g in guards),
                flush=True,
            )


if __name__ == "__main__":
    main()
