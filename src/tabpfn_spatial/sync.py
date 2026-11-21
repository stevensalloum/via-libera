"""Sync check: the source, hazard and evacuation parts agree on time, space and units.

Run before every push that touches a model: ``uv run sync-check`` checks the baselines;
``uv run sync-check --source my.module:MyLocator --hazard my.module:MyPredictor``
checks yours (built with ``backend="baseline"`` when the class takes a ``backend``, so no
API tokens are spent). It runs the closed loop at several decision times, sensor periods
and plants, validates every hand-over against ``contracts`` and the conventions in
``docs/sync.md``, times each call, and reports problems instead of crashing.

Hard problems (``problems``) break the loop; soft ones (``warnings``) are allowed but
worth a look (e.g. source probability off the pipes). Integrator-owned.
"""

import argparse
import importlib
import inspect
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tabpfn_spatial.contracts import (
    EvacPlan,
    EvacPlanner,
    HazardMap,
    HazardPredictor,
    SensorReadings,
    SourceEstimate,
    SourceLocator,
)
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.sim import Factory, readings, sample_people, sample_scenarios

HORIZON, STEP = 120.0, 10.0  # forecast t_obs + 0, 10, ..., 120 s (absolute times)
WARN = "warning: "


# ----------------------------------------------------------------- hand-over checks --
def check_readings(f: Factory, r: SensorReadings) -> list[str]:
    """Problems with sensor readings (sim -> source and hazard)."""
    out = []
    if r.positions.shape != f.sensors.shape or not np.allclose(r.positions, f.sensors):
        out.append(f"positions {r.positions.shape} do not match factory.sensors {f.sensors.shape}")
    t = np.asarray(r.times)
    if t.ndim != 1 or len(t) == 0 or np.any(np.diff(t) <= 0):
        out.append("times must be 1-D and strictly ascending (seconds since the leak started)")
    elif t[0] < 0:
        out.append("times must be >= 0 (seconds since the leak started)")
    if r.values.shape != (len(r.positions), len(t)):
        out.append(f"values {r.values.shape} != (sensors, times) {(len(r.positions), len(t))}")
    if np.shape(r.wind) != (2,) or not np.all(np.isfinite(r.wind)):
        out.append("wind must be a finite (u, v) pair in m/s")
    return out


def check_source(f: Factory, est: SourceEstimate) -> list[str]:
    """Problems with a source estimate (source -> hazard)."""
    p = np.asarray(est.prob, dtype=float)
    if p.shape != f.walls.shape:
        return [f"prob shape {p.shape} != grid (ny, nx) = {f.walls.shape}, rows are y"]
    out = []
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        out.append("prob must be finite and >= 0")
    if abs(p.sum() - 1) > 1e-6:
        out.append(f"prob must sum to 1 (sums to {p.sum():.4f})")
    if p[f.walls].sum() > 1e-9:
        out.append(f"{p[f.walls].sum():.1%} of the probability is on solid cells")
    on_sites = p[f.leak_prior > 0].sum()
    if f.pipes is not None and on_sites < 0.99:
        out.append(f"{WARN}less than 99 % of the probability is on leak sites (pipes)")
    if est.rate is not None:
        lv = None if est.rate_levels is None else np.asarray(est.rate_levels, dtype=float)
        rate = np.asarray(est.rate, dtype=float)
        if lv is None or lv.shape != rate.shape:
            out.append("rate needs rate_levels of the same length")
        elif np.any(np.diff(lv) <= 0) or lv[0] <= 0 or lv[-1] >= 1:
            out.append("rate_levels must be ascending quantile levels inside (0, 1)")
        elif np.any(np.diff(rate) < 0) or np.any(rate <= 0):
            out.append("rate quantiles must be positive and non-decreasing")
    elif f.pipes is not None:
        out.append(f"{WARN}no release rate: hazard will assume rate 1.0 (the median)")
    return out


def check_hazard(f: Factory, hz: HazardMap, times: np.ndarray) -> list[str]:
    """Problems with a hazard map (hazard -> evacuation) for the requested ``times``."""
    out = []
    if np.shape(hz.times) != np.shape(times) or not np.allclose(hz.times, times):
        out.append(
            "times must be the requested absolute times (seconds since the leak started), "
            f"got {np.round(np.asarray(hz.times)[:3], 1)}... for {np.round(times[:3], 1)}..."
        )
    want = (len(times), f.ny, f.nx)
    for name in ("p_toxic", "p_fire"):
        a = np.asarray(getattr(hz, name), dtype=float)
        if a.shape != want:
            out.append(f"{name} shape {a.shape} != (times, ny, nx) {want}")
            continue
        if not np.all(np.isfinite(a)) or a.min() < 0 or a.max() > 1:
            out.append(f"{name} has values outside [0, 1] or NaN")
        if a[:, f.walls].max(initial=0) > 1e-9:
            out.append(f"{name} is non-zero on solid cells")
    fire = np.asarray(hz.p_fire, dtype=float)
    if fire.shape == want and fire[:, ~f.ignition].max(initial=0) > 1e-9:
        out.append("p_fire is non-zero outside the ignition zones (factory.ignition)")
    return out


def check_plan(f: Factory, plan: EvacPlan, people: np.ndarray) -> list[str]:
    """Problems with an evacuation plan for ``people`` (p, 2) metres."""
    people = np.atleast_2d(people)
    if len(plan.routes) != len(people):
        return [f"one route per person expected: {len(plan.routes)} for {len(people)}"]
    out = []
    for i, (rt, xy) in enumerate(zip(plan.routes, people, strict=True)):
        c = np.asarray(rt.cells, dtype=int)
        if tuple(c[0]) != f.cell_of(xy):
            out.append(f"route {i} does not start at the person's cell")
        if not 0 <= rt.exit_index < len(f.exits):
            out.append(f"route {i}: exit_index {rt.exit_index} out of range")
        elif tuple(c[-1]) != f.cell_of(f.exits[rt.exit_index]):
            out.append(f"route {i} does not end at its exit")
        if f.walls[c[:, 0], c[:, 1]].any():
            out.append(f"route {i} crosses solid cells")
        step = np.abs(np.diff(c, axis=0))
        if len(step) and step.max() > 1:
            out.append(f"route {i} jumps more than one cell")
        if not 0 <= rt.risk <= 1:
            out.append(f"route {i}: risk {rt.risk} outside [0, 1]")
    return out


# ------------------------------------------------------------------------- the loop --
def _plant(name: str) -> Factory:
    if name == "default":
        return Factory.default()
    if name == "random":
        return Factory.random(np.random.default_rng(1))
    raise ValueError(f"plant must be 'default' or 'random', not {name!r}")


def _timed(fn, *args) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def _one(f, locator, predictor, planner, t_obs, every, seed) -> dict[str, Any]:
    """One hand-over chain; any crash becomes a problem."""
    rng = np.random.default_rng(seed)
    sc = sample_scenarios(f, 1, rng)[0]
    row: dict[str, Any] = {"locate_s": np.nan, "predict_s": np.nan, "plan_s": np.nan}
    found: list[str] = []
    stage = "readings"
    try:
        r = readings(f, sc, t_obs, rng, every=every)
        found += check_readings(f, r)
        stage = "locate"
        est, row["locate_s"] = _timed(locator.locate, f, r)
        found += check_source(f, est)
        iy, ix = est.top_cells(1)[0]
        row["source_error_m"] = float(np.linalg.norm(f.centre(int(iy), int(ix)) - sc.source))
        stage = "predict"
        times = t_obs + np.arange(0.0, HORIZON + 1e-9, STEP)
        hz, row["predict_s"] = _timed(predictor.predict, f, r, est, times)
        found += check_hazard(f, hz, times)
        stage = "plan"
        people = sample_people(f, 5, rng)
        plan, row["plan_s"] = _timed(planner.plan, f, people, hz, t_obs)
        found += check_plan(f, plan, people)
    except Exception as e:  # report, do not crash: the point is to list every break
        found.append(f"{stage} crashed: {type(e).__name__}: {e}")
    row["problems"] = [p for p in found if not p.startswith(WARN)]
    row["warnings"] = [p[len(WARN) :] for p in found if p.startswith(WARN)]
    return row


def run_sync(
    locator: SourceLocator,
    predictor: HazardPredictor,
    planner: EvacPlanner | None = None,
    decision_times: Sequence[float] = (10.0, 20.0, 30.0, 60.0),
    sample_every: Sequence[float] = (5.0,),
    seeds: Iterable[int] = range(3),
    plants: Sequence[str] = ("default", "random"),
) -> list[dict[str, Any]]:
    """One row per plant x decision time x sensor period x leak seed.

    Objects with a ``fit(factory)`` method are fitted once per plant (time in ``fit_s``).
    Cost: dominated by the plugged-in models.
    """
    planner = planner or RiskAwarePlanner()
    rows = []
    for name in plants:
        f = _plant(name)
        fit_s, fit_problem = 0.0, []
        for part in (locator, predictor):
            if hasattr(part, "fit"):
                try:
                    fit_s += _timed(part.fit, f)[1]
                except Exception as e:
                    fit_problem.append(f"fit crashed: {type(e).__name__}: {e}")
        for t_obs in decision_times:
            for every in sample_every:
                for seed in seeds:
                    row = _one(f, locator, predictor, planner, float(t_obs), float(every), seed)
                    row["problems"] = fit_problem + row["problems"]
                    rows.append(
                        {"plant": name, "t_obs": t_obs, "every": every, "seed": seed,
                         "fit_s": fit_s, **row}
                    )  # fmt: skip
    return rows


def summarise(rows: list[dict[str, Any]]) -> str:
    """Table per plant x decision time x sensor period, then every distinct problem."""
    lines = ["plant    t_obs  every  ok    locate_s  predict_s  source_err_m"]
    keys = sorted({(r["plant"], r["t_obs"], r["every"]) for r in rows})
    for k in keys:
        g = [r for r in rows if (r["plant"], r["t_obs"], r["every"]) == k]
        ok = sum(not r["problems"] for r in g)
        err = np.nanmean([r.get("source_error_m", np.nan) for r in g])
        lines.append(
            f"{k[0]:8} {k[1]:5.0f}  {k[2]:5.0f}  {ok}/{len(g)}  "
            f"{np.nanmean([r['locate_s'] for r in g]):8.3f}  "
            f"{np.nanmean([r['predict_s'] for r in g]):9.3f}  {err:12.1f}"
        )
    for label, key in (("PROBLEMS", "problems"), ("warnings", "warnings")):
        seen = sorted({p for r in rows for p in r[key]})
        lines += ["", f"{label} ({len(seen)}):", *[f"  - {p}" for p in seen]]
    return "\n".join(lines)


def load(path: str) -> type:
    """Class from ``"package.module:ClassName"``."""
    if ":" not in path:
        raise ValueError(f"expected 'module:Class', got {path!r}")
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def build(path: str, backend: str) -> Any:
    """Instantiate a class, passing ``backend`` when its constructor takes one."""
    cls = load(path)
    takes = "backend" in inspect.signature(cls).parameters
    return cls(backend=backend) if takes else cls()


# ---------------------------------------------------------------------- timeline --
def plot_timeline(path: Path, n_leaks: int = 200) -> Path:
    """One time axis with every time scale the parts must agree on (the sync figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tabpfn_spatial.contracts import HazardMap
    from tabpfn_spatial.sim import first_alarm, simulate

    f = Factory.default()
    times = np.arange(0.0, 241.0, 1.0)
    fields = simulate(f, sample_scenarios(f, n_leaks, np.random.default_rng(0)), times)
    alarms = np.array([a for a in (first_alarm(f, times, x) for x in fields) if a is not None])
    zero = np.zeros((1, f.ny, f.nx))
    open_xy = f.centres()[~f.walls]
    plan = RiskAwarePlanner().plan(f, open_xy, HazardMap(np.array([0.0]), zero, zero), 0.0)
    walk = (
        np.array(
            [np.linalg.norm(np.diff(r.cells, axis=0), axis=1).sum() * f.cell for r in plan.routes]
        )
        / RiskAwarePlanner().speed
    )
    med = float(np.median(alarms))

    fig, ax = plt.subplots(figsize=(13, 5.6))
    rows = [
        "sensors sample (every 5 s)",
        f"detector alarm ({len(alarms)} leaks)",
        "everyone walks out (nearest exit, from alarm)",
        "replan every 20 s; route lands within ~2 s",
        "hazard forecast requested (t_obs + 0..120 s)",
        "Rayan's locator today: readings at 20, 40, 60 s",
        "Ahmad's hazard today: trained on 70-180 s",
    ]
    y = np.arange(len(rows))[::-1]
    ax.vlines(np.arange(0, 241, 5), y[0] - 0.25, y[0] + 0.25, color="#1565c0", lw=1)
    ax.boxplot(alarms, positions=[y[1]], vert=False, widths=0.5, manage_ticks=False)
    ax.boxplot(med + walk, positions=[y[2]], vert=False, widths=0.5, manage_ticks=False)
    for t in np.arange(med, 241, 20):
        ax.barh(y[3], 2, left=t, height=0.4, color="#00a152")
    ax.barh(y[4], 120, left=med, height=0.4, color="#9e9e9e", alpha=0.7)
    ax.vlines([20, 40, 60], y[5] - 0.25, y[5] + 0.25, color="#d50000", lw=3)
    ax.barh(y[6], 110, left=70, height=0.4, color="#d50000", alpha=0.5)
    ax.axvline(med, c="k", ls="--", lw=1)
    ax.text(med + 1, y[0] + 0.45, f"median alarm {med:.0f} s", fontsize=8)
    ax.set_yticks(y, rows, fontsize=9)
    ax.set_xlim(0, 240)
    ax.set_xlabel("seconds since the leak started (the shared clock in contracts)")
    ax.set_title(
        "Time scales the three parts must agree on (default plant, 2 m cells, "
        "walk 1.3 m/s; alarm at 10 % LEL). Red = mismatch to fix.",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main() -> int:
    """CLI: ``uv run sync-check [--source m:C] [--hazard m:C] [--timeline out.png]``."""
    p = argparse.ArgumentParser(description="Check the parts agree on time, space and units.")
    p.add_argument("--source", default="tabpfn_spatial.source:WeightedCentroidLocator")
    p.add_argument("--hazard", default="tabpfn_spatial.hazard:ForwardSimPredictor")
    p.add_argument("--backend", default="baseline", help="for classes taking backend=")
    p.add_argument("--times", type=float, nargs="+", default=[10.0, 20.0, 30.0, 60.0])
    p.add_argument("--every", type=float, nargs="+", default=[5.0])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--plants", nargs="+", default=["default", "random"])
    p.add_argument("--timeline", type=Path, help="only draw the time-scale figure")
    a = p.parse_args()
    if a.timeline:
        print(plot_timeline(a.timeline))
        return 0
    rows = run_sync(
        build(a.source, a.backend), build(a.hazard, a.backend), None, a.times, a.every,
        range(a.seeds), a.plants,
    )  # fmt: skip
    print(summarise(rows))
    return int(any(r["problems"] for r in rows))
