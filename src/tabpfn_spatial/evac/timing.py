"""Timing study: when to decide, how often to replan, how fast the models must be.

``uv run evac-timing`` runs the live one-person evacuation (``evac.live``) on the same
leaks under different timings and draws ``outputs/timing/timing.png``:

- ``alarm``: first decision time ``t_alarm`` = how long we collect readings (``t_obs``).
  Waiting buys a better source and hazard estimate but people stand in the gas meanwhile.
- ``update``: seconds between replans.
- ``latency_walk`` / ``latency_wait``: seconds the models take to answer, when people
  walk to the nearest exit until the first route arrives, or wait for it.

Two model settings bracket what TabPFN can do: ``baseline`` (weighted-centroid source +
forward-simulation hazard, today's baselines) and ``perfect`` (true leak + true future).
Scores per leak (one person, random start, seeds = leak seeds, so settings are paired):
source error and hazard Brier at the first decision, seconds in harmful gas before the
alarm, and for each policy seconds in harm, dose and exposed (any second in harm).
Cost: one ``run_live`` per (setting, leak), about 1 s each; runs in parallel processes.
"""

import argparse
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.live import POLICIES, run_live
from tabpfn_spatial.evac.standins import OracleHazard, OracleLocator
from tabpfn_spatial.hazard import ForwardSimPredictor, brier, hazard_from_fields
from tabpfn_spatial.sim import Factory, sample_people, sample_scenarios, simulate
from tabpfn_spatial.source import WeightedCentroidLocator, localisation_error


@dataclass(frozen=True)
class Setting:
    """One timing configuration (see the module docstring)."""

    models: str = "baseline"  # baseline | perfect
    t_alarm: float = 30.0
    update_s: float = 5.0
    latency: float = 0.0
    before_plan: str = "nearest"  # nearest | wait
    drift: float = 0.05
    after: float = 120.0  # seconds simulated after the first route lands


# name -> (Setting field to vary, values, base setting, model settings)
STUDIES: dict[str, tuple[str, tuple[float, ...], Setting, tuple[str, ...]]] = {
    "alarm": ("t_alarm", (10, 20, 30, 45, 60, 90, 120), Setting(), ("baseline", "perfect")),
    "update": ("update_s", (2, 5, 10, 30, 60), Setting(drift=0.15), ("baseline", "perfect")),
    "latency_walk": ("latency", (0, 2, 5, 10, 20), Setting(), ("perfect",)),
    "latency_wait": ("latency", (0, 2, 5, 10, 20), Setting(before_plan="wait"), ("perfect",)),
}


def run_one(f: Factory, s: Setting, seed: int) -> dict[str, float]:
    """Score one leak (seed) under one setting."""
    if s.models not in ("baseline", "perfect"):
        raise ValueError(f"models must be 'baseline' or 'perfect', not {s.models!r}")
    rng = np.random.default_rng(seed)
    sc = sample_scenarios(f, 1, rng, drift=s.drift)[0]
    start = sample_people(f, 1, rng)[0]
    if s.models == "perfect":
        loc, pred = OracleLocator(sc, sigma=0.0), OracleHazard(sc)
    else:
        loc, pred = WeightedCentroidLocator(), ForwardSimPredictor()
    frames = run_live(
        f, sc, start, loc, pred, t_alarm=s.t_alarm, update_s=s.update_s,
        t_max=s.t_alarm + s.latency + s.after, rng=np.random.default_rng(seed + 1),
        latency=s.latency, before_plan=s.before_plan,
    )  # fmt: skip
    first = pre = last = None
    for fr in frames:
        if first is None and fr.replanned:
            first = fr
        if fr.t == s.t_alarm:
            pre = fr.walkers["live"].harmful_s
        last = fr
    hz = first.hazard
    toxic, fire = hazard_from_fields(f, simulate(f, [sc], hz.times)[0])
    out = {
        "source_error_m": localisation_error(f, first.source, sc.source),
        "hazard_brier": brier(1 - (1 - hz.p_toxic) * (1 - hz.p_fire), toxic | fire),
        "pre_alarm_harm_s": float(pre),
    }
    for p, w in last.walkers.items():
        out |= {
            f"{p}_harm_s": w.harmful_s,
            f"{p}_dose": w.dose,
            f"{p}_exposed": float(w.harmful_s > 0),
        }
    return out


def _job(args: tuple[Factory, Setting, int]) -> dict[str, float]:
    return run_one(*args)


def run_study(
    f: Factory,
    studies: dict[str, tuple[str, tuple[float, ...], Setting, tuple[str, ...]]],
    seeds: Iterable[int],
    workers: int = 1,
) -> dict[str, dict[str, list[dict[str, float]]]]:
    """Mean scores per study, model setting and value: ``{study: {models: [row, ...]}}``."""
    seeds = list(seeds)
    keys = [
        (name, m, float(v))
        for name, (_, values, _, models) in studies.items()
        for m in models
        for v in values
    ]
    jobs = []
    for name, m, v in keys:
        field, _, base, _ = studies[name]
        s = replace(base, models=m, **{field: v})
        jobs += [(f, s, seed) for seed in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            scores = list(pool.map(_job, jobs, chunksize=4))
    else:
        scores = [_job(j) for j in jobs]
    out: dict[str, dict[str, list[dict[str, float]]]] = {}
    for i, (name, m, v) in enumerate(keys):
        block = scores[i * len(seeds) : (i + 1) * len(seeds)]
        row = {k: float(np.mean([b[k] for b in block])) for k in block[0]}
        out.setdefault(name, {}).setdefault(m, []).append({"value": v, "n": len(seeds), **row})
    return out


def _col(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows])


def plot_timing(res: dict, path: Path) -> Path:
    """Four panels: estimate quality and harm vs alarm time, harm vs update rate and latency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    green, orange, red, grey = "#00a152", "#ef6c00", "#d50000", "#9e9e9e"
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.6))
    base, perf = res["alarm"]["baseline"], res["alarm"]["perfect"]
    x = _col(base, "value")

    a = ax[0, 0]
    a.plot(x, _col(base, "source_error_m"), "o-", c="#1565c0", label="source error (m)")
    a.set_ylabel("source error of the baseline locator (m)", color="#1565c0")
    a2 = a.twinx()
    a2.plot(x, _col(base, "hazard_brier"), "s--", c="#6a1b9a", label="hazard Brier")
    a2.set_ylabel("hazard Brier of the baseline (lower = better)", color="#6a1b9a")
    a.set_title("What waiting buys: estimates from more readings", fontsize=10)

    b = ax[0, 1]
    b.fill_between(x, 0, _col(base, "pre_alarm_harm_s"), color=grey, alpha=0.35,
                   label="in gas before the alarm (nobody moves)")  # fmt: skip
    b.plot(x, _col(base, "nearest_harm_s"), "o:", c=red, label="nearest exit")
    b.plot(x, _col(base, "live_harm_s"), "o-", c=green, label="live, baseline models")
    b.plot(x, _col(perf, "live_harm_s"), "o--", c=green, alpha=0.6, label="live, perfect models")
    b.set_ylabel("seconds in harmful gas per person")
    b.set_title("What waiting costs: harm vs first decision time", fontsize=10)

    c = ax[1, 0]
    for m, ls in (("baseline", "-"), ("perfect", "--")):
        rows = res["update"][m]
        u = _col(rows, "value")
        c.plot(u, _col(rows, "live_harm_s"), "o" + ls, c=green, label=f"live, {m} models")
        c.axhline(_col(rows, "once_harm_s").mean(), c=orange, ls=ls, lw=1,
                  label=f"planned once, {m}")  # fmt: skip
    c.plot(u, _col(res["update"]["baseline"], "nearest_harm_s"), "o:", c=red, label="nearest")
    c.set_xscale("log")
    c.set_xticks(u, [f"{v:g}" for v in u])
    c.set_xlabel("seconds between replans (alarm at 30 s, wind drift 0.15)")
    c.set_ylabel("seconds in harmful gas per person")
    c.set_title("How often to replan", fontsize=10)

    d = ax[1, 1]
    walk, wait = res["latency_walk"]["perfect"], res["latency_wait"]["perfect"]
    lx = _col(walk, "value")
    d.plot(lx, _col(walk, "live_harm_s"), "o-", c=green, label="walk to nearest exit meanwhile")
    d.plot(lx, _col(wait, "live_harm_s"), "s-", c="#00838f", label="wait for the route")
    d.plot(lx, _col(walk, "nearest_harm_s"), "o:", c=red, label="nearest exit, no model")
    d.set_xlabel("model latency: seconds from readings to route (alarm at 30 s)")
    d.set_ylabel("seconds in harmful gas per person")
    d.set_title("How fast the models must answer (perfect models)", fontsize=10)

    for axis in (ax[0, 0], ax[0, 1]):
        axis.set_xlabel("first decision time t_alarm = seconds of readings (t_obs)")
    for axis in ax.flat:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="upper left")
    a2.legend(fontsize=8, loc="upper right")
    n = base[0]["n"]
    fig.suptitle(f"Timing study: one person per leak, {n} leaks (seeds 0-{n - 1}), "
                 "same leaks in every setting", fontsize=11)  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def print_table(res: dict) -> None:
    """One line per study, model setting and value."""
    keys = ["source_error_m", "hazard_brier", "pre_alarm_harm_s"]
    keys += [f"{p}_harm_s" for p in POLICIES] + [f"{p}_exposed" for p in POLICIES]
    print("study         models    value  " + "  ".join(k[:14].rjust(14) for k in keys))
    for name, by_model in res.items():
        for m, rows in by_model.items():
            for r in rows:
                vals = "  ".join(f"{r[k]:14.3f}" for k in keys)
                print(f"{name:13} {m:8} {r['value']:6.0f}  {vals}")


def main() -> int:
    """CLI: ``uv run evac-timing [--leaks 100] [--workers 8] [--out outputs/timing]``."""
    p = argparse.ArgumentParser(description="Harm vs alarm time, update rate and latency.")
    p.add_argument("--leaks", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("outputs/timing"))
    p.add_argument("--plot-only", action="store_true", help="redraw from results.json")
    a = p.parse_args()
    data = a.out / "results.json"
    if a.plot_only:
        res = json.loads(data.read_text())["results"]
    else:
        res = run_study(Factory.default(), STUDIES, range(a.leaks), a.workers)
        a.out.mkdir(parents=True, exist_ok=True)
        meta = {n: [fld, list(v), asdict(b), list(m)] for n, (fld, v, b, m) in STUDIES.items()}
        data.write_text(json.dumps({"studies": meta, "results": res}, indent=1))
    print_table(res)
    print(plot_timing(res, a.out / "timing.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
