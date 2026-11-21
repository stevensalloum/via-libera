"""Scenario ladder: which crowd planner is best as plants and crowds grow.

``uv run evac-crowd`` runs every rung of ``RUNGS`` on the same leak seeds with the three
planners (``crowd_plan.PLANNERS``) and two model settings (``perfect``: true leak and true
future, isolates planning; ``baseline``: today's centroid + forward-simulation models),
and draws ``outputs/crowd/ladder.png`` plus ``plants.png``.

Rungs, from simple to hard: the 60 x 40 m hall with 50 then 300 people, the same with
narrow 0.8 m doors, a 120 x 80 m plant with 600 people and 8 exits, a 200 x 120 m plant
with 1,200 people and 8 exits. Workers stand mostly near machines; every leak starts on
a pipe; the detector alarm (sensors every 5 s) starts the evacuation; plans are redone
every ``update_s`` seconds.

Scores per run (``crowd_run``): mean and 95th-percentile seconds in harmful gas, % of
people exposed, time until 95 % are out, mean seconds queueing, people still inside at
``t_max``. ``aggregate`` adds the paired difference to the nearest-exit plan (same leak,
same people) with its standard error. Cost: about 0.3 s (A) to 2 min (E, baseline) per
leak; runs in parallel processes.
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.crowd import CrowdParams, sample_workers
from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_run import prepare, run_crowd
from tabpfn_spatial.sim import Factory, sample_scenarios

METRICS = ("harm_s_mean", "harm_s_p95", "exposed_pct", "out_95_s", "queued_s_mean",
           "still_inside", "t_alarm")  # fmt: skip


@dataclass(frozen=True)
class Rung:
    """One scenario: a plant, a crowd, door width and timing."""

    name: str
    label: str
    people: int
    width: float | None = None  # None: Factory.default()
    height: float = 40.0
    n_machines: int = 6
    n_hot: int = 3
    exits_per_side: int = 1
    sensor_spacing: float | None = None
    plant_seed: int = 0
    door_m: float = 1.0
    t_max: float = 400.0
    update_s: float = 15.0

    def plant(self) -> Factory:
        """The rung's factory (fixed per rung; leaks and crowds vary with the seed)."""
        if self.width is None:
            return Factory.default()
        return Factory.random(
            np.random.default_rng(self.plant_seed), self.width, self.height, 2.0,
            self.n_machines, self.n_hot, self.exits_per_side, self.sensor_spacing,
        )  # fmt: skip


RUNGS = (
    Rung("A", "60x40 m, 50 people", 50, t_max=300.0),
    Rung("B", "60x40 m, 300 people", 300),
    Rung("C", "60x40 m, 300 people, 0.8 m doors", 300, door_m=0.8),
    Rung("D", "120x80 m, 600 people, 8 exits", 600, 120.0, 80.0, 10, 4, 2, 15.0, 7,
         t_max=500.0),
    Rung("E", "200x120 m, 1200 people, 8 exits", 1200, 200.0, 120.0, 16, 6, 2, 15.0, 11,
         t_max=700.0, update_s=20.0),
)  # fmt: skip


def _job(args: tuple[Rung, int, str]) -> list[dict]:
    """All planners on one (rung, leak seed, models): same leak, alarm, maps and people."""
    rung, seed, models = args
    f = rung.plant()
    rng = np.random.default_rng(seed)
    sc = sample_scenarios(f, 1, rng)[0]
    people = sample_workers(f, rung.people, rng)
    prep = prepare(f, sc, models, rung.t_max, rung.update_s, seed)
    params = CrowdParams(door_m=rung.door_m)
    return [
        {"rung": rung.name, "seed": seed, "models": models, "planner": p,
         **run_crowd(f, prep, people, p, params).metrics}
        for p in PLANNERS
    ]  # fmt: skip


def run_ladder(
    rungs: Sequence[Rung],
    seeds: Iterable[int],
    models: Sequence[str] = ("perfect", "baseline"),
    workers: int = 1,
) -> list[dict]:
    """One row per rung x models x leak seed x planner, in that order."""
    jobs = [(r, s, m) for r in rungs for m in models for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_job, jobs))
    else:
        parts = [_job(j) for j in jobs]
    return [row for part in parts for row in part]


def aggregate(rows: list[dict]) -> dict[tuple[str, str, str], dict[str, float]]:
    """Means per (rung, models, planner) and paired differences to ``nearest`` +/- SE."""
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    keys = sorted({(r["rung"], r["models"], r["planner"]) for r in rows})
    for rung, models, planner in keys:
        mine = sorted((r for r in rows if (r["rung"], r["models"], r["planner"]) ==
                       (rung, models, planner)), key=lambda r: r["seed"])  # fmt: skip
        base = sorted((r for r in rows if (r["rung"], r["models"], r["planner"]) ==
                       (rung, models, "nearest")), key=lambda r: r["seed"])  # fmt: skip
        agg: dict[str, float] = {"n": float(len(mine))}
        for m in METRICS:
            v = np.array([r[m] for r in mine], dtype=float)
            agg[m] = float(np.nanmean(v))
            d = v - np.array([r[m] for r in base], dtype=float)
            agg[f"{m}_vs_nearest"] = float(np.nanmean(d))
            agg[f"{m}_vs_nearest_se"] = float(np.nanstd(d) / np.sqrt(max(1, np.sum(~np.isnan(d)))))
        out[(rung, models, planner)] = agg
    return out


COLOURS = {"nearest": "#d50000", "risk": "#ffa000", "ccrp": "#00a152"}


def plot_ladder(rows: list[dict], path: Path) -> Path:
    """Rows = model settings, columns = scores, x = rungs, bars = planners (+/- SE)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate(rows)
    rungs = sorted({r["rung"] for r in rows})
    model_rows = [m for m in ("perfect", "baseline") if any(r["models"] == m for r in rows)]
    panels = [("harm_s_mean", "seconds in harmful gas per person"),
              ("exposed_pct", "% of people exposed"),
              ("out_95_s", "time until 95 % are out (s since leak)"),
              ("queued_s_mean", "seconds queueing per person")]  # fmt: skip
    size = (4.6 * len(panels), 3.6 * len(model_rows))
    fig, axes = plt.subplots(len(model_rows), len(panels), figsize=size, squeeze=False)
    x = np.arange(len(rungs))
    w = 0.8 / len(PLANNERS)
    for i, models in enumerate(model_rows):
        for j, (metric, label) in enumerate(panels):
            ax = axes[i, j]
            for k, p in enumerate(PLANNERS):
                vals = [agg.get((r, models, p), {}).get(metric, np.nan) for r in rungs]
                se = [agg.get((r, models, p), {}).get(f"{metric}_vs_nearest_se", 0) for r in rungs]
                ax.bar(x + (k - 1) * w, vals, w, yerr=se if p != "nearest" else None,
                       color=COLOURS[p], label=p, capsize=2)  # fmt: skip
            ax.set_xticks(x, rungs)
            ax.set_title(f"{label}\n({models} models)", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    n = int(max(a["n"] for a in agg.values()))
    names = {r.name: r.label for r in RUNGS}
    fig.suptitle(
        f"Crowd planners on {n} leaks per rung (same leaks and people for every planner; "
        "error bars = SE of the paired difference to nearest)\n"
        + "   ".join(f"{k}: {names.get(k, k)}" for k in rungs),
        fontsize=9,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def plot_plants(rungs: Sequence[Rung], path: Path, seed: int = 0) -> Path:
    """Each rung's plant with pipes, sensors, exits and a sample of its workers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tabpfn_spatial.sim.viz import _draw_plant, draw_pipes

    fig, axes = plt.subplots(1, len(rungs), figsize=(7 * len(rungs), 5), squeeze=False)
    for ax, rung in zip(axes[0], rungs, strict=True):
        f = rung.plant()
        _draw_plant(ax, f, labels=False)
        draw_pipes(ax, f)
        xy = sample_workers(f, rung.people, np.random.default_rng(seed))
        ax.scatter(*xy.T, s=3, c="#2e7d32", alpha=0.6, zorder=6)
        ax.set_title(f"{rung.name}: {rung.label}; {len(f.sensors)} sensors, "
                     f"{len(f.leak_sites)} leak sites", fontsize=9)  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def print_table(rows: list[dict]) -> None:
    """Means and paired differences to nearest, one line per rung x models x planner."""
    agg = aggregate(rows)
    print("rung models   planner  alarm  harm_s (vs nearest)    exposed%  out95_s  queue_s  inside")
    for (rung, models, p), a in sorted(agg.items()):
        print(
            f"{rung:4} {models:8} {p:8} {a['t_alarm']:5.0f}  {a['harm_s_mean']:6.2f} "
            f"({a['harm_s_mean_vs_nearest']:+6.2f}+/-{a['harm_s_mean_vs_nearest_se']:4.2f})  "
            f"{a['exposed_pct']:7.1f}  {a['out_95_s']:7.0f}  {a['queued_s_mean']:7.1f}  "
            f"{a['still_inside']:6.1f}"
        )


def main() -> int:
    """CLI: ``uv run evac-crowd [--leaks 20] [--workers 10] [--rungs A B ...]``."""
    p = argparse.ArgumentParser(description="Crowd planners on a ladder of scenarios.")
    p.add_argument("--leaks", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rungs", nargs="+", default=[r.name for r in RUNGS])
    p.add_argument("--models", nargs="+", default=["perfect", "baseline"])
    p.add_argument("--out", type=Path, default=Path("outputs/crowd"))
    p.add_argument("--plot-only", action="store_true", help="redraw from results.json")
    a = p.parse_args()
    data = a.out / "results.json"
    rungs = [r for r in RUNGS if r.name in a.rungs]
    if a.plot_only:
        rows = json.loads(data.read_text())["rows"]
    else:
        rows = run_ladder(rungs, range(a.leaks), a.models, a.workers)
        a.out.mkdir(parents=True, exist_ok=True)
        meta = {"rungs": [asdict(r) for r in rungs], "leaks": a.leaks, "models": a.models}
        data.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print_table(rows)
    print(plot_ladder(rows, a.out / "ladder.png"))
    print(plot_plants(rungs, a.out / "plants.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
