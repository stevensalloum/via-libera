"""Unseen-shapes test: how the planners and the team's models do on new factory shapes.

``uv run evac-shapes`` draws ``--plants`` random plants of every shape (L, U, T, campus) with
seeds no template uses (``FIRST_SEED`` on), so every plant is new: its outline, rooms,
machines, pipes, detectors, exits and AGV lanes. On each plant:

- ``--leaks`` leaks with the ``perfect`` models (true leak and true future: isolates the
  planning) and the ``baseline`` models (centroid locator + forward simulation);
- ``--tabpfn-leaks`` leaks with the team's models (Rayan's locator, Ahmad's hazard
  predictor), fitted on simulated data of *that* plant, as they would be on a new site
  (``--backend baseline`` = offline k-NN stand-in, ``cloud`` = TabPFN);
- every planner (nearest exit, risk-aware, CCRP) on the same leak, alarm, model maps and
  people; the AGVs stop at the alarm and block (``sim.robots``).

People: one worker per ``M2_PER_PERSON`` m^2 of floor. Scores per run: seconds in harmful gas
per person (mean, 95th percentile), % exposed, time until 95 % are out, people still inside
at ``T_MAX``, the alarm time, and the locator's error (first update, median after it).
Results: ``outputs/shapes/results.json``, figures ``outputs/shapes/*.png``.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.crowd import CrowdParams, sample_workers
from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_run import make_models, prepare, run_crowd
from tabpfn_spatial.sim import Factory, sample_scenarios
from tabpfn_spatial.sim.layouts import KINDS, shaped
from tabpfn_spatial.sim.robots import blocked_factory

FIRST_SEED = 100  # templates use seeds 1-4: every plant here is unseen
M2_PER_PERSON = 20.0
T_MAX = 240.0  # s; evacuations of these plants finish in 1-2 min
UPDATE_S = 15.0
SIZES = {"L": (90.0, 70.0), "U": (100.0, 70.0), "T": (90.0, 80.0), "campus": (120.0, 70.0)}
METRICS = ("harm_s_mean", "harm_s_p95", "exposed_pct", "out_95_s", "still_inside")


@lru_cache(maxsize=8)
def plant(kind: str, seed: int) -> Factory:
    """The unseen plant ``(kind, seed)`` (cached per worker process)."""
    w, h = SIZES[kind]
    return shaped(np.random.default_rng(seed), kind, w, h, name=f"{kind}@{seed}")


def people_for(f: Factory) -> int:
    """Crowd size from the open floor area."""
    return round(float((~f.walls).sum()) * f.cell**2 / M2_PER_PERSON)


_FITTED: dict = {}


def job(args: tuple[str, int, int, str, str]) -> list[dict]:
    """Every planner on one (plant, leak, models)."""
    kind, plant_seed, leak_seed, models, backend = args
    f = plant(kind, plant_seed)
    rng = np.random.default_rng(leak_seed)
    sc = sample_scenarios(f, 1, rng)[0]
    fitted = None
    if models == "tabpfn":
        key = (kind, plant_seed, backend)
        if key not in _FITTED:
            _FITTED[key] = make_models(f, sc, "tabpfn", backend)
        fitted = _FITTED[key]
    t0 = time.perf_counter()
    prep = prepare(f, sc, models, T_MAX, UPDATE_S, leak_seed, fitted=fitted, backend=backend)
    f_run, stops = blocked_factory(f, prep.t_alarm)
    people = sample_workers(f_run, people_for(f), rng)
    errs = [u["source_error_m"] for u in prep.log]
    info = {
        "kind": kind, "plant_seed": plant_seed, "leak_seed": leak_seed, "models": models,
        "backend": backend if models == "tabpfn" else "-", "people": len(people),
        "exits": len(f.exits), "area_m2": float((~f.outside).sum() * f.cell**2),
        "agv_blocks": sum(s["blocks"] for s in stops), "t_alarm": prep.t_alarm,
        "loc_err_first_m": errs[0] if errs else None,
        "loc_err_median_m": float(np.median(errs[1:])) if len(errs) > 1 else None,
        "prepare_s": time.perf_counter() - t0,
    }  # fmt: skip
    params = CrowdParams()
    return [{**info, "planner": p, **run_crowd(f_run, prep, people, p, params).metrics}
            for p in PLANNERS]  # fmt: skip


def run_study(plants: int, leaks: int, tabpfn_leaks: int, backend: str, workers: int) -> list[dict]:
    """All jobs; the slow (tabpfn) ones first so the pool stays busy."""
    specs = [(k, FIRST_SEED + i) for k in KINDS for i in range(plants)]
    jobs = [(k, s, 1000 + j, "tabpfn", backend) for k, s in specs for j in range(tabpfn_leaks)]
    jobs += [(k, s, 1000 + j, m, backend) for m in ("perfect", "baseline") for k, s in specs
             for j in range(leaks)]  # fmt: skip
    rows: list[dict] = []
    with ProcessPoolExecutor(workers) as pool:
        for k, part in enumerate(pool.map(job, jobs)):
            rows += part
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                print(f"{k + 1}/{len(jobs)} runs", flush=True)
    return rows


def paired(rows: list[dict], models: str, planner: str, metric: str, by: str | None = None):
    """Per group: mean of ``planner`` - ``nearest`` on the same runs, its SE, and n."""
    out = {}
    groups = sorted({r[by] for r in rows}) if by else [None]

    def pick(p: str, g: object) -> dict:
        return {
            (r["kind"], r["plant_seed"], r["leak_seed"]): r[metric]
            for r in rows
            if r["models"] == models and r["planner"] == p and (g is None or r[by] == g)
        }

    for g in groups:
        a, b = pick(planner, g), pick("nearest", g)
        d = np.array([a[k] - b[k] for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])])
        if d.size:
            out[g] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1
                      else float("nan"), int(d.size))  # fmt: skip
    return out


def table(rows: list[dict]) -> str:
    """Markdown: means per models x planner over all unseen plants, with the paired gain."""
    lines = ["| models | planner | s in gas / person | vs nearest | exposed % | 95 % out (s) | "
             "still inside | runs |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for models in ("perfect", "baseline", "tabpfn"):
        for p in PLANNERS:
            sub = [r for r in rows if r["models"] == models and r["planner"] == p]
            if not sub:
                continue
            m = {k: float(np.nanmean([r[k] for r in sub])) for k in METRICS}
            d = paired(rows, models, p, "harm_s_mean").get(None, (0.0, 0.0, 0))
            gain = "-" if p == "nearest" else f"{d[0]:+.2f} +/- {d[1]:.2f}"
            lines.append(f"| {models} | {p} | {m['harm_s_mean']:.2f} | {gain} | "
                         f"{m['exposed_pct']:.1f} | {m['out_95_s']:.0f} | "
                         f"{m['still_inside']:.1f} | {len(sub)} |")  # fmt: skip
    return "\n".join(lines)


def main() -> int:
    """CLI: ``uv run evac-shapes [--plants 6] [--leaks 4] [--tabpfn-leaks 1] [--workers 4]``."""
    p = argparse.ArgumentParser(description="Planners and models on unseen factory shapes.")
    p.add_argument("--plants", type=int, default=6, help="unseen plants per shape")
    p.add_argument("--leaks", type=int, default=4, help="leaks per plant (perfect, baseline)")
    p.add_argument("--tabpfn-leaks", type=int, default=1, help="leaks per plant, team models")
    p.add_argument("--backend", default="baseline", choices=("baseline", "cloud"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("outputs/shapes"))
    p.add_argument("--plot-only", action="store_true", help="redraw from results.json")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "results.json"
    if a.plot_only:
        rows = json.loads(path.read_text())["rows"]
    else:
        rows = run_study(a.plants, a.leaks, a.tabpfn_leaks, a.backend, a.workers)
        path.write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()},
                                    "rows": rows}, indent=1))  # fmt: skip
    from tabpfn_spatial.evac.shapes_plots import plot_all

    print(table(rows))
    for f in plot_all(rows, a.out):
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
