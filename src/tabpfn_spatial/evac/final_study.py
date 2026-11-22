"""The final test: the whole loop in 3D on known and unseen plants (``uv run evac-final``).

Per plant, Rayan's and Ahmad's 3D models are fitted on 3D simulations of *that* plant (a new
site has no real leak data: simulate it, fit, deploy), then ``--leaks`` leaks run through
the loop of ``evac.prepare3d``: 3D gas, sensor-triggered alarm, model updates every 15 s,
and four policies on the same people: ``nearest`` (gas ignored), ``risk``, ``ccrp``
(replanned at every update) and ``once`` (CCRP planned at the alarm only). The ``perfect``
models (true future gas) give the upper bound on the same leaks.

Plants: ``known`` = the default hall and the four templates (L, U, T, campus) the viewer
shows; ``unseen`` = ``--unseen`` random plants per shape with seeds no template uses.

Per run: the policies' scores (seconds in harmful gas per person, % exposed, time to 95 %
out), the alarm time, the locator's horizontal and height error, and the hazard model's
skill at breathing height against the true harm in the next 60 s (Brier, recall and false
alarms at P >= 0.5). Saved as JSON lines (``runs.jsonl``, resumable: a restart never re-pays
a finished run); ``--figures`` draws the summary.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.crowd import CrowdParams, sample_workers
from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_run import AHEAD, run_crowd
from tabpfn_spatial.evac.prepare3d import fit_models3d, prepare3d
from tabpfn_spatial.evac.shapes_study import FIRST_SEED, people_for, plant
from tabpfn_spatial.sim import Factory, sample_scenarios
from tabpfn_spatial.sim.layouts import KINDS, template
from tabpfn_spatial.sim.robots import blocked_factory

T_MAX = 180.0
UPDATE_S = 15.0
KNOWN = ("default", "L-plant", "U-plant", "T-plant", "campus")
POLICIES = (*PLANNERS, "once")
OUT = Path("outputs/final")


def factory(name: str) -> Factory:
    """A known plant by name, or an unseen one as ``kind@seed``."""
    if name == "default":
        return Factory.default()
    if "@" in name:
        kind, seed = name.split("@")
        return plant(kind, int(seed))
    return template(name)


def hazard_skill(prep, f: Factory) -> dict:
    """Danger maps vs the true harm in the next ``AHEAD`` s (open cells, all updates)."""
    m = ~f.walls
    briers, recalls, false = [], [], []
    for t, d in zip(prep.replan_times, prep.danger, strict=True):
        k = int(t)
        truth = prep.harm[k : k + int(AHEAD) + 1].max(axis=0)
        p, y = d[m], truth[m]
        briers.append(float(np.mean((p - y) ** 2)))
        if y.any():
            recalls.append(float(np.mean(p[y] >= 0.5)))
        false.append(float(np.mean((p >= 0.5) & ~y)))
    mean = lambda a: float(np.mean(a)) if a else None  # noqa: E731
    return {"hz_brier": mean(briers), "hz_recall": mean(recalls), "hz_false": mean(false)}


def plant_job(args: tuple[str, tuple[int, ...], str, int]) -> list[dict]:
    """Fit the 3D models on one plant, then every leak x models x policy."""
    name, leaks, backend, n_train = args
    f = factory(name)
    t0 = time.perf_counter()
    fitted = fit_models3d(f, backend, n_train=n_train)
    fit_s = time.perf_counter() - t0
    rows = []
    for seed in leaks:
        for models in ("tabpfn", "perfect"):
            rng = np.random.default_rng(seed)
            sc = sample_scenarios(f, 1, rng)[0]
            t1 = time.perf_counter()
            prep, run = prepare3d(f, sc, models, T_MAX, UPDATE_S, seed,
                                  fitted if models == "tabpfn" else None, backend)  # fmt: skip
            f_run, _ = blocked_factory(f, prep.t_alarm)
            people = sample_workers(f_run, people_for(f), rng)
            log = prep.log
            info = {
                "plant": name, "split": "unseen" if "@" in name else "known", "leak": seed,
                "models": models, "backend": backend if models == "tabpfn" else "-",
                "gas": f.gas, "leak_z": run.leak.xyz[2], "people": len(people),
                "t_alarm": prep.t_alarm, "fit_s": fit_s, "prepare_s": time.perf_counter() - t1,
                "loc_err_first_m": log[0]["source_error_m"] if log else None,
                "loc_err_median_m": float(np.median([u["source_error_m"] for u in log]))
                if log else None,
                "height_err_m": float(np.median([abs(u["height_median"] - u["height_true"])
                                                 for u in log])) if log and models == "tabpfn"
                else None,
                **(hazard_skill(prep, f) if prep.danger else {}),
            }  # fmt: skip
            params = CrowdParams()
            for p in POLICIES:
                r = run_crowd(f_run, prep, people, "ccrp" if p == "once" else p, params,
                              once=p == "once")  # fmt: skip
                rows.append({**info, "policy": p, **r.metrics})
    return rows


def _safe(args: tuple) -> tuple[tuple, list[dict] | str]:
    try:
        return args, plant_job(args)
    except Exception as err:  # one bad plant must not stop the study
        return args, f"{type(err).__name__}: {err}"


def run_study(plants: list[str], leaks: int, backend: str, workers: int, n_train: int,
              log: Path) -> list[dict]:  # fmt: skip
    """One job per plant (models fitted once), each saved to ``log`` when it finishes."""
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            if "rows" in rec:
                done[rec["key"]] = rec["rows"]
    seeds = tuple(2000 + j for j in range(leaks))
    jobs = [(p, seeds, backend, n_train) for p in plants if f"{p}|{leaks}|{backend}" not in done]
    print(f"{len(done)} plants done, {len(jobs)} to go", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max(1, workers)) as pool, log.open("a") as fh:
        for args, res in pool.map(_safe, jobs):
            key = f"{args[0]}|{leaks}|{backend}"
            if isinstance(res, str):
                print(f"FAILED {key}: {res[:300]}", flush=True)
                fh.write(json.dumps({"key": key, "error": res}) + "\n")
            else:
                done[key] = res
                fh.write(json.dumps({"key": key, "rows": res}) + "\n")
                print(f"done {key}", flush=True)
            fh.flush()
    return [r for p in plants for r in done.get(f"{p}|{leaks}|{backend}", [])]


def table(rows: list[dict]) -> str:
    """Mean per split x models x policy, and the model skill per split."""
    out = ["split   models   policy    s in gas  % exposed  95% out s"]
    for split in ("known", "unseen"):
        for models in ("tabpfn", "perfect"):
            for p in POLICIES:
                r = [
                    x for x in rows if (x["split"], x["models"], x["policy"]) == (split, models, p)
                ]
                if not r:
                    continue
                g = lambda k, r=r: np.nanmean([x[k] for x in r])  # noqa: E731
                out.append(f"{split:7} {models:8} {p:8} {g('harm_s_mean'):9.2f} "
                           f"{g('exposed_pct'):10.1f} {g('out_95_s'):10.0f}")  # fmt: skip
    for split in ("known", "unseen"):
        r = [x for x in rows if x["split"] == split and x["models"] == "tabpfn"
             and x["policy"] == "nearest"]  # fmt: skip
        if r:
            g = lambda k, r=r: np.nanmedian([x[k] for x in r if x.get(k) is not None])  # noqa: E731
            out.append(f"{split}: locator {g('loc_err_median_m'):.1f} m (first "
                       f"{g('loc_err_first_m'):.1f} m), height {g('height_err_m'):.2f} m, "
                       f"hazard Brier {g('hz_brier'):.3f}, recall {g('hz_recall'):.2f}, "
                       f"false alarms {g('hz_false'):.3f}, alarm {g('t_alarm'):.0f} s")  # fmt: skip
    return "\n".join(out)


def main() -> int:
    """CLI: ``uv run evac-final --backend cloud --leaks 3 --unseen 2``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="cloud", choices=("cloud", "baseline"))
    p.add_argument("--leaks", type=int, default=3, help="leaks per plant")
    p.add_argument("--unseen", type=int, default=2, help="unseen plants per shape")
    p.add_argument("--only", nargs="*", help="plant names (default: known + unseen)")
    p.add_argument("--n-train", type=int, default=200, help="3D training leaks per model")
    p.add_argument("--workers", type=int, default=2, help="plants in parallel (cloud: 2)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--figures", action="store_true", help="only redraw figures and table")
    a = p.parse_args()
    from tabpfn_spatial.env import load_env

    load_env()
    unseen = [f"{k}@{FIRST_SEED + 50 + i}" for k in KINDS for i in range(a.unseen)]
    plants = a.only or [*KNOWN, *unseen]
    log = a.out / f"runs-{a.backend}.jsonl"
    if a.figures:
        rows = [
            r for line in log.read_text().splitlines() for r in json.loads(line).get("rows", [])
        ]
    else:
        rows = run_study(plants, a.leaks, a.backend, a.workers, a.n_train, log)
    from tabpfn_spatial.evac.final_plots import figures

    text = table(rows)
    (a.out / f"table-{a.backend}.txt").write_text(text + "\n")
    print(text)
    for path in figures(rows, a.out, a.backend):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
