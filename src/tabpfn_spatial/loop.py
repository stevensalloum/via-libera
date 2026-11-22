"""``uv run closed-loop``: one gas emergency through every part of the system.

Gas detector alarm -> Rayan's TabPFN locator (where, and how big) -> Ahmad's TabPFN hazard
forecast (now, +30 s, +60 s and beyond) -> crowd planners (nearest exit, risk-aware,
capacity-aware) -> every person scored on the true gas. Prints what each part said at
every update and how long it took; ``--still`` / ``--gif`` draw the three planners.

``--backend baseline`` (default) runs the team's models with the offline k-NN stand-in
instead of TabPFN: free, and checks that everything fits together. ``--backend cloud``
calls TabPFN (about 3 calls of >= 10,000 tokens per update); it asks for ``--yes`` first.
Integrator-owned.
"""

import argparse
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.crowd import CrowdParams, sample_workers
from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_run import make_models, prepare, run_crowd
from tabpfn_spatial.evac.crowd_study import find_rung
from tabpfn_spatial.sim import sample_scenarios


def run(rung_name: str, seed: int, backend: str, record: bool = False):
    """(prepared, results per planner) for one emergency with the team's TabPFN models."""
    rung = find_rung(rung_name)
    f = rung.plant()
    rng = np.random.default_rng(seed)
    sc = sample_scenarios(f, 1, rng)[0]
    people = sample_workers(f, rung.people, rng)
    fitted = make_models(f, sc, "tabpfn", backend)
    prep = prepare(f, sc, "tabpfn", rung.t_max, rung.update_s, seed, fitted=fitted)
    params = CrowdParams(door_m=rung.door_m)
    results = {p: run_crowd(f, prep, people, p, params, record_every=1 if record else None)
               for p in PLANNERS}  # fmt: skip
    return rung, f, prep, results


def main() -> int:
    """CLI: ``uv run closed-loop [--rung B] [--seed 500] [--backend baseline] [--still f]``."""
    p = argparse.ArgumentParser(description="One gas emergency through the whole system.")
    p.add_argument("--rung", default="B", help="scenario A-E (see evac-crowd)")
    p.add_argument("--seed", type=int, default=500, help="leak seed (500+: held out)")
    p.add_argument("--backend", default="baseline", choices=("baseline", "cloud"))
    p.add_argument("--yes", action="store_true", help="allow cloud calls (spends tokens)")
    p.add_argument("--still", type=Path)
    p.add_argument("--gif", type=Path)
    a = p.parse_args()
    if a.backend == "cloud" and not a.yes:
        rung = find_rung(a.rung)
        updates = int(rung.t_max / rung.update_s)
        print(f"cloud run: up to {updates} updates x ~3 TabPFN calls >= {updates * 30_000:,} "
              "tokens. Re-run with --yes to spend them.")  # fmt: skip
        return 1
    rung, f, prep, results = run(a.rung, a.seed, a.backend, record=bool(a.still or a.gif))
    sc = prep.scenario
    print(f"{rung.name}: {rung.label} | leak at ({sc.source[0]:.0f}, {sc.source[1]:.0f}) m, "
          f"size {sc.rate:.2f} | alarm at {prep.t_alarm} s | backend {a.backend}")  # fmt: skip
    print("  t (s)  locate s  hazard s  source error m  leak size est / true")
    for row in prep.log[:8]:
        print(
            f"  {row['t']:5.0f}  {row['locate_s']:8.2f}  {row['predict_s']:8.2f}  "
            f"{row['source_error_m']:14.1f}  {row['rate_median']:6.2f} / {row['rate_true']:.2f}"
        )
    print("  planner   harm s/person  exposed %  95 % out (s)  queue s/person")
    for name, r in results.items():
        m = r.metrics
        print(
            f"  {name:8} {m['harm_s_mean']:14.2f}  {m['exposed_pct']:9.1f}  "
            f"{m['out_95_s']:12.0f}  {m['queued_s_mean']:14.1f}"
        )
    if a.still or a.gif:
        from tabpfn_spatial.evac.crowd_viewer import Recording, save_gif, save_still

        rec = Recording(rung, a.seed, f"tabpfn ({a.backend})", f, prep,
                        {k: v.frames for k, v in results.items()},
                        {k: v.metrics for k, v in results.items()})  # fmt: skip
        if a.still:
            print(save_still(rec, a.still))
        if a.gif:
            print(save_gif(rec, a.gif))
    return 0
