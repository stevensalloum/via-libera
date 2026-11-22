"""``uv run evac-runs <run folders...>``: compare saved evac-pipeline runs.

Reads each folder's ``summary.json`` and ``cycles.csv`` (written by ``evac-pipeline``)
and draws one figure: seconds in harmful gas per person and policy, Rayan's leak error
and size estimate run by run, and each model's time per run. Prints a table too. Works
on any mix of runs (conditions, models, backends, leaks).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

POLICIES = ("live", "once", "nearest")
COLOURS = {"live": "#00a152", "once": "#ef6c00", "nearest": "#d50000"}


def load(folder: Path) -> dict:
    """Summary plus the per-run rows of one run folder."""
    s = json.loads((folder / "summary.json").read_text())
    with (folder / "cycles.csv").open() as fh:
        s["rows"] = [{k: float(v) if v not in ("", "None") else np.nan for k, v in r.items()}
                     for r in csv.DictReader(fh)]  # fmt: skip
    s["folder"] = folder.name
    m = s["meta"]
    s["label"] = f"{m.get('condition', 'standard')}\n{m.get('models')}/{m.get('backend')}"
    return s


def table(runs: list[dict]) -> str:
    """Plain-text table: one line per run."""
    lines = ["run                                   alarm  runs  Rayan s  Ahmad s  err m  "
             "size/true   live  once  nearest (s in gas)"]  # fmt: skip
    for r in runs:
        rows = r["rows"]
        loc = np.nanmean([x["locate_s"] for x in rows]) if rows else np.nan
        haz = np.nanmean([x["hazard_s"] for x in rows]) if rows else np.nan
        err = np.nanmean([x["source_error_m"] for x in rows]) if rows else np.nan
        size = rows[-1]["size_est"] if rows else np.nan
        p = r["people"]
        name = r["label"].replace("\n", " ")[:36]
        lines.append(
            f"{name:36}  {r['t_alarm'] or 0:5.0f}  {len(rows):4d}  "
            f"{loc:7.2f}  {haz:7.2f}  "
            f"{err:5.1f}  {size:4.2f}/{r['leak']['size']:4.2f}  {p['live']['harmful_s']:5.0f}  "
            f"{p['once']['harmful_s']:4.0f}  {p['nearest']['harmful_s']:7.0f}"
        )
    return "\n".join(lines)


def plot(runs: list[dict], path: Path) -> Path:
    """Four panels comparing the runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(15, 8.5))
    x = np.arange(len(runs))
    w = 0.27
    a = ax[0, 0]
    for k, p in enumerate(POLICIES):
        a.bar(x + (k - 1) * w, [r["people"][p]["harmful_s"] for r in runs], w, color=COLOURS[p],
              label={"live": "replanned every run", "once": "planned once",
                     "nearest": "nearest exit"}[p])  # fmt: skip
    a.set_xticks(x, [r["label"] for r in runs], fontsize=7)
    a.set_ylabel("seconds in harmful gas")
    a.set_title("Outcome per run and policy", fontsize=10)
    a.legend(fontsize=8)
    b = ax[0, 1]
    for r in runs:
        b.plot([q["run"] for q in r["rows"]], [q["source_error_m"] for q in r["rows"]], "o-",
               label=r["label"].replace("\n", " "))  # fmt: skip
    b.set_xlabel("model run")
    b.set_ylabel("Rayan: leak location error (m)")
    b.set_title("Where is the leak, run by run", fontsize=10)
    b.legend(fontsize=6)
    c = ax[1, 0]
    for r in runs:
        ratio = [q["size_est"] / r["leak"]["size"] for q in r["rows"]]
        c.plot([q["run"] for q in r["rows"]], ratio, "o-", label=r["label"].replace("\n", " "))
    c.axhline(1, c="k", lw=1, ls=":")
    c.set_xlabel("model run")
    c.set_ylabel("Rayan: size estimate / true size")
    c.set_title("How big is the leak, run by run (1 = exact)", fontsize=10)
    d = ax[1, 1]
    d.bar(x - 0.2, [np.nanmean([q["locate_s"] for q in r["rows"]]) for r in runs], 0.4,
          color="#19c3e6", label="Rayan (locate)")  # fmt: skip
    d.bar(x + 0.2, [np.nanmean([q["hazard_s"] for q in r["rows"]]) for r in runs], 0.4,
          color="#ff5252", label="Ahmad (hazard)")  # fmt: skip
    d.set_xticks(x, [r["label"] for r in runs], fontsize=7)
    d.set_ylabel("seconds per call (mean over runs)")
    d.set_title("How long each model takes", fontsize=10)
    d.legend(fontsize=8)
    for axis in ax.flat:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85)
    plt.close(fig)
    return path


def main() -> int:
    """CLI: ``uv run evac-runs outputs/pipeline/* [--out compare.png]``."""
    p = argparse.ArgumentParser(description="Compare saved evac-pipeline runs.")
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/pipeline_compare.png"))
    a = p.parse_args()
    runs = [load(f) for f in a.folders if (f / "summary.json").exists()]
    print(table(runs))
    print(plot(runs, a.out))
    return 0
