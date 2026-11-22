"""Markdown tables of every result file (``uv run python -m tabpfn_spatial.study3d.summary``).

The report quotes these tables, so each number in it traces back to a JSON in
``outputs/study3d/``.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tabpfn_spatial.study3d.report import _kcal_rows

RES = Path("outputs/study3d")
METRICS = ["rho", "auc_hot", "profile_r", "width_ratio"]


def _mean(rows: list[dict], key: str) -> str:
    v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])])
    return f"{v.mean():.2f}" if v.size else "-"


def ladder_table() -> str:
    """Physics level x dataset (x turbulence for Red:Vapor)."""
    lines = ["| dataset | turbulence | level | AUC hot 10 % | profile r | spread ratio |",
             "|---|---|---|---|---|---|"]  # fmt: skip

    def load(name: str) -> list[dict]:
        return json.loads((RES / f"ladder_{name}.json").read_text())

    sets = [("Red:Vapor", "redvapor", "measured", load("redvapor")),
            ("Red:Vapor", "redvapor", "calibrated", _kcal_rows("redvapor")),
            ("UCI", "uci", "assumed 10 %", load("uci")),
            ("Örebro", "orebro", "measured", load("orebro"))]  # fmt: skip
    for name, _, turb, rows in sets:
        for lv in ["L1", "L2", "L3", "L4", "L4G"]:
            sub = [r for r in rows if r["level"] == lv]
            if sub:
                vals = " | ".join(_mean(sub, m) for m in ("auc_hot", "profile_r", "width_ratio"))
                lines.append(f"| {name} | {turb} | {lv} | {vals} |")
    return "\n".join(lines)


def transfer_table(name: str) -> str:
    """Workflow means of one hazard-transfer result file."""
    p = RES / f"{name}.json"
    if not p.exists():
        return f"({name}: not run)"
    by = defaultdict(list)
    for r in json.loads(p.read_text()):
        by[r["workflow"]].append(r)
    lines = ["| workflow | rho | AUC hot 10 % | profile r | spread ratio | n |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for w, rows in by.items():
        lines.append(
            f"| {w} | " + " | ".join(_mean(rows, m) for m in METRICS) + f" | {len(rows)} |"
        )
    return "\n".join(lines)


def source_table(name: str, key: str = "error_m") -> str:
    """Median error per workflow of a source-task result file."""
    p = RES / f"{name}.json"
    if not p.exists():
        return f"({name}: not run)"
    by = defaultdict(list)
    for r in json.loads(p.read_text()):
        if np.isfinite(r.get(key, np.nan)):
            by[r["workflow"]].append(r[key])
    lines = [f"| workflow | median {key} | max | n |", "|---|---|---|---|"]
    for w, v in by.items():
        lines.append(f"| {w} | {np.median(v):.2f} | {np.max(v):.2f} | {len(v)} |")
    return "\n".join(lines)


def main() -> int:
    """Print every table."""
    print("## Physics ladder\n\n" + ladder_table())
    names = [
        "transfer_cloud_L3_kcal",
        "transfer_cloud_L1_kcal",
        "transfer_baseline_L3_k1",
        "transfer_uci_cloud_L3",
        "transfer_orebro_cloud_L3",
        "transfer_orebro_baseline_L3",
    ]
    for n in names:
        print(f"\n## {n}\n\n" + transfer_table(n))
    for n in ["source_redvapor_cloud", "source_redvapor_baseline"]:
        print(f"\n## {n}\n\n" + source_table(n))
    for n in ["source_srgdm_cloud", "source_srgdm_baseline"]:
        print(f"\n## {n}\n\n" + source_table(n, "median_cells"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
