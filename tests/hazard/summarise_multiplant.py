"""Summarise paired multi-plant results and redraw their presentation figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = (
    ("standard", "Standard"),
    ("wind_drift", "Changing\nwind"),
    ("sensor_faults", "Sensor\nfaults"),
    ("slower_mixing", "Slower\nmixing"),
    ("faster_mixing", "Faster\nmixing"),
    ("combined_stress", "Combined\nstress"),
)
METHODS = (
    ("forward", "Forward simulation", "#526175"),
    ("centroid_context", "TabPFN: centroid context", "#70aeba"),
    ("rayan_context", "TabPFN: Rayan context", "#da8023"),
)


def interval(difference):
    """Exploratory paired percentile interval over episodes at fixed layouts."""
    values = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(7761)
    draws = rng.integers(len(values), size=(5000, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975])


def aggregate(rows):
    """Mean case Brier, pooled binary counts, exposure and source diagnostics."""
    metrics = {}
    for name, _, _ in METHODS:
        points = [r[name] for r in rows]
        positive = sum(p["positive"] for p in points)
        negative = sum(p["negative"] for p in points)
        metrics[name] = {
            "brier": float(np.mean([p["brier"] for p in points])),
            "recall": sum(p["tp"] for p in points) / positive if positive else float("nan"),
            "fpr": sum(p["fp"] for p in points) / negative if negative else float("nan"),
            "exposed": sum(p["exposed"] for p in points),
        }
    metrics["n"] = len(rows)
    metrics["nearest_exposed"] = sum(r["nearest_exposed"] for r in rows)
    metrics["source_error_m"] = float(np.mean([r["source_error_m"] for r in rows]))
    metrics["source_top3_accuracy"] = float(
        np.mean([r["source_top3_contains_truth"] for r in rows])
    )
    metrics["source_activation"] = float(np.mean([r["source_activated"] for r in rows]))
    metrics["rate_abs_relative_error"] = float(
        np.mean([abs(r["estimated_rate"] - r["true_rate"]) / r["true_rate"] for r in rows])
    )
    return metrics


def make_figure(rows, output):
    """Plot Brier and threshold recall; results are from recorded cases, not live calls."""
    groups = [[r for r in rows if r["condition"] == name] for name, _ in CONDITIONS]
    if any(not group for group in groups):
        raise ValueError("the figure requires at least one case in every condition")
    summaries = [aggregate(group) for group in groups]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)
    x = np.arange(len(groups))
    width = 0.25
    for i, (name, label, color) in enumerate(METHODS):
        offset = (i - 1) * width
        axes[0].bar(
            x + offset, [s[name]["brier"] for s in summaries], width, label=label, color=color
        )
        axes[1].bar(
            x + offset, [s[name]["recall"] for s in summaries], width, label=label, color=color
        )
    axes[0].set_ylabel("Toxic Brier score\n(lower is better)")
    axes[1].set_ylabel("Toxic recall at 0.5\n(higher is better)")
    axes[1].set_xticks(
        x,
        [f"{title}\nn={len(group)}" for (_, title), group in zip(CONDITIONS, groups, strict=True)],
    )
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
    n_layouts = len({r["layout"] for r in rows})
    fig.suptitle("Paired hazard forecasts under simulated changes", fontsize=17)
    fig.text(
        0.03,
        0.015,
        f"{len(rows)} case variations · {n_layouts} layouts, uneven groups · "
        "same simulator for context and testing · correlated cells/times",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.84, bottom=0.13, hspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print("figure", output)


def main():
    """Print a deterministic summary and optionally save a figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line) for path in args.input for line in path.read_text().splitlines() if line
    ]
    for name, _ in CONDITIONS:
        group = [r for r in rows if r["condition"] == name]
        if not group:
            continue
        stats = aggregate(group)
        print("condition", name, json.dumps(stats), flush=True)
        for method in ("centroid_context", "rayan_context"):
            differences = [r[method]["brier"] - r["forward"]["brier"] for r in group]
            print(
                "paired Brier",
                name,
                method,
                "minus forward",
                round(float(np.mean(differences)), 6),
                "fixed-layout episode bootstrap 95%",
                np.round(interval(differences), 6).tolist(),
                flush=True,
            )
    for layout in sorted({r["layout"] for r in rows}):
        print(
            "layout",
            layout,
            json.dumps(aggregate([r for r in rows if r["layout"] == layout])),
            flush=True,
        )
    if args.figure:
        make_figure(rows, args.figure)


if __name__ == "__main__":
    main()
