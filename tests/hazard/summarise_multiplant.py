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

# Recorded condition aggregates, in CONDITIONS order. These are summary
# statistics for redrawing the published figure, not the simulated case data.
RECORDED = (
    (28, (0.0271871131, 0.0582493883, 0.0362785791), (0.8989855846, 0.5642285104, 0.7195942338)),
    (25, (0.0410684476, 0.0598811571, 0.0407986737), (0.8492053560, 0.5662620448, 0.7095482418)),
    (25, (0.0466447142, 0.0601433956, 0.0415089293), (0.8117160060, 0.5229553437, 0.6705970898)),
    (19, (0.0331963457, 0.0535891598, 0.0375499130), (0.8717855189, 0.5785154113, 0.6791902243)),
    (10, (0.0320000000, 0.0449186446, 0.0274450426), (0.8810013717, 0.7630315501, 0.8779149520)),
    (10, (0.0491650485, 0.0500068348, 0.0367742999), (0.7739492616, 0.6717152594, 0.7440363499)),
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
    draw_figure(summaries, len({r["layout"] for r in rows}), output)


def make_recorded_figure(output):
    """Redraw the committed figure from the documented summary statistics."""
    summaries = []
    for count, briers, recalls in RECORDED:
        row = {"n": count}
        for (name, _, _), brier, recall in zip(METHODS, briers, recalls, strict=True):
            row[name] = {"brier": brier, "recall": recall}
        summaries.append(row)
    draw_figure(summaries, 2, output)


def draw_figure(summaries, n_layouts, output):
    """Draw mean Brier and pooled recall for each recorded condition."""
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)
    x = np.arange(len(summaries))
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
        [f"{title}\nn={s['n']}" for (_, title), s in zip(CONDITIONS, summaries, strict=True)],
    )
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
    fig.suptitle("Paired hazard forecasts under simulated changes", fontsize=17)
    fig.text(
        0.03,
        0.015,
        f"{sum(s['n'] for s in summaries)} case variations · {n_layouts} layouts, uneven groups · "
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
    parser.add_argument("input", type=Path, nargs="*")
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--recorded", action="store_true", help="draw the recorded aggregates")
    args = parser.parse_args()
    if args.recorded:
        if args.input or not args.figure:
            parser.error("--recorded requires --figure and no input files")
        make_recorded_figure(args.figure)
        return
    if not args.input:
        parser.error("provide input files or --recorded with --figure")
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
