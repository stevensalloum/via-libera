"""Summarise frozen validation thresholds on an independent held-out test set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("forward", "fixed", "varied")
DIFFUSIVITIES = (0.2, 0.4, 0.8, 1.6, 3.2)
HORIZONS = (0, 10, 30, 60, 120)
THRESHOLDS = np.arange(101) / 100
SPLIT_SEEDS = {"validation": 20261201, "test": 20261202}


def aggregate(rows, method, horizon=None):
    """Pool confusion counts, excluding cases without a detector alarm."""
    detected = [r for r in rows if r["alarm"] is not None]
    if not detected:
        raise ValueError("no detected cases to score")
    counts = [r[method] if horizon is None else r[method]["horizons"][horizon] for r in detected]
    return {
        "tp": np.sum([c["tp"] for c in counts], axis=0).tolist(),
        "fp": np.sum([c["fp"] for c in counts], axis=0).tolist(),
        "positive": sum(c["positive"] for c in counts),
        "negative": sum(c["negative"] for c in counts),
        "detected": len(detected),
        "total": len(rows),
    }


def metrics(counts, threshold):
    """Recall, false-positive rate and precision for a prespecified threshold."""
    i = round(threshold * 100)
    if i < 0 or i > 100 or abs(i / 100 - threshold) > 1e-8:
        raise ValueError("threshold must be one of the 0.01 grid values")
    i = counts.get("grid", list(range(101))).index(i)
    tp, fp, pos, neg = counts["tp"][i], counts["fp"][i], counts["positive"], counts["negative"]
    return {
        "recall": tp / pos if pos else None,
        "fpr": fp / neg if neg else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "tp": tp,
        "fp": fp,
        "positive": pos,
        "negative": neg,
    }


def interval(rows, method, threshold):
    """Bootstrap episode indices jointly across layouts and spread conditions."""
    i = round(threshold * 100)
    groups = []
    for episode in sorted({r["episode"] for r in rows}):
        group = [r for r in rows if r["episode"] == episode and r["alarm"] is not None]
        if group:
            groups.append(
                [
                    sum(r[method]["tp"][i] for r in group),
                    sum(r[method]["positive"] for r in group),
                    sum(r[method]["fp"][i] for r in group),
                    sum(r[method]["negative"] for r in group),
                ]
            )
    a = np.asarray(groups)
    rng = np.random.default_rng(4881)
    samples = a[rng.integers(len(a), size=(5000, len(a)))].sum(axis=1)
    return {
        "recall": np.quantile(samples[:, 0] / samples[:, 1], [0.025, 0.975]).tolist(),
        "fpr": np.quantile(samples[:, 2] / samples[:, 3], [0.025, 0.975]).tolist(),
    }


def describe(rows, selection, label):
    """Print totals, alarm mismatches, selected vs 0.5, and condition subgroups."""
    counts = {method: aggregate(rows, method) for method in METHODS}
    print(
        label,
        "variants",
        len(rows),
        "detected",
        counts["fixed"]["detected"],
        "clean-alarm source-prior mismatch",
        sum(r["prior_at_clean"] for r in rows),
    )
    for method in METHODS:
        selected = selection["thresholds"].get(method, 0.5)
        for threshold in sorted({selected, 0.5}):
            print(
                label,
                method,
                "threshold",
                threshold,
                json.dumps(metrics(counts[method], threshold)),
                "episode-cluster interval",
                json.dumps(interval(rows, method, threshold)),
            )
    for diffusion in DIFFUSIVITIES:
        group = [r for r in rows if r["diffusion"] == diffusion]
        if not any(r["alarm"] is not None for r in group):
            continue
        print(
            label,
            "diffusivity",
            diffusion,
            "detected",
            sum(r["alarm"] is not None for r in group),
            "/",
            len(group),
            json.dumps(
                {
                    m: metrics(aggregate(group, m), selection["thresholds"][m])
                    for m in ("fixed", "varied")
                }
            ),
        )
    for layout in sorted({r["layout"] for r in rows}):
        group = [r for r in rows if r["layout"] == layout]
        if not any(r["alarm"] is not None for r in group):
            continue
        print(
            label,
            "layout",
            layout,
            "detected",
            sum(r["alarm"] is not None for r in group),
            "/",
            len(group),
            json.dumps(
                {
                    m: metrics(aggregate(group, m), selection["thresholds"][m])
                    for m in ("fixed", "varied")
                }
            ),
        )
    for j, horizon in enumerate(HORIZONS):
        print(
            label,
            "horizon",
            horizon,
            json.dumps(
                {
                    m: metrics(aggregate(rows, m, j), selection["thresholds"][m])
                    for m in ("fixed", "varied")
                }
            ),
        )
    return counts


def draw(splits, selection, output):
    """Draw recall and false-positive tradeoffs from aggregate curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (label, counts) in zip(axes, splits.items(), strict=True):
        for method, colour in (("fixed", "#287e8a"), ("varied", "#d47720")):
            score = counts[method]
            grid = score.get("grid", list(range(101)))
            recall = np.asarray(score["tp"]) / score["positive"]
            fpr = np.asarray(score["fp"]) / score["negative"]
            ax.plot(fpr, recall, color=colour, label=method.capitalize())
            idx = grid.index(round(selection["thresholds"][method] * 100))
            ax.scatter(fpr[idx], recall[idx], marker="o", color=colour, s=60, zorder=4)
            at_half = grid.index(50)
            ax.scatter(fpr[at_half], recall[at_half], marker="x", color=colour, s=55, zorder=4)
        fwd = metrics(counts["forward"], 0.5)
        ax.scatter(
            fwd["fpr"],
            fwd["recall"],
            marker="*",
            color="#526175",
            s=100,
            label="Forward at 0.5",
            zorder=4,
        )
        ax.axhline(0.9, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(label.capitalize())
        ax.set_xlabel("False-positive rate (lower is better)")
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Toxic recall (higher is better)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Threshold trade-offs: alarm-triggered simulation (zoomed)", fontsize=15)
    fig.text(
        0.5,
        0.025,
        f"Circle: selected {selection['thresholds']['fixed']:.2f} fixed / "
        f"{selection['thresholds']['varied']:.2f} varied · cross: 0.50 · dashed: 90% target\n"
        "Simulation only; cells and times are correlated. Curves join sampled thresholds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    """Read the frozen selection and split outputs; optionally write a figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--test", type=Path)
    p.add_argument("--figure", type=Path)
    p.add_argument("--recorded", action="store_true", help="draw saved aggregate curves")
    args = p.parse_args()
    if args.recorded:
        if not args.figure or args.selection or args.validation or args.test:
            p.error("--recorded needs --figure and no input files")
        from threshold_recorded import SELECTION, SPLITS

        draw(SPLITS, SELECTION, args.figure)
        return
    if not args.selection or not args.validation or not args.test:
        p.error("provide --selection, --validation and --test")
    selection = json.loads(args.selection.read_text())
    if hashlib.sha256(args.validation.read_bytes()).hexdigest() != selection["validation_sha256"]:
        p.error("validation results differ from those used to select the threshold")
    splits = {}
    for label, path in (("validation", args.validation), ("test", args.test)):
        manifest = path.with_suffix(path.suffix + ".meta.json")
        metadata = json.loads(manifest.read_text())
        if metadata != {
            "split": label,
            "seed": SPLIT_SEEDS[label],
            "settings": selection["settings"],
        }:
            p.error(f"{label} metadata does not match the frozen design")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows or any(r["split"] != label for r in rows):
            p.error(f"{label} file has incorrect split labels")
        settings = selection["settings"]
        expected = {
            (layout, episode, diffusion)
            for layout in settings["layouts"]
            for episode in range(settings["episodes"])
            for diffusion in DIFFUSIVITIES
        }
        actual = [(r["layout"], r["episode"], r["diffusion"]) for r in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            p.error(f"{label} needs one result for each planned case")
        splits[label] = describe(rows, selection, label)
    if args.figure:
        draw(splits, selection, args.figure)


if __name__ == "__main__":
    main()
