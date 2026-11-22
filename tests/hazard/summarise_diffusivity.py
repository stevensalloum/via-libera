"""Summarise a paired diffusivity study and draw its two presentation panels."""

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = (
    ("forward", "Forward simulation", "#526175"),
    ("fixed", "TabPFN: fixed training", "#70aeba"),
    ("varied", "TabPFN: varied training", "#da8023"),
)
VALUES = (0.2, 0.4, 0.8, 1.6, 3.2)
# Brier and recall from the documented 2026-11-22 cloud run, in VALUES order.
RECORDED_VALUES = (
    (
        (0.057243258893242485, 0.7694635775412231),
        (0.04173201245941137, 0.6143811312878313),
        (0.0418987341517032, 0.5914214151534126),
    ),
    (
        (0.047809396050879194, 0.7947019867549668),
        (0.03977243914377806, 0.6642752023546725),
        (0.04036818614677124, 0.635485651214128),
    ),
    (
        (0.04030195525873882, 0.8387917590084544),
        (0.040759522939013526, 0.6895674300254453),
        (0.04182694584339082, 0.6617417713206928),
    ),
    (
        (0.04064517747881642, 0.8364735985112817),
        (0.04155333020406827, 0.7148949368070093),
        (0.04155242913488211, 0.6924090873846631),
    ),
    (
        (0.0648389082977861, 0.8230435152857021),
        (0.05020458349863481, 0.7272417610491356),
        (0.04791141958670206, 0.7232393766499191),
    ),
)
RECORDED = tuple(
    {
        "n": 40,
        **{
            method[0]: {"brier": pair[0], "recall": pair[1]}
            for method, pair in zip(METHODS, row, strict=True)
        },
    }
    for row in RECORDED_VALUES
)


def aggregate(rows):
    """Mean per-episode Brier and pooled cell-time recall and false-positive rate."""
    result = {"n": len(rows), "alarmed": sum(row["source_alarmed"] for row in rows)}
    for key, _, _ in METHODS:
        scores = [r[key] for r in rows]
        pos, neg = sum(s["positive"] for s in scores), sum(s["negative"] for s in scores)
        result[key] = {
            "brier": float(np.mean([s["brier"] for s in scores])),
            "recall": sum(s["tp"] for s in scores) / pos if pos else float("nan"),
            "fpr": sum(s["fp"] for s in scores) / neg if neg else float("nan"),
        }
    return result


def paired_interval(rows):
    """Bootstrap paired episode groups jointly across the fixed layouts."""
    rng = np.random.default_rng(7761)
    layouts = sorted({row["layout"] for row in rows})
    differences = []
    for episode in sorted({r["episode"] for r in rows}):
        group = [r for r in rows if r["episode"] == episode]
        if sorted(r["layout"] for r in group) != layouts:
            raise ValueError("expected one result per layout for each paired episode")
        differences.append(np.mean([r["varied"]["brier"] - r["fixed"]["brier"] for r in group]))
    diff = np.asarray(differences)
    means = diff[rng.integers(len(diff), size=(5000, len(diff)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def draw(summaries, output, n_layouts):
    """Draw recorded Brier and recall; the bars compare the same simulated cases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x, width = np.arange(len(VALUES)), 0.25
    for i, (key, label, color) in enumerate(METHODS):
        for ax, metric in zip(axes, ("brier", "recall"), strict=True):
            ax.bar(
                x + (i - 1) * width,
                [s[key][metric] for s in summaries],
                width,
                label=label,
                color=color,
            )
    axes[0].set_ylabel("Toxic Brier score\n(lower is better)")
    axes[1].set_ylabel("Toxic recall at 0.5\n(higher is better)")
    axes[1].set_xticks(x, [f"{v}\nn={s['n']}" for v, s in zip(VALUES, summaries, strict=True)])
    axes[1].set_xlabel("Test diffusivity (m²/s); training range 0.4-1.6")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
    fig.suptitle("Does training across spreading conditions help?", fontsize=16)
    fig.text(
        0.04,
        0.01,
        f"Simulation only · {n_layouts} layouts · cases repeated across "
        "5 conditions · same simulator for context and truth",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.85, bottom=0.12, hspace=0.23)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    """Print summaries, optionally drawing from JSONL or recorded aggregates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--recorded", action="store_true")
    args = parser.parse_args()
    if args.recorded:
        if args.input or not args.figure or not RECORDED:
            parser.error("--recorded requires a figure and recorded results")
        summaries = RECORDED
        n_layouts = 2
    else:
        if not args.input:
            parser.error("provide a results file")
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
        n_layouts = len({r["layout"] for r in rows})
        summaries = []
        for value in VALUES:
            group = [r for r in rows if r["diffusion"] == value]
            if not group:
                parser.error(f"missing diffusivity {value}")
            summary = aggregate(group)
            summaries.append(summary)
            print(
                value,
                json.dumps(summary),
                "varied-minus-fixed Brier 95% interval",
                paired_interval(group),
            )
    if args.figure:
        draw(summaries, args.figure, n_layouts)


if __name__ == "__main__":
    main()
