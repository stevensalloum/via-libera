"""Plot recorded, paired synthetic 3D hazard scores without cloud calls."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from volume3d_recorded import RECORDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(path: Path) -> None:
    """Compare Brier and recall on the same held-out simulated cases."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    labels = ["Forward\nsimulation", "TabPFN"]
    colors = ["#4c8499", "#df872e"]
    briers = [np.array([row[m]["brier"] for row in RECORDS]) for m in ("forward", "tabpfn")]
    for i, values in enumerate(briers):
        axes[0].bar(i, values.mean(), color=colors[i], alpha=0.8)
        axes[0].scatter(np.full(len(values), i), values, s=14, color="#263543", alpha=0.55)
        axes[0].text(i, values.mean() + 0.001, f"{values.mean():.4f}", ha="center")
    axes[0].set_ylabel("Mean toxic Brier score (lower is better)")
    recalls = []
    for m in ("forward", "tabpfn"):
        tp = sum(row[m]["tp"] for row in RECORDS)
        fn = sum(row[m]["fn"] for row in RECORDS)
        recalls.append(100 * tp / (tp + fn))
    bars = axes[1].bar([0, 1], recalls, color=colors)
    axes[1].bar_label(bars, labels=[f"{value:.1f}%" for value in recalls], padding=3)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Toxic-cell recall at 0.50 cutoff")
    for ax in axes:
        ax.set_xticks([0, 1], labels)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Experimental 3D hazard comparison", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "12 synthetic leaks · known 3D source · one toy layout · shared simulator\n"
        "Cells and times are correlated; results are not real-world validation",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Draw the recorded case scores to a chosen figure path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, required=True)
    draw(parser.parse_args().figure)


if __name__ == "__main__":
    main()
