"""Plot recorded cloud pilots without making model calls."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Compare time-only and time-plus-sensors on each source estimator."""
    labels = ["Centroid\n+ time", "Centroid\n+ sensors", "Rayan\n+ time", "Rayan\n+ sensors"]
    colors = ["#65acb9", "#d47c25", "#65acb9", "#d47c25"]
    metrics = [
        ([0.04006, 0.03609, 0.03468, 0.03499], "Toxic Brier\nlower is better", 0.055),
        ([0.45096, 0.54306, 0.49163, 0.48804], "Recall at 0.5\nhigher is better", 0.7),
        (
            [0.00803, 0.01131, 0.00792, 0.00655],
            "False-positive rate at 0.5\nlower is better",
            0.016,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1))
    fig.suptitle("Sensor history: pilots with two source models", fontsize=16)
    for ax, (values, label, limit) in zip(axes, metrics, strict=True):
        bars = ax.bar(np.arange(4), values, color=colors, width=0.7)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_xticks(np.arange(4), labels, fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(0, limit)
        ax.axvline(1.5, color="gray", alpha=0.35)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.04,
        0.02,
        "Five simulated leaks per comparison · cloud TabPFN · 8,000 context rows · "
        "small exploratory pilot, not a safety validation",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.4)
    path = Path("docs/figures/ahmad/02-sensor-history-cloud.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
