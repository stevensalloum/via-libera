"""Redraw the recorded five-case sensor-history pilot without making model calls."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Save the small journey figure from the reported aggregate pilot values."""
    names = ["Forward", "A: spatial", "B: + time", "C: + sensors"]
    colors = ["#526175", "#007c91", "#65acb9", "#d47c25"]
    metrics = [
        ([0.05806, 0.07700, 0.07679, 0.06681], "Toxic Brier\nlower is better", 0.095),
        ([0.5215, 0.7285, 0.4761, 0.3565], "Recall at 0.5\nhigher is better", 0.9),
        ([0.0209, 0.1021, 0.1037, 0.0334], "False-positive rate at 0.5\nlower is better", 0.13),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1))
    fig.suptitle("Sensor-history pilot: mixed results", fontsize=18, fontweight="bold")
    for ax, (values, label, limit) in zip(axes, metrics, strict=True):
        bars = ax.bar(np.arange(4), values, color=colors, width=0.7)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xticks(np.arange(4), names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(0, limit)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.04,
        0.02,
        "5 simulated leaks · centroid source · 640 context rows · offline k-NN (A/B/C) · "
        "alarm assumed at leak start · no cloud calls",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.77, bottom=0.27, wspace=0.42)
    path = Path("docs/figures/ahmad/01-sensor-history-pilot.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
