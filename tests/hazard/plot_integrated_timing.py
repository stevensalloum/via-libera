"""Redraw the recorded, exploratory integrated timing comparison."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from integrated_timing_recorded import RECORDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(path: Path) -> None:
    """Compare actual TabPFN totals with separately timed forward estimates."""
    x = np.arange(len(RECORDS))
    source = np.array([r["source_s"] for r in RECORDS])
    hazard = np.array([r["hazard_s"] for r in RECORDS])
    route = np.array([r["route_s"] for r in RECORDS])
    forward = np.array([r["forward_hazard_s"] for r in RECORDS])
    forward_route = np.array([r["forward_route_s"] for r in RECORDS])
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.36
    ax.bar(x - width / 2, source, width, color="#4c8499", label="Source (shared)")
    ax.bar(x - width / 2, hazard, width, bottom=source, color="#dc8b30", label="TabPFN hazard")
    ax.bar(x - width / 2, route, width, bottom=source + hazard, color="#564b5c", label="Route")
    ax.bar(x + width / 2, source, width, color="#4c8499")
    ax.bar(x + width / 2, forward, width, bottom=source, color="#69af80", label="Forward hazard")
    ax.bar(x + width / 2, forward_route, width, bottom=source + forward, color="#564b5c")
    ax.axhline(2, color="#995656", linestyle="--", linewidth=1, label="Earlier 2 s target")
    ax.axhline(5, color="#555555", linestyle=":", linewidth=1, label="Approx. 5 s design")
    ax.set_xticks(x, [f"Leak {r['case'] + 1}, run {r['cycle']}" for r in RECORDS])
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_title("Simulated alarm-to-route component timing")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Left: measured TabPFN cycle. Right: same source + separately timed forward and route.\n"
        "One fixed person; setup, sensor delivery and guidance are excluded. Six related cycles.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Write the figure from saved aggregate measurements without cloud calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, required=True)
    draw(parser.parse_args().figure)


if __name__ == "__main__":
    main()
