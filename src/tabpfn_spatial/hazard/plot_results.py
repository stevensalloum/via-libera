"""Create presentation figures from the recorded, rounded hazard experiment results.

Run ``uv run python -m tabpfn_spatial.hazard.plot_results``. This uses the aggregate
values in PROOF_OF_CONCEPT.md; it does not run models, generate data or spend credits.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BASELINE = "#526175"
TABPFN = "#007c91"
OPTION = "#d47c25"
PALE = "#e4eaf0"
INK = "#18283b"


def style() -> None:
    """Use a readable, restrained style for slides and printable figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#aab6c2",
            "axes.axisbelow": True,
            "grid.color": PALE,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def finish(fig, out: Path, name: str, note: str) -> None:
    """Write one chart in vector SVG and high-resolution PNG formats."""
    fig.text(0.05, 0.025, note, fontsize=9, color="#526175", va="bottom")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(ax, labels, first, second, ylabel: str, limit: float, digits: int = 3) -> None:
    """Plot paired baseline and TabPFN values using a shared zero baseline."""
    x = np.arange(len(labels))
    for values, offset, color, name in (
        (first, -0.19, BASELINE, "Forward baseline"),
        (second, 0.19, TABPFN, "TabPFN"),
    ):
        bars = ax.bar(x + offset, values, 0.36, color=color, label=name)
        ax.bar_label(bars, fmt=f"%.{digits}f", padding=4, fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, limit)
    ax.grid(axis="y")


def early_quality(out: Path) -> None:
    """Compare early-alarm toxic probability error and ranking."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), width_ratios=[1.4, 1])
    fig.suptitle("Early alarm: hazard forecast quality", fontsize=19, fontweight="bold", y=0.98)
    grouped_bars(
        axes[0],
        ["0-120 s", "First 30 s"],
        [0.07186, 0.06239],
        [0.05731, 0.04352],
        "Toxic Brier score · lower is better",
        0.09,
        4,
    )
    grouped_bars(axes[1], ["0-120 s"], [0.79755], [0.94516], "Toxic AUC · higher is better", 1.12)
    axes[0].legend(frameon=False, loc="upper right", fontsize=10)
    fig.subplots_adjust(top=0.78, bottom=0.24, wspace=0.4)
    finish(
        fig,
        out,
        "01_early_forecast_quality",
        "30 paired simulated pipe leaks · one factory · centroid source · seed 0 · "
        "not real-world validation",
    )


def early_timing(out: Path) -> None:
    """Compare recorded first-route latency with the proposed response target."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.suptitle("Early alarm: time to a first route", fontsize=19, fontweight="bold", y=0.98)
    grouped_bars(
        ax,
        ["Median", "95th percentile"],
        [0.0066, 0.0091],
        [3.9187, 4.9136],
        "Source + hazard + one route plan · seconds",
        6.0,
        3,
    )
    ax.axhline(2.0, color=OPTION, linestyle="--", linewidth=2, label="Proposed 2 s target")
    ax.legend(frameon=False, ncol=3, fontsize=10, loc="upper left")
    fig.subplots_adjust(top=0.8, bottom=0.25)
    finish(
        fig,
        out,
        "02_early_route_timing",
        "30 simulated leaks · centroid source, not Rayan's model · excludes preparation · "
        "target is study-specific",
    )


def early_routes(out: Path) -> None:
    """Show exposure counts in the first-route replay."""
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    fig.suptitle("Early alarm: simulated route exposure", fontsize=19, fontweight="bold", y=0.98)
    labels = ["Nearest exit", "Forward route", "TabPFN route", "Baseline → TabPFN"]
    counts = [4, 4, 2, 1]
    bars = ax.barh(labels[::-1], counts[::-1], color=[OPTION, TABPFN, BASELINE, "#8d9aaa"])
    ax.bar_label(bars, labels=[f"{n} / 30" for n in counts[::-1]], padding=6, fontsize=12)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Simulated people exposed · lower is better")
    ax.grid(axis="x")
    fig.subplots_adjust(left=0.25, top=0.79, bottom=0.25)
    finish(
        fig,
        out,
        "03_early_route_exposure",
        "One person per leak · first-route replay · hybrid is experimental · "
        "repeat TabPFN result: 3 / 30",
    )


def previous_geometry(out: Path) -> None:
    """Compare original and optional geometry features in the older setup."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    fig.suptitle(
        "Earlier 60-second setup: effect of path features", fontsize=19, fontweight="bold", y=0.98
    )
    names = ["Forward\nbaseline", "TabPFN\noriginal", "TabPFN\n+ path"]
    colors = [BASELINE, TABPFN, OPTION]
    for ax, vals, title, ylim in (
        (axes[0], [0.04638, 0.05832, 0.05600], "Toxic Brier · lower is better", 0.075),
        (axes[1], [0.91150, 0.95749, 0.96200], "Toxic AUC · higher is better", 1.12),
    ):
        bars = ax.bar(names, vals, color=colors, width=0.62)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.set_ylabel(title)
        ax.set_ylim(0, ylim)
        ax.grid(axis="y")
    fig.subplots_adjust(top=0.78, bottom=0.25, wspace=0.35)
    finish(
        fig,
        out,
        "04_previous_geometry",
        "30 paired simulated cases · Rayan source snapshot 331bbe3 · "
        "paths approximate geometry, not airflow",
    )


def source_diagnostics(out: Path) -> None:
    """Show source-map coverage and retained probability in the older setup."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Earlier 60-second setup: source-map diagnostics", fontsize=19, fontweight="bold", y=0.98
    )
    names = ["Centroid", "Rayan snapshot"]
    for ax, vals, ylabel, suffix in (
        (axes[0], [7.06, 91.34], "Mean source probability in top 3 cells (%)", "%"),
        (axes[1], [1, 28], "True source in top 3 · cases out of 30", " / 30"),
    ):
        bars = ax.bar(names, vals, color=[BASELINE, TABPFN], width=0.58)
        labels = [f"{v:.2f}{suffix}" if suffix == "%" else f"{v}{suffix}" for v in vals]
        ax.bar_label(bars, labels=labels, padding=4)
        ax.set_ylim(0, 110 if suffix == "%" else 35)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
    fig.subplots_adjust(top=0.78, bottom=0.25, wspace=0.35)
    finish(
        fig,
        out,
        "05_source_map_diagnostics",
        "30 simulated cases · older source snapshot · "
        "probability mass is not proof of calibrated confidence",
    )


def selection(out: Path) -> None:
    """Plot two separate source-selection studies without combining their results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle(
        "Earlier setup: how many source cells to consider?", fontsize=19, fontweight="bold", y=0.98
    )
    configurations = [
        (
            ["Forward\ntop 1", "TabPFN\ntop 3", "TabPFN\ntop 10", "TabPFN\n90% target"],
            [0.04638, 0.05832, 0.05830, 0.05841],
            [BASELINE, TABPFN, TABPFN, TABPFN],
            "With Rayan's source maps",
            0.075,
        ),
        (
            ["Forward\ntop 3", "Forward\n90% target"],
            [0.09288, 0.07345],
            [BASELINE, OPTION],
            "Centroid diagnostic: forward mixtures",
            0.115,
        ),
    ]
    for ax, (names, vals, colors, title, ylim) in zip(axes, configurations, strict=True):
        bars = ax.bar(names, vals, color=colors, width=0.65)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_ylabel("Toxic Brier · lower is better")
        ax.set_ylim(0, ylim)
        ax.grid(axis="y")
    fig.subplots_adjust(top=0.74, bottom=0.26, wspace=0.3)
    finish(
        fig,
        out,
        "06_source_selection",
        "Separate studies, 30 simulated cases each · TabPFN variants share one top-10 batch · "
        "not a live latency test",
    )


def backends(out: Path) -> None:
    """Compare five-case offline and cloud model quality and hazard-call range."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.suptitle(
        "Earlier setup: offline and cloud backends", fontsize=19, fontweight="bold", y=0.98
    )
    names = ["Forward", "Offline k-NN", "Cloud TabPFN"]
    colors = [BASELINE, OPTION, TABPFN]
    vals = [0.09081, 0.10857, 0.08850]
    bars = axes[0].bar(names, vals, color=colors, width=0.6)
    axes[0].bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    axes[0].set_ylabel("Toxic Brier · lower is better")
    axes[0].set_ylim(0, 0.135)
    axes[0].grid(axis="y")
    for y, lo, hi, color in zip(
        range(3), [0.01, 4.16, 3.49], [0.02, 4.45, 4.98], colors, strict=True
    ):
        axes[1].plot([lo, hi], [y, y], color=color, linewidth=8, solid_capstyle="round")
        axes[1].text(hi + 0.09, y, f"{lo:.2f}-{hi:.2f}", va="center", fontsize=10)
    axes[1].set_yticks(range(3), names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 6.1)
    axes[1].set_xlabel("Observed hazard-call range · seconds")
    axes[1].grid(axis="x")
    fig.subplots_adjust(top=0.77, bottom=0.26, wspace=0.5)
    finish(
        fig,
        out,
        "07_backend_pilot",
        "Only 5 simulated cases · centroid source · hazard calls exclude source, routing and fit · "
        "k-NN chunk 128",
    )


def cache_fast(out: Path) -> None:
    """Plot the small cache and fast-model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Earlier setup: cloud model and cache pilot", fontsize=19, fontweight="bold", y=0.98
    )
    names = ["v3.5", "v3.5\ncache", "Fast", "Fast\ncache"]
    colors = [TABPFN, "#65acb9", OPTION, "#e7b478"]
    studies = [
        ([0.07029, 0.07034, 0.06827, 0.06825], "Toxic Brier\nlower is better", 0.09, "%.4f"),
        ([3.65, 4.68, 2.74, 4.32], "Median hazard call\nseconds", 5.8, "%.2f"),
        ([10, 30, 10, 30], "Credits per forecast\nthousands", 38, "%.0f"),
    ]
    for ax, (vals, title, ylim, fmt) in zip(axes, studies, strict=True):
        bars = ax.bar(names, vals, color=colors, width=0.65)
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=9)
        ax.set_ylabel(title)
        ax.set_ylim(0, ylim)
        ax.grid(axis="y")
    axes[0].axhline(0.04405, color=BASELINE, linestyle="--", linewidth=1.8)
    axes[0].text(0.02, 0.04405 + 0.002, "Forward baseline 0.0441", fontsize=9, color=BASELINE)
    fig.subplots_adjust(top=0.75, bottom=0.27, wspace=0.42)
    finish(
        fig,
        out,
        "08_cache_fast_pilot",
        "5 simulated cases · cached calls required 3 chunks · "
        "timing excludes source, routing and fit",
    )


def geometry_example(out: Path) -> None:
    """Illustrate a constructed pair of layouts with different simulated outcomes."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Why factory geometry can matter", fontsize=19, fontweight="bold", y=0.98)
    names = ["Door near source-target line", "Door farther from line"]
    vals = [0.13408, 0.00882]
    bars = ax.barh(names[::-1], vals[::-1], color=[BASELINE, OPTION])
    ax.bar_label(bars, fmt="%.5f", padding=5)
    ax.axvline(0.1, color=TABPFN, linestyle="--", linewidth=2, label="Simulated toxic threshold")
    ax.set_xlim(0, 0.17)
    ax.set_xlabel("Simulated concentration · relative units")
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.33, top=0.78, bottom=0.25)
    finish(
        fig,
        out,
        "09_geometry_counterexample",
        "Constructed two-layout example at 120 s · same original features · "
        "not a fitted TabPFN test",
    )


def main() -> None:
    """Render figures from recorded aggregate values without any cloud calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/hazard_results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    for draw in (
        early_quality,
        early_timing,
        early_routes,
        previous_geometry,
        source_diagnostics,
        selection,
        backends,
        cache_fast,
        geometry_example,
    ):
        draw(args.out)
    print(f"Wrote 9 figures as SVG and PNG to {args.out}")


if __name__ == "__main__":
    main()
