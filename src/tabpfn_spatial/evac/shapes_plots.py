"""Figures of the unseen-shapes test (``evac.shapes_study``)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.shapes_study import FIRST_SEED, paired, plant
from tabpfn_spatial.sim.layouts import KINDS

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COLOUR = {"nearest": PALETTE[0], "risk": PALETTE[1], "ccrp": PALETTE[2]}
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
MODEL_NAMES = {
    "perfect": "perfect models (planning only)",
    "baseline": "baseline models (centroid + forward sim)",
    "tabpfn": "team's models (Rayan + Ahmad)",
}


def _style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_gains(rows: list[dict], path: Path) -> Path:
    """Seconds in gas saved against nearest exit, per shape (paired, +/- SE)."""
    models = [m for m in ("perfect", "baseline", "tabpfn") if any(r["models"] == m for r in rows)]
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.6 * len(models), 3.8), sharey=True, squeeze=False
    )
    x = np.arange(len(KINDS) + 1)
    for ax, m in zip(axes[0], models, strict=True):
        for k, p in enumerate(("risk", "ccrp")):
            per = paired(rows, m, p, "harm_s_mean", by="kind")
            allp = paired(rows, m, p, "harm_s_mean").get(None, (np.nan, np.nan, 0))
            vals = [per.get(kd, (np.nan, np.nan, 0)) for kd in KINDS] + [allp]
            ax.errorbar(
                x + (k - 0.5) * 0.25,
                [v[0] for v in vals],
                yerr=[v[1] for v in vals],
                fmt="o",
                color=COLOUR[p],
                ms=6,
                capsize=3,
                lw=1.5,
                label=f"{p} - nearest",
            )
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x, [*KINDS, "all"])
        ax.set_title(MODEL_NAMES[m], fontsize=9, color=INK, loc="left")
        _style(ax)
    axes[0, 0].set_ylabel("change in s in gas per person\n(below 0 = better than nearest)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    title = "Unseen factory shapes: what the gas-aware planners save"
    note = " (same leak, alarm and people for every planner; bars = SE)"
    fig.suptitle(title + note, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_scores(rows: list[dict], path: Path, models: str = "tabpfn") -> Path:
    """Mean scores per shape and planner for one model setting."""
    if not any(r["models"] == models for r in rows):
        models = "baseline"
    panels = [
        ("harm_s_mean", "s in harmful gas per person"),
        ("exposed_pct", "% of people exposed"),
        ("out_95_s", "time until 95 % are out (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(KINDS))
    w = 0.8 / len(PLANNERS)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for k, p in enumerate(PLANNERS):
            vals = [
                np.nanmean(
                    [
                        r[metric]
                        for r in rows
                        if r["models"] == models and r["planner"] == p and r["kind"] == kd
                    ]
                    or [np.nan]
                )
                for kd in KINDS
            ]
            ax.bar(x + (k - 1) * w, vals, w * 0.92, color=COLOUR[p], label=p)
        ax.set_xticks(x, KINDS)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"Unseen factory shapes, {MODEL_NAMES[models]}: scores per shape",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_locator(rows: list[dict], path: Path) -> Path:
    """Locator error at the first update and afterwards, per shape and model setting."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(KINDS))
    series = [
        ("baseline", "loc_err_median_m", "baseline, after 1st update"),
        ("tabpfn", "loc_err_first_m", "team, 1st update"),
        ("tabpfn", "loc_err_median_m", "team, after 1st update"),
    ]
    for k, (m, key, label) in enumerate(series):
        for j, kd in enumerate(KINDS):
            v = [
                r[key]
                for r in rows
                if r["models"] == m
                and r["kind"] == kd
                and r["planner"] == "nearest"
                and r[key] is not None
            ]
            if v:
                ax.scatter(
                    np.full(len(v), j + (k - 1) * 0.22),
                    v,
                    s=14,
                    color=PALETTE[k + 3],
                    alpha=0.5,
                    lw=0,
                )
                ax.scatter(
                    [j + (k - 1) * 0.22],
                    [np.median(v)],
                    s=60,
                    color=PALETTE[k + 3],
                    edgecolor="white",
                    label=label if j == 0 else None,
                    zorder=3,
                )
    ax.set_xticks(x, KINDS)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("leak position error (m)")
    ax.set_title(
        "Where is the leak? Locator error on unseen plants (big dot: median)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_gallery(path: Path, per_kind: int = 2) -> Path:
    """A few of the unseen plants: outside grey, walls and machines dark, exits green."""
    fig, axes = plt.subplots(
        per_kind, len(KINDS), figsize=(4.2 * len(KINDS), 3.4 * per_kind), squeeze=False
    )
    for j, kd in enumerate(KINDS):
        for i in range(per_kind):
            f = plant(kd, FIRST_SEED + i)
            img = np.where(f.outside, 0.93, np.where(f.walls, 0.25, 1.0))
            ax = axes[i, j]
            ax.imshow(
                img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=(0, f.width, 0, f.height)
            )
            if f.pipes is not None:
                yx = np.argwhere(f.pipes)
                ax.scatter(
                    (yx[:, 1] + 0.5) * f.cell,
                    (yx[:, 0] + 0.5) * f.cell,
                    s=2,
                    color=PALETTE[3],
                    lw=0,
                )
            for lane in f.agv_lanes:
                ax.plot(*np.vstack([lane, lane[:1]]).T, color=PALETTE[1], lw=0.8)
            ax.scatter(*f.exits.T, s=24, color=PALETTE[5], zorder=3)
            ax.scatter(*f.sensors.T, s=5, color=PALETTE[0], zorder=3)
            ax.set_title(f"{kd} @ seed {FIRST_SEED + i}", fontsize=9, color=INK, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Unseen plants (examples): exits green, detectors blue, pipes yellow, AGV lanes orange",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_all(rows: list[dict], out: Path) -> list[Path]:
    """Every figure of the study."""
    return [
        plot_gallery(out / "plants.png"),
        plot_gains(rows, out / "gains.png"),
        plot_scores(rows, out / "scores.png"),
        plot_locator(rows, out / "locator.png"),
    ]
