"""Leak height on plants the method was never tuned on: random and shaped 3D plants.

The height study (``source.height_study3d``) on 7 plants and 2 gases: the default plant, two
``Factory.random`` plants (seeds 11, 12) and one unseen L, U, T and campus plant each
(``sim.layouts.shaped``, seeds 101-104; the team's templates use seeds 1-4). Every plant gets
its own simulated table (600 training leaks, 200 test leaks on its vertical pipes), as the
locator does for a new plant; nothing about the plant is hand-tuned.

Saves everything under ``<out>``: the simulated readings (``cache/<plant>/``), the cloud
predictions (``cache/<plant>/*_cloud_600.npz``), ``results.csv`` (one row per plant, gas,
layout, model, moment) and the figures ``plants.png``, ``unseen_errors_<backend>.png``,
``unseen_time_<backend>.png`` and per plant ``plants/<plant>/<gas>_2_detection_*.png``.

Run: ``uv run python -m tabpfn_spatial.source.height_plants3d --backend cloud``
(``--quote`` prints the tokens first; 56 cloud calls at 4 layouts x 2 gases x 7 plants).
Simulating all plants the first time takes ~15 min with 14 processes.
"""

import argparse
import csv
import dataclasses
import os
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from tabpfn_spatial.env import load_env
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.layouts import shaped
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.source.height_study import GAS_LABEL, WS, Result
from tabpfn_spatial.source.height_study3d import (
    COLOUR,
    LAYOUT_LABEL,
    OUT,
    detection_figure,
    prior_error,
    quote,
    run_study,
    voxel_limit,
)

PLANTS: dict[str, Callable[[], Factory]] = {
    "default": Factory.default,
    "random-11": lambda: Factory.random(np.random.default_rng(11)),
    "random-12": lambda: Factory.random(np.random.default_rng(12)),
    "L-101": lambda: shaped(np.random.default_rng(101), "L"),
    "U-102": lambda: shaped(np.random.default_rng(102), "U"),
    "T-103": lambda: shaped(np.random.default_rng(103), "T"),
    "campus-104": lambda: shaped(np.random.default_rng(104), "campus"),
}
SHOWN_AT = 15.0  # seconds after the alarm for the summary bars


def build(name: str, gas: str) -> Plant3D:
    """Plant ``name`` of ``PLANTS`` filled with ``gas``."""
    return Plant3D(dataclasses.replace(PLANTS[name](), gas=gas))


def study_all(out: Path, backend: str, workers: int) -> dict[tuple[str, str], dict[str, Result]]:
    """Results per (plant, gas); simulations and cloud predictions are cached under ``out``."""
    res = {}
    for name in PLANTS:
        for gas in GAS_LABEL:
            plant = build(name, gas)
            res[name, gas] = run_study(
                plant, backend, cache_dir=out / "cache" / name, workers=workers
            )
    return res


def table(model: dict, knn: dict, backend: str) -> list[dict]:
    """One row per plant, gas, layout, model and moment (the saved ``results.csv``)."""
    rows = []
    for (name, gas), res in model.items():
        plant = build(name, gas)
        z = res["mast_1m"].z
        prior, limit = prior_error(z, plant), voxel_limit(z, plant)
        for lay in res:
            for label, r in ((backend, res[lay]), ("knn", knn[name, gas][lay])):
                for w, e, c in zip(WS, r.mae(), r.coverage(), strict=True):
                    rows.append(
                        {
                            "plant": name,
                            "gas": gas,
                            "layout": lay,
                            "model": label,
                            "seconds_after_alarm": w,
                            "mean_error_m": round(e, 3),
                            "coverage80": round(c, 3),
                            "detectors": r.n_sensors,
                            "leaks_alarmed": r.n_test,
                            "prior_m": round(prior, 3),
                            "voxel_limit_m": round(limit, 3),
                        }
                    )
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    """Write ``rows`` as CSV."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plants_figure(path: Path) -> None:
    """Top view of every plant: walls, machines, vertical pipes, mounted detectors."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for ax, name in zip(axes.flat, PLANTS, strict=False):
        p = build(name, "methane")
        f = p.f
        ax.imshow(f.walls, origin="lower", cmap="Greys", vmax=1.6, extent=[0, f.width, 0, f.height])
        for m in p.machines:
            x0, y0, _, x1, y1, _ = m["box"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="tab:blue", alpha=0.25))
        if f.pipes is not None:
            yy, xx = np.nonzero(f.pipes)
            ax.plot((xx + 0.5) * f.cell, (yy + 0.5) * f.cell, ",", color="tab:green")
        for v in p.vertical_pipes:
            ax.plot(*v.xy, "o" if v.kind == "drop" else "s", ms=6, mfc="tab:orange", mec="k")
        det, why = p.detectors
        mounted = det[np.array(why) == "mounted"]
        ax.plot(mounted[:, 0], mounted[:, 1], "^", ms=4, color="tab:red")
        n_v = len(p.vertical_pipes)
        ax.set_title(f"{name}: {n_v} vertical pipes, {len(det)} detectors", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes.flat[-1]
    ax.axis("off")
    for mk, lab, kw in (
        ("o", "drop (rack -> machine inlet)", {"mfc": "tab:orange", "mec": "k"}),
        ("s", "riser (floor -> rack)", {"mfc": "tab:orange", "mec": "k"}),
        ("^", "mounted detector", {"color": "tab:red"}),
        ("s", "machine", {"color": "tab:blue", "alpha": 0.25}),
    ):
        ax.plot([], [], mk, ms=8, label=lab, ls="", **kw)
    ax.legend(loc="center", fontsize=11, title="every vertical pipe also gets a mast of 3")
    fig.suptitle(
        "Plants for the height study: the default plant and 6 never-seen layouts", fontsize=14
    )
    fig.savefig(path, dpi=70)
    plt.close(fig)


def errors_figure(model: dict, knn: dict, backend: str, path: Path) -> None:
    """Median height error at alarm + ``SHOWN_AT`` s per plant and layout, both gases."""
    names, lays = list(PLANTS), list(LAYOUT_LABEL)
    j = WS.index(SHOWN_AT)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), constrained_layout=True, sharey=True)
    for ax, gas in zip(axes, GAS_LABEL, strict=True):
        x = np.arange(len(names))
        for i, lay in enumerate(lays):
            dx = (i - 1.5) * 0.2
            ax.bar(
                x + dx,
                [model[n, gas][lay].mae()[j] for n in names],
                0.2,
                color=COLOUR[lay],
                label=f"{backend}: {LAYOUT_LABEL[lay]}",
            )
            ax.plot(
                x + dx,
                [knn[n, gas][lay].mae()[j] for n in names],
                "_",
                ms=14,
                mew=2,
                color="k",
                label="k-NN (same rows)" if i == 0 else None,
            )
        for k, n in enumerate(names):
            plant, z = build(n, gas), model[n, gas]["mast_1m"].z
            ax.hlines(
                prior_error(z, plant),
                k - 0.42,
                k + 0.42,
                colors="k",
                linestyles="--",
                label="guess the middle" if k == 0 else None,
            )
            ax.hlines(
                voxel_limit(z, plant),
                k - 0.42,
                k + 0.42,
                colors="k",
                linestyles=":",
                label="best possible (1 m voxels)" if k == 0 else None,
            )
        ax.set_xticks(x, names)
        ax.set(title=GAS_LABEL[gas], ylabel=f"mean height error (m), alarm + {SHOWN_AT:.0f} s")
    axes[0].legend(fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("Leak height on every plant: each plant gets its own simulated table", fontsize=14)
    fig.savefig(path, dpi=75)
    plt.close(fig)


def time_figure(model: dict, knn: dict, backend: str, path: Path) -> None:
    """Median over the plants of the height error after the alarm, per layout."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True, sharey=True)
    for ax, gas in zip(axes, GAS_LABEL, strict=True):
        for lay in LAYOUT_LABEL:
            for res, style, lab in ((model, "o-", backend), (knn, "x--", "k-NN")):
                e = np.median([res[n, gas][lay].mae() for n in PLANTS], axis=0)
                ax.plot(
                    WS,
                    e,
                    style,
                    color=COLOUR[lay],
                    alpha=1.0 if lab != "k-NN" else 0.55,
                    label=f"{lab}: {LAYOUT_LABEL[lay]}",
                )
        ax.set(
            title=GAS_LABEL[gas],
            xlabel="seconds since alarm",
            ylabel="median over 7 plants of the mean error (m)",
            ylim=(0, None),
        )
    axes[0].legend(fontsize=7)
    fig.suptitle(
        f"Leak height over time, all plants: {backend} (solid) vs k-NN (dashed)", fontsize=13
    )
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    """CLI: simulate (cached), predict, save ``results.csv`` and the figures."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="baseline", choices=("baseline", "cloud", "local"))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--quote", action="store_true", help="print the cloud tokens and stop")
    args = p.parse_args()
    load_env()
    workers = max(1, (os.cpu_count() or 2) - 2)
    args.out.mkdir(parents=True, exist_ok=True)
    plants_figure(args.out / "plants.png")
    knn = study_all(args.out, "baseline", workers)
    if args.quote:
        print(f"cloud quote: {sum(quote(r) for r in knn.values()):,} tokens, {4 * len(knn)} calls")
        return
    model = study_all(args.out, args.backend, workers) if args.backend != "baseline" else knn
    save_csv(table(model, knn, args.backend), args.out / f"results_{args.backend}.csv")
    for (name, gas), res in model.items():
        d = args.out / "plants" / name
        d.mkdir(parents=True, exist_ok=True)
        detection_figure(
            res, build(name, gas), args.backend, d / f"{gas}_2_detection_{args.backend}.png"
        )
    errors_figure(model, knn, args.backend, args.out / f"unseen_errors_{args.backend}.png")
    time_figure(model, knn, args.backend, args.out / f"unseen_time_{args.backend}.png")
    print(f"saved results_{args.backend}.csv and figures in {args.out}")


if __name__ == "__main__":
    main()
