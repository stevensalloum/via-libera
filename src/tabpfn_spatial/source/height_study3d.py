"""Leak height study on the team's 3D simulator: the stand-in study, rerun on real 3D gas.

Same question, rows, models and figures as ``source.height_study`` (which stays as the
stand-in baseline), but the data come from ``sim.plant3d`` + ``sim.gas3d`` through
``source.vertical3d``: Steven's plant heights, his mast-rule detectors, time-stepped 3D gas
with buoyancy. Light (methane) and heavy (propane) gas, three detector layouts:

- ``<gas>_1_data.png``: side view of the 3D gas, gas in height over time at the pipe, mast
  readings, for the lowest and highest leak site on one vertical pipe;
- ``<gas>_2_detection_<backend>.png``: error over time, predicted vs true, error by height;
- ``height_<backend>_vs_knn.png``: both gases, all layouts, model vs k-NN.

Run: ``uv run python -m tabpfn_spatial.source.height_study3d --backend cloud`` (6 cloud
calls; quote first with ``--quote``). The 3D runs are cached in ``<out>/cache`` (about 1,600
simulations, ~5 min with 14 processes the first time); ``--backend baseline`` makes no API
calls.
"""

import argparse
import dataclasses
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from tabpfn_spatial.env import load_env
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, simulate
from tabpfn_spatial.sim.plant3d import HALL_HEIGHT, Plant3D
from tabpfn_spatial.source.features import alarm_time, n_features
from tabpfn_spatial.source.height_study import GAS_LABEL, WS, Result, _predict
from tabpfn_spatial.source.vertical3d import (
    clean_readings,
    height_rows,
    layouts3d,
    sample_pipe_leaks,
    vertical_sites,
)

LAYOUT_LABEL = {
    "mounted": "detectors at the gas's mounting height",
    "mast_rule": "mast rule, masts on the pipe",
    "mast_1m": "mast rule, masts 1 m beside the pipe",
    "three_heights": "3 heights everywhere",
}
COLOUR = {
    "mounted": "tab:gray",
    "mast_rule": "tab:orange",
    "mast_1m": "tab:red",
    "three_heights": "tab:purple",
}
SHOWN_PIPE, SHOWN_WIND = 1, (-0.8, 0.0)  # the drop at (19, 21) m; 19 m open floor towards -x
OUT = Path("outputs/source_height3d")


def plant_for(gas: str, factory: Factory | None = None) -> Plant3D:
    """The 3D plant of ``factory`` (default plant) filled with ``gas``."""
    return Plant3D(dataclasses.replace(factory or Factory.default(), gas=gas))


def site_heights(plant: Plant3D) -> np.ndarray:
    """Distinct leak heights on the vertical pipes (m)."""
    return np.unique(vertical_sites(plant)[0][:, 2].round(3))


def prior_error(z: np.ndarray, plant: Plant3D) -> float:
    """Mean error of guessing the middle of the pipes' height range."""
    h = site_heights(plant)
    return float(np.mean(np.abs(z - (h.min() + h.max()) / 2)))


def voxel_limit(z: np.ndarray, plant: Plant3D) -> float:
    """Best possible mean error: heights inside one voxel give the same gas (1 m voxels)."""
    h = site_heights(plant)
    cell = np.floor(h / plant.grid.cell)
    best = {c: h[cell == c].mean() for c in np.unique(cell)}
    guess = np.array([best[c] for c in np.floor(z / plant.grid.cell)])
    return float(np.mean(np.abs(z - guess)))


def run_study(
    plant: Plant3D,
    backend: str = "baseline",
    n_train: int = 600,
    n_test: int = 200,
    cache_dir: Path = OUT / "cache" / "default",
    workers: int = 1,
) -> dict[str, Result]:
    """Height predictions for every layout; train seeds 100/101, test seeds 200/201."""
    gas, lay = plant.f.gas, layouts3d(plant)
    data = {}
    for split, n, seed in (("train", n_train, 100), ("test", n_test, 200)):
        leaks = sample_pipe_leaks(plant, n, np.random.default_rng(seed))
        cache = cache_dir / f"{gas}_{split}_{n}.npz"
        data[split] = (leaks, *clean_readings(plant, leaks, lay, workers=workers, cache=cache))
    h = site_heights(plant)
    out = {}
    for name, points in lay.items():
        a, b = (
            height_rows(t, c[name], lk, np.random.default_rng(s))
            for (lk, t, c), s in ((data["train"], 101), (data["test"], 201))
        )
        q = _predict(backend, a.X, a.z, b.X, cache_dir / f"{gas}_{name}_{backend}_{n_train}.npz")
        lo, med, hi = (np.clip(q.quantile(p), h.min(), h.max()) for p in (0.1, 0.5, 0.9))
        out[name] = Result(b.w, b.z, lo, med, hi, len(set(b.leak)), len(points))
    return out


def quote(res: dict[str, Result], n_train: int = 600) -> int:
    """Free cloud quote (tokens) for the TabPFN run of one gas (train rows: upper bound)."""
    from tabpfn_spatial.core.budget import cloud_quote

    return sum(
        cloud_quote(n_train * len(WS), len(r.z), n_features(r.n_sensors)) for r in res.values()
    )


def _shown_run(plant: Plant3D, z: float):
    v = plant.vertical_pipes[SHOWN_PIPE]
    vent = Ventilation(
        float(np.hypot(*SHOWN_WIND)), float(np.arctan2(SHOWN_WIND[1], SHOWN_WIND[0]))
    )
    return simulate(plant, Leak3D((*v.xy, z), 1.0, plant.f.gas), vent, t_end=70.0)


def _side_view(ax, plant: Plant3D, run, j: int, masts: np.ndarray) -> None:
    """Gas in the vertical x-z plane through the pipe at sample ``j``."""
    v = plant.vertical_pipes[SHOWN_PIPE]
    iy = int(plant.grid.index(np.array([[*v.xy, 0.0]]))[0, 1])
    side = np.maximum(run.volumes[j, :, iy, :].T.astype(float), 0.01)
    side[plant.solid[:, iy, :].T] = np.nan
    im = ax.imshow(
        side,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        extent=[0, plant.f.width, 0, HALL_HEIGHT],
        norm=LogNorm(0.01, max(np.nanmax(side), 0.02)),
    )
    ax.plot(v.xy[0], run.leak.xyz[2], "c*", ms=16, label="leak")
    ax.plot(masts[:, 0], masts[:, 2], "w^", ms=8, label="mast detectors")
    lim = sorted((v.xy[0] - 4 * np.sign(SHOWN_WIND[0]), v.xy[0] + 22 * np.sign(SHOWN_WIND[0])))
    ax.set_xlim(max(lim[0], 0.0), min(lim[1], plant.f.width))
    ax.set(
        title=f"Side view along the air, t = {run.times[j]:.0f} s (alarm + 15 s)",
        xlabel="x along the hall (m)",
        ylabel="height (m)",
    )
    ax.legend(loc="upper right", fontsize=7)
    plt.colorbar(im, ax=ax, label="concentration (log, from 0.01)")


def _mast_panels(axes, plant: Plant3D, run, masts_idx: np.ndarray) -> float:
    """Height over time in the pipe's voxel column and the mast readings; returns the alarm."""
    r = run.readings(np.random.default_rng(0))
    t_a = alarm_time(r.times, r.values) or 0.0
    v = plant.vertical_pipes[SHOWN_PIPE]
    ix, iy, _ = plant.grid.index(np.array([[*v.xy, 0.0]]))[0]
    col = run.volumes[:, ix, iy, :].T
    im = axes[0].imshow(
        col,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        extent=[0, run.times[-1], 0, HALL_HEIGHT],
    )
    for z in r.heights[masts_idx]:
        axes[0].axhline(z, color="w", ls=":", lw=1)
    axes[0].axhline(run.leak.xyz[2], color="c", lw=1.5, label="leak height")
    axes[0].axvline(t_a, color="r", lw=1.5, label="first alarm")
    axes[0].set(
        title="Gas in height over time at the pipe (1 m voxels)",
        xlabel="seconds since leak start",
        ylabel="height (m)",
    )
    axes[0].legend(loc="upper left", fontsize=7)
    plt.colorbar(im, ax=axes[0], label="concentration")
    for i, c in zip(masts_idx, ("tab:brown", "tab:green", "tab:blue"), strict=True):
        axes[1].plot(r.times, r.values[i], "o-", color=c, label=f"mast at {r.heights[i]:.2f} m")
    axes[1].axhline(0.03, color="r", ls="--", lw=1, label="alarm level")
    axes[1].axvline(t_a, color="r", lw=1)
    axes[1].set(
        title="What the three mast detectors read (noisy)",
        xlabel="seconds since leak start",
        ylabel="reading",
    )
    axes[1].legend(fontsize=7)
    return t_a


def data_figure(plant: Plant3D, path: Path) -> None:
    """The lowest and the highest leak site on the same vertical pipe."""
    det, why = plant.detectors
    v = plant.vertical_pipes[SHOWN_PIPE]
    idx = np.flatnonzero((np.array(why) == "mast") & np.all(np.isclose(det[:, :2], v.xy), axis=1))
    h = site_heights(plant)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for row, z in enumerate((h.min(), h.max())):
        run = _shown_run(plant, float(z))
        t_a = _mast_panels(ax[row, 1:], plant, run, idx)
        j = min(int(np.searchsorted(run.times, t_a + 15.0)), len(run.times) - 1)
        _side_view(ax[row, 0], plant, run, j, det[idx])
    gas = plant.f.gas
    fig.suptitle(
        f"3D simulator, {GAS_LABEL[gas]}: leak low ({h.min():.1f} m, top) vs high "
        f"({h.max():.1f} m, bottom) on the same drop pipe",
        fontsize=13,
    )
    fig.savefig(path, dpi=85)
    plt.close(fig)


def _guides(ax, z: np.ndarray, plant: Plant3D) -> None:
    ax.axhline(prior_error(z, plant), color="k", ls="--", label="guess the middle (prior)")
    ax.axhline(voxel_limit(z, plant), color="k", ls=":", label="best possible (1 m voxels)")


def detection_figure(res: dict[str, Result], plant: Plant3D, model: str, path: Path) -> None:
    """Error over time, predicted vs true (mast rule), error by true height."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    z_all = res["mast_rule"].z
    for name, r in res.items():
        ax[0].plot(
            WS,
            r.mae(),
            "o-",
            color=COLOUR[name],
            label=f"{LAYOUT_LABEL[name]} ({r.n_sensors}; {r.n_test} leaks alarmed)",
        )
    _guides(ax[0], z_all, plant)
    ax[0].set(
        title="Height error (mean) after the alarm",
        xlabel="seconds since alarm",
        ylabel="|predicted - true| height (m)",
        ylim=(0, None),
    )
    ax[0].legend(fontsize=7)
    r, m = res["mast_1m"], res["mast_1m"].w == 15.0
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, m.sum())
    ax[1].errorbar(
        r.z[m] + jitter,
        r.med[m],
        yerr=[r.med[m] - r.lo[m], r.hi[m] - r.med[m]],
        fmt="o",
        ms=3,
        alpha=0.5,
        color=COLOUR["mast_rule"],
        ecolor="0.75",
    )
    h = site_heights(plant)
    ax[1].plot([h.min(), h.max()], [h.min(), h.max()], "k-", lw=1)
    ax[1].set(
        title="Masts 1 m beside the pipe, alarm + 15 s: predicted vs true",
        xlabel="true leak height (m, the plant's sites)",
        ylabel="predicted height (m)",
    )
    edges = np.quantile(h, [1 / 3, 2 / 3])
    for i, (name, r) in enumerate(res.items()):
        m = r.w == 15.0
        band = np.digitize(r.z[m], edges)
        err = [np.mean(np.abs(r.med[m] - r.z[m])[band == b]) for b in range(3)]
        ax[2].bar(
            np.arange(3) + (i - 1.5) * 0.2, err, 0.2, color=COLOUR[name], label=LAYOUT_LABEL[name]
        )
    bands = np.split(h, np.searchsorted(h, edges))
    ax[2].set_xticks(
        range(3),
        [
            f"{n}\n{b.min():.1f}-{b.max():.1f} m"
            for n, b in zip(("low", "middle", "high"), bands, strict=True)
        ],
    )
    ax[2].set(title="Error by true leak height, alarm + 15 s", ylabel="mean error (m)")
    ax[2].legend(fontsize=7)
    fig.suptitle(
        f"3D simulator, {GAS_LABEL[plant.f.gas]}: how well the height is found ({model})",
        fontsize=13,
    )
    fig.savefig(path, dpi=85)
    plt.close(fig)


def comparison_figure(model: dict, knn: dict, name: str, path: Path) -> None:
    """Both gases: ``name`` (solid) vs k-NN (dashed) for every layout."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, sharey=True)
    for a, gas in zip(ax, GAS_LABEL, strict=True):
        for lay in LAYOUT_LABEL:
            a.plot(
                WS,
                model[gas][lay].mae(),
                "o-",
                color=COLOUR[lay],
                label=f"{name}: {LAYOUT_LABEL[lay]}",
            )
            a.plot(
                WS,
                knn[gas][lay].mae(),
                "x--",
                color=COLOUR[lay],
                alpha=0.6,
                label=f"k-NN: {LAYOUT_LABEL[lay]}",
            )
        _guides(a, model[gas]["mast_rule"].z, plant_for(gas))
        a.set(
            title=GAS_LABEL[gas],
            xlabel="seconds since alarm",
            ylabel="mean height error (m)",
            ylim=(0, None),
        )
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Leak height on the 3D simulator: {name} (solid) vs k-NN (dashed)", fontsize=13)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def _print(gas: str, res: dict[str, Result], plant: Plant3D) -> None:
    z = res["mast_rule"].z
    for lay, r in res.items():
        print(
            f"{gas:8s} {lay:14s} detectors {r.n_sensors:2d} alarmed {r.n_test:3d} "
            f"mean error {np.round(r.mae(), 2)} cover80 {np.round(r.coverage(), 2)}"
        )
    print(
        f"{gas:8s} prior {prior_error(z, plant):.2f} m, voxel limit {voxel_limit(z, plant):.2f} m"
    )


def main() -> None:
    """CLI: draw the figures and print the numbers (``--quote``: tokens only, no TabPFN)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="baseline", choices=("baseline", "cloud", "local"))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--quote", action="store_true")
    args = p.parse_args()
    load_env()
    args.out.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 2)
    knn, model = {}, {}
    for gas in GAS_LABEL:
        plant = plant_for(gas)
        data_figure(plant, args.out / f"{gas}_1_data.png")
        knn[gas] = run_study(
            plant, "baseline", cache_dir=args.out / "cache" / "default", workers=workers
        )
        if args.quote:
            print(f"{gas}: cloud quote {quote(knn[gas]):,} tokens (3 calls)")
        backend = "baseline" if args.quote else args.backend
        model[gas] = run_study(
            plant, backend, cache_dir=args.out / "cache" / "default", workers=workers
        )
        detection_figure(model[gas], plant, backend, args.out / f"{gas}_2_detection_{backend}.png")
        _print(gas, model[gas], plant)
    backend = "baseline" if args.quote else args.backend
    comparison_figure(model, knn, backend, args.out / f"height_{backend}_vs_knn.png")
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
