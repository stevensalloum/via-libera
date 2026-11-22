"""Leak height study: how well the locator's row tells the height of a leak on a vertical pipe.

Uses the stand-in in ``source.vertical`` (2D simulator x height factor). For a light
(methane) and a heavy (propane) gas it compares three detector layouts with a height
regressor on the same rows the locator uses, k-NN and optionally TabPFN, and draws:

- ``<gas>_1_data.png``: side view, gas in height over time at a mast, mast readings,
- ``<gas>_2_detection_<backend>.png``: error over time, predicted vs true, error by height,
- ``height_<backend>_vs_knn.png``: both gases, all layouts, model vs k-NN.

Run: ``uv run python -m tabpfn_spatial.source.height_study --backend cloud`` (6 cloud calls,
10,000 tokens each; predictions are cached in ``<out>/cache`` and never paid twice).
Default ``--backend baseline`` makes no API calls. ~40 s per backend on a laptop.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.core.models import BaselineRegressor, make_regressor
from tabpfn_spatial.env import load_env
from tabpfn_spatial.sim import Factory, Scenario, simulate
from tabpfn_spatial.sim.gases import GASES, LITRES_PER_RATE_UNIT, passive_distance
from tabpfn_spatial.source.features import alarm_time
from tabpfn_spatial.source.vertical import (
    HALL_HEIGHT,
    MAST_Z,
    RISER_RANGE,
    Leak3D,
    Physics,
    Sensors3D,
    height_library,
    layouts,
    noisy_series_3d,
    pick_risers,
    sample_riser_leaks,
    vertical_factor,
)

GAS_LABEL = {
    "methane": "light gas (methane, 0.55 x air)",
    "propane": "heavy gas (propane, 1.52 x air)",
}
WS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
N_RISERS = 4
LAYOUT_LABEL = {
    "one_height": "all at 1.5 m",
    "masts": "masts at vertical pipes",
    "three_heights": "3 heights everywhere",
}
COLOUR = {"one_height": "tab:gray", "masts": "tab:orange", "three_heights": "tab:purple"}
PRIOR_ERROR = (RISER_RANGE[1] - RISER_RANGE[0]) / 4  # median |z - middle| for uniform z
SHOWN_RISER, SHOWN_WIND = 3, np.array([-0.8, 0.0])  # 18 m of open floor downwind


@dataclass(frozen=True)
class Result:
    """Height predictions for the test rows of one gas and layout."""

    w: np.ndarray  # (rows,) seconds since the alarm
    z: np.ndarray  # (rows,) true height
    lo: np.ndarray  # (rows,) 10 % quantile, clipped to the pipe range
    med: np.ndarray  # (rows,) median
    hi: np.ndarray  # (rows,) 90 % quantile
    n_test: int  # test leaks that alarmed (of n_test asked)
    n_sensors: int

    def error(self) -> np.ndarray:
        """Median |predicted - true| height per moment in ``WS``."""
        return np.array([np.median(np.abs(self.med - self.z)[self.w == w]) for w in WS])

    def mae(self) -> np.ndarray:
        """Mean |predicted - true| height per moment in ``WS``.

        Use this when heights come in look-alike pairs (the 3D simulator's 1 m voxels): a
        model that picks one of two equal answers is 0 or 0.5 m off, and the median hides it.
        """
        return np.array([np.mean(np.abs(self.med - self.z)[self.w == w]) for w in WS])

    def coverage(self) -> np.ndarray:
        """Share of true heights inside the 80 % interval, per moment."""
        inside = (self.lo <= self.z) & (self.z <= self.hi)
        return np.array([inside[self.w == w].mean() for w in WS])


def _predict(backend: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, cache: Path):
    """Height quantiles from k-NN or TabPFN; TabPFN results are cached in ``cache``."""
    if backend == "baseline":
        return BaselineRegressor(k=10).fit(X, y).predict_dist(X_test)
    if cache.exists():
        d = np.load(cache)
        mean = d["mean"] if "mean" in d.files else np.median(d["values"], axis=1)
        return QuantileDist(d["levels"], d["values"], mean)
    q = make_regressor(backend).fit(X, y).predict_dist(X_test)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, levels=q.levels, values=q.values, mean=q.mean)
    return q


def run_study(
    factory: Factory,
    gas: str,
    backend: str = "baseline",
    n_train: int = 600,
    n_test: int = 200,
    cache_dir: Path | None = None,
) -> dict[str, Result]:
    """Height predictions for every layout; train seeds 100/101, test seeds 200/201."""
    risers = pick_risers(factory, N_RISERS)
    train = sample_riser_leaks(factory, risers, n_train, np.random.default_rng(100), gas)
    test = sample_riser_leaks(factory, risers, n_test, np.random.default_rng(200), gas)
    out = {}
    for name, sensors in layouts(factory, risers).items():
        a = height_library(factory, train, sensors, np.random.default_rng(101), ws=WS)
        b = height_library(factory, test, sensors, np.random.default_rng(201), ws=WS)
        cache = (cache_dir or Path("outputs/source_height/cache")) / f"{gas}_{name}_{backend}.npz"
        q = _predict(backend, a.X, a.z, b.X, cache)
        lo, med, hi = (np.clip(q.quantile(p), *RISER_RANGE) for p in (0.1, 0.5, 0.9))
        out[name] = Result(b.w, b.z, lo, med, hi, len(set(b.leak)), len(sensors.z))
    return out


def _side_view(ax, factory: Factory, field: np.ndarray, src, z_leak, t, phys, x_p) -> None:
    """Gas in the vertical plane through the leak, along the wind, at time ``t``."""
    iy, _ = factory.cell_of(src)
    xs = factory.centres()[iy, :, 0]
    zs = np.linspace(0, HALL_HEIGHT, 161)
    v = vertical_factor(zs[:, None], z_leak, np.abs(xs - src[0])[None, :], t, phys, 1.0, 0.8)
    side = np.maximum(field[iy, :][None, :] * v, 0.01)
    side[:, factory.walls[iy]] = np.nan
    im = ax.imshow(
        side,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, factory.width, 0, HALL_HEIGHT],
        norm=LogNorm(0.01, np.nanmax(side)),
    )
    ax.plot(src[0], z_leak, "c*", ms=16, label="leak")
    ax.axvspan(
        src[0] - x_p,
        src[0] + x_p,
        color="cyan",
        alpha=0.12,
        label=f"density matters (±{x_p:.1f} m)",
    )
    ax.set_xlim(src[0] - 22, src[0] + 4)
    ax.set(
        title=f"Side view along the wind, alarm + 15 s (leak at {z_leak} m)",
        xlabel="x along the hall (m)",
        ylabel="height (m)",
    )
    plt.colorbar(im, ax=ax, label="concentration (log, from 0.01 = arrival)")


def _mast_panels(axes, factory, fields, times, lk: Leak3D, mast: Sensors3D, idx) -> float:
    """Height over time at the mast, and the three mast readings; returns the alarm time."""
    _, vals, _ = noisy_series_3d(factory, [lk], mast, np.random.default_rng(0), times[-1])
    t_a = alarm_time(times, vals[0]) or 0.0
    zs = np.linspace(0, HALL_HEIGHT, 161)
    r = float(np.linalg.norm(mast.xy[idx[0]] - lk.scenario.source))
    cy, cx = factory.cell_of(mast.xy[idx[0]])
    prof = fields[:, cy, cx][None, :] * vertical_factor(
        zs[:, None], lk.z, r, times[None, :], lk.phys, 1.0, 0.8
    )
    im = axes[0].imshow(
        prof,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[times[0], times[-1], 0, HALL_HEIGHT],
    )
    for zm in MAST_Z:
        axes[0].axhline(zm, color="w", ls=":", lw=1)
    axes[0].axhline(lk.z, color="c", lw=1.5, label="leak height")
    axes[0].axvline(t_a, color="r", lw=1.5, label="first alarm")
    axes[0].set(
        title="Gas in height over time at the mast (1 m from the pipe)",
        xlabel="seconds since leak start",
        ylabel="height (m)",
    )
    axes[0].legend(loc="upper left", fontsize=7)
    plt.colorbar(im, ax=axes[0], label="concentration")
    for i, zm, c in zip(idx, MAST_Z, ("tab:brown", "tab:green", "tab:blue"), strict=True):
        axes[1].plot(times, vals[0, i], "o-", color=c, label=f"mast detector at {zm} m")
    axes[1].axhline(0.03, color="r", ls="--", lw=1, label="alarm level")
    axes[1].axvline(t_a, color="r", lw=1)
    axes[1].set(
        title="What the three mast detectors read (noisy)",
        xlabel="seconds since leak start",
        ylabel="reading",
    )
    axes[1].legend(fontsize=7)
    return t_a


def data_figure(factory: Factory, gas: str, path: Path) -> None:
    """A low (1.5 m) and a high (6 m) leak on the same vertical pipe."""
    phys = Physics(GASES[gas].sg, mixing=0.02, tilt=1.0)
    risers = pick_risers(factory, N_RISERS)
    src = risers[SHOWN_RISER]
    sc = Scenario(src.copy(), 1.0, SHOWN_WIND)
    times = np.arange(0.0, 70.0 + 1e-9, 5.0)
    fields = simulate(factory, [sc], times)[0]
    mast = layouts(factory, risers)["masts"]
    idx = len(factory.sensors) + 3 * SHOWN_RISER + np.arange(3)
    x_p = passive_distance(phys.sg, LITRES_PER_RATE_UNIT / 1000.0, 0.8)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for row, z_leak in enumerate((1.5, 6.0)):
        t_a = _mast_panels(
            ax[row, 1:], factory, fields, times, Leak3D(sc, z_leak, phys, SHOWN_RISER), mast, idx
        )
        j = int(np.searchsorted(times, t_a + 15.0))
        _side_view(ax[row, 0], factory, fields[j], src, z_leak, times[j], phys, x_p)
        ax[row, 0].plot(np.full(3, mast.xy[idx[0], 0]), MAST_Z, "w^", ms=8, label="mast")
        ax[row, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"{GAS_LABEL[gas]}: leak low (1.5 m, top) vs high (6 m, bottom) on the same vertical pipe",
        fontsize=13,
    )
    fig.savefig(path, dpi=85)
    plt.close(fig)


def detection_figure(res: dict[str, Result], gas: str, model: str, path: Path) -> None:
    """Error over time, predicted vs true (masts), error by true height."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for name, r in res.items():
        ax[0].plot(
            WS,
            r.error(),
            "o-",
            color=COLOUR[name],
            label=f"{LAYOUT_LABEL[name]} ({r.n_sensors} detectors; {r.n_test} leaks alarmed)",
        )
    ax[0].axhline(PRIOR_ERROR, color="k", ls="--", label="guess the middle (prior)")
    ax[0].set(
        title="Height error (median) after the alarm",
        xlabel="seconds since alarm",
        ylabel="|predicted - true| height (m)",
        ylim=(0, None),
    )
    ax[0].legend(fontsize=7)
    r, m = res["masts"], res["masts"].w == 15.0
    ax[1].errorbar(
        r.z[m],
        r.med[m],
        yerr=[r.med[m] - r.lo[m], r.hi[m] - r.med[m]],
        fmt="o",
        ms=3,
        alpha=0.5,
        color=COLOUR["masts"],
        ecolor="0.75",
    )
    ax[1].plot(RISER_RANGE, RISER_RANGE, "k-", lw=1)
    ax[1].set(
        title="Masts, alarm + 15 s: predicted vs true (80 % interval)",
        xlabel="true leak height (m)",
        ylabel="predicted height (m)",
    )
    for i, (name, r) in enumerate(res.items()):
        m = r.w == 15.0
        band = np.digitize(r.z[m], [2.5, 4.75])
        err = [np.median(np.abs(r.med[m] - r.z[m])[band == b]) for b in range(3)]
        ax[2].bar(
            np.arange(3) + (i - 1) * 0.27, err, 0.27, color=COLOUR[name], label=LAYOUT_LABEL[name]
        )
    ax[2].axhline(PRIOR_ERROR, color="k", ls="--")
    ax[2].set_xticks(range(3), ["low leak\n0.5-2.5 m", "middle\n2.5-4.75 m", "high leak\n4.75-7 m"])
    ax[2].set(title="Error by true leak height, alarm + 15 s", ylabel="median error (m)")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"{GAS_LABEL[gas]}: how well the height is found ({model})", fontsize=13)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def comparison_figure(model: dict, knn: dict, name: str, path: Path) -> None:
    """Both gases: ``name`` (solid) vs k-NN (dashed) for every layout."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, sharey=True)
    for a, gas in zip(ax, GAS_LABEL, strict=True):
        for lay in LAYOUT_LABEL:
            a.plot(
                WS,
                model[gas][lay].error(),
                "o-",
                color=COLOUR[lay],
                label=f"{name}: {LAYOUT_LABEL[lay]}",
            )
            a.plot(
                WS,
                knn[gas][lay].error(),
                "x--",
                color=COLOUR[lay],
                alpha=0.6,
                label=f"k-NN: {LAYOUT_LABEL[lay]}",
            )
        a.axhline(PRIOR_ERROR, color="k", ls=":", label="guess the middle")
        a.set(
            title=GAS_LABEL[gas],
            xlabel="seconds since alarm",
            ylabel="median height error (m)",
            ylim=(0, None),
        )
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Leak height: {name} (solid) vs k-NN (dashed), same rows", fontsize=13)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def main() -> None:
    """CLI: draw the figures and print the numbers."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="baseline", choices=("baseline", "cloud", "local"))
    p.add_argument("--out", type=Path, default=Path("outputs/source_height"))
    args = p.parse_args()
    load_env()
    args.out.mkdir(parents=True, exist_ok=True)
    f = Factory.default()
    knn, model = {}, {}
    for gas in GAS_LABEL:
        data_figure(f, gas, args.out / f"{gas}_1_data.png")
        knn[gas] = run_study(f, gas, "baseline", cache_dir=args.out / "cache")
        model[gas] = run_study(f, gas, args.backend, cache_dir=args.out / "cache")
        detection_figure(
            model[gas], gas, args.backend, args.out / f"{gas}_2_detection_{args.backend}.png"
        )
        for lay, r in model[gas].items():
            print(
                f"{gas:8s} {lay:14s} detectors {r.n_sensors:2d} alarmed {r.n_test:3d}/200 "
                f"error {np.round(r.error(), 2)} cover80 {np.round(r.coverage(), 2)}"
            )
    comparison_figure(model, knn, args.backend, args.out / f"height_{args.backend}_vs_knn.png")
    print(f"prior (guess the middle) error {PRIOR_ERROR} m; figures in {args.out}")


if __name__ == "__main__":
    main()
