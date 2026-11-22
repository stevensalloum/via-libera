"""Overview figures of the four real datasets.

Run: ``uv run python -m tabpfn_spatial.study3d.overview``.

One figure per dataset, in ``docs/figures/3d-study/``; each answers "what does this dataset
tell us about where gas goes, especially in height?".

Normalisations (documented in docs/research/3d-study-plan.md):
- Red:Vapor PID is relative and the closed tunnel builds up a background. Background =
  median PID of the cells **upwind** of the source (x > source x + 0.2 m: no fog from the
  release can get there); excess = PID - background, floored at 0.
- Örebro MOX are calibrated to ppm but some channels drift below 0; each sensor's
  baseline = its median over the first 5 minutes (before the source reaches it).
- UCI: response = R / R0 - 1, R0 = mean value in the 20 s before the valve opens. The stored
  value *rises* in gas although the dataset page calls it a resistance in kOhm (a MOX
  resistance falls in reducing gas), so we use the relative change and not its sign.
- SRGDM values are already normalised to 0..1 by the authors.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tabpfn_spatial.realdata import orebro, redvapor, srgdm, uci

DATA = Path("data/real")
OUT = Path("docs/figures/3d-study")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"


def _style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def redvapor_excess(run: redvapor.VoxelRun) -> np.ndarray:
    """PID above the upwind background (relative units, >= 0)."""
    upwind = run.xyz[:, 0] > run.source[0] + 0.2
    return np.clip(run.pid - np.nanmedian(run.pid[upwind]), 0, None)


def redvapor_profile(run: redvapor.VoxelRun, half_width: float = 0.3) -> tuple:
    """Mean excess per height, downwind of the source and within ``half_width`` of its axis.

    The wind blows towards -x, so "downwind" is x < source x; the axis is the line
    y = source y (the turntable rotation moves the source, not the wind).
    """
    ex = redvapor_excess(run)
    m = (run.xyz[:, 0] < run.source[0]) & (np.abs(run.xyz[:, 1] - run.source[1]) < half_width)
    z = np.round(run.xyz[m, 2], 3)
    zs = np.unique(z)
    return zs, np.array([np.nanmean(ex[m][z == v]) for v in zs])


def fig_redvapor(out: Path) -> Path:
    """Red:Vapor: vertical profiles downwind + a horizontal slice."""
    runs = redvapor.load_voxel_maps(DATA / "redvapor/voxel_maps.pkl")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[1, 1.25])
    colors = dict(zip("ABCD", PALETTE, strict=False))
    for r in runs:
        zs, prof = redvapor_profile(r)
        slow = r.wind_speed < 2
        a.plot(prof / np.nanmax(prof), zs, color=colors[r.setup], lw=2,
               ls="-" if slow else "--", marker="o", ms=3,
               label=f"exp {r.experiment}: setup {r.setup}, {r.wind_speed:.1f} m/s")  # fmt: skip
    a.axhline(0.18, color=MUTED, lw=1, ls=":")
    a.text(1.02, 0.18, "release\nheight", color=MUTED, fontsize=7, va="center")
    a.set_xlabel("fog excess, normalised to each run's peak")
    a.set_ylabel("height z (m, 1:50 model)")
    a.set_title("Vertical profile downwind of the leak", fontsize=10, color=INK, loc="left")
    a.legend(fontsize=7, frameon=False, loc="upper right")
    _style(a)
    r = next(x for x in runs if x.experiment == 4)
    ex = redvapor_excess(r)
    zband = np.abs(r.xyz[:, 2] - 0.185) < 0.05
    xs, ys = np.unique(np.round(r.xyz[zband, 0], 3)), np.unique(np.round(r.xyz[zband, 1], 3))
    grid = np.full((ys.size, xs.size), np.nan)
    ix = np.searchsorted(xs, np.round(r.xyz[zband, 0], 3))
    iy = np.searchsorted(ys, np.round(r.xyz[zband, 1], 3))
    grid[iy, ix] = ex[zband]
    im = b.pcolormesh(xs, ys, grid, cmap="Blues", shading="nearest")
    b.plot(*r.source[:2], marker="*", color=PALETTE[1], ms=14, mec="white", label="leak")
    b.annotate("", xy=(-0.9, 0.95), xytext=(-0.4, 0.95),
               arrowprops={"arrowstyle": "->", "color": INK})  # fmt: skip
    b.text(-0.65, 1.0, "wind", ha="center", fontsize=8, color=INK)
    b.set_aspect("equal")
    b.set_xlabel("x (m), +x = upwind")
    b.set_ylabel("y (m)")
    b.set_title("Exp 4 (full site C, 1.4 m/s): horizontal slice at release height",
                fontsize=10, color=INK, loc="left")  # fmt: skip
    fig.colorbar(im, ax=b, label="fog excess (relative PID units)", shrink=0.85)
    b.legend(fontsize=8, frameon=False, loc="lower left")
    _style(b)
    fig.suptitle("Red:Vapor: real fog plume around a 1:50 industrial site (wind tunnel)",
                 fontsize=11, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def orebro_layers(run: orebro.Run, skip_s: float = 300.0) -> np.ndarray:
    """Mean concentration (ppm above each sensor's start baseline) per height layer (3,)."""
    base = np.nanmedian(run.c[run.t < skip_s], 0)
    c = run.c[run.t >= skip_s] - base
    z = run.xyz[:, 2]
    return np.array([np.nanmean(c[:, z == h]) for h in orebro.HEIGHTS])


def fig_orebro(out: Path) -> Path:
    """Örebro: mean concentration per height layer, one panel per experiment."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2), sharey=True)
    for ax, e in zip(axes.flat, sorted(orebro.EXPERIMENTS), strict=True):
        run = orebro.load(DATA / "orebro", e)
        info = orebro.EXPERIMENTS[e]
        lay = orebro_layers(run)
        ax.plot(lay, orebro.HEIGHTS, color=PALETTE[0], lw=2, marker="o", ms=6)
        ax.axhline(info.source[2], color=PALETTE[1], lw=1.5, ls="--")
        ax.set_title(f"exp {e}: {info.airflow}, {info.beaker}", fontsize=8, color=INK, loc="left")
        ax.set_xlim(left=min(0, lay.min() * 1.1))
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("height (m)")
    for ax in axes[1]:
        ax.set_xlabel("mean ppm above baseline")
    axes[0, 0].plot([], [], color=PALETTE[0], lw=2, marker="o", label="mean of the 9 sensors")
    axes[0, 0].plot([], [], color=PALETTE[1], lw=1.5, ls="--", label="source height")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Örebro 3DSEN: real indoor office, 27 MOX at 3 heights; HVAC in at floor, "
                 "out at ceiling", fontsize=11, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def uci_response(d: dict) -> np.ndarray:
    """Per run and channel: mean R/R0 - 1 while gas flows (60-200 s), R0 = 0-15 s (runs, 72)."""
    s = d["sensors"].astype(np.float32)
    t = uci.GRID_S
    r0 = np.nanmean(s[:, t < 20], 1)
    return np.nanmean(s[:, (t >= 60) & (t <= 200)], 1) / r0 - 1


def fig_uci(out: Path) -> Path:
    """UCI: centre-board response vs distance, and across the tunnel at L1."""
    d = uci.load(DATA / "uci/uci251_500V_5s.npz")
    resp = uci_response(d)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (fan, rpm) in enumerate(uci.FAN_RPM.items()):
        m = d["fan"] == fan
        centre = resp[:, 32:40]  # board 5: the board on the plume axis
        med = [np.nanmedian(centre[m & (d["line"] == L)]) for L in range(1, 7)]
        q = [np.nanpercentile(centre[m & (d["line"] == L)], [25, 75]) for L in range(1, 7)]
        a.fill_between(range(1, 7), [x[0] for x in q], [x[1] for x in q], color=PALETTE[k],
                       alpha=0.15, lw=0)  # fmt: skip
        a.plot(range(1, 7), med, color=PALETTE[k], lw=2, marker="o", ms=5, label=f"fan {rpm} rpm")
        m1 = m & (d["line"] == 1)
        boards = resp[m1].reshape(-1, 9, 8)
        b.plot(range(1, 10), np.nanmedian(boards, (0, 2)), color=PALETTE[k], lw=2, marker="o",
               ms=5, label=f"fan {rpm} rpm")  # fmt: skip
    a.set_xlabel("line position (L1 nearest the gas inlet)")
    a.set_ylabel("relative change R/R0 - 1 (board 5, median, IQR)")
    a.set_title("Response along the wind", fontsize=10, color=INK, loc="left")
    b.set_xlabel("board across the tunnel (5 = plume axis)")
    b.set_ylabel("relative change R/R0 - 1 at L1 (median)")
    b.set_title("Response across the wind, nearest line", fontsize=10, color=INK, loc="left")
    for ax in (a, b):
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
    title = f"UCI 251: real MOX in an open wind tunnel, {len(d['gas'])} runs, 10 gases"
    fig.suptitle(title + " (5.0 V heater, 5 s samples)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_srgdm(out: Path) -> Path:
    """SRGDM: synthetic mean / frame / sparse sensors, and the real grid."""
    s = srgdm.load_tensor(DATA / "srgdm/30x25/valid.file")
    real = srgdm.load_tensor(DATA / "srgdm/6x5_realData/02apr.file")
    i = 11  # a central source, first season
    rows, cols = srgdm.sensor_cells()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    panels = [(s[i].mean(0), "synthetic: time mean (30 x 25)"),
              (s[i, 90], "synthetic: one frame"),
              (s[i].mean(0)[np.ix_(rows, cols)], "synthetic at the 6 x 5 sensors"),
              (real[0].mean(0), "real 6 x 5 grid: time mean")]  # fmt: skip
    for ax, (img, title) in zip(axes, panels, strict=True):
        im = ax.imshow(img, cmap="Blues", vmin=0, origin="upper")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    r, c = srgdm.source_cell(i)
    for ax in axes[:2]:
        ax.plot(c, r, marker="*", color=PALETTE[1], ms=12, mec="white")
    fig.suptitle("SRGDM: 2D plumes from measured airflow (synthetic) and a real 6 x 5 sensor "
                 "grid; values normalised 0..1 by the authors", fontsize=11, color=INK, x=0.01,
                 ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    """Write the four figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f, name in [(fig_redvapor, "redvapor"), (fig_orebro, "orebro"), (fig_uci, "uci"),
                    (fig_srgdm, "srgdm")]:  # fmt: skip
        print(f(OUT / f"data_{name}.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
