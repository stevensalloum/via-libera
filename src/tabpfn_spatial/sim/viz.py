"""Pictures of the simulator, to check it by eye: ``uv run sim-viz``.

- ``plot_layout``: the plant (machines, hot ones, wall and doors, exits, sensors, leak sites,
  ignition zones).
- ``plot_timelapse``: gas concentration over time for one or more leaks, with the toxic and
  flammable contours and the wind arrow at each moment.
- ``plot_readings``: what each sensor measured (dead sensors listed).
- ``plot_random_layouts``: several ``Factory.random`` plants side by side.
"""

import argparse
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import LEL, TOXIC_LIMIT, SensorReadings
from tabpfn_spatial.sim.factory import Factory
from tabpfn_spatial.sim.plume import Scenario, simulate, wind_at
from tabpfn_spatial.sim.scenarios import readings


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    _plt().close(fig)
    return path


def _draw_plant(ax, f: Factory, labels: bool = True) -> None:
    """Solid cells in grey, hot machines in orange, exits and sensors on top."""
    ext = (0, f.width, 0, f.height)
    ax.imshow(
        np.ma.masked_where(~f.walls, f.walls),
        origin="lower",
        extent=ext,
        cmap="Greys",
        vmin=0,
        vmax=1.4,
        alpha=0.9,
    )
    for m in f.machines:
        if m.hot:
            ax.add_patch(
                _plt().Rectangle(
                    (m.x0, m.y0), m.x1 - m.x0, m.y1 - m.y0, fill=False, ec="orangered", lw=2
                )
            )
        if labels:
            ax.text(
                (m.x0 + m.x1) / 2,
                (m.y0 + m.y1) / 2,
                m.name,
                ha="center",
                va="center",
                fontsize=7,
                color="orangered" if m.hot else "black",
            )
    ax.scatter(*f.exits.T, marker="D", s=60, c="gold", edgecolors="k", zorder=5)
    ax.scatter(*f.sensors.T, marker="s", s=28, c="deepskyblue", edgecolors="k", zorder=5)
    if labels:
        for i, (x, y) in enumerate(f.exits):
            ax.annotate(f"E{i}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        for i, (x, y) in enumerate(f.sensors):
            ax.annotate(f"S{i}", (x, y), xytext=(3, -9), textcoords="offset points", fontsize=6)
    ax.set_xlim(0, f.width)
    ax.set_ylim(0, f.height)
    ax.set_aspect("equal")


def plot_layout(f: Factory, path: Path) -> Path:
    """The plant with leak sites (small dots) and ignition zones (orange tint)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 7))
    ign = np.ma.masked_where(~f.ignition, f.ignition)
    ax.imshow(
        ign,
        origin="lower",
        extent=(0, f.width, 0, f.height),
        cmap="Oranges",
        vmin=0,
        vmax=3,
        alpha=0.6,
    )
    ax.scatter(*f.leak_sites.T, s=6, c="purple", alpha=0.5, label="possible leak sites")
    _draw_plant(ax, f)
    ax.set_title(
        f"Factory {f.width:.0f} m x {f.height:.0f} m, {f.cell:.0f} m cells: grey = solid, "
        "orange box = hot machine, orange tint = ignition zone, "
        "diamonds = exits, squares = sensors",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    return _save(fig, path)


def plot_timelapse(
    f: Factory, scenarios: dict[str, Scenario], times: np.ndarray, path: Path
) -> Path:
    """Rows = leaks, columns = times; log colour scale; white = toxic, red = flammable."""
    plt = _plt()
    from matplotlib.colors import LogNorm

    names = list(scenarios)
    fields = simulate(f, [scenarios[n] for n in names], times)
    fig, axes = plt.subplots(
        len(names), len(times), figsize=(3.6 * len(times), 2.7 * len(names)), squeeze=False
    )
    gx, gy = f.centres()[..., 0], f.centres()[..., 1]
    for i, name in enumerate(names):
        sc = scenarios[name]
        winds = wind_at(sc, times)
        for j, t in enumerate(times):
            ax = axes[i, j]
            img = np.ma.masked_where(f.walls, np.maximum(fields[i, j], 1e-3))
            ax.imshow(
                img,
                origin="lower",
                extent=(0, f.width, 0, f.height),
                cmap="viridis",
                norm=LogNorm(1e-3, 10),
            )
            ax.contour(gx, gy, fields[i, j], levels=[TOXIC_LIMIT], colors="white", linewidths=1)
            ax.contour(gx, gy, fields[i, j], levels=[LEL], colors="red", linewidths=1)
            _draw_plant(ax, f, labels=False)
            ax.scatter(*sc.source, marker="*", s=150, c="cyan", edgecolors="k", zorder=6)
            ax.arrow(4, 35, 4 * winds[j, 0], 4 * winds[j, 1], width=0.5, color="white", zorder=7)
            ax.set_title(f"{name}: t = {t:.0f} s", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Gas concentration (log scale). White line = toxic limit, red = flammable "
        "(LEL). Arrow = wind now.",
        fontsize=10,
    )
    return _save(fig, path)


def plot_readings(f: Factory, r: SensorReadings, path: Path) -> Path:
    """Each sensor's series (log scale) with the toxic limit; dead sensors in the title."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dead = [i for i in range(len(r.values)) if np.isnan(r.values[i]).all()]
    for i, series in enumerate(r.values):
        if i not in dead:
            ax.plot(r.times, np.maximum(series, 1e-4), marker=".", label=f"S{i}")
    ax.axhline(TOXIC_LIMIT, color="k", ls="--", lw=1, label="toxic limit")
    ax.set_yscale("log")
    ax.set_xlabel("seconds since the leak started")
    ax.set_ylabel("measured concentration")
    ax.set_title(
        f"Sensor readings up to t_obs = {r.t_obs:.0f} s; measured wind "
        f"({r.wind[0]:.2f}, {r.wind[1]:.2f}) m/s; dead sensors: {dead or 'none'}",
        fontsize=9,
    )
    ax.legend(ncol=8, fontsize=7, loc="upper left")
    return _save(fig, path)


def plot_random_layouts(seeds: list[int], path: Path) -> Path:
    """``Factory.random`` for each seed, side by side."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.6 * len(seeds), 3.4), squeeze=False)
    for ax, s in zip(axes[0], seeds, strict=True):
        _draw_plant(ax, Factory.random(np.random.default_rng(s)), labels=False)
        ax.set_title(f"Factory.random(seed={s})", fontsize=9)
    return _save(fig, path)


def main() -> int:
    """CLI: ``uv run sim-viz [--out outputs/sim]`` writes the four figures."""
    p = argparse.ArgumentParser(description="Draw the simulator for a visual check.")
    p.add_argument("--out", type=Path, default=Path("outputs/sim"))
    args = p.parse_args()
    f = Factory.default()
    src = np.array([23.0, 9.0])  # next to compressor A / tank farm
    leaks = {
        "steady wind": Scenario(src, 1.0, np.array([0.8, 0.4])),
        "drifting wind": Scenario(src, 1.0, np.array([0.8, 0.4]), drift=0.08, seed=3),
    }
    times = np.array([10.0, 30.0, 60.0, 120.0, 180.0])
    rng = np.random.default_rng(0)
    out = [
        plot_layout(f, args.out / "1_layout.png"),
        plot_timelapse(f, leaks, times, args.out / "2_timelapse.png"),
        plot_readings(
            f,
            readings(f, leaks["drifting wind"], 60.0, rng, dropout=0.15),
            args.out / "3_readings.png",
        ),
        plot_random_layouts([1, 2, 3, 4], args.out / "4_random_layouts.png"),
    ]
    for o in out:
        print(o)
    return 0
