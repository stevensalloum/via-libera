"""Watch a whole crowd evacuate, the three planners side by side: ``uv run evac-crowd-live``.

One leak, one crowd, the same alarm and model maps; each panel is one planner. Gas in
colour (log scale), truly harmful area as a white line, the planners' danger map as red
dashes. People: green walking, orange queueing (blocked by a door or the crowd), magenta
standing in harmful gas right now. Bottom: share of people out and mean seconds in harm
over time, per planner.

Keys: space = pause, n = next leak, q = quit. ``--gif out.gif`` records instead,
``--still out.png`` writes four moments for the journey.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.crowd import CrowdParams, sample_workers
from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_run import Prepared, prepare, run_crowd
from tabpfn_spatial.evac.crowd_study import COLOURS, RUNGS, Rung
from tabpfn_spatial.sim import Factory, sample_scenarios


@dataclass
class Recording:
    """Everything needed to draw one leak for every planner."""

    rung: Rung
    seed: int
    models: str
    f: Factory
    prep: Prepared
    frames: dict[str, list[dict]]
    metrics: dict[str, dict[str, float]]


def record(rung: Rung, seed: int, models: str = "perfect") -> Recording:
    """Run every planner on one leak, keeping a snapshot every second."""
    f = rung.plant()
    rng = np.random.default_rng(seed)
    sc = sample_scenarios(f, 1, rng)[0]
    people = sample_workers(f, rung.people, rng)
    prep = prepare(f, sc, models, rung.t_max, rung.update_s, seed)
    params = CrowdParams(door_m=rung.door_m)
    runs = {p: run_crowd(f, prep, people, p, params, record_every=1) for p in PLANNERS}
    return Recording(rung, seed, models, f, prep, {p: r.frames for p, r in runs.items()},
                     {p: r.metrics for p, r in runs.items()})  # fmt: skip


def _danger_at(prep: Prepared, t: float) -> np.ndarray | None:
    past = np.flatnonzero(prep.replan_times <= t + 1e-9)
    return prep.danger[past[-1]] if len(past) else None


def draw_panel(ax, rec: Recording, planner: str, k: int) -> None:
    """One planner at frame ``k`` (cleared first)."""
    from matplotlib.colors import LogNorm

    from tabpfn_spatial.sim.viz import draw_pipes

    f, prep = rec.f, rec.prep
    fr = rec.frames[planner][min(k, len(rec.frames[planner]) - 1)]
    ax.clear()
    ext = (0, f.width, 0, f.height)
    gas = np.ma.masked_where(f.walls, np.maximum(prep.fields[fr["k"]], 1e-3))
    ax.imshow(gas, origin="lower", extent=ext, cmap="viridis", norm=LogNorm(1e-3, 10))
    ax.imshow(np.ma.masked_where(~f.walls, f.walls), origin="lower", extent=ext,
              cmap="Greys", vmin=0, vmax=1.4)  # fmt: skip
    draw_pipes(ax, f)
    gx, gy = f.centres()[..., 0], f.centres()[..., 1]
    if prep.harm[fr["k"]].any():
        ax.contour(gx, gy, prep.harm[fr["k"]].astype(float), levels=[0.5], colors="white",
                   linewidths=1.2)  # fmt: skip
    d = _danger_at(prep, fr["t"])
    if d is not None and d.max() >= 0.5:
        ax.contour(gx, gy, d, levels=[0.5], colors="red", linewidths=1.2, linestyles="--")
    a = fr["active"]
    xy = (fr["cells"][a][:, ::-1] + 0.5) * f.cell
    xy = xy + np.random.default_rng(0).uniform(-0.4, 0.4, xy.shape) * f.cell  # spread dots
    harm_now = prep.harm[fr["k"]][fr["cells"][a, 0], fr["cells"][a, 1]]
    colour = np.where(harm_now, "#ff00ff", np.where(fr["waiting"][a], "#ff9800", "#00e676"))
    size = max(2.0, 30.0 / np.sqrt(len(fr["active"]) / 50))
    ax.scatter(*xy.T, s=size, c=colour, edgecolors="none", zorder=7)
    ax.scatter(*f.exits.T, marker="D", s=70, c="gold", edgecolors="k", zorder=8)
    ax.scatter(*prep.scenario.source, marker="*", s=160, c="cyan", edgecolors="k", zorder=8)
    out = 100 * (1 - a.mean())
    ax.set_title(f"{planner}: {out:3.0f} % out, harm {fr['harm_s'].mean():.2f} s/person",
                 fontsize=10, color=COLOURS[planner])  # fmt: skip
    ax.set_xlim(0, f.width)
    ax.set_ylim(0, f.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_curves(ax_out, ax_harm, rec: Recording, k: int) -> None:
    """Share out and mean harm-seconds over time, up to frame ``k``."""
    ends = [m["out_all_s"] for m in rec.metrics.values() if np.isfinite(m["out_all_s"])]
    right = min(rec.rung.t_max, (max(ends) if ends else rec.rung.t_max) + 15)
    for ax in (ax_out, ax_harm):
        ax.clear()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, right)
        ax.tick_params(labelsize=7)
    for p in PLANNERS:
        fr = rec.frames[p][: k + 1]
        t = [x["t"] for x in fr]
        ax_out.plot(t, [100 * (1 - x["active"].mean()) for x in fr], c=COLOURS[p], label=p)
        ax_harm.plot(t, [x["harm_s"].mean() for x in fr], c=COLOURS[p], label=p)
    if rec.prep.t_alarm is not None:
        for ax in (ax_out, ax_harm):
            ax.axvline(rec.prep.t_alarm, c="k", ls=":", lw=1)
    ax_out.set_ylabel("% out", fontsize=8)
    ax_harm.set_ylabel("harm s / person", fontsize=8)
    ax_harm.set_xlabel("seconds since the leak started (dotted: alarm)", fontsize=8)
    ax_out.legend(fontsize=7, loc="lower right")


def _figure(rec: Recording, scale: float = 1.0):
    import matplotlib.pyplot as plt

    aspect = rec.f.height / rec.f.width
    fig = plt.figure(figsize=(16 * scale, (16 / 3 * aspect + 2.6) * scale))
    grid = fig.add_gridspec(3, 3, height_ratios=(16 / 3 * aspect, 1.1, 1.1))
    maps = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_out = fig.add_subplot(grid[1, :])
    ax_harm = fig.add_subplot(grid[2, :], sharex=ax_out)
    return fig, maps, ax_out, ax_harm


def _draw_all(fig, maps, ax_out, ax_harm, rec: Recording, k: int) -> None:
    for ax, p in zip(maps, PLANNERS, strict=True):
        draw_panel(ax, rec, p, k)
    draw_curves(ax_out, ax_harm, rec, k)
    t = rec.frames[PLANNERS[0]][min(k, len(rec.frames[PLANNERS[0]]) - 1)]["t"]
    fig.suptitle(
        f"{rec.rung.name}: {rec.rung.label} | leak seed {rec.seed} | {rec.models} models | "
        f"t = {t:.0f} s | green walking, orange queueing, magenta in harmful gas",
        fontsize=10,
    )


def save_gif(rec: Recording, path: Path, every: int = 3, fps: int = 10) -> Path:
    """Record the evacuation (every ``every``-th second) until everyone is out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    fig, maps, ax_out, ax_harm = _figure(rec, scale=0.75)
    last = max(int(np.nanmax([m["out_all_s"] for m in rec.metrics.values()] or [0])) + 5, 30)
    last = min(last, len(rec.frames[PLANNERS[0]]) - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PillowWriter(fps=fps).saving(fig, str(path), dpi=60) as w:
        for k in range(0, last + 1, every):
            _draw_all(fig, maps, ax_out, ax_harm, rec, k)
            w.grab_frame()
    plt.close(fig)
    return path


def save_still(rec: Recording, path: Path, moment: float | None = None) -> Path:
    """One moment (default: 40 s after the alarm) with the curves for the whole run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alarm = rec.prep.t_alarm or 0.0
    k = int(min(moment if moment is not None else alarm + 40, rec.rung.t_max))
    fig, maps, ax_out, ax_harm = _figure(rec)
    _draw_all(fig, maps, ax_out, ax_harm, rec, k)
    draw_curves(ax_out, ax_harm, rec, len(rec.frames[PLANNERS[0]]) - 1)
    for ax in (ax_out, ax_harm):
        ax.axvline(k, c="grey", lw=1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def main() -> int:
    """CLI: ``uv run evac-crowd-live [--rung B] [--seed 0] [--models perfect]``."""
    p = argparse.ArgumentParser(description="Watch the three crowd planners side by side.")
    p.add_argument("--rung", default="B", choices=[r.name for r in RUNGS])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", default="perfect", choices=("perfect", "baseline"))
    p.add_argument("--gif", type=Path)
    p.add_argument("--still", type=Path)
    p.add_argument("--at", type=float, help="moment for --still (s since the leak)")
    a = p.parse_args()
    rung = next(r for r in RUNGS if r.name == a.rung)
    rec = record(rung, a.seed, a.models)
    for pl, m in rec.metrics.items():
        print(f"{pl:8} " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()))
    if a.gif:
        print(save_gif(rec, a.gif))
        return 0
    if a.still:
        print(save_still(rec, a.still, a.at))
        return 0
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, maps, ax_out, ax_harm = _figure(rec)
    state = {"k": 0, "paused": False, "rec": rec}

    def tick(_):
        r = state["rec"]
        if not state["paused"]:
            state["k"] = min(state["k"] + 2, len(r.frames[PLANNERS[0]]) - 1)
        _draw_all(fig, maps, ax_out, ax_harm, r, state["k"])

    def key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "n":
            r = state["rec"]
            state["rec"], state["k"] = record(r.rung, r.seed + 1, r.models), 0

    fig.canvas.mpl_connect("key_press_event", key)
    anim = FuncAnimation(fig, tick, interval=80, cache_frame_data=False)
    plt.show()
    del anim
    return 0
