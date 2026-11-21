"""Watch one person evacuate while the team's models run back to back: ``uv run evac-pipeline``.

Left: the hall with the true gas, the person (``live`` follows every new route; ``once``
only the first; ``nearest`` ignores the gas), the route currently followed (the best way
the planner found), Ahmad's predicted danger and Rayan's leak guess. Right: what each
model said at the last run: Rayan's leak map, Ahmad's P(harm) now, +30 s and +60 s.
Bottom: the pipeline over time: each run's locator time (cyan) and hazard time (red),
the moment its route reached the person, and the readings it used.

``--models team`` (default): Rayan's ``TabPFNLocator`` (live mode: map first) and Ahmad's
``TabPFNHazardPredictor``; ``--backend baseline`` runs them with the offline stand-in (no
tokens), ``--backend cloud --yes`` with TabPFN (about 3 calls of >= 10,000 tokens per
run). ``--models perfect`` / ``baseline`` swap in the oracle or the simple baselines.
The run is computed first (the terminal shows each run as it lands), then played.
"""

import argparse
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.live import POLICIES, PipelineFrame, run_pipeline
from tabpfn_spatial.evac.standins import OracleHazard, OracleLocator
from tabpfn_spatial.sim import Factory, Scenario, sample_people, sample_scenarios
from tabpfn_spatial.sim.viz import draw_pipes

STYLE = {"live": ("#00e676", "-"), "once": ("#ffa726", "--"), "nearest": ("#ff1744", ":")}


def models_for(name: str, sc: Scenario, f: Factory, backend: str):
    """(locator, predictor) for ``team`` / ``perfect`` / ``baseline``."""
    if name == "perfect":
        return OracleLocator(sc, 0.0), OracleHazard(sc)
    if name == "baseline":
        from tabpfn_spatial.hazard import ForwardSimPredictor
        from tabpfn_spatial.source import WeightedCentroidLocator

        return WeightedCentroidLocator(), ForwardSimPredictor()
    if name != "team":
        raise ValueError(f"models must be team, perfect or baseline, not {name!r}")
    from tabpfn_spatial.hazard import TabPFNHazardPredictor
    from tabpfn_spatial.source import TabPFNLocator

    return TabPFNLocator(backend=backend, live=True).fit(f), TabPFNHazardPredictor(
        backend=backend
    ).fit(f)


def compute(f: Factory, seed: int, start, models: str, backend: str, max_runs: int, t_max: float,
            log=print) -> tuple[Scenario, list[PipelineFrame]]:  # fmt: skip
    """Run the whole emergency; print each run when its route lands."""
    rng = np.random.default_rng(seed)
    sc = sample_scenarios(f, 1, rng)[0]
    start = sample_people(f, 1, rng)[0] if start is None else np.asarray(start, float)
    log(f"leak at ({sc.source[0]:.0f}, {sc.source[1]:.0f}) m, size {sc.rate:.2f}")
    log(f"person at ({start[0]:.0f}, {start[1]:.0f}) m; fitting the {models} models ({backend})")
    loc, pred = models_for(models, sc, f, backend)
    frames, seen = [], 0
    for fr in run_pipeline(f, sc, start, loc, pred, max_runs=max_runs, t_max=t_max,
                           rng=np.random.default_rng(seed + 1)):  # fmt: skip
        frames.append(fr)
        if len(fr.cycles) > seen:
            c = fr.cycles[-1]
            w = fr.walkers["live"]
            rayan = f"leak {c.source_error_m:.1f} m off, size {c.rate_est:.2f} (true {sc.rate:.2f})"
            log(f"run {c.index + 1}: readings 0-{c.t_start:.0f} s ({c.samples} per sensor) | "
                f"Rayan {c.locate_s:.2f} s: {rayan} | Ahmad {c.predict_s:.2f} s | "
                f"route lands at {c.t_land:.0f} s -> exit E{w.exit_index}")  # fmt: skip
            seen = len(fr.cycles)
    last = frames[-1].walkers
    for p in POLICIES:
        w = last[p]
        out = f"out at {w.out_at:.0f} s" if w.out_at is not None else "still inside"
        log(
            f"{p:8} exit E{w.exit_index}, {out}, {w.harmful_s:.0f} s in harmful gas, "
            f"dose {w.dose:.2f}"
        )
    return sc, frames


def _plant(ax, f: Factory, labels: bool = True) -> None:
    ext = (0, f.width, 0, f.height)
    ax.imshow(np.ma.masked_where(~f.walls, f.walls), origin="lower", extent=ext, cmap="Greys",
              vmin=0, vmax=1.4)  # fmt: skip
    draw_pipes(ax, f)
    ax.scatter(*f.exits.T, marker="D", s=70, c="gold", edgecolors="k", zorder=6)
    if labels:
        for i, (x, y) in enumerate(f.exits):
            off = (-16 if x > f.width / 2 else 6, -14 if y > f.height / 2 else 6)
            ax.annotate(f"E{i}", (x, y), xytext=off, textcoords="offset points", fontsize=8,
                        color="white", weight="bold")  # fmt: skip
    ax.set_xlim(0, f.width)
    ax.set_ylim(0, f.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _danger_at(fr: PipelineFrame, t: float) -> np.ndarray | None:
    hz = fr.hazard
    if hz is None:
        return None
    i = int(np.argmin(np.abs(hz.times - t)))
    return 1 - (1 - hz.p_toxic[i]) * (1 - hz.p_fire[i])


def draw(fig, axes, f: Factory, sc: Scenario, frames: list[PipelineFrame], k: int, t_max: float):
    """Everything at frame ``k``."""
    from matplotlib.colors import LogNorm

    fr = frames[k]
    ax, src, d0, d30, d60, tl = axes
    for a in axes:
        a.clear()
    ext = (0, f.width, 0, f.height)
    gx, gy = f.centres()[..., 0], f.centres()[..., 1]
    ax.imshow(np.ma.masked_where(f.walls, np.maximum(fr.field, 1e-3)), origin="lower", extent=ext,
              cmap="viridis", norm=LogNorm(1e-3, 10))  # fmt: skip
    _plant(ax, f)
    if fr.harm.any():
        ax.contour(gx, gy, fr.harm.astype(float), levels=[0.5], colors="white", linewidths=1.4)
    d = _danger_at(fr, fr.t + 30)
    if d is not None and d.max() >= 0.5:
        ax.contour(gx, gy, d, levels=[0.5], colors="red", linewidths=1.4, linestyles="--")
    if fr.source is not None:
        iy, ix = fr.source.top_cells(1)[0]
        ax.scatter(*f.centre(int(iy), int(ix)), marker="o", s=260, facecolors="none",
                   edgecolors="cyan", linewidths=2.2, zorder=7)  # fmt: skip
    ax.scatter(*sc.source, marker="*", s=200, c="cyan", edgecolors="k", zorder=7)
    for p in POLICIES[::-1]:
        w, (col, ls) = fr.walkers[p], STYLE[p]
        if w.trail:
            ax.plot(*np.array([*w.trail, w.pos]).T, c=col, lw=1, alpha=0.6)
        if w.path is not None and w.out_at is None:
            ax.plot(*w.path.T, c=col, lw=2.4 if p == "live" else 1.6, ls=ls)
        ax.scatter(*w.pos, s=110 if p == "live" else 60, c=col, edgecolors="k", zorder=8)
    ax.set_title(f"t = {fr.t:.0f} s  |  {fr.phase}", fontsize=10, loc="left")

    last = fr.cycles[-1] if fr.cycles else None
    _plant(src, f, labels=False)
    if last is not None:
        p = np.ma.masked_where(f.walls, np.maximum(last.source.prob, 1e-5))
        src.imshow(p, origin="lower", extent=ext, cmap="magma", norm=LogNorm(1e-4, 1))
        src.scatter(*sc.source, marker="*", s=120, c="cyan", edgecolors="k", zorder=7)
        title = f"Rayan, run {last.index + 1}: where is the leak?\n"
        title += f"{last.source_error_m:.1f} m off, size {last.rate_est:.2f} (true {sc.rate:.2f})"
        src.set_title(title, fontsize=8)
    else:
        src.set_title("Rayan: waiting for the alarm", fontsize=8)
    for a, ahead, name in ((d0, 0, "now"), (d30, 30, "+30 s"), (d60, 60, "+60 s")):
        _plant(a, f, labels=False)
        m = _danger_at(fr, fr.t + ahead)
        if m is not None:
            a.imshow(np.ma.masked_where(f.walls, m), origin="lower", extent=ext, cmap="Reds",
                     vmin=0, vmax=1)  # fmt: skip
        a.set_title(f"Ahmad: P(harm) {name}", fontsize=8)

    tl.set_xlim(0, t_max)
    tl.set_ylim(-0.6, max(len(frames[-1].cycles), 1) - 0.4)
    tl.invert_yaxis()
    for c in frames[-1].cycles:
        if c.t_start > fr.t:
            break
        tl.barh(c.index, c.locate_s, left=c.t_start, color="#19c3e6", height=0.6)
        tl.barh(c.index, c.t_land - c.t_start - c.locate_s, left=c.t_start + c.locate_s,
                color="#ff5252", height=0.6)  # fmt: skip
        tl.text(c.t_land + 0.5, c.index, f"run {c.index + 1}: {c.samples} readings/sensor",
                va="center", fontsize=7)  # fmt: skip
    if fr.t_alarm is not None:
        tl.axvline(fr.t_alarm, c="k", ls=":", lw=1)
    tl.axvline(fr.t, c="#555", lw=1.5)
    tl.set_yticks([])
    tl.set_xlabel("seconds since the leak (dotted: alarm). Cyan = Rayan's model, red = Ahmad's "
                  "model; each run starts when the last route has landed", fontsize=8)  # fmt: skip
    tl.tick_params(labelsize=7)


def save_gif(f, sc, frames, fig, axes, path: Path, end: int, t_max: float, fps: int = 6) -> Path:
    """Every simulated second up to ``end`` as a GIF."""
    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=60):
        for k in range(end + 1):
            draw(fig, axes, f, sc, frames, k, t_max)
            writer.grab_frame()
    return path


def figure(f: Factory):
    """Figure and its six axes: hall, Rayan's map, Ahmad now / +30 / +60, timeline."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8.6))
    g = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.55), width_ratios=(1.25, 1.25, 1, 1))
    cells = (g[0:2, 0:2], g[0, 2], g[0, 3], g[1, 2], g[1, 3], g[2, :])
    axes = [fig.add_subplot(c) for c in cells]
    fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.07, wspace=0.08, hspace=0.25)
    return fig, axes


def main() -> int:
    """CLI: ``uv run evac-pipeline [--models team] [--backend baseline] [--seed 500]``."""
    p = argparse.ArgumentParser(description="One person, the team's models running back to back.")
    p.add_argument("--models", default="team", choices=("team", "perfect", "baseline"))
    p.add_argument("--backend", default="baseline", choices=("baseline", "cloud"))
    p.add_argument("--yes", action="store_true", help="allow cloud calls (spends tokens)")
    p.add_argument("--seed", type=int, default=500, help="leak seed (500+: held out)")
    p.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"), help="person start (m)")
    p.add_argument("--max-runs", type=int, default=8)
    p.add_argument("--t-max", type=float, default=120.0)
    p.add_argument("--gif", type=Path)
    p.add_argument("--still", type=Path)
    a = p.parse_args()
    if a.backend == "cloud" and a.models == "team" and not a.yes:
        print(f"cloud run: up to {a.max_runs} runs x ~3 TabPFN calls (>= {a.max_runs * 30_000:,} "
              "tokens). Re-run with --yes to spend them.")  # fmt: skip
        return 1
    f = Factory.default()
    sc, frames = compute(f, a.seed, a.at, a.models, a.backend, a.max_runs, a.t_max)
    import matplotlib

    if a.gif or a.still:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = figure(f)
    done = [
        k for k, fr in enumerate(frames) if all(w.out_at is not None for w in fr.walkers.values())
    ]
    end = done[0] if done else len(frames) - 1
    end = min(len(frames) - 1, end + 5)
    if a.still:
        k = frames[-1].cycles[0].t_land if frames[-1].cycles else end
        draw(fig, axes, f, sc, frames, int(min(k + 3, end)), a.t_max)
        fig.savefig(a.still, dpi=80)
        print(a.still)
    if a.gif:
        print(save_gif(f, sc, frames, fig, axes, a.gif, end, a.t_max))
    if not (a.gif or a.still):
        from matplotlib.animation import FuncAnimation

        state = {"k": 0, "pause": False}

        def tick(_):
            if not state["pause"]:
                state["k"] = min(state["k"] + 1, end)
            draw(fig, axes, f, sc, frames, state["k"], a.t_max)

        def key(e):
            if e.key == " ":
                state["pause"] = not state["pause"]
            if e.key == "right":
                state["k"] = min(state["k"] + 1, end)
            if e.key == "left":
                state["k"] = max(state["k"] - 1, 0)

        fig.canvas.mpl_connect("key_press_event", key)
        anim = FuncAnimation(fig, tick, interval=250, cache_frame_data=False)
        plt.show()
        del anim
    return 0
