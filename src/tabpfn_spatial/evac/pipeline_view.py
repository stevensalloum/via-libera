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
from tabpfn_spatial.sim import CONDITIONS, Factory, Scenario, sample_people, sample_scenarios
from tabpfn_spatial.sim.viz import draw_pipes

STYLE = {"live": ("#00e676", "-"), "once": ("#ffa726", "--"), "nearest": ("#ff1744", ":")}
LABELS = {
    "live": "replanned at every model run (follows the newest route)",
    "once": "planned once, on the first model run, never updated",
    "nearest": "nearest exit, gas ignored (standard plan)",
}


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


def compute(
    f: Factory,
    seed: int,
    start,
    models: str,
    backend: str,
    max_runs: int,
    t_max: float,
    log=print,
    condition: str = "standard",
) -> tuple[Scenario, list[PipelineFrame]]:
    """Run the whole emergency under ``condition``; print each run when its route lands."""
    from dataclasses import replace

    cond = CONDITIONS[condition]
    rng = np.random.default_rng(seed)
    sc = replace(sample_scenarios(f, 1, rng)[0], drift=cond.drift, seed=seed)
    start = sample_people(f, 1, rng)[0] if start is None else np.asarray(start, float)
    log(f"leak at ({sc.source[0]:.0f}, {sc.source[1]:.0f}) m, size {sc.rate:.2f}")
    log(f"person at ({start[0]:.0f}, {start[1]:.0f}) m; fitting the {models} models ({backend})")
    loc, pred = models_for(models, sc, f, backend)
    frames, seen = [], 0
    log(
        f"condition: {condition} (mixing {cond.diffusivity} m2/s, wind drift {cond.drift}, "
        f"dead sensors {cond.dropout:.0%}, stuck {cond.stuck:.0%})"
    )
    for fr in run_pipeline(
        f,
        sc,
        start,
        loc,
        pred,
        max_runs=max_runs,
        t_max=t_max,
        rng=np.random.default_rng(seed + 1),
        diffusivity=cond.diffusivity,
        dropout=cond.dropout,
        stuck=cond.stuck,
    ):
        frames.append(fr)
        if len(fr.cycles) > seen:
            c = fr.cycles[-1]
            w = fr.walkers["live"]
            rayan = f"leak {c.source_error_m:.1f} m off, size {c.rate_est:.2f} (true {sc.rate:.2f})"
            log(
                f"run {c.index + 1}: readings 0-{c.t_start:.0f} s ({c.samples} per sensor) | "
                f"Rayan {c.locate_s:.2f} s: {rayan} | Ahmad {c.predict_s:.2f} s | "
                f"route lands at {c.t_land:.0f} s -> exit E{w.exit_index}"
            )
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
    ax.imshow(
        np.ma.masked_where(~f.walls, f.walls),
        origin="lower",
        extent=ext,
        cmap="Greys",
        vmin=0,
        vmax=1.4,
    )
    draw_pipes(ax, f)
    ax.scatter(*f.exits.T, marker="D", s=70, c="gold", edgecolors="k", zorder=6)
    if labels:
        for i, (x, y) in enumerate(f.exits):
            off = (-16 if x > f.width / 2 else 6, -14 if y > f.height / 2 else 6)
            ax.annotate(
                f"E{i}",
                (x, y),
                xytext=off,
                textcoords="offset points",
                fontsize=8,
                color="white",
                weight="bold",
            )
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
    ax.imshow(
        np.ma.masked_where(f.walls, np.maximum(fr.field, 1e-3)),
        origin="lower",
        extent=ext,
        cmap="viridis",
        norm=LogNorm(1e-3, 10),
    )
    _plant(ax, f)
    if fr.harm.any():
        ax.contour(gx, gy, fr.harm.astype(float), levels=[0.5], colors="white", linewidths=1.4)
    d = _danger_at(fr, fr.t + 30)
    if d is not None and d.max() >= 0.5:
        ax.contour(gx, gy, d, levels=[0.5], colors="red", linewidths=1.4, linestyles="--")
    if fr.source is not None:
        iy, ix = fr.source.top_cells(1)[0]
        ax.scatter(
            *f.centre(int(iy), int(ix)),
            marker="o",
            s=260,
            facecolors="none",
            edgecolors="cyan",
            linewidths=2.2,
            zorder=7,
        )
    ax.scatter(*sc.source, marker="*", s=200, c="cyan", edgecolors="k", zorder=7)
    for r in fr.routes[:-1]:  # earlier routes of the replanned person, faded
        ax.plot(*r.T, c=STYLE["live"][0], lw=1.2, alpha=0.3, ls="-")
    for p in POLICIES[::-1]:
        w, (col, ls) = fr.walkers[p], STYLE[p]
        if w.trail:
            ax.plot(*np.array([*w.trail, w.pos]).T, c=col, lw=1, alpha=0.6)
        if w.path is not None and w.out_at is None:
            ax.plot(*w.path.T, c=col, lw=2.4 if p == "live" else 1.6, ls=ls)
        ax.scatter(*w.pos, s=110 if p == "live" else 60, c=col, edgecolors="k", zorder=8)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], c=STYLE[p][0], ls=STYLE[p][1], lw=2, marker="o", label=LABELS[p])
        for p in POLICIES
    ]
    handles += [
        Line2D(
            [],
            [],
            c=STYLE["live"][0],
            alpha=0.3,
            lw=1.2,
            label="earlier routes of the replanned person",
        ),
        Line2D([], [], c="white", lw=1.4, label="truly harmful now"),
        Line2D([], [], c="red", ls="--", lw=1.4, label="Ahmad: harmful in 30 s"),
        Line2D([], [], c="cyan", marker="o", ls="", label="Rayan's best guess (star = true leak)"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=6.5, framealpha=0.85)
    leg.set_zorder(20)
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
            a.imshow(
                np.ma.masked_where(f.walls, m),
                origin="lower",
                extent=ext,
                cmap="Reds",
                vmin=0,
                vmax=1,
            )
        a.set_title(f"Ahmad: P(harm) {name}", fontsize=8)

    tl.set_xlim(0, t_max)
    tl.set_ylim(-0.6, max(len(frames[-1].cycles), 1) - 0.4)
    tl.invert_yaxis()
    for c in frames[-1].cycles:
        if c.t_start > fr.t:
            break
        tl.barh(c.index, c.locate_s, left=c.t_start, color="#19c3e6", height=0.6)
        tl.barh(
            c.index,
            c.t_land - c.t_start - c.locate_s,
            left=c.t_start + c.locate_s,
            color="#ff5252",
            height=0.6,
        )
        tl.text(
            c.t_land + 0.5,
            c.index,
            f"run {c.index + 1}: {c.samples} readings/sensor",
            va="center",
            fontsize=7,
        )
    if fr.t_alarm is not None:
        tl.axvline(fr.t_alarm, c="k", ls=":", lw=1)
    tl.axvline(fr.t, c="#555", lw=1.5)
    tl.set_yticks([])
    tl.set_xlabel(
        "seconds since the leak (dotted: alarm). Cyan = Rayan's model, red = Ahmad's "
        "model; each run starts when the last route has landed",
        fontsize=8,
    )
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


RUNS = Path("outputs/pipeline")


def save_run(folder: Path, sc: Scenario, frames: list[PipelineFrame], meta: dict) -> Path:
    """Pickle a computed run so it can be replayed later without calling the models."""
    import datetime as dt
    import pickle

    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"run-{stamp}-{meta.get('models', 'x')}-seed{meta.get('seed', 0)}.pkl"
    path.write_bytes(pickle.dumps({"scenario": sc, "frames": frames, "meta": meta}))
    return path


def save_run_folder(
    root: Path,
    f: Factory,
    sc: Scenario,
    frames: list[PipelineFrame],
    meta: dict,
    t_max: float,
    video: bool = True,
) -> Path:
    """One folder per run with everything needed for plots later.

    ``run.pkl`` (replay), ``summary.json`` (settings, leak, every model run, outcomes),
    ``cycles.csv`` (one row per model run), ``track.csv`` (every second: each person's
    position, exit, in gas, dose; the phase), ``maps.npz`` (true gas per second, Rayan's
    map and Ahmad's forecasts per run), ``still.png`` and ``video.mp4``.
    """
    import csv
    import datetime as dt
    import json
    import pickle

    cond = CONDITIONS[meta.get("condition", "standard")]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = root / f"{stamp}-{meta.get('models', 'x')}-{cond.name}-seed{meta.get('seed', 0)}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.pkl").write_bytes(pickle.dumps({"scenario": sc, "frames": frames, "meta": meta}))
    cycles = frames[-1].cycles
    exits = [
        int(fr.walkers["live"].exit_index) for c in cycles for fr in frames if fr.t == c.t_land
    ][: len(cycles)]
    rows = [
        {
            "run": c.index + 1,
            "t_start": c.t_start,
            "t_land": c.t_land,
            "readings_per_sensor": c.samples,
            "locate_s": round(c.locate_s, 3),
            "hazard_s": round(c.predict_s, 3),
            "source_error_m": round(c.source_error_m, 2),
            "size_est": round(c.rate_est, 3),
            "size_true": round(sc.rate, 3),
            "exit_after": exits[i] if i < len(exits) else None,
            "top_guesses": [
                {"xy": f.centre(int(a), int(b)).tolist(), "p": round(float(c.source.prob[a, b]), 4)}
                for a, b in c.source.top_cells(3)
            ],
        }
        for i, c in enumerate(cycles)
    ]
    last = frames[-1].walkers
    people = {
        p: {
            "exit": int(w.exit_index) if w.exit_index is not None else None,
            "out_at_s": w.out_at,
            "harmful_s": w.harmful_s,
            "dose": round(w.dose, 4),
            "walked_m": round(w.walked, 2),
        }
        for p, w in last.items()
    }
    summary = {
        "meta": meta,
        "condition": cond.__dict__,
        "leak": {
            "xy": sc.source.tolist(),
            "size": sc.rate,
            "wind": sc.wind.tolist(),
            "drift": sc.drift,
        },
        "t_alarm": frames[-1].t_alarm,
        "cycles": rows,
        "people": people,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with (run / "cycles.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, [k for k in rows[0] if k != "top_guesses"] if rows else ["run"])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: v for k, v in r.items() if k != "top_guesses"})
    with (run / "track.csv").open("w", newline="") as fh:
        cols = ["t", "phase", "alarm", "runs_landed"]
        for p in POLICIES:
            cols += [f"{p}_x", f"{p}_y", f"{p}_exit", f"{p}_in_gas", f"{p}_dose", f"{p}_out"]
        wr = csv.DictWriter(fh, cols)
        wr.writeheader()
        for fr in frames:
            row = {
                "t": fr.t,
                "phase": fr.phase,
                "alarm": int(fr.alarm),
                "runs_landed": len(fr.cycles),
            }
            for p, w in fr.walkers.items():
                iy, ix = f.cell_of(w.pos)
                row |= {
                    f"{p}_x": round(float(w.pos[0]), 2),
                    f"{p}_y": round(float(w.pos[1]), 2),
                    f"{p}_exit": w.exit_index,
                    f"{p}_in_gas": int(fr.harm[iy, ix] and w.out_at is None),
                    f"{p}_dose": round(w.dose, 4),
                    f"{p}_out": int(w.out_at is not None),
                }
            wr.writerow(row)
    np.savez_compressed(
        run / "maps.npz",
        times=np.array([fr.t for fr in frames]),
        true_gas=np.stack([fr.field for fr in frames]).astype(np.float32),
        walls=f.walls,
        cycle_start=np.array([c.t_start for c in cycles]),
        source_prob=np.stack([c.source.prob for c in cycles]).astype(np.float32)
        if cycles
        else np.zeros((0, f.ny, f.nx)),
        hazard_times=np.stack([c.hazard.times for c in cycles]) if cycles else np.zeros((0, 0)),
        p_toxic=np.stack([c.hazard.p_toxic for c in cycles]).astype(np.float32)
        if cycles
        else np.zeros((0,)),
        p_fire=np.stack([c.hazard.p_fire for c in cycles]).astype(np.float32)
        if cycles
        else np.zeros((0,)),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = figure(f)
    end = end_frame(frames)
    k = frames[-1].cycles[0].t_land if frames[-1].cycles else end
    draw(fig, axes, f, sc, frames, int(min(k + 3, end)), t_max)
    fig.savefig(run / "still.png", dpi=80)
    plt.close(fig)
    if video:
        save_video(f, sc, frames, run / "video.mp4", t_max)
    return run


def load_run(path: Path) -> tuple[Scenario, list[PipelineFrame], dict]:
    """A run saved by ``save_run`` or ``save_run_folder`` (a folder or its ``run.pkl``).

    Only load files you made yourself: it is a pickle.
    """
    import pickle

    path = Path(path)
    if path.is_dir():
        path = path / "run.pkl"
    d = pickle.loads(path.read_bytes())
    return d["scenario"], d["frames"], d["meta"]


def end_frame(frames: list[PipelineFrame]) -> int:
    """Last frame worth showing: 5 s after everyone is out."""
    done = [
        k for k, fr in enumerate(frames) if all(w.out_at is not None for w in fr.walkers.values())
    ]
    return min(len(frames) - 1, (done[0] if done else len(frames) - 1) + 5)


def save_video(
    f: Factory,
    sc: Scenario,
    frames: list[PipelineFrame],
    path: Path,
    t_max: float,
    end: int | None = None,
    fps: int = 4,
) -> Path:
    """MP4 of the run (one frame per simulated second); GIF if ffmpeg is missing."""
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter, PillowWriter

    end = end_frame(frames) if end is None else end
    if path.suffix == ".mp4" and not shutil.which("ffmpeg"):
        path = path.with_suffix(".gif")
    writer = FFMpegWriter(fps=fps, bitrate=4000) if path.suffix == ".mp4" else PillowWriter(fps=fps)
    fig, axes = figure(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(path), dpi=90 if path.suffix == ".mp4" else 60):
        for k in range(end + 1):
            draw(fig, axes, f, sc, frames, k, t_max)
            writer.grab_frame()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def play(f: Factory, sc: Scenario, frames: list[PipelineFrame], t_max: float, video: Path) -> None:
    """Window with Play/Pause, Restart, a time slider and Save video (keys: space, arrows, r)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, Slider

    end = end_frame(frames)
    fig, axes = figure(f)
    fig.subplots_adjust(bottom=0.12)
    state = {"k": 0, "play": True}
    slider = Slider(fig.add_axes((0.30, 0.012, 0.45, 0.025)), "t (s)", 0, end, valinit=0, valstep=1)
    b_play = Button(fig.add_axes((0.02, 0.008, 0.08, 0.035)), "Pause")
    b_restart = Button(fig.add_axes((0.11, 0.008, 0.08, 0.035)), "Restart")
    b_video = Button(fig.add_axes((0.20, 0.008, 0.08, 0.035)), "Save video")

    def show(k: int) -> None:
        state["k"] = int(max(0, min(end, k)))
        draw(fig, axes, f, sc, frames, state["k"], t_max)
        if slider.val != state["k"]:
            slider.eventson = False
            slider.set_val(state["k"])
            slider.eventson = True
        fig.canvas.draw_idle()

    def toggle(_=None) -> None:
        state["play"] = not state["play"]
        if state["play"] and state["k"] >= end:
            state["k"] = 0
        b_play.label.set_text("Pause" if state["play"] else "Play")

    def restart(_=None) -> None:
        state["play"] = True
        b_play.label.set_text("Pause")
        show(0)

    def save(_=None) -> None:
        b_video.label.set_text("saving...")
        fig.canvas.draw_idle()
        print("video:", save_video(f, sc, frames, video, t_max, end))
        b_video.label.set_text("Saved")

    def tick(_) -> None:
        if state["play"]:
            if state["k"] >= end:
                toggle()
            else:
                show(state["k"] + 1)

    def key(e) -> None:
        if e.key == " ":
            toggle()
        elif e.key == "right":
            state["play"] = False
            show(state["k"] + 1)
        elif e.key == "left":
            state["play"] = False
            show(state["k"] - 1)
        elif e.key == "r":
            restart()

    slider.on_changed(lambda v: (state.update(play=False), show(int(v))))
    b_play.on_clicked(toggle)
    b_restart.on_clicked(restart)
    b_video.on_clicked(save)
    fig.canvas.mpl_connect("key_press_event", key)
    show(0)
    anim = FuncAnimation(fig, tick, interval=300, cache_frame_data=False)
    plt.show()
    del anim


def main() -> int:
    """CLI: ``uv run evac-pipeline [--at X Y] [--backend cloud --yes] [--load run.pkl]``."""
    p = argparse.ArgumentParser(description="One person, the team's models running back to back.")
    p.add_argument("--models", default="team", choices=("team", "perfect", "baseline"))
    p.add_argument("--backend", default="baseline", choices=("baseline", "cloud"))
    p.add_argument("--yes", action="store_true", help="allow cloud calls (spends tokens)")
    p.add_argument("--seed", type=int, default=500, help="leak seed (500+: held out)")
    p.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"), help="person start (m)")
    p.add_argument("--max-runs", type=int, default=8)
    p.add_argument("--t-max", type=float, default=120.0)
    p.add_argument(
        "--condition",
        default="standard",
        choices=list(CONDITIONS),
        help="stress: mixing, wind drift, sensor faults (Ahmad's scenarios)",
    )
    p.add_argument("--no-video", action="store_true", help="skip the MP4 in the run folder")
    p.add_argument("--no-window", action="store_true", help="save the run and exit (batch runs)")
    p.add_argument(
        "--load", type=Path, help="replay a saved run folder (no model calls, no tokens)"
    )
    p.add_argument("--video", type=Path, help="save the run as an MP4 and exit")
    p.add_argument("--gif", type=Path)
    p.add_argument("--still", type=Path)
    a = p.parse_args()
    f = Factory.default()
    if a.load:
        sc, frames, meta = load_run(a.load)
        print(f"replaying {a.load} ({meta})")
        stem = (a.load if a.load.is_dir() else a.load.parent) / "video"
    else:
        if a.backend == "cloud" and a.models == "team" and not a.yes:
            print(
                f"cloud run: up to {a.max_runs} runs x ~3 TabPFN calls (>= "
                f"{a.max_runs * 30_000:,} tokens). Re-run with --yes to spend them."
            )
            return 1
        sc, frames = compute(
            f, a.seed, a.at, a.models, a.backend, a.max_runs, a.t_max, condition=a.condition
        )
        meta = {
            "models": a.models,
            "backend": a.backend,
            "seed": a.seed,
            "at": a.at,
            "max_runs": a.max_runs,
            "condition": a.condition,
        }
        if not a.no_video:
            print("saving the run folder and its video (about a minute)...")
        run = save_run_folder(RUNS, f, sc, frames, meta, a.t_max, video=not a.no_video)
        stem = run / "video"
        print(
            f"saved: {run}/  (summary.json, cycles.csv, track.csv, maps.npz, still.png"
            f"{', video.mp4' if not a.no_video else ''}, run.pkl)"
        )
        print(f"replay for free: uv run evac-pipeline --load {run}")
    if a.video:
        print("video:", save_video(f, sc, frames, a.video, a.t_max))
    if a.gif or a.still:
        import matplotlib

        matplotlib.use("Agg")
        fig, axes = figure(f)
        end = end_frame(frames)
        if a.still:
            k = frames[-1].cycles[0].t_land if frames[-1].cycles else end
            draw(fig, axes, f, sc, frames, int(min(k + 3, end)), a.t_max)
            fig.savefig(a.still, dpi=80)
            print(a.still)
        if a.gif:
            print(save_gif(f, sc, frames, fig, axes, a.gif, end, a.t_max))
    if not (a.video or a.gif or a.still or a.no_window):
        play(f, sc, frames, a.t_max, stem.with_suffix(".mp4"))
    return 0
