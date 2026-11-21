"""Watch one evacuation live: ``uv run evac-live``.

A window plays the leak second by second: true gas (colour), truly harmful area (white
line), what the models predict for the next minute (red dashes), the source estimate
(cyan), and three people leaving from the same spot (see ``evac.live``). The side panel
lists every exit as the planner sees it at the latest update, and the dose each person has
breathed so far.

Keys: click the map = move the person there and restart; space = pause; n = next leak;
up / down = replan more / less often; q = quit.
``--gif out.gif`` records the run instead of opening a window; ``--story out.png`` writes
the four-moment still used in the journey.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.live import POLICIES, Frame, run_live
from tabpfn_spatial.evac.standins import OracleHazard, OracleLocator
from tabpfn_spatial.sim import Factory, Scenario, sample_people, sample_scenarios
from tabpfn_spatial.sim.viz import draw_pipes

STYLE = {  # policy -> (colour, line style, label)
    "live": ("#00e676", "-", "live: replanned every update"),
    "once": ("#ffa726", "--", "once: planned at the alarm only"),
    "nearest": ("#ff1744", ":", "nearest exit, gas ignored"),
}


@dataclass
class Config:
    """Everything that defines one run (the viewer's knobs)."""

    seed: int = 0
    start: np.ndarray | None = None  # metres; None = random open spot
    t_alarm: float | None = None  # None = first gas detector alarm
    update_s: float = 5.0
    t_max: float = 180.0
    drift: float = 0.05
    source: str = "oracle"  # oracle (stand-in, error sigma) | centroid (baseline)
    sigma: float = 3.0
    hazard: str = "forward"  # forward (baseline forward simulation) | oracle (perfect)
    history: list[tuple[float, dict[str, float]]] = field(default_factory=list)


def make_run(f: Factory, cfg: Config):
    """(scenario, start, frame iterator) for a config."""
    from tabpfn_spatial.hazard import ForwardSimPredictor
    from tabpfn_spatial.source import WeightedCentroidLocator

    rng = np.random.default_rng(cfg.seed)
    sc = sample_scenarios(f, 1, rng, drift=cfg.drift)[0]
    start = cfg.start if cfg.start is not None else sample_people(f, 1, rng)[0]
    loc = (
        OracleLocator(sc, cfg.sigma, np.random.default_rng(cfg.seed + 1))
        if cfg.source == "oracle"
        else WeightedCentroidLocator()
    )
    pred = OracleHazard(sc) if cfg.hazard == "oracle" else ForwardSimPredictor()
    frames = run_live(
        f, sc, start, loc, pred, t_alarm=cfg.t_alarm, update_s=cfg.update_s, t_max=cfg.t_max,
        rng=np.random.default_rng(cfg.seed + 2),
    )  # fmt: skip
    return sc, start, frames


def _predicted_danger(fr: Frame, ahead: float = 60.0) -> np.ndarray | None:
    """Max predicted P(harm) over the next ``ahead`` seconds, or None before the alarm."""
    if fr.hazard is None:
        return None
    hz = fr.hazard
    keep = (hz.times >= fr.t - 1e-9) & (hz.times <= fr.t + ahead)
    if not keep.any():
        keep = hz.times == hz.times.max()
    p = 1 - (1 - hz.p_toxic[keep]) * (1 - hz.p_fire[keep])
    return p.max(axis=0)


def draw_map(ax, f: Factory, sc: Scenario, fr: Frame) -> None:
    """One moment of the run on ``ax`` (cleared first)."""
    from matplotlib.colors import LogNorm

    ax.clear()
    ext = (0, f.width, 0, f.height)
    gas = np.ma.masked_where(f.walls, np.maximum(fr.field, 1e-3))
    ax.imshow(gas, origin="lower", extent=ext, cmap="viridis", norm=LogNorm(1e-3, 10))
    ax.imshow(np.ma.masked_where(~f.walls, f.walls), origin="lower", extent=ext,
              cmap="Greys", vmin=0, vmax=1.4)  # fmt: skip
    draw_pipes(ax, f)
    gx, gy = f.centres()[..., 0], f.centres()[..., 1]
    if fr.harm.any():
        ax.contour(gx, gy, fr.harm.astype(float), levels=[0.5], colors="white", linewidths=1.5)
    danger = _predicted_danger(fr)
    if danger is not None and danger.max() >= 0.5:
        ax.contour(gx, gy, danger, levels=[0.5], colors="red", linewidths=1.5,
                   linestyles="--")  # fmt: skip
    if fr.source is not None:
        p = fr.source.prob
        ax.contour(gx, gy, p, levels=[0.25 * p.max()], colors="cyan", linewidths=1.2)
    ax.scatter(*sc.source, marker="*", s=200, c="cyan", edgecolors="k", zorder=6)
    ax.scatter(*f.sensors.T, marker="s", s=22, c="deepskyblue", edgecolors="k", zorder=5)
    ax.scatter(*f.exits.T, marker="D", s=90, c="gold", edgecolors="k", zorder=6)
    for i, (x, y) in enumerate(f.exits):
        off = (-16 if x > f.width / 2 else 6, -14 if y > f.height / 2 else 6)
        ax.annotate(f"E{i}", (x, y), xytext=off, textcoords="offset points", fontsize=9,
                    color="white", weight="bold")  # fmt: skip
    for pol in POLICIES[::-1]:
        w, (col, ls, _) = fr.walkers[pol], STYLE[pol]
        if w.trail:
            ax.plot(*np.array([*w.trail, w.pos]).T, c=col, lw=1, alpha=0.6)
        if w.path is not None and w.out_at is None:
            ax.plot(*w.path.T, c=col, lw=2.2, ls=ls)
        ax.scatter(*w.pos, s=90, c=col, edgecolors="k", zorder=8)
    ax.set_xlim(0, f.width)
    ax.set_ylim(0, f.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    status = "alarm: evacuating" if fr.alarm else "leak not yet detected, nobody moves"
    ax.set_title(f"t = {fr.t:5.0f} s   {status}", fontsize=11, loc="left")


def add_legend(fig) -> None:
    """One legend for the whole figure, along the bottom edge."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], c=c, ls=ls, lw=2.2, label=lab) for c, ls, lab in STYLE.values()]
    handles += [
        Line2D([], [], c="black", lw=1.5, label="white line: truly harmful now"),
        Line2D([], [], c="red", lw=1.5, ls="--", label="predicted harmful in the next 60 s"),
        Line2D([], [], c="cyan", lw=1.5, label="source estimate (cyan star = true leak)"),
        Line2D([], [], c="#5d2e8c", lw=2.2, marker="o", label="gas pipe / fitting"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)


def side_text(fr: Frame, cfg: Config) -> str:
    """Exit table of the latest update and each person's score so far."""
    how = "fixed" if cfg.t_alarm is not None else "detector"
    when = f"{fr.t_alarm:.0f} s ({how})" if fr.t_alarm is not None else "no detector alarm"
    lines = [f"replan every {cfg.update_s:.0f} s  |  alarm at {when}", ""]
    if fr.options:
        best = min(fr.options, key=lambda o: o.cost).exit_index
        lines.append("exits at last update   walk   ETA    P(harm)")
        for o in fr.options:
            mark = "<- best" if o.exit_index == best else ""
            lines.append(
                f"  E{o.exit_index}                {o.length_m:4.0f} m {o.eta_s:4.0f} s"
                f"   {o.risk:4.2f}  {mark}"
            )
    else:
        lines.append("no plan yet (waiting for the alarm)")
    lines += ["", "person     exit  out at   dose  in harm"]
    for pol in POLICIES:
        w = fr.walkers[pol]
        out = f"{w.out_at:4.0f} s" if w.out_at is not None else "  ... "
        ex = f"E{w.exit_index}" if w.exit_index is not None else "-"
        lines.append(f"{pol:9}  {ex:>4}  {out}  {w.dose:5.1f}  {w.harmful_s:4.0f} s")
    return "\n".join(lines)


def draw_dose(ax, history: list[tuple[float, dict[str, float]]], t_max: float) -> None:
    """Cumulative dose of each person against time."""
    ax.clear()
    if history:
        t = [h[0] for h in history]
        for pol in POLICIES:
            col, ls, _ = STYLE[pol]
            ax.plot(t, [h[1][pol] for h in history], c=col, ls=ls, lw=2, label=pol)
    ax.set_xlim(0, t_max)
    ax.set_xlabel("seconds since the leak started", fontsize=8)
    ax.set_ylabel("dose breathed (conc. x s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)


class LiveView:
    """The figure and its event handling; one ``Frame`` per animation tick."""

    def __init__(self, f: Factory, cfg: Config) -> None:
        """Build the figure and start the first run."""
        import matplotlib.pyplot as plt

        self.f, self.cfg, self.paused = f, cfg, False
        self.fig = plt.figure(figsize=(15, 7.2))
        grid = self.fig.add_gridspec(2, 2, width_ratios=(2.3, 1), height_ratios=(1.25, 1))
        self.ax_map = self.fig.add_subplot(grid[:, 0])
        self.ax_text = self.fig.add_subplot(grid[0, 1])
        self.ax_dose = self.fig.add_subplot(grid[1, 1])
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.12, wspace=0.12)
        add_legend(self.fig)
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.restart()

    def restart(self) -> None:
        """New run from the current config."""
        self.sc, self.start, self.frames = make_run(self.f, self.cfg)
        self.cfg.start = self.start
        self.cfg.history = []
        self.last: Frame | None = None

    def tick(self, _=None) -> None:
        """Advance one frame (or hold the last one) and redraw."""
        if not self.paused:
            nxt = next(self.frames, None)
            if nxt is not None:
                self.last = nxt
                self.cfg.history.append((nxt.t, {p: w.dose for p, w in nxt.walkers.items()}))
        if self.last is None:
            return
        draw_map(self.ax_map, self.f, self.sc, self.last)
        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(0, 1, side_text(self.last, self.cfg), family="monospace",
                          fontsize=9, va="top")  # fmt: skip
        self.ax_text.set_title(
            f"leak seed {self.cfg.seed}  |  source: {self.cfg.source}  |  hazard: "
            f"{self.cfg.hazard}\nclick = move person, space = pause, n = next leak, "
            "up/down = update rate",
            fontsize=8,
            loc="left",
        )
        draw_dose(self.ax_dose, self.cfg.history, self.cfg.t_max)

    def _click(self, event) -> None:
        if event.inaxes is self.ax_map and event.xdata is not None:
            p = np.array([event.xdata, event.ydata])
            if not self.f.walls[self.f.cell_of(p)]:
                self.cfg.start = p
                self.restart()

    def _key(self, event) -> None:
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "n":
            self.cfg.seed += 1
            self.cfg.start = None
            self.restart()
        elif event.key in ("up", "down"):
            steps = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
            i = steps.index(self.cfg.update_s) if self.cfg.update_s in steps else 2
            i = max(0, i - 1) if event.key == "up" else min(len(steps) - 1, i + 1)
            self.cfg.update_s = steps[i]
            self.restart()


def save_gif(f: Factory, cfg: Config, path: Path, every: int = 2, fps: int = 10) -> Path:
    """Record the whole run (every ``every``-th second) as a GIF."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.animation import PillowWriter

    view = LiveView(f, cfg)
    view.fig.set_size_inches(12, 5.8)
    writer = PillowWriter(fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(view.fig, str(path), dpi=70):
        for k in range(int(cfg.t_max) + 1):
            view.tick()
            if k % every == 0:
                writer.grab_frame()
    return path


def save_story(f: Factory, cfg: Config, moments: list[float] | None, path: Path) -> Path:
    """Four moments of one run side by side, plus the dose curves (the journey figure).

    ``moments=None``: the alarm, 20 s and 45 s later, and the end of the run.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc, _, run = make_run(f, cfg)
    frames = list(run)
    if moments is None:
        ta = frames[0].t_alarm if frames[0].t_alarm is not None else cfg.t_max / 4
        moments = [ta, min(ta + 20, cfg.t_max), min(ta + 45, cfg.t_max), cfg.t_max]
    history, shots = [], {}
    for fr in frames:
        history.append((fr.t, {p: w.dose for p, w in fr.walkers.items()}))
        if any(abs(fr.t - m) < 1e-9 for m in moments):
            shots[fr.t] = fr
        last = fr
    fig = plt.figure(figsize=(16, 8.4))
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.75))
    for i, m in enumerate(moments):
        draw_map(fig.add_subplot(grid[i // 2, i % 2]), f, sc, shots[m])
    ax_t = fig.add_subplot(grid[0, 2])
    ax_t.axis("off")
    ax_t.text(0, 1, side_text(last, cfg), family="monospace", fontsize=8.5, va="top")
    draw_dose(fig.add_subplot(grid[1, 2]), history, cfg.t_max)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.1, wspace=0.08, hspace=0.18)
    add_legend(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=72)
    plt.close(fig)
    return path


def compare(f: Factory, cfg: Config, n: int) -> dict[str, dict[str, float]]:
    """Run leaks ``cfg.seed .. cfg.seed + n - 1`` (random start each) and score each policy.

    Returns per policy: ``exposed`` (people with any second in harm), ``harm_s`` (total
    seconds in harm), ``mean_dose``; plus ``pre_alarm_harm_s`` (seconds in harm before
    anyone could move, the same for every policy) and ``live_other_exit`` (runs where
    replanning changed the exit). Cost: n runs of ``run_live``.
    """
    out = {p: {"exposed": 0.0, "harm_s": 0.0, "mean_dose": 0.0} for p in POLICIES}
    pre = changed = 0.0
    for s in range(cfg.seed, cfg.seed + n):
        run = Config(s, None, cfg.t_alarm, cfg.update_s, cfg.t_max, cfg.drift, cfg.source,
                     cfg.sigma, cfg.hazard)  # fmt: skip
        _, _, frames = make_run(f, run)
        for fr in frames:
            if fr.t == fr.t_alarm:
                pre += fr.walkers["live"].harmful_s
            last = fr
        for p, w in last.walkers.items():
            out[p]["exposed"] += w.harmful_s > 0
            out[p]["harm_s"] += w.harmful_s
            out[p]["mean_dose"] += w.dose / n
        changed += last.walkers["live"].exit_index != last.walkers["once"].exit_index
    out["all"] = {"pre_alarm_harm_s": pre, "live_other_exit": changed, "runs": float(n)}
    return out


def main() -> int:
    """CLI: ``uv run evac-live [--seed S] [--at X Y] [--update 5] [--gif out.gif]``."""
    p = argparse.ArgumentParser(description="Watch one evacuation replanned live.")
    p.add_argument("--seed", type=int, default=0, help="which leak")
    p.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"), help="person start (m)")
    p.add_argument("--alarm", type=float, help="alarm at this second (default: first detector)")
    p.add_argument("--update", type=float, default=5.0, help="seconds between replans")
    p.add_argument("--t-max", type=float, default=180.0)
    p.add_argument("--drift", type=float, default=0.05, help="wind direction wander")
    p.add_argument("--source", choices=("oracle", "centroid"), default="oracle")
    p.add_argument("--sigma", type=float, default=3.0, help="oracle source error (m)")
    p.add_argument("--hazard", choices=("forward", "oracle"), default="forward")
    p.add_argument("--gif", type=Path, help="record a GIF instead of opening a window")
    p.add_argument("--story", type=Path, help="write the four-moment still instead")
    p.add_argument("--compare", type=int, metavar="N", help="score the policies on N leaks")
    a = p.parse_args()
    cfg = Config(a.seed, None if a.at is None else np.array(a.at), a.alarm, a.update,
                 a.t_max, a.drift, a.source, a.sigma, a.hazard)  # fmt: skip
    f = Factory.default()
    if a.gif:
        print(save_gif(f, cfg, a.gif))
        return 0
    if a.compare:
        for k, v in compare(f, cfg, a.compare).items():
            print(f"{k:8}", "  ".join(f"{m} {x:.2f}" for m, x in v.items()))
        return 0
    if a.story:
        print(save_story(f, cfg, None, a.story))
        return 0
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    view = LiveView(f, cfg)
    anim = FuncAnimation(view.fig, view.tick, interval=60, cache_frame_data=False)
    plt.show()
    del anim
    return 0
