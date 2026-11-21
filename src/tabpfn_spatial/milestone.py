"""``uv run milestone M1``: one local folder with everything a milestone needs.

Creates ``milestones/<name>-<date>/`` (git-ignored: videos and big files stay local):

- ``figures/``: every journey figure, the simulator figures, the sync timeline, stills;
- ``animations/``: GIF + MP4 (if ffmpeg is installed) of one person evacuated live, a crowd
  on the 120 x 80 m plant, and one emergency through the team's TabPFN models;
- ``data/``: the study results (timing, crowd ladder, team-model comparison) and the 3D
  replay, when they exist in ``outputs/``;
- ``videos/``: where screen recordings of the 3D world go (how-to inside);
- ``README.md``: what is in the folder, the headline numbers and how to view it all.

``--fast`` skips the run through the team's models (about 2 minutes offline).
Integrator-owned.
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT_FILES = Path("docs/figures")


def _mp4(gif: Path) -> Path | None:
    """GIF -> H.264 MP4 next to it (None without ffmpeg)."""
    if not shutil.which("ffmpeg"):
        return None
    out = gif.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(gif), "-movflags", "faststart",
           "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)]  # fmt: skip
    subprocess.run(cmd, check=True)
    return out


def _figures(dst: Path) -> list[Path]:
    from tabpfn_spatial.sim import Factory, Scenario
    from tabpfn_spatial.sim.viz import plot_layout, plot_random_layouts, plot_timelapse
    from tabpfn_spatial.sync import plot_timeline

    dst.mkdir(parents=True, exist_ok=True)
    out = [shutil.copy(p, dst / p.name) for p in sorted(ROOT_FILES.rglob("*.png"))]
    f = Factory.default()
    leak = Scenario(f.leak_sites[3], 1.2, np.array([0.8, 0.3]), drift=0.06, seed=2)
    out += [
        plot_layout(f, dst / "plant_layout.png"),
        plot_timelapse(
            f, {"pipe leak": leak}, np.array([10.0, 30.0, 60.0, 120.0]), dst / "gas_timelapse.png"
        ),
        plot_random_layouts([1, 2, 3], dst / "random_plants.png"),
        plot_timeline(dst / "time_scales.png"),
    ]
    return [Path(p) for p in out]


def _animations(dst: Path, fast: bool) -> list[Path]:
    from tabpfn_spatial.evac.crowd_study import RUNGS
    from tabpfn_spatial.evac.crowd_viewer import Recording, record, save_gif, save_still
    from tabpfn_spatial.evac.viewer import Config, save_story
    from tabpfn_spatial.evac.viewer import save_gif as save_one_gif
    from tabpfn_spatial.sim import Factory

    dst.mkdir(parents=True, exist_ok=True)
    f = Factory.default()
    out = [save_story(f, Config(seed=30, t_max=120.0), None, dst / "one_person_live.png"),
           save_one_gif(f, Config(seed=30, t_max=100.0), dst / "one_person_live.gif")]  # fmt: skip
    rec = record(next(r for r in RUNGS if r.name == "D"), 509, "perfect")
    out += [
        save_still(rec, dst / "crowd_D_120x80m.png", 60.0),
        save_gif(rec, dst / "crowd_D_120x80m.gif"),
    ]
    if not fast:
        from tabpfn_spatial.loop import run

        rung, plant, prep, results = run("B", 500, "baseline", record=True)
        rec = Recording(rung, 500, "tabpfn (offline stand-in)", plant, prep,
                        {k: v.frames for k, v in results.items()},
                        {k: v.metrics for k, v in results.items()})  # fmt: skip
        out += [save_still(rec, dst / "team_models_B_300_people.png"),
                save_gif(rec, dst / "team_models_B_300_people.gif")]  # fmt: skip
    out += [m for g in list(out) if g.suffix == ".gif" and (m := _mp4(g))]
    return out


def _data(dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for name, src in [("timing_study.json", "outputs/timing/results.json"),
                      ("crowd_ladder.json", "outputs/crowd/results.json"),
                      ("team_models_vs_baselines.json", "outputs/crowd_tabpfn/results.json"),
                      ("replay_3d.json", "outputs/world/replay.json"),
                      ("scene_3d.json", "outputs/world/scene.json")]:  # fmt: skip
        if Path(src).exists():
            out.append(Path(shutil.copy(src, dst / name)))
    world = Path("outputs/world")
    if (world / "replay.json").exists():  # ready for: uv run world --keep --out <dst>/world
        (dst / "world").mkdir(exist_ok=True)
        for f in ("scene.json", "replay.json"):
            out.append(Path(shutil.copy(world / f, dst / "world" / f)))
    for src in ("outputs/crowd_tabpfn/ladder.png", "outputs/crowd/ladder.png"):
        if Path(src).exists():
            name = "team_models_vs_baselines.png" if "tabpfn" in src else "crowd_ladder.png"
            out.append(Path(shutil.copy(src, dst.parent / "figures" / name)))
    return out


def _table(path: Path) -> str:
    """Markdown table of a crowd results file: means per rung, models, planner."""
    from tabpfn_spatial.evac.crowd_study import aggregate

    rows = json.loads(path.read_text())["rows"]
    lines = ["| rung | models | planner | s in gas / person | exposed % | 95 % out (s) |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for (rung, models, planner), a in sorted(aggregate(rows).items()):
        lines.append(f"| {rung} | {models} | {planner} | {a['harm_s_mean']:.2f} | "
                     f"{a['exposed_pct']:.1f} | {a['out_95_s']:.0f} |")  # fmt: skip
    return "\n".join(lines)


def _readme(root: Path, name: str, files: list[Path]) -> Path:
    data = root / "data"
    note = (
        "**Read this first:** rows with models `tabpfn` ran the team's models with the offline "
        "k-NN stand-in (`--backend baseline`) unless the run used `--backend cloud`: they test "
        "that the parts fit together, not how good TabPFN is. `perfect` = true leak and true "
        "future (upper bound); `baseline` = centroid locator + forward simulation.\n\n"
    )
    tables = "\n\n".join(
        f"### {p.stem.replace('_', ' ')}\n\n{_table(p)}"
        for p in sorted(data.glob("*.json"))
        if p.name in ("crowd_ladder.json", "team_models_vs_baselines.json")
    )
    index = "\n".join(f"- `{p.relative_to(root)}`" for p in sorted(files))
    text = f"""# {name}: gas leak in a factory, closed loop with TabPFN

Made {dt.date.today()} by `uv run milestone {name}`.

## What works at this milestone

1. **Simulated plant:** 60 x 40 m hall (and random plants up to 200 x 120 m), gas pipes,
   leaks on the pipes, 2D advection-diffusion gas, 14 gas detectors read every 5 s.
2. **Detector alarm** starts everything (median 10 s after the leak on the default hall).
3. **Leak source (Rayan, TabPFN):** location and leak size from the readings, every 5 s
   from the alarm. Cloud numbers: 0.7 m error from 5 s after the alarm; size error 13 % at 10 s.
4. **Hazard (Ahmad, TabPFN):** probability of toxic / flammable gas now, +30 s, +60 s.
   Cloud Brier 0.057 vs 0.072 for the forward simulation; ~4 s per call.
5. **Evacuation (Steven):** whole-plant evacuation at the alarm; crowd simulator with
   door capacity and queues; nearest exit vs risk-aware vs capacity-aware (CCRP-style)
   routing; staged routes while the models answer.
6. **Integration:** `uv run sync-check` with both TabPFN models: 0 problems, 0 warnings.
7. **3D world:** the same plant in a navigable 3D factory, with a replay of a whole run.

## How to view it

```bash
cd ~/dev/tabpfn-spatial
uv run world --keep --out {root}/data/world   # this milestone's 3D replay
uv run world --replay --rung B --leak 500     # compute a fresh 3D replay (1-2 min)
uv run world                                   # 3D factory only, walk through it
uv run closed-loop --rung B --seed 500         # one emergency, every part, in the terminal
uv run evac-crowd-live --rung D --seed 509     # 2D: three planners side by side
uv run evac-live                               # 2D: one person, click to move them
```

Videos: in the 3D replay press **Record video** (saves a .webm to Downloads; labels are
not in it), or record the screen with **Cmd+Shift+5** and drop the file in `videos/`.

## Headline numbers

{note + tables if tables else "(run the studies first: uv run evac-crowd ...)"}

## Files

{index}
"""
    path = root / "README.md"
    path.write_text(text)
    return path


def main() -> int:
    """CLI: ``uv run milestone M1 [--out milestones] [--fast]``."""
    p = argparse.ArgumentParser(description="Collect a milestone: figures, animations, data.")
    p.add_argument("name", help="e.g. M1")
    p.add_argument("--out", type=Path, default=Path("milestones"))
    p.add_argument("--fast", action="store_true", help="skip the run through the team's models")
    a = p.parse_args()
    root = a.out / f"{a.name}-{dt.date.today()}"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "HOW_TO.md").write_text(
        "Put screen recordings of `uv run world --replay` here (Cmd+Shift+5 on macOS),\n"
        "or the .webm saved by the Record video button in the 3D replay.\n"
    )
    files = (
        _figures(root / "figures") + _animations(root / "animations", a.fast) + _data(root / "data")
    )
    print(_readme(root, a.name, files))
    print(f"{len(files)} files in {root}")
    return 0
