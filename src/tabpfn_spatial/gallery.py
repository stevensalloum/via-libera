"""``uv run gallery Final``: one page with every video, graph and 3D replay, in pitch order.

Creates ``milestones/<name>-<date>/`` (git-ignored) with:
- ``videos/``: every replay video recorded in the 3D viewer (``outputs/world/replay-*/videos``),
  converted to H.264 MP4 (ffmpeg), captioned from the run's own data;
- ``figures/``: the figures of ``docs/figures`` in the order of the pitch (``SECTIONS``),
  named ``<section>-<folder>-<file>``;
- 2D videos of ``evac-pipeline`` runs (``outputs/pipeline/*/video.mp4``);
- ``replays/``: each replay's ``scene.json`` + ``replay.json``, to open again in 3D with
  ``uv run world --keep --out <folder>``;
- ``index.html``: the gallery (open it in a browser).
"""

import argparse
import datetime as dt
import html
import json
import shutil
import subprocess
from pathlib import Path

FIGURES = Path("docs/figures")
REPLAYS = Path("outputs/world")
PIPELINE = Path("outputs/pipeline")  # 2D runs of ``evac-pipeline`` (video.mp4 each)
# The pitch, in order: (section, [(figure folder, file, caption)]). Missing files are skipped.
SECTIONS = (
    ("1 · The setting: a gas leak in a factory", [
        ("shapes", "plants.png", "Plants of every shape: exits, detectors, gas pipes, AGV lanes."),
        ("steven", "01-pipes.png", "Leaks happen on the pipes; detectors on walls and masts."),
        ("3d-study", "data_redvapor.png", "Real gas: a fog plume around a 1:50 industrial site."),
    ]),
    ("2 · Today: fixed signs to the nearest exit", [
        ("final", "1-exposure.png", "Same leaks, same people: nearest exit vs Via Libera."),
        ("steven", "02-evac-live.png", "Nearest exit (red) walks into the gas; the live route "
                                       "(green) goes around it."),
    ]),
    ("3 · Via Libera: find the leak, predict the gas, route everyone", [
        ("steven", "06-crowd-run.png", "A whole crowd, every planner, the same leak."),
        ("shapes", "gains.png", "What the gas-aware planners save, per plant shape."),
    ]),
    ("4 · How: three steps, each a TabPFN model", [
        ("rayan", "height3d-3-tabpfn-vs-knn.png", "1 · Source localisation: position and "
                                                    "height from the detectors."),
        ("ahmad", "08-hazard-3d.png", "2 · Hazard forecast: where the air will be harmful, "
                                      "in 3D."),
        ("steven", "05-crowd-ladder.png", "3 · Crowd routing: CCRP plans every "
                                          "person around the predicted gas."),
        ("steven", "04-sync-timeline.png", "The loop: detectors every 5 s, models every 15 s."),
    ]),
    ("5 · Scaling to 3D", [
        ("3d-study", "verdict.png", "Simulation vs real data: where each workflow works."),
        ("3d-study", "ladder.png", "Physics ladder up to CFD: calibrated simple physics suffices."),
        ("3d-study", "heights.png", "Light gas rises, heavy gas sinks, only near the leak."),
        ("rayan", "height3d-2-detection-methane.png", "Masts at vertical pipes find the height."),
    ]),
    ("6 · Why TabPFN", [
        ("final", "3-accuracy-real.png", "Real gas data: TabPFN vs a classic k-NN."),
        ("final", "2-accuracy-factory.png", "3D factory: TabPFN vs k-NN, per plant."),
        ("final", "4-compute.png", "No training, cheap physics; update latency still above 15 s."),
        ("shapes", "locator.png", "Leak position error on unseen plants."),
    ]),
    ("7 · Next", [
        ("final", "5-stress.png", "The hard case: low leaks near people, updates every 5 s."),
        ("ahmad", "05-threshold-tradeoff.png", "Alarm threshold: early vs false alarms."),
    ]),
)  # fmt: skip


def _mp4(webm: Path, out: Path) -> Path | None:
    """WebM -> H.264 MP4 (None without ffmpeg)."""
    if not shutil.which("ffmpeg"):
        return None
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-crf", "22",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "faststart",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)]  # fmt: skip
    subprocess.run(cmd, check=True)
    return out


def _caption(rep: dict, planner: str) -> str:
    """One line from a replay: plant, gas, models, alarm, and each planner's result."""
    s = rep["scene"]
    alarm = f"alarm at {rep['t_alarm']:.0f} s" if rep.get("t_alarm") is not None else "no alarm"
    parts = [rep["rung"], f"gas: {s.get('gas', 'methane')}", f"models: {rep['models']}", alarm]
    res = []
    for p, d in rep["planners"].items():
        m = d["metrics"]
        mark = "<b>" if p == planner else ""
        end = "</b>" if p == planner else ""
        res.append(
            f"{mark}{p}: {m['harm_s_mean']:.2f} s in gas, {m['exposed_pct']:.1f} % exposed{end}"
        )
    return html.escape(" · ".join(parts)) + "<br>" + " · ".join(res)


def collect(root: Path) -> dict:
    """Copy videos, figures and replays into ``root``; return what the page shows."""
    out: dict = {"videos": [], "groups": [], "replays": []}
    (root / "videos").mkdir(parents=True, exist_ok=True)
    for folder in sorted(REPLAYS.glob("replay-*")):
        if not (folder / "replay.json").exists():
            continue
        rep = json.loads((folder / "replay.json").read_text())
        dst = root / "replays" / folder.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("scene.json", "replay.json"):
            shutil.copy(folder / f, dst / f)
        out["replays"].append(dst)
        for webm in sorted((folder / "videos").glob("*.webm")):
            planner = next((p for p in rep["planners"] if p in webm.stem), "ccrp")
            mp4 = _mp4(webm, root / "videos" / f"{folder.name}-{webm.stem}.mp4")
            path = mp4 or Path(shutil.copy(webm, root / "videos" / f"{folder.name}-{webm.name}"))
            out["videos"].append({"path": path.relative_to(root), "title": rep["rung"],
                                  "caption": _caption(rep, planner)})  # fmt: skip
    for title, items in SECTIONS:
        shown = []
        for sub, name, cap in items:
            src = FIGURES / sub / name
            if not src.exists():
                continue
            dst = root / "figures" / f"{title[0]}-{sub}-{name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            shown.append({"path": dst.relative_to(root), "caption": cap})
        out["groups"].append({"title": title, "items": shown})
    for run in sorted(PIPELINE.glob("*/video.mp4")):  # 2D single-person runs
        dst = root / "videos" / f"2d-{run.parent.name}.mp4"
        shutil.copy(run, dst)
        cap = "One person, the TabPFN models back to back (2D)."
        title = f"2D · {run.parent.name}"
        out["videos"].append({"path": dst.relative_to(root), "title": title, "caption": cap})
    return out


PAGE = (Path(__file__).parent / "gallery.html").read_text()  # str.format template


def _figure_card(src: str, cap: str) -> str:
    return (f'<div class="card"><a href="{src}"><img src="{src}" loading="lazy" alt="{cap}"></a>'
            f'<div class="cap">{cap}</div></div>')  # fmt: skip


def page(name: str, items: dict) -> str:
    """The gallery HTML."""
    vids = "".join(
        f'<div class="card"><video src="{html.escape(str(v["path"]))}" controls muted loop '
        f'playsinline preload="metadata"></video><div class="cap"><b>{html.escape(v["title"])}'
        f"</b><br>{v['caption']}</div></div>"
        for v in items["videos"]
    )
    n = len(items["videos"])
    videos = (
        f'<h2>3D replays ({n} videos)</h2><div class="grid">{vids}</div>'
        if vids
        else "<h2>3D replays</h2><p class='lead'>No videos recorded yet.</p>"
    )
    groups = "".join(
        f"<h2>{html.escape(g['title'])}</h2><div class='grid'>"
        + "".join(
            _figure_card(html.escape(str(i["path"])), html.escape(i["caption"])) for i in g["items"]
        )
        + "</div>"
        for g in items["groups"]
    )
    reps = "".join(f"<li><code>uv run world --keep --out {html.escape(str(r))}</code></li>"
                   for r in items["replays"]) or "<li>(none)</li>"  # fmt: skip
    return PAGE.format(name=html.escape(name), date=dt.date.today(), videos=videos, groups=groups,
                       replays=reps)  # fmt: skip


def main() -> int:
    """CLI: ``uv run gallery M2 [--out milestones]``."""
    p = argparse.ArgumentParser(description="One page with every video, graph and replay.")
    p.add_argument("name", help="e.g. M2")
    p.add_argument("--out", type=Path, default=Path("milestones"))
    a = p.parse_args()
    root = a.out / f"{a.name}-{dt.date.today()}"
    items = collect(root)
    index = root / "index.html"
    index.write_text(page(a.name, items))
    shutil.copy(Path(__file__).parent / "world" / "web" / "logo.webp", root / "logo.webp")
    print(f"{len(items['videos'])} videos, {sum(len(g['items']) for g in items['groups'])} "
          f"figures, {len(items['replays'])} replays")  # fmt: skip
    print(index.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
