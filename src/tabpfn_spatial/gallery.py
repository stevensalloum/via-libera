"""``uv run gallery M2``: one page with every video, graph and 3D replay of a milestone.

Creates ``milestones/<name>-<date>/`` (git-ignored) with:
- ``videos/``: every replay video recorded in the 3D viewer (``outputs/world/replay-*/videos``),
  converted to H.264 MP4 (ffmpeg), captioned from the run's own data;
- ``figures/``: the study figures from ``docs/figures`` (grouped and captioned);
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
GROUPS = (
    ("Unseen factory shapes (24 new plants, cloud TabPFN)", "shapes", {
        "plants.png": "Examples of the unseen plants: exits, detectors, pipes, AGV lanes.",
        "gains.png": "What the gas-aware planners save against nearest exit, per shape.",
        "scores.png": "Seconds in gas, % exposed, time to 95 % out, per shape (team's models).",
        "locator.png": "Leak position error of the locators on unseen plants.",
    }),
    ("3D study: real data vs simulation", "3d-study", {
        "verdict.png": "Where each workflow works (hazard task, cloud TabPFN).",
        "ladder.png": "Physics ladder (plume to CFD and GADEN) against real data.",
        "profiles.png": "Does height come out right? Real vs simulated vertical profiles.",
        "transfer_hazard.png": "Hazard task: real -> real, A, B, hybrid, physics.",
        "transfer_source.png": "Source task: leak position error per workflow.",
        "heights.png": "Within how many metres of a leak gas density decides height.",
        "data_redvapor.png": "Red:Vapor: real fog plume around a 1:50 industrial site.",
        "data_orebro.png": "Örebro: real office with HVAC, 27 sensors at 3 heights.",
        "data_uci.png": "UCI 251: MOX response along and across a wind tunnel.",
        "data_srgdm.png": "SRGDM: plumes from measured airflow and a real 6 x 5 grid.",
    }),
    ("Evacuation (Steven)", "steven", {}),
    ("Hazard (Ahmad)", "ahmad", {}),
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
    for title, sub, captions in GROUPS:
        src = FIGURES / sub
        if not src.exists():
            continue
        items = []
        for png in sorted(src.glob("*.png"), key=lambda q: (q.name not in captions, q.name)):
            dst = root / "figures" / sub / png.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(png, dst)
            items.append(
                {"path": dst.relative_to(root), "caption": captions.get(png.name, png.stem)}
            )
        out["groups"].append({"title": title, "items": items})
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
    print(f"{len(items['videos'])} videos, {sum(len(g['items']) for g in items['groups'])} "
          f"figures, {len(items['replays'])} replays")  # fmt: skip
    print(index.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
