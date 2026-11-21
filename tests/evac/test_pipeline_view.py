from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tabpfn_spatial.evac.pipeline_view import compute, draw, figure, save_gif
from tabpfn_spatial.sim import Factory


def test_pipeline_view_runs_and_draws(tmp_path: Path) -> None:
    f = Factory.default()
    lines = []
    sc, frames = compute(f, 500, None, "perfect", "baseline", 3, 40.0, log=lines.append)
    assert any(line.startswith("run 1:") for line in lines)
    assert frames[-1].cycles
    fig, axes = figure(f)
    draw(fig, axes, f, sc, frames, len(frames) - 1, 40.0)
    fig.savefig(tmp_path / "s.png", dpi=50)
    assert (tmp_path / "s.png").stat().st_size > 20_000
    assert save_gif(f, sc, frames, fig, axes, tmp_path / "g.gif", 4, 40.0).stat().st_size > 5_000


def test_runs_are_saved_and_replay_without_the_models(tmp_path: Path) -> None:
    import shutil

    from tabpfn_spatial.evac.pipeline_view import load_run, save_run, save_video

    f = Factory.default()
    sc, frames = compute(f, 500, None, "perfect", "baseline", 2, 30.0, log=lambda *_: None)
    path = save_run(tmp_path, sc, frames, {"models": "perfect", "seed": 500})
    assert path.exists() and path.suffix == ".pkl"
    sc2, frames2, meta = load_run(path)
    assert meta["seed"] == 500 and len(frames2) == len(frames)
    assert frames2[-1].cycles[0].t_land == frames[-1].cycles[0].t_land
    out = save_video(f, sc2, frames2, tmp_path / "run.mp4", 30.0, end=5)
    assert out.exists() and out.stat().st_size > 5_000
    if shutil.which("ffmpeg"):
        assert out.suffix == ".mp4"
