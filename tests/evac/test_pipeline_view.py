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


def test_every_run_gets_a_folder_with_all_its_data(tmp_path: Path) -> None:
    import csv
    import json

    import numpy as np

    from tabpfn_spatial.evac.pipeline_view import load_run, save_run_folder

    f = Factory.default()
    sc, frames = compute(f, 500, [43, 37], "perfect", "baseline", 3, 50.0, log=lambda *_: None,
                         condition="combined_stress")  # fmt: skip
    meta = {"models": "perfect", "backend": "baseline", "seed": 500, "condition": "combined_stress"}
    run = save_run_folder(tmp_path, f, sc, frames, meta, 50.0, video=False)
    names = {p.name for p in run.iterdir()}
    assert {"run.pkl", "summary.json", "cycles.csv", "track.csv", "maps.npz", "still.png"} <= names
    summary = json.loads((run / "summary.json").read_text())
    assert summary["meta"]["condition"] == "combined_stress"
    assert summary["condition"]["diffusivity"] == 0.4
    assert len(summary["cycles"]) == len(frames[-1].cycles)
    assert set(summary["people"]) == {"live", "once", "nearest"}
    rows = list(csv.DictReader((run / "track.csv").open()))
    assert len(rows) == len(frames) and {"t", "live_x", "nearest_in_gas", "phase"} <= set(rows[0])
    maps = np.load(run / "maps.npz")
    assert maps["true_gas"].shape[0] == len(frames)
    assert maps["source_prob"].shape[0] == len(frames[-1].cycles)
    assert maps["p_toxic"].shape[:1] == maps["source_prob"].shape[:1]
    _sc2, frames2, meta2 = load_run(run)  # a folder replays like a .pkl
    assert meta2["condition"] == "combined_stress" and len(frames2) == len(frames)


def test_the_view_has_a_legend_and_draws_earlier_routes() -> None:
    f = Factory.default()
    sc, frames = compute(f, 500, [43, 37], "perfect", "baseline", 4, 60.0, log=lambda *_: None)
    fig, axes = figure(f)
    k = len(frames) - 1
    draw(fig, axes, f, sc, frames, k, 60.0)
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert any("replanned" in t for t in labels) and any("nearest" in t for t in labels)
