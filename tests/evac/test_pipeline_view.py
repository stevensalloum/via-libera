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
