from pathlib import Path

import numpy as np

from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.sim.viz import plot_layout, plot_random_layouts, plot_readings, plot_timelapse


def _ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 5_000


def test_all_simulator_figures_render(tmp_path: Path) -> None:
    f = Factory.default()
    sc = Scenario(np.array([34.0, 20.0]), 1.0, np.array([0.8, 0.3]), drift=0.1, seed=1)
    assert _ok(plot_layout(f, tmp_path / "layout.png"))
    assert _ok(plot_timelapse(f, {"drift": sc}, np.array([10.0, 60.0]), tmp_path / "t.png"))
    r = readings(f, sc, 60.0, np.random.default_rng(0), dropout=0.2)
    assert _ok(plot_readings(f, r, tmp_path / "r.png"))
    assert _ok(plot_random_layouts([1, 2], tmp_path / "rand.png"))
