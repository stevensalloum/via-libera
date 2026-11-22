from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tabpfn_spatial.evac.compare_runs import load, plot, table
from tabpfn_spatial.evac.pipeline_view import compute, save_run_folder
from tabpfn_spatial.sim import Factory


def test_compare_two_saved_runs(tmp_path: Path) -> None:
    f = Factory.default()
    runs = []
    for cond in ("standard", "slower_mixing"):
        sc, frames = compute(f, 500, [43, 37], "perfect", "baseline", 3, 45.0, log=lambda *_: None,
                             condition=cond)  # fmt: skip
        meta = {"models": "perfect", "backend": "baseline", "seed": 500, "condition": cond}
        runs.append(load(save_run_folder(tmp_path, f, sc, frames, meta, 45.0, video=False)))
    text = table(runs)
    assert "standard" in text and "slower_mixing" in text
    assert plot(runs, tmp_path / "c.png").stat().st_size > 30_000
