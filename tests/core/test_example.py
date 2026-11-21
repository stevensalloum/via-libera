from pathlib import Path

import numpy as np

from tabpfn_spatial.core.example import hidden_field, run_example
from tabpfn_spatial.core.grid import Grid2D


def test_hidden_field_is_deterministic_and_bounded() -> None:
    g = Grid2D((0, 10), (0, 10), 20, 20)
    a, b = hidden_field(g.points()), hidden_field(g.points())
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 1


def test_run_example_baseline_end_to_end(tmp_path: Path) -> None:
    report = run_example(backend="baseline", n_readings=150, resolution=24, out_dir=tmp_path)
    assert report.backend == "baseline"
    assert 0 < report.crps < 0.3
    assert 0.5 < report.coverage90 <= 1.0
    assert report.figure.exists() and report.figure.stat().st_size > 1000
