"""Guard the synthetic benchmark against training/test seed reuse."""

import runpy
from pathlib import Path

import numpy as np
import pytest


def test_3d_benchmark_uses_separate_scenarios_and_explicit_heights():
    study = runpy.run_path(str(Path(__file__).with_name("benchmark_volume3d.py")))
    grid, candidates = study["example_layout"]()
    with pytest.raises(ValueError, match="test seed"):
        study["make_cases"](grid, candidates, 3, study["TRAIN_SEED"])
    cases = study["make_cases"](grid, candidates, 12, study["TEST_SEED"])
    assert len({float(case["source"][2]) for case in cases}) > 1
    for case in cases:
        iz, iy, ix = grid.indices(case["source"][None])[0]
        assert candidates[iz, iy, ix] and not grid.solid[iz, iy, ix]
        assert case["diffusivity"] > 0
        assert np.isfinite(case["wind"]).all()
