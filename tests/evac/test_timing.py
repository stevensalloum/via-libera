"""Timing study: harm against alarm time, update interval and model latency."""

import json
from pathlib import Path

import numpy as np
import pytest

from tabpfn_spatial.evac.live import POLICIES
from tabpfn_spatial.evac.timing import STUDIES, Setting, plot_timing, run_one, run_study
from tabpfn_spatial.sim import Factory

F = Factory.default()


def test_run_one_scores_the_models_and_every_policy() -> None:
    out = run_one(F, Setting(models="baseline", t_alarm=20.0, after=40.0), seed=3)
    assert out["source_error_m"] >= 0 and 0 <= out["hazard_brier"] <= 1
    assert out["pre_alarm_harm_s"] <= 20.0
    for p in POLICIES:
        assert out[f"{p}_harm_s"] >= out["pre_alarm_harm_s"]
        assert out[f"{p}_exposed"] in (0.0, 1.0)


def test_perfect_models_have_no_source_error_and_no_brier() -> None:
    out = run_one(F, Setting(models="perfect", t_alarm=20.0, after=40.0), seed=3)
    assert out["source_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert out["hazard_brier"] == 0.0


def test_unknown_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="models"):
        run_one(F, Setting(models="tabpfn"), seed=0)


def test_study_is_paired_and_reproducible(tmp_path: Path) -> None:
    tiny = {"alarm": ("t_alarm", (10.0, 20.0), Setting(after=30.0), ("baseline",))}
    a = run_study(F, tiny, seeds=range(2), workers=1)
    b = run_study(F, tiny, seeds=range(2), workers=1)
    assert a == b
    rows = a["alarm"]["baseline"]
    assert [r["value"] for r in rows] == [10.0, 20.0]
    assert rows[0]["n"] == 2
    json.dumps(a)  # saved as JSON


def test_the_four_studies_are_defined() -> None:
    assert set(STUDIES) == {"alarm", "update", "latency_walk", "latency_wait"}


def test_plot_renders_from_results(tmp_path: Path) -> None:
    def rows(values):
        return [
            {"value": v, "n": 5, "source_error_m": 10 - v / 20, "hazard_brier": 0.08,
             "pre_alarm_harm_s": v / 10, **{f"{p}_harm_s": v / 5 + i for i, p in
             enumerate(POLICIES)}, **{f"{p}_exposed": 0.1 for p in POLICIES}}
            for v in values
        ]  # fmt: skip

    res = {
        name: {m: rows(values) for m in models} for name, (_, values, _, models) in STUDIES.items()
    }
    path = plot_timing(res, tmp_path / "timing.png")
    assert path.stat().st_size > 20_000
    assert np.isfinite(path.stat().st_size)
