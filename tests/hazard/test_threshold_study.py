import runpy
from pathlib import Path

import numpy as np
import pytest

STUDY = runpy.run_path(str(Path(__file__).with_name("threshold_study.py")))


def test_threshold_counts_include_ties_and_zero_probabilities():
    thresholds = np.array([0.0, 0.1, 0.5, 1.0])
    p = np.array([0.0, 0.1, 0.5, 1.0])
    y = np.array([True, False, True, False])
    result = STUDY["threshold_counts"](p, y, thresholds)
    assert result["tp"] == [2, 1, 1, 0]
    assert result["fp"] == [2, 2, 1, 1]
    assert result["positive"] == 2 and result["negative"] == 2


def test_threshold_counts_allow_only_floating_point_roundoff_outside_probability_range():
    rounded = STUDY["threshold_counts"](np.array([-1e-16, 1 + 1e-15]), [False, True], [0, 1])
    assert rounded["tp"] == [1, 1]
    with pytest.raises(ValueError, match="probabilities"):
        STUDY["threshold_counts"](np.array([1.01]), [True], [0, 1])


def test_selection_meets_validation_target_and_does_not_use_test_counts():
    thresholds = [0.0, 0.1, 0.5, 1.0]
    rows = [{"positive": 10, "negative": 10, "tp": [10, 9, 8, 1], "fp": [10, 4, 2, 0]}]
    assert STUDY["choose_threshold"](rows, thresholds, 0.9) == 0.1
    with pytest.raises(ValueError, match="positive"):
        STUDY["choose_threshold"](
            [dict(positive=0, negative=3, tp=[0] * 4, fp=[3] * 4)], thresholds, 0.9
        )


def test_split_seeds_do_not_overlap_training_or_each_other():
    seeds = STUDY["SPLIT_SEEDS"]
    assert len(set(seeds.values()) | {1001, 12345, 20261129}) == len(seeds) + 3


def test_alarm_triggered_case_uses_same_history_and_has_alarmed():
    from tabpfn_spatial.sim import Factory
    from tabpfn_spatial.source.features import alarm_time

    cases = STUDY["make_cases"](Factory.default(), 1, STUDY["SPLIT_SEEDS"]["validation"])
    assert len(cases) == 5
    for case in cases:
        if case["alarm"] is not None:
            assert alarm_time(case["obs"].times, case["obs"].values) == case["alarm"]
            assert case["obs"].t_obs == case["alarm"]
            assert case["times"][0] == case["alarm"]


def test_summary_excludes_cases_without_an_alarm():
    analysis = runpy.run_path(str(Path(__file__).with_name("summarise_threshold.py")))
    counts = STUDY["threshold_counts"](np.array([0.2, 0.7]), [True, False])
    rows = [{"alarm": 5, "fixed": counts}, {"alarm": None}]
    summary = analysis["aggregate"](rows, "fixed")
    assert summary["detected"] == 1 and summary["total"] == 2
    result = analysis["metrics"](summary, 0.5)
    assert result["recall"] == 0 and result["fpr"] == 1


def test_selection_rejects_incomplete_or_duplicate_case_sets():
    rows = [
        {"layout": "default", "episode": 0, "diffusion": diffusion} for diffusion in STUDY["VALUES"]
    ]
    STUDY["require_complete"](rows, ["default"], 1)
    with pytest.raises(ValueError, match="complete"):
        STUDY["require_complete"](rows[:-1], ["default"], 1)
    with pytest.raises(ValueError, match="complete"):
        STUDY["require_complete"]([*rows, rows[0]], ["default"], 1)
