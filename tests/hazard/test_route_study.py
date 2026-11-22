import runpy
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import HazardMap
from tabpfn_spatial.sim import Factory

STUDY = runpy.run_path(str(Path(__file__).with_name("route_study.py")))


def test_overlay_does_not_change_the_path_and_counts_false_positives():
    factory = Factory.default()
    start = factory.leak_sites[0]
    times = np.array([10.0, 130.0])
    probabilities = np.full((2, factory.ny, factory.nx), 0.3) * ~factory.walls
    hazard = HazardMap(times, probabilities, np.zeros_like(probabilities))
    truth = np.zeros((200, factory.ny, factory.nx))
    outcome = STUDY["replay"](factory, start, 10, truth, hazard, delay=0)
    assert outcome["exposed"] is False
    assert outcome["completed"] is True
    assert outcome["overlay"]["0.19"]["false_positive_s"] > 0
    assert outcome["overlay"]["0.50"]["false_positive_s"] == 0
    assert outcome["overlay"]["0.19"]["false_negative_s"] == 0


def test_replay_measures_harm_at_alarm_and_unfinished_walks():
    factory = Factory.default()
    start = factory.leak_sites[0]
    iy, ix = factory.cell_of(start)
    truth = np.zeros((11, factory.ny, factory.nx))
    truth[10, iy, ix] = 1.0
    probabilities = np.zeros((1, factory.ny, factory.nx))
    hazard = HazardMap(np.array([10.0]), probabilities, probabilities)
    result = STUDY["replay"](factory, start, 10, truth, hazard, delay=0)
    assert result["harmful_s"] == 1
    assert result["exposed"] is True
    assert result["completed"] is False
    assert result["overlay"]["0.19"]["false_negative_s"] == 1


def test_route_aggregates_exclude_unalarmed_cases_and_incomplete_distance():
    summary = runpy.run_path(str(Path(__file__).with_name("summarise_routes.py")))
    one = dict(exposed=True, harmful_s=3, walked_m=12.0, completed=True, beyond_horizon_s=0)
    two = dict(exposed=False, harmful_s=0, walked_m=99.0, completed=False, beyond_horizon_s=0)
    rows = [
        {"alarm": 10.0, "initial_harm": [True, False], "routes": {"fixed": [one, two]}},
        {"alarm": None},
    ]
    stats = summary["aggregate"](rows, "fixed")
    assert stats["exposed"] == 1 and stats["people"] == 2
    assert stats["completed"] == 1 and stats["mean_completed_m"] == 12.0
    assert stats["mean_harm_s"] == 1.5
    assert stats["safe_at_alarm"] == 1 and stats["exposed_after_safe_start"] == 0


def test_forecasts_start_from_the_same_alarmed_sensor_history():
    from tabpfn_spatial.source.features import alarm_time

    cases = STUDY["make_cases"](Factory.default(), 1, 2)
    for case in cases:
        if case["alarm"] is not None:
            assert alarm_time(case["obs"].times, case["obs"].values) == case["alarm"]
            assert case["times"][0] == case["alarm"]


def test_route_seed_is_independent_of_training_and_threshold_studies():
    assert STUDY["SEED"] not in {1001, 12345, 20261129, 20261201, 20261202}


def test_paired_exposure_counts_compare_the_same_people():
    summary = runpy.run_path(str(Path(__file__).with_name("summarise_routes.py")))
    rows = [
        {
            "alarm": 10,
            "initial_harm": [False, True],
            "routes": {
                "fixed": [{"exposed": True}, {"exposed": True}],
                "forward": [{"exposed": False}, {"exposed": True}],
            },
        }
    ]
    counts = summary["paired_counts"](rows, "fixed", "forward", safe_start=True)
    assert counts == {"both": 0, "test_only": 1, "reference_only": 0, "neither": 0}
