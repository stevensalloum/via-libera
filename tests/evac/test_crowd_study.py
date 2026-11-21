"""Scenario ladder: plants and crowds of growing size, every planner on the same leaks."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_study import RUNGS, aggregate, plot_ladder, plot_plants, run_ladder


def _tiny():
    return [replace(RUNGS[0], people=12, t_max=150.0)]


def test_the_ladder_grows_in_size_and_crowd() -> None:
    assert [r.name for r in RUNGS] == ["A", "B", "C", "D", "E"]
    people = [r.people for r in RUNGS]
    assert people[0] < people[1] <= people[3] < people[4]
    assert RUNGS[2].door_m < RUNGS[1].door_m
    plants = [r.plant() for r in RUNGS]
    areas = [f.width * f.height for f in plants]
    assert areas[3] > areas[0] and areas[4] > areas[3]
    assert len(plants[4].exits) > len(plants[0].exits)


def test_run_ladder_pairs_planners_on_the_same_leaks_and_is_reproducible() -> None:
    rows = run_ladder(_tiny(), seeds=range(2), models=("perfect",), workers=1)
    assert len(rows) == 2 * len(PLANNERS)
    assert {r["planner"] for r in rows} == set(PLANNERS)
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], set()).add(r["t_alarm"])
    assert all(len(v) == 1 for v in by_seed.values())  # same leak for every planner
    assert rows == run_ladder(_tiny(), seeds=range(2), models=("perfect",), workers=1)


def test_aggregate_gives_means_and_paired_differences_to_nearest() -> None:
    rows = run_ladder(_tiny(), seeds=range(3), models=("perfect",), workers=1)
    agg = aggregate(rows)
    a = agg[("A", "perfect", "nearest")]
    assert a["n"] == 3
    assert agg[("A", "perfect", "nearest")]["harm_s_mean_vs_nearest"] == 0.0
    for p in PLANNERS:
        assert np.isfinite(agg[("A", "perfect", p)]["harm_s_mean"])


def test_figures_render(tmp_path: Path) -> None:
    rows = run_ladder(_tiny(), seeds=range(2), models=("perfect",), workers=1)
    assert plot_ladder(rows, tmp_path / "ladder.png").stat().st_size > 20_000
    assert plot_plants(RUNGS[3:], tmp_path / "plants.png").stat().st_size > 20_000
