"""The sync check catches every way the three parts can disagree on time, space and units."""

from dataclasses import replace

import numpy as np
import pytest

from tabpfn_spatial.contracts import EvacPlan, HazardMap, Route, SourceEstimate
from tabpfn_spatial.evac import RiskAwarePlanner
from tabpfn_spatial.hazard import ForwardSimPredictor
from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.source import WeightedCentroidLocator
from tabpfn_spatial.sync import (
    check_hazard,
    check_plan,
    check_readings,
    check_source,
    load,
    run_sync,
)

F = Factory.default()
SC = Scenario(F.leak_sites[4], 1.0, np.array([0.8, 0.2]))
R = readings(F, SC, 30.0, np.random.default_rng(0))
TIMES = 30.0 + np.arange(0.0, 121.0, 10.0)


def _uniform() -> SourceEstimate:
    return SourceEstimate(F.leak_prior, np.array([0.7, 1.0, 1.4]), np.array([0.1, 0.5, 0.9]))


def test_good_outputs_pass() -> None:
    assert check_readings(F, R) == []
    assert check_source(F, _uniform()) == []
    hz = ForwardSimPredictor().predict(F, R, _uniform(), TIMES)
    assert check_hazard(F, hz, TIMES) == []
    plan = RiskAwarePlanner().plan(F, np.array([[5.0, 15.0]]), hz, 30.0)
    assert check_plan(F, plan, np.array([[5.0, 15.0]])) == []


def test_readings_problems_are_found() -> None:
    bad = replace(R, times=R.times[::-1].copy())
    assert any("ascending" in p for p in check_readings(F, bad))
    assert any("positions" in p for p in check_readings(F, replace(R, positions=R.positions[:3])))


@pytest.mark.parametrize(
    ("prob", "word"),
    [
        (np.ones((F.ny, F.nx)), "sum"),
        (np.ones((F.nx, F.ny)) / (F.nx * F.ny), "shape"),
        (np.where(F.walls, 1.0, 0.0) / F.walls.sum(), "solid"),
    ],
)
def test_source_problems_are_found(prob: np.ndarray, word: str) -> None:
    assert any(word in p for p in check_source(F, SourceEstimate(prob)))


def test_source_off_the_pipes_and_bad_rate_quantiles_are_flagged() -> None:
    off = np.where(F.walls | (F.leak_prior > 0), 0.0, 1.0)
    assert any("leak sites" in p for p in check_source(F, SourceEstimate(off / off.sum())))
    est = SourceEstimate(F.leak_prior, np.array([2.0, 1.0]), np.array([0.1, 0.9]))
    assert any("rate" in p for p in check_source(F, est))


def test_hazard_problems_are_found() -> None:
    z = np.zeros((len(TIMES), F.ny, F.nx))
    relative = HazardMap(TIMES - 30.0, z, z)
    assert any("absolute" in p for p in check_hazard(F, relative, TIMES))
    over = HazardMap(TIMES, z + 1.5, z)
    assert any("[0, 1]" in p for p in check_hazard(F, over, TIMES))
    fire_everywhere = HazardMap(TIMES, z, z + 0.5)
    assert any("ignition" in p for p in check_hazard(F, fire_everywhere, TIMES))


def test_plan_problems_are_found() -> None:
    people = np.array([[5.0, 15.0]])
    wall = np.argwhere(F.walls)[0]
    through = Route(np.array([F.cell_of(people[0]), tuple(wall)]), 0, 0.0)
    probs = check_plan(F, EvacPlan([through]), people)
    assert any("solid" in p for p in probs) and any("exit" in p for p in probs)
    assert any("one route" in p for p in check_plan(F, EvacPlan([]), people))


def test_run_sync_with_the_baselines_reports_no_problems() -> None:
    rows = run_sync(
        WeightedCentroidLocator(),
        ForwardSimPredictor(),
        decision_times=(10.0, 30.0),
        sample_every=(5.0,),
        seeds=range(2),
        plants=("default", "random"),
    )
    assert len(rows) == 2 * 2 * 1 * 2
    assert all(r["problems"] == [] for r in rows), [r["problems"] for r in rows]
    assert all(r["locate_s"] >= 0 and r["predict_s"] >= 0 for r in rows)


def test_run_sync_reports_a_crash_instead_of_raising() -> None:
    class Broken:
        def locate(self, factory, readings):
            raise RuntimeError("model needs readings at 60 s")

    rows = run_sync(Broken(), ForwardSimPredictor(), decision_times=(10.0,), seeds=range(1),
                    plants=("default",))  # fmt: skip
    assert "model needs readings at 60 s" in rows[0]["problems"][0]


def test_load_finds_classes_by_import_path() -> None:
    assert load("tabpfn_spatial.source:WeightedCentroidLocator") is WeightedCentroidLocator
    with pytest.raises(ValueError, match="module:Class"):
        load("WeightedCentroidLocator")
