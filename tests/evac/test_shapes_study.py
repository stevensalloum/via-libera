"""The unseen-shapes study: plants are new, runs are paired, AGVs and crowd sizes apply."""

import numpy as np

from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.shapes_study import FIRST_SEED, job, paired, people_for, plant
from tabpfn_spatial.sim.layouts import TEMPLATES


def test_plants_are_unseen() -> None:
    assert max(t["seed"] for t in TEMPLATES.values()) < FIRST_SEED
    a, b = plant("U", FIRST_SEED), plant("U", FIRST_SEED + 1)
    assert not np.array_equal(a.walls, b.walls)


def test_one_job_runs_every_planner_on_the_same_leak() -> None:
    rows = job(("L", FIRST_SEED, 1000, "perfect", "baseline"))
    assert [r["planner"] for r in rows] == list(PLANNERS)
    assert len({(r["t_alarm"], r["people"], r["agv_blocks"]) for r in rows}) == 1
    assert rows[0]["people"] == people_for(plant("L", FIRST_SEED))
    gain = paired(rows, "perfect", "ccrp", "harm_s_mean")[None]
    assert gain[2] == 1
