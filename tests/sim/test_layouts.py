"""Shaped plants, AGVs and the height zone: properties every plant must have."""

import numpy as np
import pytest

from tabpfn_spatial.sim.factory import _all_reach
from tabpfn_spatial.sim.gases import GASES, height_zone, passive_distance
from tabpfn_spatial.sim.layouts import KINDS, TEMPLATES, shaped, template
from tabpfn_spatial.sim.robots import AGV_SPEED, agv_xy, blocked_factory
from tabpfn_spatial.world.export import outline, scene


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_template_is_evacuable_and_consistent(name: str) -> None:
    f = template(name)
    assert f.outside is not None and f.outside.any()
    assert (f.walls[f.outside]).all()  # outside is solid for the planner
    exits = [f.cell_of(e) for e in f.exits]
    assert all(not f.walls[c] and not f.outside[c] for c in exits)
    assert _all_reach(f.walls, exits)  # every open cell reaches an exit
    assert not f.walls[tuple(np.array([f.cell_of(s) for s in f.sensors]).T)].any()
    assert f.pipes is not None and len(f.leak_sites) > 10
    for lane in f.agv_lanes:  # AGV lanes on open cells only
        assert not any(f.walls[f.cell_of(p)] for p in lane)


def test_template_is_reproducible() -> None:
    a, b = template("U-plant"), template("U-plant")
    assert np.array_equal(a.walls, b.walls) and a.machines == b.machines


@pytest.mark.parametrize("kind", KINDS)
def test_unseen_random_shapes_build(kind: str) -> None:
    for seed in (11, 12):
        f = shaped(np.random.default_rng(seed), kind)
        assert _all_reach(f.walls, [f.cell_of(e) for e in f.exits])


def test_outline_is_closed() -> None:
    f = template("T-plant")
    ends = {}
    for r in outline(f):
        for p in (tuple(r["from"]), tuple(r["to"])):
            ends[p] = ends.get(p, 0) + 1
    assert all(v % 2 == 0 for v in ends.values())  # every corner joins two runs


def test_agvs_move_at_their_speed_and_stop_at_the_alarm() -> None:
    f = template("L-plant")
    xy = agv_xy(f, np.arange(0.0, 20.0), t_stop=10.0)
    step = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    assert np.allclose(step[:, :9], AGV_SPEED, atol=1e-6)
    assert np.allclose(step[:, 10:], 0.0)


def test_stopped_agvs_block_but_never_trap_anyone() -> None:
    f = template("campus")
    g, stops = blocked_factory(f, 23.0)
    assert int(g.walls.sum() - f.walls.sum()) == sum(s["blocks"] for s in stops)
    assert _all_reach(g.walls, [g.cell_of(e) for e in g.exits])


def test_height_zone_physics() -> None:
    light, heavy, air = GASES["methane"], GASES["chlorine"], GASES["CO"]
    assert light.advice == "stay low" and heavy.advice == "stay high" and air.advice == "none"
    small = passive_distance(heavy.sg, 0.001, 0.3)
    big = passive_distance(heavy.sg, 0.1, 0.3)
    assert big == pytest.approx(10 * small)  # x_p grows with sqrt(q0)
    assert passive_distance(heavy.sg, 0.01, 3.0) < passive_distance(heavy.sg, 0.01, 0.3)
    assert height_zone("propane", 1.0, 0.5)["radius_m"] > 0


def test_scene_of_a_rectangular_plant_is_unchanged_in_shape() -> None:
    from tabpfn_spatial.sim import Factory

    s = scene(Factory.default())
    assert not s["shaped"] and s["outline"] == [] and s["agv_lanes"] == []
    assert {e["side"] for e in s["exits"]} == {"west", "east", "south", "north"}
