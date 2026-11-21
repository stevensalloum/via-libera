"""Scene export for the 3D world: everything drawn comes from the planner's factory."""

import json

import numpy as np
import pytest

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.world.export import kind_of, pipe_runs, scene, wall_runs

F = Factory.default()


def _cells_of_runs(runs, cell: float, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r in runs:
        (x0, y0), (x1, y1) = r["from"], r["to"]
        a = (int(y0 // cell), int(x0 // cell))
        b = (int(y1 // cell), int(x1 // cell))
        mask[min(a[0], b[0]) : max(a[0], b[0]) + 1, min(a[1], b[1]) : max(a[1], b[1]) + 1] = True
    return mask


def test_scene_is_json_and_carries_the_plant() -> None:
    s = json.loads(json.dumps(scene(F)))
    assert s["width"] == 60 and s["height"] == 40 and s["cell"] == 2
    assert len(s["machines"]) == len(F.machines)
    assert len(s["exits"]) == len(F.exits) and len(s["sensors"]) == len(F.sensors)
    assert np.array(s["solid"]).shape == F.walls.shape
    np.testing.assert_array_equal(np.array(s["solid"], dtype=bool), F.walls)


def test_pipe_runs_rebuild_the_pipe_mask_exactly() -> None:
    runs = pipe_runs(F)
    np.testing.assert_array_equal(_cells_of_runs(runs, F.cell, F.walls.shape), F.pipes)
    for r in runs:
        (x0, y0), (x1, y1) = r["from"], r["to"]
        assert x0 == x1 or y0 == y1  # straight runs


def test_wall_runs_cover_the_solid_cells_that_are_not_machines() -> None:
    runs = wall_runs(F)
    mask = _cells_of_runs(runs, F.cell, F.walls.shape)
    c = F.centres()
    machine = np.zeros(F.walls.shape, dtype=bool)
    for m in F.machines:
        machine |= (
            (c[..., 0] >= m.x0) & (c[..., 0] <= m.x1) & (c[..., 1] >= m.y0) & (c[..., 1] <= m.y1)
        )
    np.testing.assert_array_equal(mask, F.walls & ~machine)
    assert mask.sum() > 0  # the partition wall


def test_exits_know_which_outer_wall_they_are_on() -> None:
    sides = [e["side"] for e in scene(F)["exits"]]
    assert sorted(sides) == ["east", "north", "south", "west"]


@pytest.mark.parametrize(
    ("name", "kind"),
    [("tank farm", "tank"), ("furnace", "furnace"), ("kiln", "furnace"), ("boiler", "furnace"),
     ("compressor A", "compressor"), ("chiller", "compressor"), ("press line", "press"),
     ("generator", "generator"), ("dryer", "drum"), ("mixer", "drum"), ("lathe", "machine")],
)  # fmt: skip
def test_machine_kinds_drive_the_3d_model(name: str, kind: str) -> None:
    assert kind_of(name) == kind


def test_machines_keep_their_exact_footprint_and_get_a_height() -> None:
    for m, d in zip(F.machines, scene(F)["machines"], strict=True):
        assert d["rect"] == [m.x0, m.y0, m.x1, m.y1]
        assert d["hot"] == m.hot and d["height"] > 0


def test_random_plants_export_too() -> None:
    s = scene(Factory.random(np.random.default_rng(3), width=120.0, height=80.0, n_machines=10,
                             n_hot=4, exits_per_side=2, sensor_spacing=15.0))  # fmt: skip
    assert len(s["exits"]) == 8 and s["pipes"]
    json.dumps(s)
