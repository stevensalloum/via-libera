import numpy as np

from tabpfn_spatial.contracts import HazardMap
from tabpfn_spatial.evac import RiskAwarePlanner, route_exposed
from tabpfn_spatial.sim import Factory


def _calm(f: Factory) -> HazardMap:
    z = np.zeros((1, f.ny, f.nx))
    return HazardMap(np.array([60.0]), z, z)


def test_route_reaches_an_exit_through_open_cells() -> None:
    f = Factory.default()
    plan = RiskAwarePlanner().plan(f, np.array([[5.0, 21.0]]), _calm(f), 60.0)
    route = plan.routes[0]
    assert tuple(route.cells[0]) == f.cell_of(np.array([5.0, 21.0]))
    assert tuple(route.cells[-1]) == f.cell_of(f.exits[route.exit_index])
    assert not f.walls[route.cells[:, 0], route.cells[:, 1]].any()
    steps = np.abs(np.diff(route.cells, axis=0)).max(axis=1)
    assert np.all(steps == 1)  # 8-connected, no jumps
    assert route.risk == 0.0


def test_planner_detours_around_a_toxic_corridor() -> None:
    f = Factory.default()
    start = np.array([[5.0, 21.0]])
    calm = RiskAwarePlanner().plan(f, start, _calm(f), 60.0).routes[0]
    hz = _calm(f).p_toxic.copy()
    for iy, ix in calm.cells[1:]:
        hz[0, iy, ix] = 0.9  # poison the shortest route
    risky = HazardMap(np.array([60.0]), hz, np.zeros_like(hz))
    detour = RiskAwarePlanner().plan(f, start, risky, 60.0).routes[0]
    on_poison = hz[0, detour.cells[1:, 0], detour.cells[1:, 1]] > 0
    assert on_poison.mean() < 0.5 and detour.risk < 0.9


def test_route_exposed_uses_arrival_times() -> None:
    f = Factory.default()
    route = RiskAwarePlanner().plan(f, np.array([[5.0, 21.0]]), _calm(f), 0.0).routes[0]
    times = np.array([0.0, 1000.0])
    toxic = np.zeros((2, f.ny, f.nx), dtype=bool)
    toxic[1] = True  # everything toxic, but only long after the person left
    assert not route_exposed(f, route, times, toxic, t_start=0.0)
    toxic[0] = True
    assert route_exposed(f, route, times, toxic, t_start=0.0)
