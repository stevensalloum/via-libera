"""Cost-to-exit fields: one reverse Dijkstra per exit, shared by every person.

For exit ``e``, ``cost[iy, ix]`` is the cheapest cost of walking from that cell to the
exit, where entering a cell costs its step length (m) plus ``risk_weight`` x its danger
(P(harm) in [0, 1]); ``length`` is the walking distance of that cheapest path and
``next`` the cell to step to. Moves are 8-connected without cutting solid corners (as in
``RiskAwarePlanner``). With zero risk this is the shortest path to that exit.

Danger is static per plan: the worst forecast P(harm) over the next ``ahead`` seconds
(``danger_window``), a conservative stand-in for the time-dependent search of the
one-person planner that keeps crowd planning O(exits * N log N) for N cells, whatever the
number of people.
"""

import heapq
from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.contracts import HazardMap
from tabpfn_spatial.sim import Factory

_MOVES = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


@dataclass(frozen=True)
class ExitField:
    """Cost-to-go towards one exit."""

    exit_index: int
    exit_cell: tuple[int, int]
    cost: np.ndarray  # (ny, nx), inf on solid / unreachable cells
    length: np.ndarray  # (ny, nx) metres along the cheapest path
    next: np.ndarray  # (ny, nx, 2) next cell; -1 at the exit and on solid cells


def danger_window(hz: HazardMap, t: float, ahead: float = 60.0) -> np.ndarray:
    """(ny, nx) worst P(harm) over forecasts in ``[t, t + ahead]`` (latest one if none)."""
    keep = (hz.times >= t - 1e-9) & (hz.times <= t + ahead + 1e-9)
    if not keep.any():
        keep = hz.times == hz.times.max()
    p = 1 - (1 - hz.p_toxic[keep]) * (1 - hz.p_fire[keep])
    return p.max(axis=0)


def _field(f: Factory, e: int, danger: np.ndarray, risk_weight: float) -> ExitField:
    goal = f.cell_of(f.exits[e])
    cost = np.full(f.walls.shape, np.inf)
    length = np.full(f.walls.shape, np.inf)
    nxt = np.full((*f.walls.shape, 2), -1, dtype=int)
    cost[goal] = length[goal] = 0.0
    heap = [(0.0, goal)]
    while heap:
        c, v = heapq.heappop(heap)
        if c > cost[v]:
            continue
        enter = risk_weight * danger[v]  # paid by whoever steps into v
        for dy, dx in _MOVES:
            u = (v[0] + dy, v[1] + dx)
            if not (0 <= u[0] < f.ny and 0 <= u[1] < f.nx) or f.walls[u]:
                continue
            if dy and dx and (f.walls[v[0], u[1]] or f.walls[u[0], v[1]]):
                continue  # u -> v would cut a solid corner
            step = f.cell * (1.4142135623730951 if dy and dx else 1.0)
            new = c + step + enter
            if new < cost[u]:
                cost[u], length[u], nxt[u] = new, length[v] + step, v
                heapq.heappush(heap, (new, u))
    return ExitField(e, goal, cost, length, nxt)


def exit_fields(f: Factory, danger: np.ndarray, risk_weight: float = 200.0) -> list[ExitField]:
    """One ``ExitField`` per exit. O(exits * N log N) for N grid cells."""
    return [_field(f, e, danger, risk_weight) for e in range(len(f.exits))]


def follow(fe: ExitField, start: tuple[int, int]) -> np.ndarray:
    """(m, 2) cells from ``start`` to the field's exit (inclusive)."""
    path = [tuple(start)]
    while path[-1] != fe.exit_cell:
        nxt = tuple(int(v) for v in fe.next[path[-1]])
        if nxt[0] < 0:
            raise ValueError(f"cell {path[-1]} cannot reach exit {fe.exit_index}")
        path.append(nxt)
    return np.array(path)
