"""Baseline evacuation planner and the shared exposure check.

Dijkstra on the 8-connected grid from each person to every exit (a diagonal step is only
allowed when both cells it squeezes past are open: no cutting machine corners). A step
costs its length plus ``risk_weight`` x the hazard probability of the cell at the time the
person arrives (walking at ``speed``). With an all-zero hazard map this is the
conventional fixed plan (nearest exit). Cost: O(N log N) per person for N grid cells.
"""

import heapq

import numpy as np

from tabpfn_spatial.contracts import EvacPlan, HazardMap, Route
from tabpfn_spatial.sim import Factory

WALK_SPEED = 1.3  # m/s
_MOVES = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _danger(hazard: HazardMap) -> np.ndarray:
    """(h, ny, nx) P(harm) combining toxic and fire, assuming independence."""
    return 1 - (1 - hazard.p_toxic) * (1 - hazard.p_fire)


class RiskAwarePlanner:
    """Shortest path where hazardous cells cost extra."""

    def __init__(self, risk_weight: float = 200.0, speed: float = WALK_SPEED) -> None:
        """``risk_weight``: metres of detour worth avoiding a certain hazard."""
        self.risk_weight, self.speed = risk_weight, speed

    def plan(
        self, factory: Factory, people: np.ndarray, hazard: HazardMap, t_start: float
    ) -> EvacPlan:
        """One route per person to whichever exit is cheapest."""
        danger = _danger(hazard)
        exits = {factory.cell_of(e): i for i, e in enumerate(factory.exits)}
        routes = [
            self._route(factory, factory.cell_of(p), exits, danger, hazard.times, t_start)
            for p in np.atleast_2d(people)
        ]
        return EvacPlan(routes)

    def _route(self, f, start, exits, danger, times, t_start) -> Route:
        best = {start: 0.0}
        length = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        heap = [(0.0, start)]
        while heap:
            cost, cell = heapq.heappop(heap)
            if cell in exits:
                return self._finish(f, cell, exits[cell], parent, danger, times, t_start)
            if cost > best[cell]:
                continue
            for dy, dx in _MOVES:
                nxt = (cell[0] + dy, cell[1] + dx)
                if not (0 <= nxt[0] < f.ny and 0 <= nxt[1] < f.nx) or f.walls[nxt]:
                    continue
                if dy and dx and (f.walls[cell[0] + dy, cell[1]] or f.walls[cell[0], cell[1] + dx]):
                    continue  # would cut the corner of a solid cell
                step = f.cell * (1.4142 if dy and dx else 1.0)
                arrive = t_start + (length[cell] + step) / self.speed
                k = int(np.argmin(np.abs(times - arrive)))
                new = cost + step + self.risk_weight * danger[k][nxt]
                if new < best.get(nxt, np.inf):
                    best[nxt], length[nxt], parent[nxt] = new, length[cell] + step, cell
                    heapq.heappush(heap, (new, nxt))
        raise RuntimeError(f"no exit reachable from cell {start}")

    def _finish(self, f, cell, exit_index, parent, danger, times, t_start) -> Route:
        path = [cell]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        cells = np.array(path[::-1])
        p = _danger_at_arrival(f, cells, danger, times, t_start, self.speed)
        return Route(cells, exit_index, float(1 - np.prod(1 - p)))

    def cost(self, f: Factory, route: Route, hazard: HazardMap, t_start: float) -> float:
        """The planner's own cost of ``route``: metres + ``risk_weight`` x P(harm) per step."""
        p = _danger_at_arrival(f, route.cells, _danger(hazard), hazard.times, t_start, self.speed)
        return float(_arrival_lengths(f, route.cells)[-1] + self.risk_weight * p[1:].sum())


def _danger_at_arrival(f, cells, danger, times, t_start, speed) -> np.ndarray:
    """(m,) P(harm) of each route cell at the forecast time nearest to the arrival."""
    arrive = t_start + _arrival_lengths(f, cells) / speed
    k = np.argmin(np.abs(np.asarray(times)[None] - arrive[:, None]), axis=1)
    return danger[k, cells[:, 0], cells[:, 1]]


def _arrival_lengths(f: Factory, cells: np.ndarray) -> np.ndarray:
    """Distance walked (metres) when reaching each cell of a route."""
    steps = np.linalg.norm(np.diff(cells, axis=0), axis=1) * f.cell
    return np.concatenate([[0.0], np.cumsum(steps)])


def route_exposed(
    f: Factory,
    route: Route,
    times: np.ndarray,
    harmful: np.ndarray,
    t_start: float,
    speed: float = WALK_SPEED,
) -> bool:
    """True if any cell is harmful (true sim) when the person reaches it.

    ``harmful``: (T, ny, nx) boolean ground truth at ``times`` (nearest time is used).
    """
    arrive = t_start + _arrival_lengths(f, route.cells) / speed
    k = np.argmin(np.abs(np.asarray(times)[None] - arrive[:, None]), axis=1)
    return bool(harmful[k, route.cells[:, 0], route.cells[:, 1]].any())
