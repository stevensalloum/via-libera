"""Crowd simulator: many people walking cell to cell, queueing at doors and in crowds.

A deliberately simple, standard model (a cell queue model, in the spirit of floor-field
cellular automata) that is enough to see congestion:

- **Speed falls with density** (Weidmann 1993): ``v = v0 (1 - exp(-gamma (1/rho - 1/rho_max)))``
  with the density ``rho`` (people/m^2) of the cell being entered.
- **Flow is capped**: at most ``specific_flow`` people per metre of width per second leave
  a cell (cell width = ``factory.cell``) or pass an exit door (``door_m`` wide). About
  1.3 people/(m s) is the usual handbook value for doors and corridors.
- **Cells fill up**: at most ``max_occupancy`` people per cell (90 % of jam density).

Time steps of ``dt`` = 1 s; a person who cannot move waits (and keeps breathing the gas).
Ready people move oldest-waiting first, so queues are first come, first served.
Cost per step: O(people) in Python; 1,000 people x 300 s takes about a second.
"""

from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.evac.planner import WALK_SPEED
from tabpfn_spatial.sim import Factory


@dataclass(frozen=True)
class CrowdParams:
    """Walking and capacity constants (defaults from pedestrian-flow handbooks)."""

    speed: float = WALK_SPEED  # free walking speed, m/s
    min_speed: float = 0.1  # crawl speed in a jam, m/s
    door_m: float = 1.0  # clear width of each exit door, m
    specific_flow: float = 1.3  # people per metre of width per second (doors, corridors)
    rho_max: float = 5.4  # jam density, people/m^2 (Weidmann)
    gamma: float = 1.913  # Weidmann's fit, 1/m^2


def weidmann_speed(rho: float, p: CrowdParams) -> float:
    """Walking speed (m/s) at density ``rho`` (people/m^2)."""
    if rho >= p.rho_max:
        return p.min_speed
    v = p.speed * (1 - np.exp(-p.gamma * (1 / max(rho, 1e-6) - 1 / p.rho_max)))
    return float(max(v, p.min_speed))


def sample_workers(
    f: Factory, n: int, rng: np.random.Generator, near_share: float = 0.75, near_m: float = 6.0
) -> np.ndarray:
    """(n, 2) positions on open floor: ``near_share`` work within ``near_m`` of a machine."""
    c = f.centres()
    near = np.zeros(f.walls.shape, dtype=bool)
    for m in f.machines:
        near |= (
            (c[..., 0] > m.x0 - near_m) & (c[..., 0] < m.x1 + near_m)
            & (c[..., 1] > m.y0 - near_m) & (c[..., 1] < m.y1 + near_m)
        )  # fmt: skip
    near &= ~f.walls
    pools = (c[near], c[~f.walls]) if near.any() else (c[~f.walls], c[~f.walls])
    k = rng.binomial(n, near_share)
    pick = [p[rng.integers(len(p), size=m)] for p, m in zip(pools, (k, n - k), strict=True)]
    jitter = rng.uniform(-0.4, 0.4, size=(n, 2)) * f.cell
    return np.vstack(pick) + jitter


class Crowd:
    """State of everyone in the plant; ``step`` advances one ``dt``."""

    def __init__(self, f: Factory, positions: np.ndarray, p: CrowdParams = CrowdParams()):
        """People start at ``positions`` (n, 2) metres, standing, without a route."""
        self.f, self.p = f, p
        self.cells = np.array([f.cell_of(xy) for xy in np.atleast_2d(positions)], dtype=int)
        n = len(self.cells)
        self.routes: list[np.ndarray | None] = [None] * n  # route[0] = current cell
        self.exit_index = np.full(n, -1)
        self.wait = np.zeros(n)
        self.ready_since = np.zeros(n)
        self.out_at = np.full(n, np.nan)
        self.dose = np.zeros(n)
        self.harm_s = np.zeros(n)
        self.occupancy = np.zeros(f.walls.shape, dtype=int)
        np.add.at(self.occupancy, (self.cells[:, 0], self.cells[:, 1]), 1)
        self.max_occupancy = int(0.9 * p.rho_max * f.cell**2)
        self._cap_cell = p.specific_flow * f.cell
        self._cap_exit = p.specific_flow * p.door_m
        self._budget = np.zeros(f.walls.shape)
        self._exit_budget = np.zeros(len(f.exits))

    @property
    def active(self) -> np.ndarray:
        """(n,) still inside the plant."""
        return np.isnan(self.out_at)

    def set_route(self, i: int, path: np.ndarray, exit_index: int) -> None:
        """Give person ``i`` a path (m, 2) cells starting at their current cell."""
        if tuple(path[0]) != tuple(self.cells[i]):
            raise ValueError(f"route of person {i} must start at their cell {self.cells[i]}")
        self.routes[i], self.exit_index[i] = np.asarray(path, dtype=int), exit_index

    def expose(self, field: np.ndarray, harm: np.ndarray, dt: float = 1.0) -> None:
        """Everyone still inside breathes ``field`` at their cell for ``dt`` seconds."""
        a = self.active
        iy, ix = self.cells[a, 0], self.cells[a, 1]
        self.dose[a] += field[iy, ix] * dt
        self.harm_s[a] += harm[iy, ix] * dt

    def _speed_into(self, cell: tuple[int, int]) -> float:
        return weidmann_speed((self.occupancy[cell] + 1) / self.f.cell**2, self.p)

    def step(self, t: float, dt: float = 1.0) -> None:
        """Advance ``dt`` seconds: queue-ordered moves under flow and occupancy limits."""
        # unused flow carries over up to one person (no hoarding while a cell is idle)
        self._budget = np.minimum(self._budget, 1.0) + self._cap_cell * dt
        self._exit_budget = np.minimum(self._exit_budget, 1.0) + self._cap_exit * dt
        moving = self.active & np.array([r is not None for r in self.routes], dtype=bool)
        was_waiting = self.wait > 0
        self.wait[moving] -= dt
        newly = moving & was_waiting & (self.wait <= 1e-9)
        self.ready_since[newly] = t
        ready = np.flatnonzero(moving & (self.wait <= 1e-9))
        for i in ready[np.lexsort((ready, self.ready_since[ready]))]:
            self._try_move(int(i), t, dt)

    def _try_move(self, i: int, t: float, dt: float) -> None:
        here, route = tuple(self.cells[i]), self.routes[i]
        if len(route) == 1:  # standing in the exit cell: leave if the door lets us
            e = self.exit_index[i]
            if self._exit_budget[e] >= 1:
                self._exit_budget[e] -= 1
                self.occupancy[here] -= 1
                self.out_at[i] = t + dt
            return
        nxt = tuple(route[1])
        if self._budget[here] < 1 or self.occupancy[nxt] >= self.max_occupancy:
            self.wait[i] = 0.0  # blocked: queue, no speed credit
            return
        self._budget[here] -= 1
        self.occupancy[here] -= 1
        self.occupancy[nxt] += 1
        diag = here[0] != nxt[0] and here[1] != nxt[1]
        step = self.f.cell * (1.4142135623730951 if diag else 1.0)
        self.wait[i] = max(self.wait[i], -dt) + step / self._speed_into(nxt)
        self.cells[i] = nxt
        self.routes[i] = route[1:]
