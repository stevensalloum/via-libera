"""Factory layout: 2D grid of floor, walls and machines, with exits, sensors and gas pipes.

Coordinates are metres, origin bottom-left; grid cell ``(iy, ix)`` covers
``[ix * cell, (ix + 1) * cell) x [iy * cell, (iy + 1) * cell)``.

Leaks happen where gas is: along the pipes, and most often at their fittings (machine
connections, tees, bends, valves), which industrial leak records name as the usual
release points. A plant built with ``pipes=False`` keeps the v0 rule (any cell next to a
machine) so old results can be reproduced.
"""

import heapq
from dataclasses import dataclass

import numpy as np

FITTING_WEIGHT = 5.0  # a fitting is this many times likelier to leak than a plain pipe cell


@dataclass(frozen=True)
class Machine:
    """Solid rectangle ``[x0, x1] x [y0, y1]`` in metres; ``hot`` = ignition source."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    hot: bool = False


@dataclass(frozen=True)
class Factory:
    """Grid layout. ``walls``: (ny, nx) solid cells (walls and machines)."""

    width: float
    height: float
    cell: float
    walls: np.ndarray
    machines: tuple[Machine, ...]
    exits: np.ndarray  # (e, 2) metres, on open cells at the boundary
    sensors: np.ndarray  # (k, 2) metres, on open cells
    pipes: np.ndarray | None = None  # (ny, nx) bool: cells a gas pipe runs over
    fittings: np.ndarray | None = None  # (ny, nx) bool: joints on the pipes (subset)

    @property
    def ny(self) -> int:
        """Rows."""
        return self.walls.shape[0]

    @property
    def nx(self) -> int:
        """Columns."""
        return self.walls.shape[1]

    def cell_of(self, xy: np.ndarray) -> tuple[int, int]:
        """Grid index (iy, ix) of a point in metres (clipped to the grid)."""
        ix = int(np.clip(xy[0] // self.cell, 0, self.nx - 1))
        iy = int(np.clip(xy[1] // self.cell, 0, self.ny - 1))
        return iy, ix

    def centre(self, iy: int, ix: int) -> np.ndarray:
        """Centre of a cell in metres."""
        return np.array([(ix + 0.5) * self.cell, (iy + 0.5) * self.cell])

    def centres(self) -> np.ndarray:
        """(ny, nx, 2) cell centres in metres."""
        xs = (np.arange(self.nx) + 0.5) * self.cell
        ys = (np.arange(self.ny) + 0.5) * self.cell
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx, gy], axis=-1)

    @property
    def ignition(self) -> np.ndarray:
        """(ny, nx) open cells within 2 cells of a hot machine (where gas can ignite)."""
        c = self.centres()
        near = np.zeros(self.walls.shape, dtype=bool)
        pad = 2 * self.cell
        for m in self.machines:
            if m.hot:
                near |= (
                    (c[..., 0] >= m.x0 - pad)
                    & (c[..., 0] <= m.x1 + pad)
                    & (c[..., 1] >= m.y0 - pad)
                    & (c[..., 1] <= m.y1 + pad)
                )
        return near & ~self.walls

    def _leak_mask(self) -> np.ndarray:
        """(ny, nx) open cells where a leak can start."""
        if self.pipes is not None:
            return self.pipes & ~self.walls
        c = self.centres()
        near = np.zeros(self.walls.shape, dtype=bool)
        for m in self.machines:
            near |= (
                (c[..., 0] >= m.x0 - self.cell)
                & (c[..., 0] <= m.x1 + self.cell)
                & (c[..., 1] >= m.y0 - self.cell)
                & (c[..., 1] <= m.y1 + self.cell)
            )
        return near & ~self.walls

    @property
    def leak_sites(self) -> np.ndarray:
        """(n, 2) centres of the cells where leaks start (row-major order).

        With pipes: every open pipe cell. Without (v0 plant): open cells within 1 cell of a
        machine.
        """
        return self.centres()[self._leak_mask()]

    @property
    def leak_weights(self) -> np.ndarray:
        """(n,) relative leak likelihood of each ``leak_sites`` entry (fittings weigh more)."""
        mask = self._leak_mask()
        if self.fittings is None:
            return np.ones(int(mask.sum()))
        return np.where(self.fittings[mask], FITTING_WEIGHT, 1.0)

    @property
    def leak_prior(self) -> np.ndarray:
        """(ny, nx) P(leak starts in this cell) before any reading: the source prior."""
        prior = np.zeros(self.walls.shape)
        prior[self._leak_mask()] = self.leak_weights
        return prior / prior.sum()

    @classmethod
    def default(cls, pipes: bool = True) -> "Factory":
        """60 m x 40 m hall, 2 m cells: two production halls split by a wall with doors.

        Gas comes from the tank farm along one main at y = 21 m: west to the compressor and
        the furnace, east through the wall to the generator and the dryer.
        ``pipes=False``: the v0 plant.
        """
        width, height, cell = 60.0, 40.0, 2.0
        machines = (
            Machine("compressor A", 8, 8, 14, 14),
            Machine("furnace", 8, 26, 16, 32, hot=True),
            Machine("tank farm", 20, 14, 26, 26),
            Machine("press line", 36, 6, 50, 10),
            Machine("generator", 40, 28, 46, 34, hot=True),
            Machine("dryer", 50, 18, 56, 24, hot=True),
        )
        walls = _rasterise(width, height, cell, machines, 30.0, ((6, 12), (26, 32)))
        exits = np.array([[1.0, 21.0], [59.0, 13.0], [21.0, 39.0], [45.0, 1.0]])
        sensors = np.array(
            [
                [5, 5],
                [17, 5],
                [29, 5],
                [41, 15],
                [55, 5],
                [5, 21],
                [17, 21],
                [27, 31],
                [37, 21],
                [55, 29],
                [5, 37],
                [21, 35],
                [37, 37],
                [53, 37],
            ],
            dtype=float,
        )
        if not pipes:
            return cls(width, height, cell, walls, machines, exits, sensors)
        pipe, fit = _pipes_from_segments(_DEFAULT_PIPES, cell, walls.shape)
        return cls(width, height, cell, walls, machines, exits, sensors, pipe, fit)

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        width: float = 60.0,
        height: float = 40.0,
        cell: float = 2.0,
        n_machines: int = 6,
        n_hot: int = 3,
    ) -> "Factory":
        """Random plant: machines, a dividing wall with two doors, 4 exits, 16 sensors, pipes.

        Pipes run from a supply point on the outer wall to every hot machine and one other
        machine (see ``_route_pipes``). Retries until every open cell can reach an exit and
        every pipe can be routed. Reproducible for a given ``rng``.
        """
        for _ in range(200):
            machines = _random_machines(rng, width, height, n_machines, n_hot)
            if machines is None:
                continue
            wall_x = float(rng.uniform(0.35, 0.65) * width)
            doors = tuple(
                (float(y), float(y) + 6.0)
                for y in (rng.uniform(2, height / 2 - 8), rng.uniform(height / 2, height - 8))
            )
            walls = _rasterise(width, height, cell, machines, wall_x, doors)
            exits = _random_exits(rng, walls, cell, width, height)
            if exits is None or not _all_reach(walls, [_cell(e, cell, walls) for e in exits]):
                continue
            sensors = _snap_sensors(rng, walls, cell, width, height)
            network = _route_pipes(rng, walls, machines, cell)
            if network is None:
                continue
            return cls(width, height, cell, walls, machines, exits, sensors, *network)
        raise RuntimeError("could not build a connected random factory")


def _rasterise(width, height, cell, machines, wall_x, doors) -> np.ndarray:
    """Solid mask (ny, nx) from machine rectangles and a vertical wall with door gaps."""
    ny, nx = int(height / cell), int(width / cell)
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell)
    walls = np.zeros((ny, nx), dtype=bool)
    for m in machines:
        walls |= (gx >= m.x0) & (gx <= m.x1) & (gy >= m.y0) & (gy <= m.y1)
    gap = np.zeros_like(walls)
    for y0, y1 in doors:
        gap |= (gy > y0) & (gy < y1)
    walls |= (gx > wall_x) & (gx < wall_x + cell) & ~gap
    return walls


_NAMES = (
    "compressor",
    "furnace",
    "tank farm",
    "press line",
    "generator",
    "dryer",
    "boiler",
    "mixer",
    "kiln",
    "chiller",
)


def _random_machines(rng, width, height, n, n_hot) -> tuple[Machine, ...] | None:
    """Non-overlapping rectangles 4-14 m, 4 m apart and 4 m from the walls (or None)."""
    boxes: list[Machine] = []
    hot = set(rng.choice(n, size=min(n_hot, n), replace=False).tolist())
    for i in range(n):
        for _ in range(100):
            w, h = rng.uniform(4, 14), rng.uniform(4, 10)
            x0, y0 = rng.uniform(4, width - 4 - w), rng.uniform(4, height - 4 - h)
            box = Machine(_NAMES[i % len(_NAMES)], x0, y0, x0 + w, y0 + h, hot=i in hot)
            if all(
                box.x1 + 4 < b.x0 or b.x1 + 4 < box.x0 or box.y1 + 4 < b.y0 or b.y1 + 4 < box.y0
                for b in boxes
            ):
                boxes.append(box)
                break
        else:
            return None
    return tuple(boxes)


def _cell(xy: np.ndarray, cell: float, walls: np.ndarray) -> tuple[int, int]:
    ny, nx = walls.shape
    return int(np.clip(xy[1] // cell, 0, ny - 1)), int(np.clip(xy[0] // cell, 0, nx - 1))


def _random_exits(rng, walls, cell, width, height) -> np.ndarray | None:
    """One exit on each side of the hall, on an open boundary cell (or None)."""
    ny, nx = walls.shape
    sides = [
        (lambda i: (i, 0), ny),
        (lambda i: (i, nx - 1), ny),
        (lambda i: (0, i), nx),
        (lambda i: (ny - 1, i), nx),
    ]
    exits = []
    for at, n in sides:
        free = [at(i) for i in range(2, n - 2) if not walls[at(i)]]
        if not free:
            return None
        iy, ix = free[rng.integers(len(free))]
        exits.append([(ix + 0.5) * cell, (iy + 0.5) * cell])
    return np.array(exits)


def _all_reach(walls: np.ndarray, starts: list[tuple[int, int]]) -> bool:
    """True if every open cell is 4-connected to one of ``starts``. O(ny * nx)."""
    seen = np.zeros_like(walls)
    stack = list(starts)
    for c in stack:
        seen[c] = True
    while stack:
        y, x = stack.pop()
        for n in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if (
                0 <= n[0] < walls.shape[0]
                and 0 <= n[1] < walls.shape[1]
                and not walls[n]
                and not seen[n]
            ):
                seen[n] = True
                stack.append(n)
    return bool(np.array_equal(seen, ~walls))


def _snap_sensors(rng, walls, cell, width, height, rows=4, cols=4) -> np.ndarray:
    """A jittered rows x cols grid of sensors, each moved to the nearest open cell."""
    open_yx = np.argwhere(~walls)
    out = []
    for gy in (np.arange(rows) + 0.5) / rows * height:
        for gx in (np.arange(cols) + 0.5) / cols * width:
            target = np.array([gx, gy]) + rng.uniform(-3, 3, size=2)
            centres = (open_yx[:, ::-1] + 0.5) * cell
            out.append(centres[np.argmin(((centres - target) ** 2).sum(axis=1))])
    return np.array(out)


# ------------------------------------------------------------------------ pipes --
# Default network as straight runs between cell centres (metres). Every run end is a
# fitting: a machine connection, a tee or a tank outlet. One gas main runs along y = 21 m
# from the compressor/furnace tee through the tank farm (its manifold) and the partition
# wall to the dryer. Solid cells can carry pipe but cannot leak into the hall.
_DEFAULT_PIPES = (
    ((11, 21), (19, 21)),  # west header: compressor/furnace tee -> tank farm
    ((19, 21), (27, 21)),  # tank farm manifold (inside the tanks)
    ((27, 21), (49, 21)),  # east header: tank farm -> through the wall -> dryer
    ((43, 21), (43, 27)),  # branch to the generator
    ((11, 21), (11, 15)),  # branch to compressor A
    ((11, 21), (11, 25)),  # branch to the furnace
)


def _pipes_from_segments(segments, cell: float, shape) -> tuple[np.ndarray, np.ndarray]:
    """Pipe and fitting masks from axis-aligned runs between cell centres."""
    pipes, fittings = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    for (x0, y0), (x1, y1) in segments:
        a = (int(y0 // cell), int(x0 // cell))
        b = (int(y1 // cell), int(x1 // cell))
        if a[0] != b[0] and a[1] != b[1]:
            raise ValueError(f"pipe run {(x0, y0)} -> {(x1, y1)} is not axis-aligned")
        pipes[min(a[0], b[0]) : max(a[0], b[0]) + 1, min(a[1], b[1]) : max(a[1], b[1]) + 1] = True
        fittings[a] = fittings[b] = True
    return pipes, fittings


_TURN_COST = 4.0  # extra cost of a bend, so pipes run straight like real ones


def _route(walls: np.ndarray, sources: set, targets: set) -> list | None:
    """Cheapest 4-connected path over open cells from any source to any target.

    Dijkstra on (cell, heading) states; each step costs 1 and each bend ``_TURN_COST``.
    O(N log N) for N open cells.
    """
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    heap = [(0.0, c, -1) for c in sorted(sources)]
    best, parent = {(c, -1): 0.0 for c in sources}, {}
    while heap:
        cost, cell, h = heapq.heappop(heap)
        if cell in targets:
            path, state = [cell], (cell, h)
            while state in parent:
                state = parent[state]
                path.append(state[0])
            return path[::-1]
        if cost > best[(cell, h)]:
            continue
        for k, (dy, dx) in enumerate(moves):
            n = (cell[0] + dy, cell[1] + dx)
            if not (0 <= n[0] < walls.shape[0] and 0 <= n[1] < walls.shape[1]) or walls[n]:
                continue
            new = cost + 1.0 + (_TURN_COST if h not in (-1, k) else 0.0)
            if new < best.get((n, k), np.inf):
                best[(n, k)], parent[(n, k)] = new, (cell, h)
                heapq.heappush(heap, (new, n, k))
    return None


def _connections(walls: np.ndarray, m: Machine, cell: float) -> set:
    """Open cells sharing an edge with the machine's footprint (where a pipe can attach)."""
    ny, nx = walls.shape
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell)
    inside = (gx >= m.x0) & (gx <= m.x1) & (gy >= m.y0) & (gy <= m.y1)
    grown = inside.copy()
    grown[1:] |= inside[:-1]
    grown[:-1] |= inside[1:]
    grown[:, 1:] |= inside[:, :-1]
    grown[:, :-1] |= inside[:, 1:]
    return {tuple(c) for c in np.argwhere(grown & ~walls)}


def _route_pipes(rng, walls, machines, cell) -> tuple[np.ndarray, np.ndarray] | None:
    """Pipe tree from a supply cell on the outer wall to the gas users (or None).

    Users: every hot machine plus one other machine. Each joins the existing network by
    its cheapest route, nearest user first, so branches share headers like a real
    distribution tree. Fittings: the supply, every machine connection, tees and bends.
    """
    ny, nx = walls.shape
    edge = [
        c for c in map(tuple, np.argwhere(~walls)) if c[0] in (0, ny - 1) or c[1] in (0, nx - 1)
    ]
    supply = edge[rng.integers(len(edge))]
    cold = [m for m in machines if not m.hot]
    users = [m for m in machines if m.hot] + ([cold[rng.integers(len(cold))]] if cold else [])
    pipes, fittings = np.zeros_like(walls), np.zeros_like(walls)
    pipes[supply] = fittings[supply] = True
    targets = [_connections(walls, m, cell) for m in users]
    while targets:
        net = {tuple(c) for c in np.argwhere(pipes)}
        paths = [_route(walls, net, t) for t in targets]
        if any(p is None for p in paths):
            return None
        i = min(range(len(paths)), key=lambda j: len(paths[j]))
        path = paths[i]
        for a, b, c in zip(path, path[1:], path[2:], strict=False):
            if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
                fittings[b] = True  # bend
        fittings[path[0]] = fittings[path[-1]] = True  # tee and machine connection
        for c in path:
            pipes[c] = True
        targets.pop(i)
    return pipes, fittings
