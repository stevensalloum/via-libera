"""Factory layout: a 2D grid of open floor, walls and machines, with exits and gas sensors.

Coordinates are metres, origin bottom-left; grid cell ``(iy, ix)`` covers
``[ix * cell, (ix + 1) * cell) x [iy * cell, (iy + 1) * cell)``.
"""

from dataclasses import dataclass

import numpy as np


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

    @property
    def leak_sites(self) -> np.ndarray:
        """(n, 2) open cell centres within 1 cell of any machine: where leaks occur."""
        c = self.centres()
        near = np.zeros(self.walls.shape, dtype=bool)
        for m in self.machines:
            near |= (
                (c[..., 0] >= m.x0 - self.cell)
                & (c[..., 0] <= m.x1 + self.cell)
                & (c[..., 1] >= m.y0 - self.cell)
                & (c[..., 1] <= m.y1 + self.cell)
            )
        return c[near & ~self.walls]

    @classmethod
    def default(cls) -> "Factory":
        """60 m x 40 m hall, 2 m cells: two production halls split by a wall with doors."""
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
        return cls(width, height, cell, walls, machines, exits, sensors)

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
        """Random plant: machines, a dividing wall with two doors, 4 exits, 16 sensors.

        Retries until every open cell can reach an exit. Reproducible for a given ``rng``.
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
            return cls(width, height, cell, walls, machines, exits, sensors)
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
