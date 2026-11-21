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
        ny, nx = int(height / cell), int(width / cell)
        c_x = (np.arange(nx) + 0.5) * cell
        c_y = (np.arange(ny) + 0.5) * cell
        gx, gy = np.meshgrid(c_x, c_y)
        walls = np.zeros((ny, nx), dtype=bool)
        for m in machines:
            walls |= (gx >= m.x0) & (gx <= m.x1) & (gy >= m.y0) & (gy <= m.y1)
        partition = (gx > 30) & (gx < 32)
        doors = ((gy > 6) & (gy < 12)) | ((gy > 26) & (gy < 32))
        walls |= partition & ~doors
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
