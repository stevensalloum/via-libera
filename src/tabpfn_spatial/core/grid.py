"""Regular 2D grids: a whole map becomes one batched query, and back into an image.

One ``predict`` call over ``grid.points()`` is far cheaper than one call per cell
(each TabPFN cloud call costs at least 10,000 tokens).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """``nx`` x ``ny`` cell centres spanning ``x_range`` x ``y_range`` (inclusive)."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    nx: int
    ny: int

    def __post_init__(self) -> None:
        """Reject empty grids."""
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be >= 1")

    @property
    def size(self) -> int:
        """Number of cells."""
        return self.nx * self.ny

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` for ``matplotlib.imshow(origin="lower")``."""
        return (*self.x_range, *self.y_range)

    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(*self.x_range, self.nx), np.linspace(*self.y_range, self.ny)

    def points(self, extra: Sequence[float] = ()) -> np.ndarray:
        """(size, 2 + len(extra)) rows ``[x, y, *extra]``, row-major in y. O(size)."""
        xs, ys = self._axes()
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        if len(extra):
            pts = np.column_stack([pts, np.broadcast_to(np.asarray(extra), (len(pts), len(extra)))])
        return pts

    def to_image(self, values: np.ndarray) -> np.ndarray:
        """Reshape per-cell values (size,) to an (ny, nx) image."""
        values = np.asarray(values)
        if values.shape != (self.size,):
            raise ValueError(f"expected ({self.size},), got {values.shape}")
        return values.reshape(self.ny, self.nx)

    def cell_index(self, xy: np.ndarray) -> np.ndarray:
        """Flat index of the nearest cell for each (x, y) row. O(len(xy))."""
        xy = np.asarray(xy, dtype=float)

        def nearest(v: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
            if n == 1:
                return np.zeros(len(v), dtype=int)
            return np.clip(np.rint((v - lo) / (hi - lo) * (n - 1)), 0, n - 1).astype(int)

        ix = nearest(xy[:, 0], *self.x_range, self.nx)
        iy = nearest(xy[:, 1], *self.y_range, self.ny)
        return iy * self.nx + ix
