"""Voxel grids and STL geometry for the 3D physics ladder.

A ``Grid`` is a box of cubic cells; ``solid`` marks cells filled by buildings or walls. STL
models (binary, metres) are voxelised by casting one vertical ray per column of cells and
counting crossings (parity): a cell is solid if its centre lies between an odd and the next
even crossing. Columns with an odd count (open meshes) are filled from the lowest to the
highest crossing, which is right for buildings standing on the floor.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Cells of size ``cell`` (m) from ``lo`` (x, y, z of the lower corner), ``shape`` cells."""

    lo: tuple[float, float, float]
    cell: float
    shape: tuple[int, int, int]

    def axis(self, k: int) -> np.ndarray:
        """Cell-centre coordinates along axis ``k``."""
        return self.lo[k] + self.cell * (np.arange(self.shape[k]) + 0.5)

    def index(self, xyz: np.ndarray) -> np.ndarray:
        """(n, 3) integer cell indices of points (clipped into the grid)."""
        i = np.floor((np.asarray(xyz) - np.array(self.lo)) / self.cell).astype(int)
        return np.clip(i, 0, np.array(self.shape) - 1)

    @staticmethod
    def box(lo: tuple[float, float, float], hi: tuple[float, float, float], cell: float) -> "Grid":
        """The grid covering ``lo``..``hi`` with cells of ``cell`` metres."""
        n = tuple(round((h - l) / cell) for l, h in zip(lo, hi, strict=True))  # noqa: E741
        return Grid(lo, cell, n)  # type: ignore[arg-type]


def read_stl(path: Path) -> np.ndarray:
    """Triangles of a binary STL file, (n, 3, 3) float64."""
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(b[84 : 84 + n * 50], dtype=rec)["v"].astype(float)


def voxelise(tri: np.ndarray, g: Grid, floor_eps: float = 1e-3) -> np.ndarray:
    """Solid mask (nx, ny, nz) of the closed surfaces in ``tri`` (column ray parity).

    Triangles lying flat on the floor (all z < ``floor_eps``) are dropped: they are the
    ground plate, not an obstacle.
    """
    tri = tri[~(tri[:, :, 2] < floor_eps).all(1)]
    xs, ys, zs = g.axis(0), g.axis(1), g.axis(2)
    hits: dict[tuple[int, int], list[float]] = {}
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    for t in np.flatnonzero(np.abs(det) > 1e-12):  # vertical triangles never hit a vertical ray
        lo, hi = tri[t, :, :2].min(0), tri[t, :, :2].max(0)
        ix = np.flatnonzero((xs >= lo[0]) & (xs <= hi[0]))
        iy = np.flatnonzero((ys >= lo[1]) & (ys <= hi[1]))
        if not ix.size or not iy.size:
            continue
        px, py = np.meshgrid(xs[ix], ys[iy], indexing="ij")
        l1 = ((b[t, 1] - c[t, 1]) * (px - c[t, 0]) + (c[t, 0] - b[t, 0]) * (py - c[t, 1])) / det[t]
        l2 = ((c[t, 1] - a[t, 1]) * (px - c[t, 0]) + (a[t, 0] - c[t, 0]) * (py - c[t, 1])) / det[t]
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        z = l1 * a[t, 2] + l2 * b[t, 2] + l3 * c[t, 2]
        for i, j in zip(*np.nonzero(inside), strict=True):
            hits.setdefault((ix[i], iy[j]), []).append(float(z[i, j]))
    solid = np.zeros(g.shape, bool)
    for (i, j), zz in hits.items():
        zz = np.unique(np.round(zz, 6))
        if zz.size % 2:  # open column: fill from the floor to the highest crossing
            solid[i, j] = zs <= zz.max()
            continue
        for z0, z1 in zip(zz[::2], zz[1::2], strict=True):
            solid[i, j] |= (zs >= z0) & (zs <= z1)
    return solid


def fill_enclosed(solid: np.ndarray) -> np.ndarray:
    """``solid`` with every air pocket apart from the largest one filled (sealed voids)."""
    from scipy import ndimage

    lab, n = ndimage.label(~solid)
    if n <= 1:
        return solid
    keep = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return solid | ((lab > 0) & (lab != keep))
