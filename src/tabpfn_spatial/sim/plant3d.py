"""The 3D plant: every factory as a voxel world with heights, 3D pipes and 3D detectors.

Built from any ``Factory`` (rectangular or shaped), so the 2D planner keeps its grid and the
3D models get, from one place:

- **hall:** ``HALL_HEIGHT`` (eaves) over every inside cell; outside cells and interior walls
  are solid to the roof;
- **machines:** axis-aligned boxes ``(x0, y0, 0, x1, y1, h)``, ``h`` by machine kind
  (``MACHINE_HEIGHTS``);
- **pipes in 3D:** the 2D network runs on the overhead rack at ``PIPE_HEIGHT``. Vertical
  sections: a **drop** from the rack to each machine inlet (fittings next to a machine) and
  a **supply riser** from the floor to the rack where the main enters the building (fittings
  on the outline);
- **leak sites in 3D** ``(n, 3)`` with weights: rack cells at ``PIPE_HEIGHT``, and every
  ``LEAK_DZ`` along each vertical section (fittings weigh ``FITTING_WEIGHT`` more, as in 2D);
- **detectors in 3D** ``(k, 3)`` by the **mast rule:**
  1. every 2D detector at the mounting height for the plant's gas (gas-detector practice:
     light gases high, near the rack; heavy gases low; neutral ones at breathing height);
  2. every vertical pipe section of at least ``MAST_MIN_M`` gets a **mast of 3 detectors**
     beside it at the bottom, middle and top of the section (Rayan's height study: masts at
     vertical pipes locate leak height best, 0.26-0.28 m median error, with fewer detectors
     than "3 heights everywhere").

Voxels are cubes of ``VOXEL`` m. Arrays follow ``study3d.physics`` (x, y, z) order; use
``to_zyx`` for (z, y, x) volumes.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from tabpfn_spatial.sim.factory import FITTING_WEIGHT, Factory
from tabpfn_spatial.study3d.geometry import Grid

HALL_HEIGHT = 9.0  # m, eaves
PIPE_HEIGHT = 4.2  # m, overhead gas pipe rack
VOXEL = 1.0  # m
LEAK_DZ = 0.5  # m between leak sites along a vertical pipe
MAST_MIN_M = 1.5  # m: shorter vertical sections get no mast
MACHINE_HEIGHTS = {"tank": 6.5, "furnace": 5.0, "compressor": 2.6, "press": 3.6, "generator": 3.0,
                   "drum": 3.4, "machine": 2.5, "conveyor": 1.1, "rack": 5.5, "robot": 2.4,
                   "cnc": 2.3}  # fmt: skip
_KINDS = (("tank", "tank"), ("furnace", "furnace"), ("kiln", "furnace"), ("boiler", "furnace"),
          ("compressor", "compressor"), ("chiller", "compressor"), ("press", "press"),
          ("generator", "generator"), ("dryer", "drum"), ("mixer", "drum"),
          ("conveyor", "conveyor"), ("rack", "rack"), ("robot", "robot"),
          ("cnc", "cnc"))  # fmt: skip
# detector mounting height by gas density (light / heavy / about air)
MOUNT_Z = {"light": 4.0, "heavy": 0.45, "neutral": 1.6}


def kind_of(name: str) -> str:
    """Machine kind from its name (``machine`` if nothing matches)."""
    low = name.lower()
    return next((k for key, k in _KINDS if key in low), "machine")


def gas_class(gas: str) -> str:
    """``light``, ``heavy`` or ``neutral`` for a gas of ``sim.gases.GASES``."""
    from tabpfn_spatial.sim.gases import GASES

    sg = GASES[gas].sg if gas in GASES else 1.0
    return "light" if sg < 0.9 else "heavy" if sg > 1.1 else "neutral"


@dataclass(frozen=True)
class VerticalPipe:
    """A vertical pipe section at ``xy`` from ``z0`` to ``z1`` (m)."""

    xy: tuple[float, float]
    z0: float
    z1: float
    kind: str  # "drop" (rack -> machine inlet) or "riser" (floor -> rack)


@dataclass(frozen=True)
class Plant3D:
    """A factory with heights. Build with ``Plant3D(factory)``."""

    f: Factory

    # ------------------------------------------------------------------ geometry --
    @cached_property
    def grid(self) -> Grid:
        """1 m voxels over the plant, floor to eaves."""
        return Grid.box((0.0, 0.0, 0.0), (self.f.width, self.f.height, HALL_HEIGHT), VOXEL)

    @cached_property
    def machines(self) -> list[dict]:
        """Machine boxes: name, kind, hot, ``box`` = (x0, y0, z0, x1, y1, z1) in metres."""
        out = []
        for m in self.f.machines:
            k = kind_of(m.name)
            out.append({"name": m.name, "kind": k, "hot": m.hot,
                        "box": (m.x0, m.y0, 0.0, m.x1, m.y1, MACHINE_HEIGHTS[k])})  # fmt: skip
        return out

    @cached_property
    def solid(self) -> np.ndarray:
        """(nx, ny, nz) solid voxels: walls and outside to the roof, machines to their height."""
        g, f = self.grid, self.f
        xs, ys, zs = g.axis(0), g.axis(1), g.axis(2)
        # 2D walls without machines: full height; cell lookup on the factory's grid
        mach2d = np.zeros(f.walls.shape, dtype=bool)
        c = f.centres()
        for m in f.machines:
            mach2d |= ((c[..., 0] >= m.x0) & (c[..., 0] <= m.x1) & (c[..., 1] >= m.y0)
                       & (c[..., 1] <= m.y1))  # fmt: skip
        walls2d = f.walls & ~mach2d
        iy = np.clip((ys // f.cell).astype(int), 0, f.ny - 1)
        ix = np.clip((xs // f.cell).astype(int), 0, f.nx - 1)
        wall_xy = walls2d[np.ix_(iy, ix)].T  # (nx, ny)
        solid = np.repeat(wall_xy[:, :, None], len(zs), axis=2)
        for m in self.machines:
            x0, y0, _, x1, y1, h = m["box"]
            sx = (xs >= x0) & (xs <= x1)
            sy = (ys >= y0) & (ys <= y1)
            solid[np.ix_(sx, sy, zs <= h)] = True
        return solid

    def to_zyx(self, vol_xyz: np.ndarray) -> np.ndarray:
        """(nx, ny, nz[, ...]) -> (nz, ny, nx[, ...]) (Ahmad's HazardVolume order)."""
        return np.moveaxis(vol_xyz, (0, 1, 2), (2, 1, 0))

    # ------------------------------------------------------------------- pipes --
    def _machine_near(self, x: float, y: float) -> dict | None:
        for m in self.machines:
            x0, y0, _, x1, y1, _ = m["box"]
            if x0 - 2.2 < x < x1 + 2.2 and y0 - 2.2 < y < y1 + 2.2:
                return m
        return None

    def _on_outline(self, iy: int, ix: int) -> bool:
        f = self.f
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny_, nx_ = iy + dy, ix + dx
            if not (0 <= ny_ < f.ny and 0 <= nx_ < f.nx):
                return True
            if f.outside is not None and f.outside[ny_, nx_]:
                return True
        return False

    @cached_property
    def vertical_pipes(self) -> list[VerticalPipe]:
        """Drops from the rack to machine inlets and supply risers from the floor."""
        f = self.f
        if f.fittings is None:
            return []
        out = []
        for iy, ix in np.argwhere(f.fittings & ~f.walls):
            x, y = (float(v) for v in f.centre(int(iy), int(ix)))
            m = self._machine_near(x, y)
            if m is not None:
                inlet = min(1.6, 0.6 * m["box"][5])
                out.append(VerticalPipe((x, y), inlet, PIPE_HEIGHT, "drop"))
            elif self._on_outline(int(iy), int(ix)):
                out.append(VerticalPipe((x, y), 0.15, PIPE_HEIGHT, "riser"))
        return out

    @cached_property
    def pipe_segments(self) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Every pipe as 3D segments: rack runs between neighbouring pipe cells + verticals."""
        f = self.f
        segs = []
        if f.pipes is not None:
            for iy, ix in np.argwhere(f.pipes):
                for dy, dx in ((0, 1), (1, 0)):
                    ny_, nx_ = iy + dy, ix + dx
                    if ny_ < f.ny and nx_ < f.nx and f.pipes[ny_, nx_]:
                        a, b = f.centre(int(iy), int(ix)), f.centre(int(ny_), int(nx_))
                        segs.append(((*a, PIPE_HEIGHT), (*b, PIPE_HEIGHT)))
        for v in self.vertical_pipes:
            segs.append(((*v.xy, v.z0), (*v.xy, v.z1)))
        return [(tuple(map(float, a)), tuple(map(float, b))) for a, b in segs]

    @cached_property
    def leak_sites(self) -> tuple[np.ndarray, np.ndarray]:
        """(n, 3) leak positions (m) and (n,) relative weights (fittings weigh more)."""
        f = self.f
        pts, w = [], []
        if f.pipes is not None:
            mask = f.pipes & ~f.walls
            fit = f.fittings if f.fittings is not None else np.zeros_like(mask)
            for iy, ix in np.argwhere(mask):
                x, y = f.centre(int(iy), int(ix))
                pts.append((x, y, PIPE_HEIGHT))
                w.append(FITTING_WEIGHT if fit[iy, ix] else 1.0)
        for v in self.vertical_pipes:
            for z in np.arange(v.z0, v.z1 - 1e-9, LEAK_DZ):
                pts.append((*v.xy, float(z)))
                w.append(FITTING_WEIGHT if abs(z - v.z0) < 1e-9 else 1.0)  # inlet/foot joint
        return np.array(pts, float), np.array(w, float)

    # --------------------------------------------------------------- detectors --
    @cached_property
    def detectors(self) -> tuple[np.ndarray, list[str]]:
        """(k, 3) detector positions and why each is there (``wall`` / ``mast``)."""
        z = MOUNT_Z[gas_class(self.f.gas)]
        pts = [(float(x), float(y), z) for x, y in self.f.sensors]
        why = ["mounted"] * len(pts)
        for v in self.vertical_pipes:
            if v.z1 - v.z0 < MAST_MIN_M:
                continue
            for frac in (0.1, 0.5, 0.9):
                pts.append((v.xy[0], v.xy[1], v.z0 + frac * (v.z1 - v.z0)))
                why.append("mast")
        return np.array(pts, float), why

    def summary(self) -> dict:
        """Counts for a quick check."""
        sites, _ = self.leak_sites
        det, why = self.detectors
        return {"voxels": self.solid.shape, "solid_frac": float(self.solid.mean()),
                "machines": len(self.machines), "vertical_pipes": len(self.vertical_pipes),
                "leak_sites": len(sites), "detectors": len(det),
                "masts": why.count("mast") // 3, "gas": self.f.gas}  # fmt: skip
