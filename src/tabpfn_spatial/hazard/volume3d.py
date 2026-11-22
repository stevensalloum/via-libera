"""Experimental transient 3D advection-diffusion on an explicitly supplied voxel grid.

Arrays use (z, y, x); coordinates use (x, y, z) in metres. This extends the
existing 2D proof-of-concept transport with vertical diffusion. Concentrations
and source rates remain simulated relative units, not calibrated gas levels.
It does not model buoyancy, HVAC, vertical wind or heat-driven airflow.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.contracts import HazardMap
from tabpfn_spatial.sim.plume import _divergence


@dataclass(frozen=True)
class MachineBox3D:
    """Explicit axis-aligned machine dimensions (metres) and ignition status."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    hot: bool = False

    def __post_init__(self) -> None:
        """Reject missing heights, reversed bounds and nonfinite dimensions."""
        lower, upper = np.asarray(self.lower, float), np.asarray(self.upper, float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower < 0)
            or np.any(upper <= lower)
        ):
            raise ValueError("machine needs finite positive-volume (x, y, z) bounds")


@dataclass(frozen=True)
class VolumeGrid:
    """Explicit solid and ignition voxels; no source or obstacle heights are inferred."""

    solid: np.ndarray
    ignition: np.ndarray
    cell: float
    dz: float

    @classmethod
    def from_layout(
        cls,
        walls: np.ndarray,
        machines: Sequence[MachineBox3D],
        *,
        cell: float,
        dz: float,
        ignition_radius: float,
    ) -> "VolumeGrid":
        """Voxelise supplied 3D walls and boxes; a caller sets the hot-zone radius.

        Machine voxels intersecting a box are solid. A hot machine marks open
        voxel centres within the supplied radius of its box as ignition zones.
        This simplified geometry is not a validated combustion model.
        """
        walls = np.asarray(walls, dtype=bool)
        if not np.isfinite(ignition_radius) or ignition_radius < 0:
            raise ValueError("ignition radius must be finite and nonnegative")
        if any(m.hot for m in machines) and ignition_radius <= 0:
            raise ValueError("hot machines require a positive ignition radius")
        empty = cls(walls, np.zeros_like(walls), cell, dz)
        xyz = empty.centres()
        half = np.array([cell / 2, cell / 2, dz / 2])
        bounds = np.array(walls.shape[::-1]) * np.array([cell, cell, dz])
        solid, ignition = walls.copy(), np.zeros_like(walls)
        for machine in machines:
            low, high = np.asarray(machine.lower), np.asarray(machine.upper)
            if np.any(high > bounds):
                raise ValueError("machine extends outside the volume")
            intersects = ((xyz + half > low) & (xyz - half < high)).all(axis=-1)
            solid |= intersects
            if machine.hot:
                delta = np.maximum(np.maximum(low - xyz, xyz - high), 0.0)
                ignition |= np.linalg.norm(delta, axis=-1) <= ignition_radius
        return cls(solid, ignition & ~solid, cell, dz)

    def __post_init__(self) -> None:
        """Reject incompatible grids and nonphysical cell sizes."""
        solid = np.asarray(self.solid, dtype=bool)
        ignition = np.asarray(self.ignition, dtype=bool)
        if solid.ndim != 3 or not solid.size or ignition.shape != solid.shape:
            raise ValueError("solid and ignition must be matching nonempty (z, y, x) grids")
        if any(n == 0 for n in solid.shape) or solid.all():
            raise ValueError("the volume needs at least one open voxel")
        if not np.isfinite([self.cell, self.dz]).all() or min(self.cell, self.dz) <= 0:
            raise ValueError("cell sizes must be positive and finite")
        if np.any(ignition & solid):
            raise ValueError("ignition zones must be open voxels")
        object.__setattr__(self, "solid", solid.copy())
        object.__setattr__(self, "ignition", ignition.copy())

    def indices(self, xyz: np.ndarray) -> np.ndarray:
        """Map (n, 3) positions to (n, 3) indices in z, y, x order; reject outside."""
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or np.any(~np.isfinite(xyz)):
            raise ValueError("source positions must have finite x, y and z coordinates")
        izyx = np.floor(xyz[:, ::-1] / [self.dz, self.cell, self.cell]).astype(int)
        if np.any(izyx < 0) or np.any(izyx >= np.array(self.solid.shape)):
            raise ValueError("position outside the volume")
        return izyx

    def centres(self) -> np.ndarray:
        """Voxel centres in metres, shape (z, y, x, 3) with x, y, z coordinates."""
        nz, ny, nx = self.solid.shape
        z, y, x = np.meshgrid(
            (np.arange(nz) + 0.5) * self.dz,
            (np.arange(ny) + 0.5) * self.cell,
            (np.arange(nx) + 0.5) * self.cell,
            indexing="ij",
        )
        return np.stack((x, y, z), axis=-1)


def validate_times(times: np.ndarray) -> np.ndarray:
    """Accept strictly increasing nonnegative simulated times."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or np.any(~np.isfinite(times))
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):  # fmt: skip
        raise ValueError("times must be finite, nonnegative and strictly increasing")
    return times


def simulate_volume(
    grid: VolumeGrid,
    source_xyz: np.ndarray,
    rate: float,
    wind: np.ndarray,
    times: np.ndarray,
    diffusivity: float = 0.8,
) -> np.ndarray:
    """Simulate a relative concentration field at each time, (t, z, y, x).

    Source and rate are explicit. Floor, ceiling and solid faces have zero flux;
    lateral boundaries use the 2D simulator's open-boundary rule. Uniform wind
    is horizontal. Time O(steps * nz * ny * nx), memory O(times * volume).
    """
    times = validate_times(times)
    source = np.asarray(source_xyz, dtype=float)
    if source.shape != (3,):
        raise ValueError("an explicit three-dimensional source position is required")
    iz, iy, ix = grid.indices(source[None])[0]
    if grid.solid[iz, iy, ix]:
        raise ValueError("source position is inside a solid voxel")
    wind = np.asarray(wind, dtype=float)
    if wind.shape != (2,) or np.any(~np.isfinite(wind)):
        raise ValueError("wind must be a finite horizontal (u, v) vector")
    if not np.isfinite(rate) or rate < 0 or not np.isfinite(diffusivity) or diffusivity <= 0:
        raise ValueError("rate must be nonnegative and diffusivity must be positive")
    bound = (2 * diffusivity * (2 / grid.cell**2 + 1 / grid.dz**2)
             + np.abs(wind).sum() / grid.cell)  # fmt: skip
    max_step = 0.9 / bound
    open_ = ~grid.solid
    concentration = np.zeros(grid.solid.shape)
    out = np.empty((len(times), *grid.solid.shape))
    current = 0.0
    for k, target in enumerate(times):
        while current < target - 1e-9:
            step = min(max_step, target - current)
            horizontal = np.stack(
                [
                    _divergence(layer[None], mask, wind[None], grid.cell, diffusivity)[0]
                    for layer, mask in zip(concentration, open_, strict=True)
                ]
            )
            vertical = np.zeros_like(concentration)
            flux = (diffusivity / grid.dz**2 * (concentration[1:] - concentration[:-1])
                    * open_[1:] * open_[:-1])  # fmt: skip
            vertical[:-1] += flux
            vertical[1:] -= flux
            concentration += step * (horizontal + vertical)
            concentration[iz, iy, ix] += step * rate
            concentration[grid.solid] = 0.0
            np.maximum(concentration, 0.0, out=concentration)
            current += step
        out[k] = concentration
    return out


@dataclass(frozen=True)
class HazardVolume:
    """Experimental toxic and fire probabilities, with axes (time, z, y, x)."""

    times: np.ndarray
    p_toxic: np.ndarray
    p_fire: np.ndarray

    def __post_init__(self) -> None:
        """Reject malformed or non-probabilistic forecast volumes."""
        times = validate_times(self.times)
        toxic, fire = np.asarray(self.p_toxic, float), np.asarray(self.p_fire, float)
        if toxic.ndim != 4 or fire.shape != toxic.shape or toxic.shape[0] != len(times):
            raise ValueError("hazard volumes must match (time, z, y, x)")
        if (
            any(n == 0 for n in toxic.shape)
            or np.any(~np.isfinite(toxic))
            or np.any(~np.isfinite(fire))
        ):
            raise ValueError("hazard volumes must be nonempty and finite")
        if np.any((toxic < 0) | (toxic > 1)) or np.any((fire < 0) | (fire > 1)):
            raise ValueError("hazard probabilities must be between zero and one")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "p_toxic", toxic.copy())
        object.__setattr__(self, "p_fire", fire.copy())

    def at_height(self, height_index: int) -> HazardMap:
        """Return a selected horizontal slice as the existing 2D hazard contract."""
        if (
            not isinstance(height_index, (int, np.integer))
            or not 0 <= height_index < self.p_toxic.shape[1]
        ):
            raise ValueError("height index outside the volume")
        return HazardMap(self.times, self.p_toxic[:, height_index], self.p_fire[:, height_index])
