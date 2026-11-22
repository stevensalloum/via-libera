"""Real experiments as simulation cases: geometry, leak, flow (measured) and observations.

Red:Vapor (8 sampling experiments):
- geometry: the setup STL, voxelised at 4 cm;
- leak: the hole of the tank model (realdata.redvapor.source_xyz);
- flow: wind blows towards -x; the speed, sigma_v and sigma_w profiles are the medians per
  height of the voxels measured **upwind of all buildings** (x > 0.95 m), from the 10 Hz
  anemometer (realdata.redvapor.extract_stats). Note: each voxel is a 5 s dwell, so
  meandering slower than ~5 s is not in these sigmas;
- observation: PID excess over the time-local tunnel background (realdata.redvapor.excess).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tabpfn_spatial.realdata import redvapor
from tabpfn_spatial.study3d import physics as ph
from tabpfn_spatial.study3d.geometry import Grid, fill_enclosed, read_stl, voxelise

DATA = Path("data/real")
RV_GRID = Grid.box((-1.2, -1.2, 0.0), (1.2, 1.2, 1.4), 0.04)
UPWIND_X = 0.95  # all Red:Vapor buildings end below this x


@dataclass(frozen=True)
class Case:
    """One real experiment, ready for every physics level and for TabPFN."""

    dataset: str
    name: str
    grid: Grid
    solid: np.ndarray
    source: np.ndarray
    flow: ph.Flow
    xyz: np.ndarray  # observation points (n, 3)
    obs: np.ndarray  # observed concentration (dataset units), (n,)
    info: dict = field(default_factory=dict)


def _profile(xyz: np.ndarray, s: dict[str, np.ndarray], mask: np.ndarray) -> ph.Flow:
    z = np.round(xyz[mask, 2], 3)
    zs = np.unique(z)

    def med(col: str, sign: float = 1.0) -> np.ndarray:
        return np.array([np.nanmedian(sign * s[col][mask][z == v]) for v in zs])

    return ph.Flow(direction=(-1.0, 0.0), z=zs, speed=med("u", -1.0), sigma_v=med("sv"),
                   sigma_w=med("sw"), open_faces=ph.OPEN_TUNNEL)  # fmt: skip


def redvapor_cases() -> list[Case]:
    """The 8 Red:Vapor sampling experiments."""
    runs = redvapor.load_voxel_maps(DATA / "redvapor/voxel_maps.pkl")
    stats = DATA / "redvapor/voxel_stats.npz"
    solids: dict[str, np.ndarray] = {}
    out = []
    for r in runs:
        if r.setup not in solids:
            stl = DATA / f"redvapor/Setups/Setup{r.setup}/setup{r.setup}.stl"
            solids[r.setup] = fill_enclosed(voxelise(read_stl(stl), RV_GRID))
        xyz, ex, s = redvapor.excess(stats, r.experiment)
        flow = _profile(xyz, s, xyz[:, 0] > UPWIND_X)
        out.append(Case("redvapor", f"exp{r.experiment}", RV_GRID, solids[r.setup], r.source,
                        flow, xyz, ex, {"setup": r.setup, "wind_speed": r.wind_speed,
                                         "experiment": r.experiment}))  # fmt: skip
    return out
