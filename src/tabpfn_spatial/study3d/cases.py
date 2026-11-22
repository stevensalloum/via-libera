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


# Örebro 3DSEN: an office, HVAC supply at the floor and extraction at the ceiling.
OREBRO_CEILING = 2.7  # m; ASSUMED (typical office; README gives none; top sensors at 2.23 m)
OREBRO_ACH = 2.0  # air changes per hour; ASSUMED (typical office ventilation, 1.5-3 /h)
OREBRO_EDDY = 0.5  # m; ASSUMED largest eddy in a room (~0.1-0.2 x room size)
OREBRO_GRID = Grid.box((0.0, 0.0, 0.0), (5.94, 4.56, OREBRO_CEILING), 0.09)


def orebro_cases(skip_s: float = 300.0) -> list[Case]:
    """The 10 Örebro experiments: time-mean ppm above each sensor's start baseline.

    Flow: no mean horizontal wind (closed room; the single centre anemometer shows
    recirculating air, direction consistency R 0.1-0.9); turbulence sigma = the
    anemometer's 90th-percentile speed (floored at 0.01 m/s, its resolution); the
    ventilation gives an updraft ACH x H / 3600. The L1 plume, which needs a direction, uses
    the measured mean direction and median speed.
    """
    from tabpfn_spatial.realdata import orebro

    out = []
    solid = np.zeros(OREBRO_GRID.shape, bool)
    for e, info in orebro.EXPERIMENTS.items():
        run = orebro.load(DATA / "orebro", e)
        base = np.nanmedian(run.c[run.t < skip_s], 0)
        obs = np.nanmean(run.c[run.t >= skip_s], 0) - base
        sig = max(float(np.nanpercentile(run.wind_speed, 90)), 0.01)
        rad = np.deg2rad(run.wind_dir)
        ang = np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad)))
        z = np.array([0.0, OREBRO_CEILING])
        flow = ph.Flow(direction=(float(np.cos(ang)), float(np.sin(ang))), z=z,
                       speed=np.full(2, max(float(np.nanmedian(run.wind_speed)), 0.01)),
                       sigma_v=np.full(2, sig), sigma_w=np.full(2, sig),
                       updraft=OREBRO_ACH * OREBRO_CEILING / 3600, l_max=OREBRO_EDDY,
                       open_faces=ph.VENTILATED_ROOM)  # fmt: skip
        out.append(Case("orebro", f"exp{e}", OREBRO_GRID, solid, np.array(info.source, float),
                        flow, run.xyz, obs,
                        {"experiment": e, "airflow": info.airflow, "beaker": info.beaker,
                         "setup": f"exp{e}"}))  # fmt: skip
    return out


# UCI 251: 2.5 x 1.2 x 0.4 m tunnel, fan-driven flow along x.
UCI_SPEED = {"000": 0.10, "060": 0.21, "100": 0.34}  # m/s on the tunnel axis (published)
UCI_LINES = (0.25, 0.50, 0.98, 1.18, 1.40, 1.45)  # m from the inlet; UNVERIFIED (recalled)
UCI_SOURCE = (0.0, 0.6, 0.2)  # gas outlet; ASSUMED at the inlet, centre of the section
UCI_SENSOR_Z = 0.2  # m; ASSUMED mid-height
UCI_TI = 0.10  # turbulence intensity; ASSUMED (no measurement published)
UCI_GRID = Grid.box((0.0, 0.0, 0.0), (2.5, 1.2, 0.4), 0.025)


def uci_cases() -> list[Case]:
    """One case per fan speed: normalised response of the 9 boards at the 6 lines.

    Per gas and fan, the median response (R/R0 - 1 during release) per line and board over
    repetitions, divided by its maximum; then the median over the 10 gases. So the
    observation is the *shape* of the plume (gas sensitivity and dose divided out).
    """
    from tabpfn_spatial.realdata import uci
    from tabpfn_spatial.study3d.overview import uci_response

    d = uci.load(DATA / "uci/uci251_500V_5s.npz")
    resp = uci_response(d).reshape(-1, 9, 8).mean(2)  # (runs, boards): mean of 8 sensors
    ys = 1.2 * (np.arange(9) + 0.5) / 9
    xyz = np.array([(x, y, UCI_SENSOR_Z) for x in UCI_LINES for y in ys])
    solid = np.zeros(UCI_GRID.shape, bool)
    out = []
    for fan, u in UCI_SPEED.items():
        shapes = []
        for gas in np.unique(d["gas"]):
            m = (d["gas"] == gas) & (d["fan"] == fan)
            grid = np.array([[np.nanmedian(resp[m & (d["line"] == L), b]) for b in range(9)]
                             for L in range(1, 7)])  # fmt: skip
            if np.isfinite(grid).all() and np.nanmax(grid) > 0:
                shapes.append(grid / np.nanmax(grid))
        obs = np.nanmedian(shapes, 0).ravel()
        z = np.array([0.0, 0.4])
        flow = ph.Flow(direction=(1.0, 0.0), z=z, speed=np.full(2, u),
                       sigma_v=np.full(2, UCI_TI * u), sigma_w=np.full(2, UCI_TI * u),
                       l_max=0.2, open_faces=ph.CLOSED_DUCT)  # fmt: skip
        out.append(Case("uci", f"fan{fan}", UCI_GRID, solid, np.array(UCI_SOURCE), flow, xyz,
                        obs, {"fan": fan, "speed": u, "setup": f"fan{fan}",
                              "gases": len(shapes)}))  # fmt: skip
    return out
