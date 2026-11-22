"""Red:Vapor wind-tunnel dataset (Hinsen et al., Sci Data 2026; Zenodo 16414473, CC BY 4.0).

A 1:50 industrial site on a turntable. Frame: origin at the turntable centre on the floor,
+x upwind (the wind blows towards -x), z up, metres. A fog plume is released through a
hole in a tank model at z = 0.18 m.

The leak position is not in the paper's text; we take it from the setup 3D models
(``Setups/Setup?/setup?.stl``): the hole in setup A's tank top is at (0.783, 0.017) m;
setups C and D are the same model turned by -22.5 and +22.5 degrees (the hole found in
their models matches this rotation within 0.5 degrees and 1 cm).

Concentration: the PID "ppm" is a relative, uncalibrated, concentration-proportional
signal (assumed 55 mV/ppm sensitivity); use it relative within an experiment.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOLE_A = np.array([0.783, 0.017])
RELEASE_Z = 0.18
TURNTABLE_DEG = {"A": 0.0, "B": 0.0, "C": -22.5, "D": 22.5}


def source_xyz(setup: str) -> np.ndarray:
    """Leak position (x, y, z) in metres for setup A-D."""
    a = np.radians(TURNTABLE_DEG[setup])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.array([*(rot @ HOLE_A), RELEASE_Z])


@dataclass(frozen=True)
class VoxelRun:
    """One rasterised sampling experiment: time-averaged values per voxel."""

    experiment: int
    setup: str
    wind_speed: float  # mean tunnel wind, m/s
    xyz: np.ndarray  # (n, 3) voxel centres, m
    pid: np.ndarray  # (n,) relative PID ppm (reference instrument)
    mos: np.ndarray  # (n,) MiCS-5524 ppm (low-cost MOX, datasheet curve)
    wind: np.ndarray  # (n, 3) measured u, v, w, m/s
    dwell_s: float  # seconds sampled per voxel

    @property
    def source(self) -> np.ndarray:
        """Leak position for this run's setup."""
        return source_xyz(self.setup)


def load_voxel_maps(path: Path) -> list[VoxelRun]:
    """The 8 sampling experiments from ``voxel_maps.pkl`` (9 MB)."""
    runs = []
    for item in pickle.loads(Path(path).read_bytes()):
        info, v = item["experiment_info"], item["voxel_map"]
        runs.append(
            VoxelRun(
                experiment=int(item["path"].rsplit("_", 1)[1]),
                setup=info["setup"],
                wind_speed=float(info["mean wind speed [m/s]"]),
                xyz=v[["x", "y", "z"]].to_numpy(float),
                pid=v[("PID-sensor", "ppm")].to_numpy(float),
                mos=v[("MiCS5524", "ppm")].to_numpy(float),
                wind=v[["wind-u", "wind-v", "wind-w"]].to_numpy(float),
                dwell_s=float(info["sampling time [s]"]),
            )
        )
    return runs
