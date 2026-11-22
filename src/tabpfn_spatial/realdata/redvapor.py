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


STAT_COLS = ["u", "v", "w", "su", "sv", "sw", "pid", "pid_sd", "pid_p90", "n", "t"]


def extract_stats(pkl: Path, out: Path) -> Path:
    """Per-voxel turbulence and intermittency from the 10 Hz series (experiments.pkl, 1.3 GB).

    For each sampling experiment and each of the probe's 4 layers, rows are grouped by the
    gantry position (NaN while moving). Per voxel: mean and standard deviation of the 3D
    wind (u, v, w, su, sv, sw in m/s), PID ppm mean / standard deviation / 90th percentile,
    the number of 10 Hz samples and the mean time t (s from the run start: the raster is
    sequential and the tunnel background grows, so each voxel needs its own background).
    Also the upstream PID at 1 Hz (background build-up). Writes one small ``.npz``; the
    big pickle can then be deleted.
    """
    import pickle

    import pandas as pd

    with open(pkl, "rb") as fh:
        data = pickle.load(fh)["Sampling_Experiments"]
    arrays: dict[str, np.ndarray] = {}
    for k, e in enumerate(data, start=1):
        rows = []
        t0 = e["upstream"][0].index[0]
        for df in e["layers"]:
            d = pd.DataFrame({
                "t": (df.index - t0).total_seconds(),
                "x": df[("x", "")], "y": df[("y", "")], "z": df[("z", "")],
                "u": df[("wind-u", "")], "v": df[("wind-v", "")], "w": df[("wind-w", "")],
                "pid": df[("PID-sensor", "ppm")],
            }).dropna(subset=["x", "y", "z"])  # fmt: skip
            g = d.groupby(["x", "y", "z"])
            s = pd.concat([g[["u", "v", "w"]].mean(),
                           g[["u", "v", "w"]].std().add_prefix("s"),
                           g["pid"].mean(), g["pid"].std().rename("pid_sd"),
                           g["pid"].quantile(0.9).rename("pid_p90"),
                           g["pid"].size().rename("n"), g["t"].mean()],
                          axis=1).reset_index()  # fmt: skip
            rows.append(s)
        s = pd.concat(rows, ignore_index=True)
        arrays[f"e{k}_xyz"] = s[["x", "y", "z"]].to_numpy(np.float32)
        arrays[f"e{k}_stats"] = s[STAT_COLS].to_numpy(np.float32)
        up = e["upstream"][0]
        pid = up[("PID-sensor", "ppm")]
        sec = (pid.index - pid.index[0]).total_seconds().astype(int)
        arrays[f"e{k}_upstream_pid_1hz"] = pid.groupby(sec).mean().to_numpy(np.float32)
    np.savez_compressed(out, **arrays)
    return out


def load_stats(path: Path, experiment: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(xyz (n, 3), {column: (n,)}) of ``extract_stats`` for one sampling experiment."""
    with np.load(path) as d:
        xyz, st = d[f"e{experiment}_xyz"], d[f"e{experiment}_stats"]
        cols = {c: st[:, i] for i, c in enumerate(STAT_COLS)}
        cols["upstream_pid_1hz"] = d[f"e{experiment}_upstream_pid_1hz"]
    return xyz, cols


def background(t: np.ndarray, pid: np.ndarray, half_window: float = 450.0,
               q: float = 10.0) -> np.ndarray:  # fmt: skip
    """Tunnel background at each voxel's time.

    It is the ``q``-th percentile of all PID readings within +-``half_window`` s.

    The closed tunnel fills with fog over a run (upstream sensor: ~0 to 1-2 relative ppm), so
    a fixed background is wrong. The raster sweeps mostly plume-free air, so the lower
    envelope of what was measured around time t is the background at t. Checked against the
    upstream sensor cross-calibrated on upwind voxels where that fit is good (exp 4-7).
    """
    order = np.argsort(t)
    ts, ps = t[order], pid[order]
    lo = np.searchsorted(ts, t - half_window)
    hi = np.searchsorted(ts, t + half_window, side="right")
    return np.array([np.percentile(ps[a:b], q) for a, b in zip(lo, hi, strict=True)])


def excess(path: Path, experiment: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """(xyz, PID excess above the time-local background, stats) for one experiment."""
    xyz, s = load_stats(path, experiment)
    return xyz, s["pid"] - background(s["t"], s["pid"]), s
