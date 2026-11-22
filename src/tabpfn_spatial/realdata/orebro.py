"""Örebro 3DSEN indoor dataset (Burgués et al. 2018/2020; GitHub jburgues/Orebro3DSEN, GPL-3.0).

An office of 5.94 m (x) by 4.56 m (y) with HVAC air supplied at floor level and extracted
at the ceiling. 27 MOX sensors, calibrated to ppm, on 9 poles with 3 heights each:
z = 0.45, 1.20, 2.23 m (from ``wsn_lite.py``). Sampled at 2 Hz.

Assumption: the README only says the poles are "evenly distributed", with grid indices
0-2 in x and y, so we place the pole centres at x = 5.94 (i + 0.5) / 3 and
y = 4.56 (j + 0.5) / 3. The source positions, beaker size, airflow and window come from
the README's Table 1. The chemical is not named in the README.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOM = (5.94, 4.56)
HEIGHTS = (0.45, 1.20, 2.23)


@dataclass(frozen=True)
class Experiment:
    """One row of the README's Table 1."""

    source: tuple[float, float, float]
    beaker: str
    airflow: str
    window: str


EXPERIMENTS = {
    1: Experiment((2.70, 0.50, 0.90), "small", "off", "closed"),
    2: Experiment((2.70, 0.50, 0.90), "large", "off", "closed"),
    3: Experiment((5.95, 1.60, 0.00), "large", "off", "closed"),
    4: Experiment((0.10, 4.50, 2.00), "large", "off", "open"),
    5: Experiment((2.70, 2.70, 0.00), "large + bubbler", "off", "closed"),
    6: Experiment((2.70, 0.50, 0.90), "small", "DC fan", "closed"),
    7: Experiment((2.70, 0.10, 2.20), "small", "DC fan", "closed"),
    8: Experiment((2.70, 0.50, 0.90), "small", "tower fan I", "closed"),
    9: Experiment((2.70, 0.50, 0.90), "large", "tower fan I", "closed"),
    10: Experiment((2.70, 3.15, 1.75), "large", "tower fan III", "closed"),
}


def sensor_xyz(order: list[str] | None = None) -> np.ndarray:
    """(27, 3) sensor positions in metres, in log column order ``C_ijk`` (i: x, j: y, k: z)."""
    order = order or [f"C_{i}{j}{k}" for i in range(3) for j in range(3) for k in range(3)]
    out = []
    for name in order:
        i, j, k = (int(c) for c in name[2:5])
        out.append([ROOM[0] * (i + 0.5) / 3, ROOM[1] * (j + 0.5) / 3, HEIGHTS[k]])
    return np.array(out)


@dataclass(frozen=True)
class Run:
    """One experiment's log."""

    experiment: int
    t: np.ndarray  # (T,) s
    c: np.ndarray  # (T, 27) ppm
    xyz: np.ndarray  # (27, 3) m
    wind_speed: np.ndarray  # (T,) m/s at 1.3 m, room centre
    wind_dir: np.ndarray  # (T,) deg
    info: Experiment

    @property
    def source(self) -> tuple[float, float, float]:
        """Source position (x, y, z) in metres."""
        return self.info.source


def load(folder: Path, experiment: int) -> Run:
    """Read ``Exp{nn}.csv`` (about 13,000 rows of 2 Hz data)."""
    path = Path(folder) / f"Exp{experiment:02d}.csv"
    with path.open(encoding="mac_roman") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip().lstrip("# ").split(" ")[0] for h in rows[0]]
    data = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=float)
    cols = [k for k, h in enumerate(header) if h.startswith("C_")]
    return Run(
        experiment=experiment,
        t=data[:, 0],
        c=data[:, cols],
        xyz=sensor_xyz([header[k] for k in cols]),
        wind_speed=data[:, header.index("Wspd")],
        wind_dir=data[:, header.index("Wdir")],
        info=EXPERIMENTS[experiment],
    )
