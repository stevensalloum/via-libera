"""UCI 251 "Gas sensor arrays in open sampling settings" (Vergara et al. 2013, CC BY 4.0).

Wind tunnel 2.5 m x 1.2 m x 0.4 m, 72 MOX sensors on 9 boards placed across the tunnel
at one of 6 line positions L1 (nearest the gas inlet) to L6. 10 gases, 3 fan settings,
5 heater voltages, 100 Hz for 260 s. No height information.

The zip is 8.4 GB and 107 GB unpacked, so ``convert`` streams it file by file, keeps one
heater voltage (default 5.0 V, the middle setting), resamples to one sample every 5 s
(our detectors' period) and stores the 72 sensor channels plus the flow and climate
columns in one ``.npz`` of a few tens of MB.

Columns of each raw file (92): time (ms), 10 control/climate columns (fan, MFC set and
read values, temperature, humidity), then 9 boards x (a "1" marker + 8 sensors).
"""

import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NAME = re.compile(
    r"WTD_upload/([^/]+)/(L\d)/(\d+)_board_setPoint_(\d+)V_fan_setPoint_(\d+)_mfc_setPoint_"
    r"(\w+?)_(\d+)ppm_p(\d)"
)
SENSOR_COLS = [11 + b * 9 + 1 + s for b in range(9) for s in range(8)]  # skip board markers
FAN_RPM = {"000": 1500, "060": 3900, "100": 5500}  # set-points from the dataset page


@dataclass(frozen=True)
class Meta:
    """What a raw file's name says about its run."""

    folder: str
    gas: str
    ppm: int
    line: int  # 1..6, distance index from the source (1 = nearest)
    heater: int  # board set-point in volts x 100
    fan: str  # "000" | "060" | "100"


def parse_name(name: str) -> Meta | None:
    """Metadata from a zip member name (None if it is not a run file)."""
    m = NAME.match(name)
    if not m:
        return None
    folder, line, _, heater, fan, gas, ppm, _ = m.groups()
    return Meta(folder, gas, int(ppm), int(line[1]), int(heater), fan)


STEP_S = 5  # one sample every 5 s: the reading period of our factory detectors
GRID_S = np.arange(0, 261, STEP_S)  # 53 samples over the 260 s runs
KEEP_EVERY = 25  # raw rows are ~20 ms apart: parse every 25th (~0.5 s), then resample


def _read(args: tuple[str, str]) -> tuple[np.ndarray, np.ndarray] | None:
    """One zip member on the 5 s grid: sensors (53, 72) float16, control (53, 11).

    Rows are about 20 ms apart and runs last 130-260 s; each grid second takes the last
    row at or before it, and seconds after the run's end are NaN. Returns None for a
    broken file (fewer than 10 rows or the wrong number of columns).
    """
    zip_path, name = args
    with zipfile.ZipFile(zip_path) as z, z.open(name) as fh:
        lines = fh.read().splitlines()[::KEEP_EVERY]
    a = np.loadtxt(lines, dtype=np.float32, ndmin=2)
    if a.shape[0] < 10 or a.shape[1] < max(SENSOR_COLS) + 1:
        return None
    t = a[:, 0] / 1000.0
    k = np.clip(np.searchsorted(t, GRID_S, side="right") - 1, 0, len(t) - 1)
    out = a[k]
    out[t[-1] < GRID_S] = np.nan
    return out[:, SENSOR_COLS].astype(np.float16), out[:, :11]


def convert(zip_path: Path, out: Path, heater: int = 500, workers: int = 4,
            limit: int | None = None) -> Path:  # fmt: skip
    """Stream the zip, keep one heater voltage, downsample, write ``out`` (.npz)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [(n, m) for n in z.namelist() if (m := parse_name(n)) and m.heater == heater]
    members = members[:limit] if limit else members
    jobs = [(str(zip_path), n) for n, _ in members]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_read, jobs, chunksize=8))
    skipped = [n for (n, _), r in zip(members, parts, strict=True) if r is None]
    members = [mn for mn, r in zip(members, parts, strict=True) if r is not None]
    parts = [r for r in parts if r is not None]
    sensors = np.stack([s for s, _ in parts])
    meta_cols = np.stack([m for _, m in parts])
    metas = [m for _, m in members]
    np.savez_compressed(
        out,
        sensors=sensors,
        control=meta_cols,
        gas=np.array([m.gas for m in metas]),
        ppm=np.array([m.ppm for m in metas]),
        line=np.array([m.line for m in metas]),
        fan=np.array([m.fan for m in metas]),
        heater=np.array([m.heater for m in metas]),
        names=np.array([n for n, _ in members]),
        skipped=np.array(skipped, dtype=str),
    )
    return out


def load(path: Path) -> dict[str, np.ndarray]:
    """The converted runs: ``sensors`` (runs, 53, 72) every 5 s and their metadata."""
    with np.load(path) as d:
        return {k: d[k] for k in d.files}
