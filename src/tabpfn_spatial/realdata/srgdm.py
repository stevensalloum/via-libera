"""SRGDM (Winkler et al., Sens. Actuators B 419 (2024) 136267; github.com/BAMresearch/SRGDM, MIT).

Two parts, both 2D gas maps with values normalised to 0..1:

- **Synthetic** ``30x25/valid.file``: [120, 180, 30, 25] = 30 source positions x 4 measured
  airflow recordings ("seasons"), 180 frames each, on a 30 x 25 cell grid. Plumes are
  simulated from *measured* wind. The sources sit on a regular grid of cells
  (rows 1, 6, ..., 26, columns 1, 6, ..., 21), in the order of ``SOURCE_CELLS``: read off the
  hottest cell of the mean map, identical for all 4 seasons (checked in the tests).
- **Real** ``6x5_realData/*.file``: [30, 3800, 6, 5] per recording day, a 6 x 5 sensor grid.
  The repository does not document what the first axis is, and the maps show no single
  source cell, so we do **not** use them as source labels: only for field statistics
  (intermittency, spatial correlation) against simulation.

The files are PyTorch archives (a zip with a tiny pickle + raw float32 storage). They are
read here with numpy; the unpickler accepts only the tensor-rebuild call and refuses
anything else, so no code from the file can run.
"""

import pickle
import zipfile
from pathlib import Path

import numpy as np

SEASONS = 4
SOURCES = 30


class _TensorOnly(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, *_: (storage, offset, size, stride)
        if (module, name) == ("torch", "FloatStorage"):
            return "<f4"
        if (module, name) == ("collections", "OrderedDict"):
            return dict
        raise pickle.UnpicklingError(f"refused {module}.{name}")

    def persistent_load(self, pid):
        return pid  # ("storage", dtype, key, device, numel)


def load_tensor(path: Path) -> np.ndarray:
    """A float tensor saved with ``torch.save`` as a numpy array (read-only view copied)."""
    with zipfile.ZipFile(path) as z:
        root = z.namelist()[0].split("/")[0]
        (_, dtype, key, _, numel), offset, size, stride = _TensorOnly(
            z.open(f"{root}/data.pkl")
        ).load()
        flat = np.frombuffer(z.read(f"{root}/data/{key}"), dtype, count=numel)
    item = flat.itemsize
    return np.lib.stride_tricks.as_strided(
        flat[offset:], shape=size, strides=[s * item for s in stride]
    ).copy()


# Not row-major: (6, 21), (16, 21) and (26, 21) come last (likely the original file sort).
_LAST = [(6, 21), (16, 21), (26, 21)]
SOURCE_CELLS = [
    (r, c) for r in (1, 6, 11, 16, 21, 26) for c in (1, 6, 11, 16, 21) if (r, c) not in _LAST
] + _LAST


def source_cell(index: int) -> tuple[int, int]:
    """(row, column) cell of the source of synthetic simulation ``index`` (0..119)."""
    return SOURCE_CELLS[index % SOURCES]


def sensor_cells() -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of the 6 x 5 sparse sensors inside the 30 x 25 map (SRGDM mask)."""
    return np.arange(2, 30, 5), np.arange(2, 25, 5)
