"""Features describing a cell relative to a leak and its wind."""

import numpy as np

from tabpfn_spatial.sim import Factory


def cell_features(
    factory: Factory,
    source_xy: np.ndarray,
    wind: np.ndarray,
    rate: float,
    t: float,
    cells: np.ndarray,
) -> np.ndarray:
    """Return (n, 7) features; line-of-sight sampling costs O(n * grid diameter)."""
    cells = np.asarray(cells, dtype=float).reshape(-1, 2)
    source = np.asarray(source_xy, dtype=float)
    wind = np.asarray(wind, dtype=float)
    speed = float(np.linalg.norm(wind))
    direction = wind / speed if speed > 1e-12 else np.array([1.0, 0.0])
    offset = cells - source
    downwind = offset @ direction
    crosswind = offset @ np.array([-direction[1], direction[0]])
    distance = np.linalg.norm(offset, axis=1)
    steps = max(2, int(np.ceil(np.hypot(factory.width, factory.height) / factory.cell)) + 1)
    points = source + np.linspace(0, 1, steps)[None, :, None] * offset[:, None, :]
    ix = np.clip((points[..., 0] / factory.cell).astype(int), 0, factory.nx - 1)
    iy = np.clip((points[..., 1] / factory.cell).astype(int), 0, factory.ny - 1)
    open_fraction = (~factory.walls[iy, ix]).mean(axis=1)
    return np.column_stack(
        [
            downwind,
            crosswind,
            distance,
            np.full(len(cells), t),
            np.full(len(cells), speed),
            np.full(len(cells), rate),
            open_fraction,
        ]
    )
