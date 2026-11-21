"""Experimental grid-path features; distances are not a gas-flow simulation."""

from collections import deque

import numpy as np

from tabpfn_spatial.sim import Factory


def path_features(factory: Factory, source_xy: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return open-grid path distance and reachability for each cell.

    Paths use four-neighbour movement and count metres. A finite sentinel larger
    than any simple path denotes unreachable or solid targets; the second column
    is 1 for reachable cells and 0 otherwise. O(grid cells + queried cells).
    This is a geometric proxy, not a model of airflow or an evacuation route.
    """
    source_xy = np.asarray(source_xy, dtype=float)
    cells = np.asarray(cells, dtype=float).reshape(-1, 2)
    if source_xy.shape != (2,) or np.any(~np.isfinite(source_xy)):
        raise ValueError("source must be a finite two-dimensional position")
    if np.any(source_xy < 0) or source_xy[0] >= factory.width or source_xy[1] >= factory.height:
        raise ValueError("source must be inside the factory")
    if np.any(~np.isfinite(cells)) or np.any(cells < 0):
        raise ValueError("cells must be finite positions inside the factory")
    if np.any(cells[:, 0] >= factory.width) or np.any(cells[:, 1] >= factory.height):
        raise ValueError("cells must be finite positions inside the factory")
    start = factory.cell_of(source_xy)
    if factory.walls[start]:
        raise ValueError("source must be on open floor")
    sentinel = float(factory.cell * (factory.nx * factory.ny + 1))
    distance = np.full(factory.walls.shape, sentinel)
    distance[start] = 0.0
    queue = deque([start])
    while queue:
        iy, ix = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = iy + dy, ix + dx
            if not (0 <= y < factory.ny and 0 <= x < factory.nx):
                continue
            if factory.walls[y, x] or distance[y, x] != sentinel:
                continue
            distance[y, x] = distance[iy, ix] + factory.cell
            queue.append((y, x))
    ix = (cells[:, 0] / factory.cell).astype(int)
    iy = (cells[:, 1] / factory.cell).astype(int)
    result = distance[iy, ix]
    return np.column_stack([result, (result < sentinel).astype(float)])
