"""Scene description for the 3D world, built only from the planner's ``Factory``.

Whatever the 3D viewer draws as an obstacle must be an obstacle for the planner too, so
everything comes from the factory: the solid grid (exported as is, the viewer uses it
for walking collisions and the "model grid" overlay), machine rectangles (exact
footprints; height and look chosen from the machine's name), wall runs (solid cells that
are not machines), gas pipe runs and fittings, sensors, exits (with the outer wall they
sit in), leak sites, ignition zones.

Coordinates stay in metres, factory frame (x right, y up, origin bottom-left). The viewer
maps (x, y) to Three.js (x - W/2, 0, H/2 - y).
"""

import numpy as np

from tabpfn_spatial.sim import Factory

# name keyword -> 3D model kind; height (m) per kind
_KINDS = (
    ("tank", "tank"),
    ("furnace", "furnace"),
    ("kiln", "furnace"),
    ("boiler", "furnace"),
    ("compressor", "compressor"),
    ("chiller", "compressor"),
    ("press", "press"),
    ("generator", "generator"),
    ("dryer", "drum"),
    ("mixer", "drum"),
)
HEIGHTS = {"tank": 6.5, "furnace": 5.0, "compressor": 2.6, "press": 3.6, "generator": 3.0,
           "drum": 3.4, "machine": 2.5}  # fmt: skip
HALL_HEIGHT = 9.0  # m, eaves of the hall
PIPE_HEIGHT = 4.2  # m, overhead gas pipe rack


def kind_of(name: str) -> str:
    """3D model kind for a machine name (``"machine"`` if nothing matches)."""
    low = name.lower()
    return next((kind for key, kind in _KINDS if key in low), "machine")


def _runs(mask: np.ndarray, cell: float) -> list[dict]:
    """Straight runs covering ``mask`` exactly: horizontal runs, then vertical leftovers.

    A cell that belongs to a horizontal run of length >= 2 is drawn there; isolated cells
    and vertical chains become vertical runs. Endpoints are cell centres in metres.
    """
    ny, nx = mask.shape
    used = np.zeros_like(mask)
    runs = []
    for iy in range(ny):
        ix = 0
        while ix < nx:
            if mask[iy, ix]:
                j = ix
                while j + 1 < nx and mask[iy, j + 1]:
                    j += 1
                if j > ix:
                    runs.append(((iy, ix), (iy, j)))
                    used[iy, ix : j + 1] = True
                ix = j + 1
            else:
                ix += 1
    for ix in range(nx):
        iy = 0
        while iy < ny:
            if mask[iy, ix] and not used[iy, ix]:
                j = iy
                while j + 1 < ny and mask[j + 1, ix] and not used[j + 1, ix]:
                    j += 1
                runs.append(((iy, ix), (j, ix)))
                iy = j + 1
            else:
                iy += 1
    c = lambda i: (i + 0.5) * cell  # noqa: E731
    return [{"from": [c(a[1]), c(a[0])], "to": [c(b[1]), c(b[0])]} for a, b in runs]


def _machine_mask(f: Factory) -> np.ndarray:
    c = f.centres()
    mask = np.zeros(f.walls.shape, dtype=bool)
    for m in f.machines:
        mask |= (
            (c[..., 0] >= m.x0) & (c[..., 0] <= m.x1) & (c[..., 1] >= m.y0) & (c[..., 1] <= m.y1)
        )
    return mask


def wall_runs(f: Factory) -> list[dict]:
    """Inner walls: solid cells that are not machines, as straight runs."""
    return _runs(f.walls & ~_machine_mask(f), f.cell)


def pipe_runs(f: Factory) -> list[dict]:
    """Gas pipes as straight runs between cell centres (empty if the plant has none)."""
    return [] if f.pipes is None else _runs(f.pipes, f.cell)


def _side(f: Factory, xy: np.ndarray) -> str:
    """Outer wall an exit sits in."""
    d = {"west": xy[0], "east": f.width - xy[0], "south": xy[1], "north": f.height - xy[1]}
    return min(d, key=d.get)


def scene(f: Factory) -> dict:
    """Everything the viewer needs, JSON-ready. O(ny * nx)."""
    cells = lambda mask: [f.centre(int(iy), int(ix)).tolist() for iy, ix in np.argwhere(mask)]  # noqa: E731
    return {
        "width": f.width,
        "height": f.height,
        "cell": f.cell,
        "hall_height": HALL_HEIGHT,
        "pipe_height": PIPE_HEIGHT,
        "solid": f.walls.astype(int).tolist(),
        "machines": [
            {
                "name": m.name,
                "kind": kind_of(m.name),
                "rect": [m.x0, m.y0, m.x1, m.y1],
                "hot": m.hot,
                "height": HEIGHTS[kind_of(m.name)],
            }
            for m in f.machines
        ],
        "walls": wall_runs(f),
        "pipes": pipe_runs(f),
        "fittings": [] if f.fittings is None else cells(f.fittings),
        "leak_sites": f.leak_sites.tolist(),
        "ignition": cells(f.ignition),
        "sensors": f.sensors.tolist(),
        "exits": [{"xy": e.tolist(), "side": _side(f, e)} for e in f.exits],
    }
