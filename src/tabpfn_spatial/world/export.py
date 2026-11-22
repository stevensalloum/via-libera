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
    ("conveyor", "conveyor"),
    ("rack", "rack"),
    ("robot", "robot"),
    ("cnc", "cnc"),
)
HEIGHTS = {"tank": 6.5, "furnace": 5.0, "compressor": 2.6, "press": 3.6, "generator": 3.0,
           "drum": 3.4, "machine": 2.5, "conveyor": 1.1, "rack": 5.5, "robot": 2.4,
           "cnc": 2.3}  # fmt: skip
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
    """Inner walls: solid cells that are not machines (nor outside the building), as runs."""
    inner = f.walls & ~_machine_mask(f)
    if f.outside is not None:
        inner &= ~f.outside
    return _runs(inner, f.cell)


def _inside(f: Factory) -> np.ndarray:
    return np.ones(f.walls.shape, dtype=bool) if f.outside is None else ~f.outside


def outline(f: Factory) -> list[dict]:
    """Outer walls of a shaped plant, as straight runs.

    Cell edges between inside and outside, merged along their line; ``normal`` points out
    of the building. O(ny * nx).
    """
    inside = _inside(f)
    pad = np.pad(inside, 1, constant_values=False)
    c = f.cell
    runs = []
    # (dy, dx) of the outside neighbour -> outward normal (x, y)
    for (dy, dx), normal in (((-1, 0), (0, -1)), ((1, 0), (0, 1)), ((0, -1), (-1, 0)),
                             ((0, 1), (1, 0))):  # fmt: skip
        nb = pad[1 + dy : pad.shape[0] - 1 + dy, 1 + dx : pad.shape[1] - 1 + dx]
        edge = inside & ~nb
        if dx == 0:  # horizontal edges: merge along x
            for iy in range(f.ny):
                ix = 0
                while ix < f.nx:
                    if edge[iy, ix]:
                        j = ix
                        while j + 1 < f.nx and edge[iy, j + 1]:
                            j += 1
                        y = (iy + (1 if dy > 0 else 0)) * c
                        runs.append({"from": [ix * c, y], "to": [(j + 1) * c, y], "normal": normal})
                        ix = j + 1
                    else:
                        ix += 1
        else:  # vertical edges: merge along y
            for ix in range(f.nx):
                iy = 0
                while iy < f.ny:
                    if edge[iy, ix]:
                        j = iy
                        while j + 1 < f.ny and edge[j + 1, ix]:
                            j += 1
                        x = (ix + (1 if dx > 0 else 0)) * c
                        runs.append({"from": [x, iy * c], "to": [x, (j + 1) * c], "normal": normal})
                        iy = j + 1
                    else:
                        iy += 1
    return runs


def _exit_info(f: Factory, xy: np.ndarray) -> dict:
    """Exit position, the outer wall it sits in (``side``) and its outward normal."""
    if f.outside is None:
        side = _side(f, xy)
        normal = {"west": (-1, 0), "east": (1, 0), "south": (0, -1), "north": (0, 1)}[side]
        return {"xy": xy.tolist(), "side": side, "normal": normal}
    iy, ix = f.cell_of(xy)
    inside = _inside(f)
    sides = (((0, -1), (-1, 0), "south"), ((0, 1), (1, 0), "north"),
             ((-1, 0), (0, -1), "west"), ((1, 0), (0, 1), "east"))  # fmt: skip
    for normal, (dy, dx), side in sides:
        ny_, nx_ = iy + dy, ix + dx
        if not (0 <= ny_ < f.ny and 0 <= nx_ < f.nx) or not inside[ny_, nx_]:
            return {"xy": xy.tolist(), "side": side, "normal": normal}
    return {"xy": xy.tolist(), "side": _side(f, xy), "normal": (0, -1)}


def leak_heights(f: Factory) -> list[float]:
    """Height (m) of each leak site.

    The overhead rack, or the machine inlet where a fitting next to a machine drops down to
    feed it (the drop the viewer draws).
    """
    out = []
    mask = f.pipes & ~f.walls if f.pipes is not None else None
    if mask is None:
        return [0.0] * len(f.leak_sites)
    fit = f.fittings if f.fittings is not None else np.zeros_like(mask)
    for iy, ix in np.argwhere(mask):
        x, y = f.centre(int(iy), int(ix))
        z = PIPE_HEIGHT
        if fit[iy, ix]:
            for m in f.machines:
                if m.x0 - 2.2 < x < m.x1 + 2.2 and m.y0 - 2.2 < y < m.y1 + 2.2:
                    z = min(1.6, HEIGHTS[kind_of(m.name)] * 0.6)
                    break
        out.append(float(z))
    return out


def pipe_runs(f: Factory) -> list[dict]:
    """Gas pipes as straight runs between cell centres (empty if the plant has none)."""
    return [] if f.pipes is None else _runs(f.pipes, f.cell)


def _side(f: Factory, xy: np.ndarray) -> str:
    """Outer wall an exit sits in."""
    d = {"west": xy[0], "east": f.width - xy[0], "south": xy[1], "north": f.height - xy[1]}
    return min(d, key=d.get)


def scene(f: Factory, people: int = 0, seed: int = 0) -> dict:
    """Everything the viewer needs, JSON-ready. O(ny * nx).

    ``people``: workers placed with ``evac.crowd.sample_workers`` (seeded), the same way
    the crowd simulator places them.
    """
    from tabpfn_spatial.evac.crowd import sample_workers

    crowd = sample_workers(f, people, np.random.default_rng(seed)) if people else np.zeros((0, 2))
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
        "exits": [_exit_info(f, e) for e in f.exits],
        "people": crowd.tolist(),
        "leak_z": leak_heights(f),
        "name": f.name,
        "gas": f.gas,
        "shaped": f.outside is not None,
        "outside": None if f.outside is None else f.outside.astype(int).tolist(),
        "wings": [list(w) for w in f.wings],
        "outline": outline(f) if f.outside is not None else [],
        "agv_lanes": [lane.tolist() for lane in f.agv_lanes],
    }
