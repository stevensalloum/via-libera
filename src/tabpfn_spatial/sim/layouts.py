"""Shaped plants: L, U and T halls, and a campus of two halls joined by a corridor.

A shaped plant is still one ``Factory`` grid, so the gas simulation, the models, the
planner and the 3D viewer keep sharing every obstacle:

- **wings:** the building is a union of rectangles; cells outside it are ``outside`` (and
  solid). The viewer draws the outer walls along the outline.
- **interior walls with doors:** fire walls between wings, office and control rooms along a
  corridor (each with a door).
- **machines:** only in production areas, 4 m apart and 4 m from the outline, so aisles stay
  open; more kinds than the rectangular plant (conveyors, storage racks, robot cells).
- **exits:** spread along the outline (one per ~``exit_spacing`` m of wall).
- **pipes:** the same tree router as ``Factory.random`` (supply on the outline).
- **AGV lanes:** closed loops through the aisles of the production wings
  (``sim.robots`` moves AGVs on them; at the alarm they stop and block their cell).

Proportions are jittered by the seed, so ``shaped(rng, kind)`` draws new, unseen shapes;
``template(name)`` fixes a seed for setups that are rerun and compared.
"""

from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.sim.factory import (
    Factory,
    Machine,
    _all_reach,
    _route,
    _route_pipes,
)

KINDS = ("L", "U", "T", "campus")
# (name, width range, depth range, hot?) for machines of a shaped plant
_CATALOGUE = (
    ("tank farm", (6, 10), (6, 10), False),
    ("furnace", (6, 10), (5, 8), True),
    ("compressor", (4, 7), (4, 6), False),
    ("press line", (10, 16), (3, 5), False),
    ("generator", (5, 8), (4, 6), True),
    ("dryer", (5, 8), (4, 7), True),
    ("conveyor", (12, 20), (2, 2.5), False),
    ("storage rack", (10, 16), (2, 2.5), False),
    ("robot cell", (6, 8), (6, 8), False),
    ("cnc machine", (4, 6), (3, 5), False),
    ("boiler", (4, 6), (4, 6), True),
    ("mixer", (4, 6), (4, 6), False),
)
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Plan:
    """Building outline and interior walls before machines are placed."""

    wings: tuple[Rect, ...]
    production: tuple[Rect, ...]  # where machines may stand
    walls: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, ...]], ...]
    # each wall: ((x0, y0), (x1, y1), door centres along it)


def _j(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.uniform(a, b))


def plan(kind: str, w: float, h: float, rng: np.random.Generator) -> Plan:
    """Wings, production areas and interior walls of a ``kind`` building in ``w`` x ``h`` m."""
    snap = lambda v: 2.0 * round(v / 2.0)  # noqa: E731 - walls on the 2 m grid
    if kind == "L":
        yb, xa = snap(h * _j(rng, 0.42, 0.55)), snap(w * _j(rng, 0.38, 0.5))
        wings = ((0, 0, w, yb), (0, 0, xa, h))
        prod = wings
        walls = (((xa, 0), (xa, yb), (yb * 0.3, yb * 0.75)),)  # fire wall, two doors
    elif kind == "U":
        yb = snap(h * _j(rng, 0.32, 0.42))
        xa, xb = snap(w * _j(rng, 0.28, 0.36)), snap(w * _j(rng, 0.64, 0.72))
        wings = ((0, 0, w, yb), (0, 0, xa, h), (xb, 0, w, h))
        prod = wings
        walls = (((0, yb), (xa, yb), (xa * 0.5,)), ((xb, yb), (w, yb), ((xb + w) * 0.5,)))
    elif kind == "T":
        yt = snap(h * _j(rng, 0.5, 0.62))
        xa, xb = snap(w * _j(rng, 0.3, 0.38)), snap(w * _j(rng, 0.62, 0.7))
        wings = ((0, yt, w, h), (xa, 0, xb, h))
        prod = wings
        walls = (((xa, yt), (xb, yt), ((xa + xb) * 0.5 - 6, (xa + xb) * 0.5 + 6)),)
    elif kind == "campus":
        xa, xb = snap(w * _j(rng, 0.36, 0.42)), snap(w * _j(rng, 0.58, 0.64))
        yc = snap(h * _j(rng, 0.4, 0.5))
        office = (xa, yc + 4, xb, yc + 16)  # rooms north of the corridor
        corridor = (xa, yc, xb, yc + 4)
        wings = ((0, 0, xa, h), (xb, snap(h * 0.15), w, h), corridor, office)
        prod = ((0, 0, xa, h), (xb, snap(h * 0.15), w, h))
        mid = snap((xa + xb) / 2)
        walls = (
            ((xa, yc + 4), (xb, yc + 4), ((xa + mid) / 2, (mid + xb) / 2)),  # rooms | corridor
            ((mid, yc + 4), (mid, yc + 16), ()),  # between the two rooms
            ((xa, 0), (xa, h), (yc + 2,)),  # west hall | corridor (door = corridor end)
            ((xb, snap(h * 0.15)), (xb, h), (yc + 2,)),  # corridor | east hall
        )
    else:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    return Plan(tuple(map(tuple, wings)), tuple(map(tuple, prod)), walls)  # type: ignore[arg-type]


def _rect_mask(rects, gx, gy) -> np.ndarray:
    m = np.zeros(gx.shape, dtype=bool)
    for x0, y0, x1, y1 in rects:
        m |= (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1)
    return m


def _wall_mask(p: Plan, gx, gy, cell: float) -> np.ndarray:
    """Interior wall cells (one cell thick) minus 2-cell door gaps."""
    m = np.zeros(gx.shape, dtype=bool)
    for (x0, y0), (x1, y1), doors in p.walls:
        if x0 == x1:  # vertical wall on the cell column just east of x0
            on = (gx > x0) & (gx < x0 + cell) & (gy > min(y0, y1)) & (gy < max(y0, y1))
            gap = np.zeros_like(on)
            for d in doors:
                gap |= np.abs(gy - d) < cell
        else:
            on = (gy > y0) & (gy < y0 + cell) & (gx > min(x0, x1)) & (gx < max(x0, x1))
            gap = np.zeros_like(on)
            for d in doors:
                gap |= np.abs(gx - d) < cell
        m |= on & ~gap
    return m


def _machines(rng, prod, free, gx, gy, cell, n, n_hot) -> tuple[Machine, ...]:
    """Machines from the catalogue inside production areas, on free cells, 4 m apart."""
    placed: list[Machine] = []
    taken = np.zeros(free.shape, dtype=bool)
    order = list(rng.permutation(len(_CATALOGUE)))
    while len(order) < n:
        order += list(rng.permutation(len(_CATALOGUE)))
    hot_left = n_hot
    for k in order[:n]:
        name, wr, dr, hot = _CATALOGUE[k]
        for _ in range(200):
            w, d = _j(rng, *wr), _j(rng, *dr)
            if rng.random() < 0.5:
                w, d = d, w
            x0, y0, x1, y1 = prod[rng.integers(len(prod))]
            if x1 - x0 < w + 8 or y1 - y0 < d + 8:
                continue
            mx, my = _j(rng, x0 + 4, x1 - 4 - w), _j(rng, y0 + 4, y1 - 4 - d)
            body = (gx >= mx) & (gx <= mx + w) & (gy >= my) & (gy <= my + d)
            halo = (gx >= mx - 4) & (gx <= mx + w + 4) & (gy >= my - 4) & (gy <= my + d + 4)
            if not body.any() or not free[halo].all() or taken[halo].any():
                continue
            is_hot = hot and hot_left > 0
            hot_left -= is_hot
            placed.append(Machine(name, mx, my, mx + w, my + d, hot=bool(is_hot)))
            taken |= body
            break
    return tuple(placed)


def _outline_cells(inside: np.ndarray) -> np.ndarray:
    """Inside cells touching the outside or the grid edge: (n, 2) (iy, ix)."""
    pad = np.pad(inside, 1, constant_values=False)
    nb_out = (~pad[:-2, 1:-1]) | (~pad[2:, 1:-1]) | (~pad[1:-1, :-2]) | (~pad[1:-1, 2:])
    return np.argwhere(inside & nb_out)


def _exits(rng, walls, inside, cell, spacing) -> np.ndarray | None:
    """Exits on open outline cells, greedy-spread so neighbours are ~``spacing`` m apart."""
    cand = [c for c in _outline_cells(inside) if not walls[tuple(c)]]
    if not cand:
        return None
    cand = np.array(cand)
    n = max(3, round(len(cand) * cell / spacing))
    pick = [int(rng.integers(len(cand)))]
    d = np.full(len(cand), np.inf)
    for _ in range(n - 1):
        d = np.minimum(d, np.abs(cand - cand[pick[-1]]).sum(1))
        pick.append(int(np.argmax(d)))
    return (cand[pick][:, ::-1] + 0.5) * cell


def _sensors(rng, walls, cell, w, h, spacing) -> np.ndarray:
    open_yx = np.argwhere(~walls)
    centres = (open_yx[:, ::-1] + 0.5) * cell
    out = []
    for gy in np.arange(spacing / 2, h, spacing):
        for gx in np.arange(spacing / 2, w, spacing):
            target = np.array([gx, gy]) + rng.uniform(-2, 2, size=2)
            dist = ((centres - target) ** 2).sum(1)
            k = int(np.argmin(dist))
            if dist[k] < (spacing / 2) ** 2:
                out.append(centres[k])
    return np.unique(np.array(out), axis=0)


def _agv_loop(rng, walls, prod_rect, cell) -> np.ndarray | None:
    """A closed AGV loop through 4 waypoints near the corners of a production area."""
    x0, y0, x1, y1 = prod_rect
    ny, nx = walls.shape
    pts = []
    for fx, fy in ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)):
        tx, ty = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        best = None
        for _ in range(60):  # nearest open cell to a jittered corner
            px, py = tx + rng.uniform(-4, 4), ty + rng.uniform(-4, 4)
            c = (int(np.clip(py // cell, 0, ny - 1)), int(np.clip(px // cell, 0, nx - 1)))
            if not walls[c]:
                best = c
                break
        if best is None:
            return None
        pts.append(best)
    loop: list = []
    for a, b in zip(pts, pts[1:] + pts[:1], strict=True):
        path = _route(walls, {a}, {b})
        if path is None:
            return None
        loop += path[:-1]
    if len(loop) < 8:
        return None
    return (np.array(loop)[:, ::-1] + 0.5) * cell


def shaped(
    rng: np.random.Generator,
    kind: str,
    width: float = 90.0,
    height: float = 70.0,
    cell: float = 2.0,
    n_machines: int = 12,
    n_hot: int = 4,
    exit_spacing: float = 45.0,
    sensor_spacing: float = 12.0,
    agv_loops: int = 2,
    gas: str = "methane",
    name: str = "",
) -> Factory:
    """A random ``kind`` plant (L, U, T or campus); retries until it is fully evacuable."""
    ny, nx = int(height / cell), int(width / cell)
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell)
    for _ in range(100):
        p = plan(kind, width, height, rng)
        inside = _rect_mask(p.wings, gx, gy)
        walls = ~inside | _wall_mask(p, gx, gy, cell)
        prod = _rect_mask(p.production, gx, gy) & ~walls
        machines = _machines(rng, p.production, prod, gx, gy, cell, n_machines, n_hot)
        if len(machines) < max(4, n_machines // 2):
            continue
        for m in machines:
            walls = walls | ((gx >= m.x0) & (gx <= m.x1) & (gy >= m.y0) & (gy <= m.y1))
        exits = _exits(rng, walls, inside, cell, exit_spacing)
        if exits is None:
            continue
        starts = [(int(e[1] // cell), int(e[0] // cell)) for e in exits]
        if not _all_reach(walls, starts):
            continue
        network = _route_pipes(rng, walls, machines, cell)
        if network is None:
            continue
        loops = []
        for k in range(agv_loops):
            loop = _agv_loop(rng, walls, p.production[k % len(p.production)], cell)
            if loop is not None:
                loops.append(loop)
        sensors = _sensors(rng, walls, cell, width, height, sensor_spacing)
        return Factory(width, height, cell, walls, machines, exits, sensors, *network,
                       outside=~inside, wings=p.wings, agv_lanes=tuple(loops), gas=gas,
                       name=name or f"{kind}-plant")  # fmt: skip
    raise RuntimeError(f"could not build a {kind} plant in {width} x {height} m")


# Fixed setups (rerun and compared); other seeds give unseen shapes for testing.
TEMPLATES = {
    "L-plant": {"kind": "L", "width": 90.0, "height": 70.0, "seed": 1, "gas": "methane"},
    "U-plant": {"kind": "U", "width": 100.0, "height": 70.0, "seed": 2, "gas": "propane"},
    "T-plant": {"kind": "T", "width": 90.0, "height": 80.0, "seed": 3, "gas": "hydrogen"},
    "campus": {"kind": "campus", "width": 120.0, "height": 70.0, "seed": 4, "gas": "methane"},
}


def template(name: str) -> Factory:
    """One of ``TEMPLATES`` (same plant every time)."""
    t = dict(TEMPLATES[name])
    rng = np.random.default_rng(t.pop("seed"))
    return shaped(rng, t.pop("kind"), name=name, **t)
