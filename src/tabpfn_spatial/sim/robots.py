"""AGVs on their lanes, and the emergency stop at the gas alarm.

Each closed lane (``Factory.agv_lanes``) carries ``AGV_PER_LANE`` vehicles spaced evenly,
driving at ``AGV_SPEED`` (typical indoor AGV speeds are 1-2 m/s). At the alarm every AGV
stops where it is, as real AGVs do on an emergency stop, and stays there: its cell becomes
an obstacle for the evacuation planner and the crowd. A stop that would cut part of the
plant off from every exit is not blocked (people squeeze past a stopped vehicle there);
``blocked_factory`` reports which stops were skipped.
"""

from dataclasses import replace

import numpy as np

from tabpfn_spatial.sim.factory import Factory, _all_reach

AGV_SPEED = 1.0  # m/s
AGV_PER_LANE = 2


def _lane_geometry(lane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(np.vstack([lane, lane[:1]]), axis=0), axis=1)
    return seg, np.concatenate([[0.0], np.cumsum(seg)])


def agv_xy(f: Factory, t: np.ndarray | float, t_stop: float | None = None) -> np.ndarray:
    """Positions (n_agv, len(t), 2) in metres; frozen from ``t_stop`` on."""
    times = np.atleast_1d(np.asarray(t, dtype=float))
    if t_stop is not None:
        times = np.minimum(times, t_stop)
    out = []
    for lane in f.agv_lanes:
        seg, cum = _lane_geometry(lane)
        total = cum[-1]
        closed = np.vstack([lane, lane[:1]])
        for k in range(AGV_PER_LANE):
            s = (k * total / AGV_PER_LANE + AGV_SPEED * times) % total
            i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
            frac = (s - cum[i]) / np.where(seg[i] > 0, seg[i], 1.0)
            out.append(closed[i] + frac[:, None] * (closed[i + 1] - closed[i]))
    return np.array(out).reshape(-1, len(times), 2)


def blocked_factory(f: Factory, t_alarm: float | None) -> tuple[Factory, list[dict]]:
    """The plant with the AGVs stopped at the alarm as obstacles, and each AGV's stop.

    Exits stay reachable from every open cell: a stop that would break that is skipped.
    """
    if not f.agv_lanes or t_alarm is None:
        return f, []
    stops = agv_xy(f, t_alarm)[:, 0]
    walls = f.walls.copy()
    exits = [f.cell_of(e) for e in f.exits]
    report = []
    for xy in stops:
        c = f.cell_of(xy)
        blocks = False
        if not walls[c] and c not in exits:
            walls[c] = True
            blocks = _all_reach(walls, exits)
            if not blocks:
                walls[c] = False
        report.append({"xy": xy.tolist(), "cell": [int(c[0]), int(c[1])], "blocks": blocks})
    return replace(f, walls=walls), report
