"""Crowd planners: who goes to which exit, and by which path.

All three share the exit cost fields (``evac.fields``), so planning costs a few reverse
searches whatever the number of people:

- ``nearest``: shortest walk, gas ignored (the conventional evacuation plan).
- ``risk``: cheapest path of length + ``risk_weight`` x danger, each person on their own
  (the one-person planner applied to everyone): it ignores that others use the same door.
- ``ccrp``: capacity-aware risk planning in the spirit of CCRP (Lu, George & Shekhar 2005).
  People are taken earliest-arrival first; each picks the exit with the lowest cost of
  path + risk + the queue it would meet, where an exit door serves ``specific_flow x
  door_m`` people per second and the queue is what the people before them booked. Only
  exit doors are modelled as bottlenecks in planning (the simulator models every cell).

Replanning keeps the current exit unless another is better by more than ``switch_m``
metres-equivalent, so people do not flip back and forth. Cost: O(exits * N log N) for the
fields plus O(people * exits).
"""

import numpy as np

from tabpfn_spatial.evac.crowd import Crowd, CrowdParams
from tabpfn_spatial.evac.fields import exit_fields, follow
from tabpfn_spatial.sim import Factory

PLANNERS = ("nearest", "risk", "ccrp")


def _costs(f: Factory, crowd: Crowd, danger: np.ndarray, name: str, risk_weight: float):
    """(people, exits) path cost and walking length from each person's cell, plus fields."""
    fields = exit_fields(f, danger if name != "nearest" else np.zeros_like(danger),
                         0.0 if name == "nearest" else risk_weight)  # fmt: skip
    iy, ix = crowd.cells[:, 0], crowd.cells[:, 1]
    cost = np.stack([fe.cost[iy, ix] for fe in fields], axis=1)
    length = np.stack([fe.length[iy, ix] for fe in fields], axis=1)
    return cost, length, fields


def _queue_aware(
    f: Factory, crowd: Crowd, cost, length, danger, t, p: CrowdParams, risk_weight: float
) -> np.ndarray:
    """Sequential booking of exit doors, earliest arrivals first (CCRP-style)."""
    arrive = t + length / p.speed  # (n, e) free-walk arrival time
    order = np.argsort(np.nanmin(np.where(np.isfinite(arrive), arrive, np.inf), axis=1))
    service = 1.0 / (p.specific_flow * p.door_m)
    free = np.full(len(f.exits), t)
    exit_danger = np.array([danger[f.cell_of(xy)] for xy in f.exits])
    per_s = p.speed * (1 + risk_weight * exit_danger / f.cell)  # waiting cost per second
    choice = np.full(len(crowd.cells), -1)
    for i in order:
        start = np.maximum(arrive[i], free)
        total = cost[i] + (start - arrive[i]) * per_s
        e = int(np.argmin(total))
        choice[i], free[e] = e, start[e] + service
    return choice


def assign(
    name: str,
    f: Factory,
    crowd: Crowd,
    danger: np.ndarray,
    t: float,
    risk_weight: float = 200.0,
    switch_m: float = 4.0,
) -> None:
    """Give every person still inside a route under planner ``name``."""
    if name not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, not {name!r}")
    cost, length, fields = _costs(f, crowd, danger, name, risk_weight)
    keep = crowd.exit_index >= 0
    cost = cost + switch_m * keep[:, None]
    cost[keep, crowd.exit_index[keep]] -= switch_m  # staying put is free
    if name == "ccrp":
        choice = _queue_aware(f, crowd, cost, length, danger, t, crowd.p, risk_weight)
    else:
        choice = np.argmin(cost if name == "risk" else length, axis=1)
    for i in np.flatnonzero(crowd.active):
        e = int(choice[i])
        if crowd.routes[i] is None or e != crowd.exit_index[i] or name != "nearest":
            crowd.set_route(i, follow(fields[e], tuple(crowd.cells[i])), e)
