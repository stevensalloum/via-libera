"""One crowd evacuation, scored on the true gas.

``prepare`` does everything that does not depend on the planner, once per leak: the true
simulation (1 s steps), the detector alarm, and the danger map from the models at every
replan time (alarm, then every ``update_s``). ``run_crowd`` then evacuates a crowd with
one planner, so planners are compared on exactly the same leak, alarm and maps.

Models: ``perfect`` (true leak + true future: isolates planning) or ``baseline``
(centroid source + forward-simulation hazard: today's realistic models).
"""

from dataclasses import dataclass, field

import numpy as np

from tabpfn_spatial.evac.crowd import DEFAULT_PARAMS, Crowd, CrowdParams
from tabpfn_spatial.evac.crowd_plan import PLANNERS, assign
from tabpfn_spatial.evac.fields import danger_window
from tabpfn_spatial.evac.standins import OracleHazard, OracleLocator
from tabpfn_spatial.hazard import ForwardSimPredictor, hazard_from_fields
from tabpfn_spatial.sim import Factory, Scenario, first_alarm, readings, simulate
from tabpfn_spatial.source import WeightedCentroidLocator

AHEAD = 60.0  # seconds of forecast folded into each danger map


@dataclass(frozen=True)
class Prepared:
    """Planner-independent part of one leak."""

    scenario: Scenario
    times: np.ndarray  # (T,) 1 s steps
    fields: np.ndarray  # (T, ny, nx) true concentration
    harm: np.ndarray  # (T, ny, nx) truly toxic or flammable
    t_alarm: float | None
    replan_times: np.ndarray
    danger: list[np.ndarray]  # model danger map at each replan time


@dataclass
class CrowdResult:
    """Scores, plus snapshots for drawing when asked."""

    metrics: dict[str, float]
    frames: list[dict] = field(default_factory=list)


def prepare(
    f: Factory,
    sc: Scenario,
    models: str,
    t_max: float = 300.0,
    update_s: float = 15.0,
    seed: int = 0,
    sample_every: float = 5.0,
) -> Prepared:
    """Truth, alarm and model danger maps for one leak. Cost: dominated by the models."""
    if models not in ("perfect", "baseline"):
        raise ValueError(f"models must be 'perfect' or 'baseline', not {models!r}")
    times = np.arange(0.0, t_max + 0.5, 1.0)
    fields = simulate(f, [sc], times)[0]
    toxic, fire = hazard_from_fields(f, fields)
    t_alarm = first_alarm(f, times, fields, every=sample_every)
    replan = np.arange(t_alarm, t_max + 1e-9, update_s) if t_alarm is not None else np.array([])
    if models == "perfect":
        loc, pred = OracleLocator(sc, sigma=0.0), OracleHazard(sc)
    else:
        loc, pred = WeightedCentroidLocator(), ForwardSimPredictor()
    rng, danger = np.random.default_rng(seed), []
    for t in replan:
        r = readings(f, sc, float(t), rng, every=sample_every)
        hz = pred.predict(f, r, loc.locate(f, r), t + np.arange(0.0, 120.0 + 1e-9, 10.0))
        danger.append(danger_window(hz, float(t), AHEAD))
    return Prepared(sc, times, fields, toxic | fire, t_alarm, replan, danger)


def _snapshot(t: float, k: int, crowd: Crowd) -> dict:
    return {
        "t": t,
        "k": k,
        "cells": crowd.cells.copy(),
        "active": crowd.active.copy(),
        "exit": crowd.exit_index.copy(),
        "harm_s": crowd.harm_s.copy(),
        "waiting": crowd.active & (crowd.wait <= 1e-9) & (crowd.exit_index >= 0),
    }


def run_crowd(
    f: Factory,
    prep: Prepared,
    positions: np.ndarray,
    planner: str,
    params: CrowdParams = DEFAULT_PARAMS,
    record_every: int | None = None,
) -> CrowdResult:
    """Evacuate ``positions`` with ``planner``; every planner replans at the same times."""
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, not {planner!r}")
    crowd = Crowd(f, positions, params)
    replans = {round(float(t)): d for t, d in zip(prep.replan_times, prep.danger, strict=True)}
    exit_cells = np.array([f.cell_of(xy) for xy in f.exits])
    out = CrowdResult({})
    queue = 0
    for k, t in enumerate(prep.times):
        if round(float(t)) in replans and crowd.active.any():
            assign(planner, f, crowd, replans[round(float(t))], float(t))
        crowd.expose(prep.fields[k], prep.harm[k])
        if record_every and k % record_every == 0:
            out.frames.append(_snapshot(float(t), k, crowd))
        crowd.step(float(t))
        queue = max(queue, int(crowd.occupancy[exit_cells[:, 0], exit_cells[:, 1]].max()))
    out.metrics = _metrics(crowd, prep, queue)
    return out


def _metrics(crowd: Crowd, prep: Prepared, queue: int) -> dict[str, float]:
    n = len(crowd.out_at)
    done = np.sort(crowd.out_at[~np.isnan(crowd.out_at)])
    k95 = int(np.ceil(0.95 * n)) - 1
    return {
        "people": float(n),
        "t_alarm": float(prep.t_alarm) if prep.t_alarm is not None else float("nan"),
        "out_95_s": float(done[k95]) if len(done) > k95 else float("nan"),
        "out_all_s": float(done[-1]) if len(done) == n else float("nan"),
        "still_inside": float(n - len(done)),
        "harm_s_mean": float(crowd.harm_s.mean()),
        "harm_s_p95": float(np.percentile(crowd.harm_s, 95)),
        "exposed_pct": float(100 * (crowd.harm_s > 0).mean()),
        "dose_mean": float(crowd.dose.mean()),
        "max_exit_queue": float(queue),
    }
