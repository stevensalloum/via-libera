"""One crowd evacuation, scored on the true gas.

``prepare`` does everything that does not depend on the planner, once per leak: the true
simulation (1 s steps), the detector alarm, and the danger map from the models at every
replan time (alarm, then every ``update_s``). ``run_crowd`` then evacuates a crowd with
one planner, so planners are compared on exactly the same leak, alarm and maps.

Models: ``perfect`` (true leak + true future: isolates planning), ``baseline`` (centroid
source + forward-simulation hazard) or ``tabpfn`` (the team's TabPFN locator, leak size and
hazard models; offline k-NN stand-in unless ``backend="cloud"``).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from tabpfn_spatial.evac.crowd import DEFAULT_PARAMS, Crowd, CrowdParams
from tabpfn_spatial.evac.crowd_plan import DETOUR_WEIGHT, PLANNERS, assign
from tabpfn_spatial.evac.fields import danger_window
from tabpfn_spatial.evac.standins import OracleHazard, OracleLocator
from tabpfn_spatial.hazard import ForwardSimPredictor, hazard_from_fields
from tabpfn_spatial.sim import Factory, Scenario, sensor_history, simulate
from tabpfn_spatial.source import WeightedCentroidLocator
from tabpfn_spatial.source.features import alarm_time

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
    log: list[dict] = field(default_factory=list)  # per update: model times and errors


@dataclass
class CrowdResult:
    """Scores, plus snapshots for drawing when asked."""

    metrics: dict[str, float]
    frames: list[dict] = field(default_factory=list)


MODELS = ("perfect", "baseline", "tabpfn")


def make_models(
    f: Factory, sc: Scenario, models: str, backend: str = "baseline", n_train: int = 1200
):
    """(locator, predictor) for a model setting; ``tabpfn`` fits the team's models on ``f``.

    ``tabpfn`` = Rayan's ``TabPFNLocator`` (location + leak size) and Ahmad's
    ``TabPFNHazardPredictor``; ``backend="baseline"`` swaps TabPFN for the offline k-NN
    stand-in (no tokens), ``"cloud"`` uses TabPFN. Fitting simulates their training
    tables: reuse the pair across leaks of the same plant.
    """
    if models not in MODELS:
        raise ValueError(f"models must be one of {MODELS}, not {models!r}")
    if models == "perfect":
        return OracleLocator(sc, sigma=0.0), OracleHazard(sc)
    if models == "baseline":
        return WeightedCentroidLocator(), ForwardSimPredictor()
    from tabpfn_spatial.hazard import TabPFNHazardPredictor
    from tabpfn_spatial.source import TabPFNLocator

    return (TabPFNLocator(backend=backend, n_train=n_train).fit(f),
            TabPFNHazardPredictor(backend=backend).fit(f))  # fmt: skip


def prepare(
    f: Factory,
    sc: Scenario,
    models: str,
    t_max: float = 300.0,
    update_s: float = 15.0,
    seed: int = 0,
    sample_every: float = 5.0,
    fitted: tuple | None = None,
    backend: str = "baseline",
) -> Prepared:
    """Truth, alarm and model danger maps for one leak. Cost: dominated by the models.

    The alarm and model inputs use the same noisy sensor history.
    ``fitted``: a (locator, predictor) pair to reuse (perfect models are bound to ``sc``,
    so they are always rebuilt). Every update is logged: model times, source error,
    median leak size.
    """
    if models not in MODELS:
        raise ValueError(f"models must be one of {MODELS}, not {models!r}")
    times = np.arange(0.0, t_max + 0.5, 1.0)
    fields = simulate(f, [sc], times)[0]
    toxic, fire = hazard_from_fields(f, fields)
    rng, danger, log = np.random.default_rng(seed), [], []
    history = sensor_history(f, sc, t_max, rng, every=sample_every)
    full = history(t_max)
    t_alarm = alarm_time(full.times, full.values)
    replan = np.arange(t_alarm, t_max + 1e-9, update_s) if t_alarm is not None else np.array([])
    if t_alarm is None:
        return Prepared(sc, times, fields, toxic | fire, None, replan, danger, log)
    if models == "perfect" or fitted is None:
        loc, pred = make_models(f, sc, models, backend)
    else:
        loc, pred = fitted
    for t in replan:
        r = history(float(t))  # one fixed history: old readings never change
        t0 = time.perf_counter()
        est = loc.locate(f, r)
        t1 = time.perf_counter()
        hz = pred.predict(f, r, est, t + np.arange(0.0, 120.0 + 1e-9, 10.0))
        t2 = time.perf_counter()
        danger.append(danger_window(hz, float(t), AHEAD))
        iy, ix = est.top_cells(1)[0]
        rate = 1.0
        if est.rate is not None and est.rate_levels is not None:
            rate = float(np.interp(0.5, est.rate_levels, est.rate))
        top = est.top_cells(3)
        guesses = [
            {"xy": f.centre(int(a), int(b)).tolist(), "p": float(est.prob[a, b])} for a, b in top
        ]
        log.append({"t": float(t), "locate_s": t1 - t0, "predict_s": t2 - t1, "source_top": guesses,
                    "source_error_m": float(np.linalg.norm(f.centre(int(iy), int(ix)) - sc.source)),
                    "rate_median": rate, "rate_true": float(sc.rate)})  # fmt: skip
    return Prepared(sc, times, fields, toxic | fire, t_alarm, replan, danger, log)


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
    detour_weight: float = DETOUR_WEIGHT,
) -> CrowdResult:
    """Evacuate ``positions`` with ``planner``; every planner replans at the same times."""
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, not {planner!r}")
    crowd = Crowd(f, positions, params)
    replans = {round(float(t)): d for t, d in zip(prep.replan_times, prep.danger, strict=True)}
    out = CrowdResult({})
    for k, t in enumerate(prep.times):
        if round(float(t)) in replans and crowd.active.any():
            assign(planner, f, crowd, replans[round(float(t))], float(t),
                   detour_weight=detour_weight)  # fmt: skip
        crowd.expose(prep.fields[k], prep.harm[k])
        if record_every and k % record_every == 0:
            out.frames.append(_snapshot(float(t), k, crowd))
        crowd.step(float(t))
    out.metrics = _metrics(crowd, prep)
    return out


def _metrics(crowd: Crowd, prep: Prepared) -> dict[str, float]:
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
        "queued_s_mean": float(crowd.queued_s.mean()),
        "queued_s_p95": float(np.percentile(crowd.queued_s, 95)),
    }
