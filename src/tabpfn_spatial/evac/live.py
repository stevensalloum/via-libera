"""Live evacuation of one person: replan every ``update_s`` seconds as the leak grows.

Timeline: the leak starts at t = 0 and the person stands still. The alarm sounds at
``t_alarm``: a fixed time, or with ``None`` when a gas detector (sampled every
``sample_every`` s) first reads above ``sim.ALARM_LEVEL``. Then the whole plant evacuates.
Until the first route arrives the person walks to the nearest exit
(``before_plan="nearest"``) or waits (``"wait"``). Every ``update_s`` seconds the system
reads the sensors and asks the source and hazard models for new maps; the models take
``latency`` seconds, so each route lands that much later, planned from where the person
is by then. Three people start at the same point so the
policies can be compared on the same leak:

- ``live``: replanned at every update (what we build);
- ``once``: planned at the alarm with the same models, never updated;
- ``nearest``: the conventional plan, nearest exit, gas ignored.

Everyone is scored on the true simulation: ``dose`` (concentration x seconds breathed) and
``harmful_s`` (seconds spent in truly toxic or flammable cells). Cost per update: one
readings simulation, the two models, and ``1 + exits`` Dijkstra runs (O(N log N) each).
"""

import copy
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import numpy as np

from tabpfn_spatial.contracts import (
    HazardMap,
    HazardPredictor,
    SensorReadings,
    SourceEstimate,
    SourceLocator,
)
from tabpfn_spatial.evac.planner import WALK_SPEED, RiskAwarePlanner
from tabpfn_spatial.hazard import hazard_from_fields
from tabpfn_spatial.sim import (
    Factory,
    Scenario,
    sensor_history,
    simulate,
)
from tabpfn_spatial.sim.plume import DIFFUSIVITY
from tabpfn_spatial.source.features import alarm_time

POLICIES = ("live", "once", "nearest")
FORECAST_STEP = 10.0  # seconds between forecast maps
HORIZON = 120.0  # forecast this far ahead at each update


@dataclass
class Walker:
    """A person following a polyline of waypoints (metres) at ``speed`` m/s."""

    pos: np.ndarray
    speed: float = WALK_SPEED
    path: np.ndarray | None = None  # remaining waypoints, path[0] == pos
    exit_index: int | None = None
    walked: float = 0.0
    out_at: float | None = None  # time the exit was reached
    dose: float = 0.0
    harmful_s: float = 0.0
    trail: list[np.ndarray] = field(default_factory=list)

    def follow(self, waypoints: np.ndarray, exit_index: int) -> None:
        """Head for ``waypoints[1:]`` straight from where we stand."""
        self.path = np.vstack([self.pos, waypoints[1:]])
        self.exit_index = exit_index

    def step(self, dt: float, t: float) -> None:
        """Walk ``speed * dt`` metres along the path; record ``out_at`` on arrival."""
        if self.path is None or self.out_at is not None:
            return
        self.trail.append(self.pos.copy())
        budget = self.speed * dt
        while len(self.path) > 1 and budget > 0:
            seg = self.path[1] - self.pos
            d = float(np.linalg.norm(seg))
            if d <= budget:
                self.pos, budget, self.walked = self.path[1].copy(), budget - d, self.walked + d
                self.path = self.path[1:]
            else:
                self.pos = self.pos + seg / d * budget
                self.walked, budget = self.walked + budget, 0.0
                self.path[0] = self.pos
        if len(self.path) == 1:
            self.out_at = t + dt - budget / self.speed
            self.trail.append(self.pos.copy())


@dataclass(frozen=True)
class ExitOption:
    """What the planner thinks of one exit from where the person stands now."""

    exit_index: int
    length_m: float
    eta_s: float  # absolute time of arrival
    risk: float  # P(harm) along the route
    cost: float  # the planner's cost (metres + risk_weight x P(harm) per step)


@dataclass(frozen=True)
class Frame:
    """State of the run at time ``t`` (everything a viewer draws)."""

    t: float
    field: np.ndarray  # (ny, nx) true concentration
    harm: np.ndarray  # (ny, nx) bool, truly toxic or flammable
    walkers: dict[str, Walker]
    readings: SensorReadings | None
    source: SourceEstimate | None
    hazard: HazardMap | None
    options: list[ExitOption]
    replanned: bool
    alarm: bool
    t_alarm: float | None  # when the alarm sounds (None: no detector has alarmed)


def exit_options(
    f: Factory, planner: RiskAwarePlanner, pos: np.ndarray, hz: HazardMap, t: float
) -> list[ExitOption]:
    """Best route to each exit separately (unreachable exits are left out)."""
    out = []
    for e in range(len(f.exits)):
        g = replace(f, exits=f.exits[e : e + 1])
        try:
            rt = planner.plan(g, pos[None], hz, t).routes[0]
        except RuntimeError:
            continue
        length = float(np.linalg.norm(np.diff(rt.cells, axis=0), axis=1).sum() * f.cell)
        rt = replace(rt, exit_index=e)
        out.append(
            ExitOption(e, length, t + length / planner.speed, rt.risk, planner.cost(f, rt, hz, t))
        )
    return out


def _steer(w: Walker, f: Factory, planner: RiskAwarePlanner, hz: HazardMap, t: float) -> None:
    """Replace the walker's path by the planner's best route from its position."""
    if w.out_at is not None:
        return
    rt = planner.plan(f, w.pos[None], hz, t).routes[0]
    w.follow((rt.cells[:, ::-1] + 0.5) * f.cell, rt.exit_index)


def run_live(
    factory: Factory,
    scenario: Scenario,
    start: np.ndarray,
    locator: SourceLocator,
    predictor: HazardPredictor,
    planner: RiskAwarePlanner | None = None,
    t_alarm: float | None = None,
    update_s: float = 5.0,
    t_max: float = 180.0,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
    latency: float = 0.0,
    before_plan: str = "nearest",
    sample_every: float = 5.0,
) -> Iterator[Frame]:
    """Yield one ``Frame`` every ``dt`` seconds from 0 to ``t_max``.

    ``t_alarm=None``: the first noisy sensor alarm starts everything (no alarm, nobody moves).
    ``sample_every``: seconds between sensor samples (alarm check and readings).
    A supplied ``t_alarm`` is an explicit simulation override. Sensor history is
    generated once and the readings that trigger the alarm are passed to the locator.

    ``latency``: seconds the source + hazard models take; a route made from the readings
    at t reaches the person at t + latency. ``before_plan``: what ``live`` and ``once`` do
    between the alarm and their first route, ``"nearest"`` (walk to the nearest exit) or
    ``"wait"`` (stay put). ``nearest`` always walks.
    """
    if before_plan not in ("nearest", "wait"):
        raise ValueError(f"before_plan must be 'nearest' or 'wait', not {before_plan!r}")
    planner = planner or RiskAwarePlanner()
    rng = rng or np.random.default_rng()
    times = np.arange(0.0, t_max + dt / 2, dt)
    fields = simulate(factory, [scenario], times)[0]
    toxic, fire = hazard_from_fields(factory, fields)
    harm = toxic | fire
    people = {p: Walker(np.asarray(start, float).copy(), planner.speed) for p in POLICIES}
    history = sensor_history(factory, scenario, t_max, rng, every=sample_every)
    if t_alarm is None:
        full = history(t_max)
        t_alarm = alarm_time(full.times, full.values)
    k_alarm = round(t_alarm / dt) if t_alarm is not None else len(times)
    k_every = max(1, round(update_s / dt))
    k_lat = round(latency / dt)
    r = est = hz = None
    options: list[ExitOption] = []
    pending: list[tuple[int, HazardMap, bool]] = []  # (step it lands, maps, first plan?)
    for k, t in enumerate(times):
        t = float(t)
        replanned = k >= k_alarm and (k - k_alarm) % k_every == 0
        if k == k_alarm:  # alarm: who walks to the nearest exit right away
            calm = np.zeros((1, factory.ny, factory.nx))
            walk = POLICIES if before_plan == "nearest" else ("nearest",)
            for pol in walk:
                _steer(people[pol], factory, planner, HazardMap(np.array([t]), calm, calm), t)
        if replanned:
            r = history(t)
            est = locator.locate(factory, r)
            times_f = t + np.arange(0.0, HORIZON + 1e-9, FORECAST_STEP)
            hz = predictor.predict(factory, r, est, times_f)
            pending.append((k + k_lat, hz, k == k_alarm))
        for _due, maps, first in [p for p in pending if p[0] == k]:
            if people["live"].out_at is None:
                options = exit_options(factory, planner, people["live"].pos, maps, t)
            _steer(people["live"], factory, planner, maps, t)
            if first:
                _steer(people["once"], factory, planner, maps, t)
        pending = [p for p in pending if p[0] > k]
        snap = {p: copy.deepcopy(w) for p, w in people.items()}
        yield Frame(
            t, fields[k], harm[k], snap, r, est, hz, options, replanned, k >= k_alarm, t_alarm
        )
        for w in people.values():
            if w.out_at is None:
                iy, ix = factory.cell_of(w.pos)
                w.dose += float(fields[k, iy, ix]) * dt
                w.harmful_s += dt * bool(harm[k, iy, ix])
                if k >= k_alarm:
                    w.step(dt, t)


# ------------------------------------------------------------ event-driven pipeline --
@dataclass(frozen=True)
class Cycle:
    """One run of the pipeline: readings up to ``t_start`` -> locate -> hazard -> route.

    The route reaches the person at ``t_land`` = ``t_start`` + the time the models took
    (rounded up to whole simulated seconds). ``samples``: readings per sensor it used.
    """

    index: int
    t_start: float
    t_land: float
    samples: int
    locate_s: float
    predict_s: float
    source: SourceEstimate
    hazard: HazardMap
    source_error_m: float
    rate_est: float


@dataclass(frozen=True)
class PipelineFrame(Frame):
    """``Frame`` plus the pipeline: runs that have landed and what is being computed now."""

    cycles: tuple[Cycle, ...] = ()
    phase: str = ""
    running: Cycle | None = None
    routes: tuple[np.ndarray, ...] = ()  # the replanned person's route as each run landed


def _rate(est: SourceEstimate) -> float:
    if est.rate is None or est.rate_levels is None:
        return 1.0
    return float(np.interp(0.5, est.rate_levels, est.rate))


def _phase(t: float, alarm: bool, running: Cycle | None, landed: list, max_runs: int) -> str:
    if not alarm:
        return "no alarm yet: nobody moves"
    if running is not None:
        n = running.index + 1
        if t < running.t_start + running.locate_s:
            return f"run {n}: source model locating the leak ({running.locate_s:.1f} s)"
        return f"run {n}: hazard model forecasting the gas ({running.predict_s:.1f} s)"
    if not landed:
        return "alarm: walking to the nearest exit"
    if len(landed) >= max_runs:
        return f"all {max_runs} runs done: following the route of run {len(landed)}"
    return f"route from run {len(landed)}"


def run_pipeline(
    factory: Factory,
    scenario: Scenario,
    start: np.ndarray,
    locator: SourceLocator,
    predictor: HazardPredictor,
    planner: RiskAwarePlanner | None = None,
    t_alarm: float | None = None,
    max_runs: int = 10,
    t_max: float = 180.0,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
    sample_every: float = 5.0,
    fixed_latency: dict[str, float] | None = None,
    diffusivity: float = DIFFUSIVITY,
    dropout: float = 0.0,
    stuck: float = 0.0,
) -> Iterator[PipelineFrame]:
    """Yield a frame every ``dt`` s; the models run back to back from the alarm.

    Unless ``t_alarm`` is supplied explicitly, the first noisy sensor alarm starts the
    loop; the locator receives the same sensor history. No fixed update period: at the
    alarm everyone heads for the nearest exit and run 1
    starts with the readings so far (locator, then hazard on its output). When both have
    answered, the route lands and run 2 starts at once with all readings up to then, and
    so on for ``max_runs`` runs (after that, everyone is assumed out or on a final route).
    Time spent in the models is their measured wall time (cloud or offline), unless
    ``fixed_latency`` = {"locate": s, "predict": s} pins it (tests, reproducible studies).
    ``diffusivity``, ``dropout``, ``stuck`` stress the truth and the sensors (``sim.CONDITIONS``;
    wind drift lives in the scenario); the models keep their own assumptions.
    """
    if max_runs < 1:
        raise ValueError("max_runs must be >= 1")
    planner = planner or RiskAwarePlanner()
    rng = rng or np.random.default_rng()
    times = np.arange(0.0, t_max + dt / 2, dt)
    fields = simulate(factory, [scenario], times, d=diffusivity)[0]
    toxic, fire = hazard_from_fields(factory, fields)
    harm = toxic | fire
    history = sensor_history(factory, scenario, t_max, rng, every=sample_every,
                             diffusivity=diffusivity, dropout=dropout, stuck=stuck)  # fmt: skip
    if t_alarm is None:
        full = history(t_max)
        t_alarm = alarm_time(full.times, full.values)
    people = {p: Walker(np.asarray(start, float).copy(), planner.speed) for p in POLICIES}
    routes: list[np.ndarray] = []
    started: list[Cycle] = []
    landed: list[Cycle] = []
    running: Cycle | None = None
    r = None
    options: list[ExitOption] = []
    for k, t in enumerate(times):
        t = float(t)
        alarm = t_alarm is not None and t >= t_alarm - 1e-9
        if alarm and t_alarm is not None and abs(t - t_alarm) < dt / 2:
            calm = np.zeros((1, factory.ny, factory.nx))
            for w in people.values():
                _steer(w, factory, planner, HazardMap(np.array([t]), calm, calm), t)
        if running is not None and t >= running.t_land - 1e-9:  # the route lands
            if people["live"].out_at is None:
                options = exit_options(factory, planner, people["live"].pos, running.hazard, t)
            _steer(people["live"], factory, planner, running.hazard, t)
            if people["live"].path is not None:
                routes.append(people["live"].path.copy())
            if not landed:
                _steer(people["once"], factory, planner, running.hazard, t)
            landed.append(running)
            running = None
        if alarm and running is None and len(started) < max_runs and people["live"].out_at is None:
            r = history(t)  # every reading so far; old ones never change
            t0 = time.perf_counter()
            est = locator.locate(factory, r)
            t1 = time.perf_counter()
            hz = predictor.predict(
                factory, r, est, t + np.arange(0.0, HORIZON + 1e-9, FORECAST_STEP)
            )
            t2 = time.perf_counter()
            loc_s, pred_s = (t1 - t0, t2 - t1) if fixed_latency is None else (
                fixed_latency["locate"], fixed_latency["predict"])  # fmt: skip
            iy, ix = est.top_cells(1)[0]
            err = float(np.linalg.norm(factory.centre(int(iy), int(ix)) - scenario.source))
            t_land = t + max(dt, float(np.ceil((loc_s + pred_s) / dt - 1e-9)) * dt)
            running = Cycle(
                len(started), t, t_land, len(r.times), loc_s, pred_s, est, hz, err, _rate(est)
            )
            started.append(running)
        last = landed[-1] if landed else None
        snap = {p: copy.deepcopy(w) for p, w in people.items()}
        yield PipelineFrame(
            t, fields[k], harm[k], snap, r, last.source if last else None,
            last.hazard if last else None, options, bool(landed) and landed[-1].t_land == t,
            alarm, t_alarm, tuple(landed), _phase(t, alarm, running, landed, max_runs), running,
            tuple(routes),
        )  # fmt: skip
        for w in people.values():
            if w.out_at is None:
                iy, ix = factory.cell_of(w.pos)
                w.dose += float(fields[k, iy, ix]) * dt
                w.harmful_s += dt * bool(harm[k, iy, ix])
                if alarm:
                    w.step(dt, t)
