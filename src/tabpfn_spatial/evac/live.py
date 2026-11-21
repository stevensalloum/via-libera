"""Live evacuation of one person: replan every ``update_s`` seconds as the leak grows.

Timeline: the leak starts at t = 0 and the person stands still. At ``t_alarm`` the alarm
sounds and the person starts walking. From then on, every ``update_s`` seconds the system
reads the sensors, asks the source and hazard models for new maps and replans. Three
people start at the same point so the policies can be compared on the same leak:

- ``live``: replanned at every update (what we build);
- ``once``: planned at the alarm with the same models, never updated;
- ``nearest``: the conventional plan, nearest exit, gas ignored.

Everyone is scored on the true simulation: ``dose`` (concentration x seconds breathed) and
``harmful_s`` (seconds spent in truly toxic or flammable cells). Cost per update: one
readings simulation, the two models, and ``1 + exits`` Dijkstra runs (O(N log N) each).
"""

import copy
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
from tabpfn_spatial.sim import Factory, Scenario, readings, simulate

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
    t_alarm: float = 30.0,
    update_s: float = 5.0,
    t_max: float = 180.0,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Iterator[Frame]:
    """Yield one ``Frame`` every ``dt`` seconds from 0 to ``t_max``."""
    planner = planner or RiskAwarePlanner()
    rng = rng or np.random.default_rng()
    times = np.arange(0.0, t_max + dt / 2, dt)
    fields = simulate(factory, [scenario], times)[0]
    toxic, fire = hazard_from_fields(factory, fields)
    harm = toxic | fire
    people = {p: Walker(np.asarray(start, float).copy(), planner.speed) for p in POLICIES}
    k_alarm, k_every = round(t_alarm / dt), max(1, round(update_s / dt))
    r = est = hz = None
    options: list[ExitOption] = []
    for k, t in enumerate(times):
        replanned = k >= k_alarm and (k - k_alarm) % k_every == 0
        if replanned:
            r = readings(factory, scenario, float(t), rng)
            est = locator.locate(factory, r)
            hz = predictor.predict(
                factory, r, est, t + np.arange(0.0, HORIZON + 1e-9, FORECAST_STEP)
            )
            if people["live"].out_at is None:
                options = exit_options(factory, planner, people["live"].pos, hz, float(t))
            _steer(people["live"], factory, planner, hz, float(t))
            if k == k_alarm:
                _steer(people["once"], factory, planner, hz, float(t))
                zero = np.zeros_like(hz.p_toxic)
                _steer(people["nearest"], factory, planner, HazardMap(hz.times, zero, zero), t)
        snap = {p: copy.deepcopy(w) for p, w in people.items()}
        yield Frame(
            float(t), fields[k], harm[k], snap, r, est, hz, options, replanned, k >= k_alarm
        )
        for w in people.values():
            if w.out_at is None:
                iy, ix = factory.cell_of(w.pos)
                w.dose += float(fields[k, iy, ix]) * dt
                w.harmful_s += dt * bool(harm[k, iy, ix])
                if k >= k_alarm:
                    w.step(dt, float(t))
