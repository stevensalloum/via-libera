"""Shared data contracts between the four areas (integrator-owned; change only via PR).

Flow of one emergency (all positions in metres, times in seconds since the leak started):

    sim.Factory + sim.Scenario
        -> SensorReadings          (what the plant measures up to ``t_obs``)
        -> source  -> SourceEstimate   (where is the leak? a probability map)
        -> hazard  -> HazardMap        (where is it toxic / flammable at future times?)
        -> evac    -> EvacPlan         (which route is safest for each person?)

Each area owns one ``Protocol`` below and ships a baseline plus a TabPFN version.
Arrays use grid order ``[..., iy, ix]`` (row = y, column = x), matching ``Factory``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from tabpfn_spatial.sim.factory import Factory

# Concentration units are relative (1.0 = the source cell after 1 s of release at rate 1).
TOXIC_LIMIT = 0.1  # above this a person is exposed (~15 % of the floor 1-3 min after a leak)
LEL, UEL = 0.3, 3.0  # flammable range (lower / upper explosive limit), near the leak


@dataclass(frozen=True)
class SensorReadings:
    """Gas sensor time series up to ``t_obs`` plus the measured wind.

    ``positions``: (k, 2) sensor x, y. ``times``: (t,) sample times.
    ``values``: (k, t) noisy concentrations. ``wind``: (2,) mean wind u, v in m/s.
    ``heights``: optional (k,) detector heights in m (3D plants, ``sim.plant3d``).
    """

    positions: np.ndarray
    times: np.ndarray
    values: np.ndarray
    wind: np.ndarray
    heights: np.ndarray | None = None

    @property
    def t_obs(self) -> float:
        """Time of the latest reading."""
        return float(self.times[-1])


@dataclass(frozen=True)
class SourceEstimate:
    """Posterior over the leak location as a probability map on the factory grid.

    ``prob``: (ny, nx), sums to 1. ``rate``: optional (q,) quantiles of the release rate
    at ``rate_levels``. Use ``top_cells`` to sample or enumerate likely sources.
    ``height``: optional (q,) quantiles of the leak height in m at ``height_levels`` (3D).
    """

    prob: np.ndarray
    rate: np.ndarray | None = None
    rate_levels: np.ndarray | None = None
    height: np.ndarray | None = None
    height_levels: np.ndarray | None = None

    def top_cells(self, n: int = 5) -> np.ndarray:
        """(n, 2) grid indices (iy, ix) of the n most likely cells, best first."""
        flat = np.argsort(self.prob.ravel())[::-1][:n]
        return np.column_stack(np.unravel_index(flat, self.prob.shape))


@dataclass(frozen=True)
class HazardMap:
    """Probabilities per cell at future times ``times`` (absolute seconds).

    ``p_toxic``: (h, ny, nx) P(concentration > TOXIC_LIMIT).
    ``p_fire``: (h, ny, nx) P(flammable mixture next to an ignition source).
    """

    times: np.ndarray
    p_toxic: np.ndarray
    p_fire: np.ndarray

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Maps at the forecast time closest to ``t``."""
        i = int(np.argmin(np.abs(self.times - t)))
        return self.p_toxic[i], self.p_fire[i]


@dataclass(frozen=True)
class Route:
    """One person's path: (m, 2) grid cells (iy, ix) from start to an exit."""

    cells: np.ndarray
    exit_index: int
    risk: float  # planner's own estimate of P(harm) along the route


@dataclass(frozen=True)
class EvacPlan:
    """One route per person, plus exits judged unusable."""

    routes: list[Route]
    blocked_exits: list[int] = field(default_factory=list)


# --------------------------------------------------------------- area interfaces --
class SourceLocator(Protocol):
    """Leak source detection (owner: see AGENTS.md)."""

    def locate(self, factory: "Factory", readings: SensorReadings) -> SourceEstimate:
        """Probability map of the leak location from the readings."""
        ...


class HazardPredictor(Protocol):
    """Hazard prediction (owner: see AGENTS.md)."""

    def predict(
        self,
        factory: "Factory",
        readings: SensorReadings,
        source: SourceEstimate,
        times: np.ndarray,
    ) -> HazardMap:
        """Toxic and fire probability maps at the requested absolute times."""
        ...


class EvacPlanner(Protocol):
    """Emergency evacuation planning (owner: see AGENTS.md)."""

    def plan(
        self, factory: "Factory", people: np.ndarray, hazard: HazardMap, t_start: float
    ) -> EvacPlan:
        """Routes for people at (p, 2) positions in metres, leaving at ``t_start``."""
        ...
