"""Gas dispersion: 2D advection-diffusion on the factory grid, many scenarios at once.

Finite volumes with upwind advection; walls and machines are no-flux, the outer boundary
lets gas out (ventilation). The leak releases ``rate`` units per second into its cell
from t = 0. Units are relative (see ``contracts``).

Cost: O(B * ny * nx) per step, steps = t_max / dt with
dt = 0.9 / (4 D / dx^2 + (|u| + |v|) / dx). Default hall, 500 scenarios, 120 s: ~2 s.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.sim.factory import Factory

DIFFUSIVITY = 0.8  # m^2/s, turbulent mixing indoors


@dataclass(frozen=True)
class Scenario:
    """One leak: ``source`` (2,) metres, ``rate`` units/s, ``wind`` (2,) m/s (u, v) at t = 0.

    ``drift``: random walk of the wind direction in rad per sqrt(second) (0 = steady);
    ``seed`` makes the walk reproducible. Speed stays constant.
    """

    source: np.ndarray
    rate: float
    wind: np.ndarray
    drift: float = 0.0
    seed: int = 0


_WIND_STEP = 1.0  # seconds between wind direction updates


def wind_at(scenario: Scenario, times: np.ndarray) -> np.ndarray:
    """(T, 2) wind vector at ``times`` (seconds): ``wind`` rotated by the drift walk. O(t_max)."""
    if scenario.drift < 0:
        raise ValueError("drift must be >= 0")
    times = np.asarray(times, dtype=float)
    k = np.floor(times / _WIND_STEP).astype(int)
    steps = int(k.max(initial=0)) + 1
    walk = np.random.default_rng(scenario.seed).standard_normal(steps)
    angle = np.concatenate([[0.0], np.cumsum(walk[1:])]) * scenario.drift * np.sqrt(_WIND_STEP)
    a = angle[k]
    u, v = scenario.wind
    return np.column_stack([u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)])


def _divergence(c: np.ndarray, open_: np.ndarray, wind: np.ndarray, dx: float, d: float):
    """Net inflow per cell (B, ny, nx) from advection and diffusion fluxes."""
    b, ny, nx = c.shape
    cp = np.zeros((b, ny + 2, nx + 2))  # zero gas outside the hall (np.pad is ~2x slower)
    cp[:, 1:-1, 1:-1] = c
    op = np.ones((ny + 2, nx + 2))
    op[1:-1, 1:-1] = open_
    out = np.zeros_like(c)
    for axis, vel in ((2, wind[:, 0]), (1, wind[:, 1])):
        if axis == 2:
            lo, hi = cp[:, 1:-1, :-1], cp[:, 1:-1, 1:]
            o = op[1:-1, :-1] * op[1:-1, 1:]
        else:
            lo, hi = cp[:, :-1, 1:-1], cp[:, 1:, 1:-1]
            o = op[:-1, 1:-1] * op[1:, 1:-1]
        v = vel[:, None, None]
        flux = (v * np.where(v > 0, lo, hi) - d * (hi - lo) / dx) * o
        diff = np.diff(flux, axis=axis)
        out -= diff / dx
    return out


def simulate(
    factory: Factory, scenarios: Sequence[Scenario], times: np.ndarray, d: float = DIFFUSIVITY
) -> np.ndarray:
    """Concentration fields (B, T, ny, nx) at the requested times (seconds, ascending)."""
    times = np.asarray(times, dtype=float)
    b = len(scenarios)
    rate = np.array([s.rate for s in scenarios], dtype=float)
    src = np.array([factory.cell_of(s.source) for s in scenarios]).reshape(b, 2)
    if factory.walls[src[:, 0], src[:, 1]].any():
        raise ValueError("leak source inside a solid cell (wall or machine)")
    grid_t = np.arange(0.0, times.max(initial=0.0) + _WIND_STEP, _WIND_STEP)
    winds = np.stack([wind_at(s, grid_t) for s in scenarios], axis=1)  # (steps, B, 2)
    open_ = (~factory.walls).astype(float)
    dx = factory.cell
    speed = np.sqrt(2) * np.linalg.norm(winds, axis=-1).max(initial=0.0)  # rotation bound
    dt = 0.9 / (4 * d / dx**2 + speed / dx)
    c = np.zeros((b, factory.ny, factory.nx))
    out = np.zeros((b, len(times), factory.ny, factory.nx))
    t, idx = 0.0, np.arange(b)
    for k, target in enumerate(times):
        while t < target - 1e-9:
            h = min(dt, target - t)
            wind = winds[min(int(t / _WIND_STEP), len(grid_t) - 1)]
            c += h * _divergence(c, open_, wind, dx, d)
            c[idx, src[:, 0], src[:, 1]] += h * rate
            c *= open_
            np.maximum(c, 0.0, out=c)
            t += h
        out[:, k] = c
    return out
