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
    """One leak: ``source`` (2,) metres, ``rate`` units/s, ``wind`` (2,) m/s (u, v)."""

    source: np.ndarray
    rate: float
    wind: np.ndarray


def _divergence(c: np.ndarray, open_: np.ndarray, wind: np.ndarray, dx: float, d: float):
    """Net inflow per cell (B, ny, nx) from advection and diffusion fluxes."""
    cp = np.pad(c, ((0, 0), (1, 1), (1, 1)))
    op = np.pad(open_, 1, constant_values=1.0)
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
    wind = np.array([s.wind for s in scenarios], dtype=float).reshape(b, 2)
    rate = np.array([s.rate for s in scenarios], dtype=float)
    src = np.array([factory.cell_of(s.source) for s in scenarios]).reshape(b, 2)
    if factory.walls[src[:, 0], src[:, 1]].any():
        raise ValueError("leak source inside a solid cell (wall or machine)")
    open_ = (~factory.walls).astype(float)
    dx = factory.cell
    speed = np.abs(wind).sum(axis=1).max(initial=0.0)
    dt = 0.9 / (4 * d / dx**2 + speed / dx)
    c = np.zeros((b, factory.ny, factory.nx))
    out = np.zeros((b, len(times), factory.ny, factory.nx))
    t, idx = 0.0, np.arange(b)
    for k, target in enumerate(times):
        while t < target - 1e-9:
            h = min(dt, target - t)
            c += h * _divergence(c, open_, wind, dx, d)
            c[idx, src[:, 0], src[:, 1]] += h * rate
            c *= open_
            np.maximum(c, 0.0, out=c)
            t += h
        out[:, k] = c
    return out
