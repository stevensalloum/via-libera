"""Random leaks, people and sensor readings for training tables and episodes."""

import numpy as np

from tabpfn_spatial.contracts import SensorReadings
from tabpfn_spatial.sim.factory import Factory
from tabpfn_spatial.sim.plume import Scenario, simulate, wind_at


def sample_scenarios(
    factory: Factory, n: int, rng: np.random.Generator, drift: float = 0.0
) -> list[Scenario]:
    """Leaks on the pipes (fittings likelier); rate log-uniform 0.5-2; wind 0.2-1.5 m/s.

    Sites are drawn from ``factory.leak_sites`` with probability ``leak_weights``.

    ``drift`` > 0 makes the wind direction wander (see ``Scenario``); each leak gets its own
    seed from ``rng``, so a fixed ``rng`` seed reproduces the whole set.
    """
    sites = factory.leak_sites
    w = factory.leak_weights
    picks = sites[rng.choice(len(sites), size=n, p=w / w.sum())]
    rates = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=n))
    speed = rng.uniform(0.2, 1.5, size=n)
    angle = rng.uniform(0, 2 * np.pi, size=n)
    winds = np.column_stack([speed * np.cos(angle), speed * np.sin(angle)])
    seeds = rng.integers(2**31, size=n)
    return [
        Scenario(p, float(r), w, drift, int(sd))
        for p, r, w, sd in zip(picks, rates, winds, seeds, strict=True)
    ]


def sample_people(factory: Factory, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2) random positions (metres) on open floor."""
    open_cells = factory.centres()[~factory.walls]
    jitter = rng.uniform(-0.4, 0.4, size=(n, 2)) * factory.cell
    return open_cells[rng.integers(len(open_cells), size=n)] + jitter


def sensor_values(factory: Factory, fields: np.ndarray) -> np.ndarray:
    """Clean sensor values (..., k) from fields (..., ny, nx)."""
    cells = np.array([factory.cell_of(s) for s in factory.sensors])
    return fields[..., cells[:, 0], cells[:, 1]]


def readings(
    factory: Factory,
    scenario: Scenario,
    t_obs: float,
    rng: np.random.Generator,
    every: float = 5.0,
    noise: float = 0.1,
    floor: float = 0.001,
    wind_noise: float = 0.1,
    dropout: float = 0.0,
    stuck: float = 0.0,
) -> SensorReadings:
    """Noisy sensor series 0..t_obs and the measured wind (mean over the window).

    Faults (off by default): each sensor is dead with probability ``dropout`` (its whole row
    is NaN) or stuck with probability ``stuck`` (repeats its first value). Consumers must
    treat NaN as "no reading", e.g. ``np.nan_to_num`` or a missing-value flag column.
    """
    times = np.arange(0.0, t_obs + 1e-9, every)
    clean = sensor_values(factory, simulate(factory, [scenario], times)[0]).T  # (k, t)
    noisy = clean * (1 + noise * rng.standard_normal(clean.shape))
    noisy += floor * rng.standard_normal(clean.shape)
    values = np.maximum(noisy, 0.0)
    k = len(values)
    is_stuck = rng.uniform(size=k) < stuck
    values[is_stuck] = values[is_stuck, :1]
    values[rng.uniform(size=k) < dropout] = np.nan
    wind = wind_at(scenario, times).mean(axis=0) + wind_noise * rng.standard_normal(2)
    return SensorReadings(factory.sensors.copy(), times, values, wind)
