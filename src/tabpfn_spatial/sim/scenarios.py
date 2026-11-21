"""Random leaks, people and sensor readings for training tables and episodes."""

import numpy as np

from tabpfn_spatial.contracts import SensorReadings
from tabpfn_spatial.sim.factory import Factory
from tabpfn_spatial.sim.plume import Scenario, simulate


def sample_scenarios(factory: Factory, n: int, rng: np.random.Generator) -> list[Scenario]:
    """Leaks at random machine-adjacent cells; rate log-uniform 0.5-2; wind 0.2-1.5 m/s."""
    sites = factory.leak_sites
    picks = sites[rng.integers(len(sites), size=n)]
    rates = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=n))
    speed = rng.uniform(0.2, 1.5, size=n)
    angle = rng.uniform(0, 2 * np.pi, size=n)
    winds = np.column_stack([speed * np.cos(angle), speed * np.sin(angle)])
    return [Scenario(p, float(r), w) for p, r, w in zip(picks, rates, winds, strict=True)]


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
) -> SensorReadings:
    """Noisy sensor series 0..t_obs (multiplicative + additive noise) and measured wind."""
    times = np.arange(0.0, t_obs + 1e-9, every)
    clean = sensor_values(factory, simulate(factory, [scenario], times)[0]).T  # (k, t)
    noisy = clean * (1 + noise * rng.standard_normal(clean.shape))
    noisy += floor * rng.standard_normal(clean.shape)
    wind = scenario.wind + wind_noise * rng.standard_normal(2)
    return SensorReadings(factory.sensors.copy(), times, np.maximum(noisy, 0.0), wind)
