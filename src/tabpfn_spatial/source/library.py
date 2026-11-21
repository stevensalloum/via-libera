"""Training library for simulation-based inference: simulated leaks -> (features, zone).

Leaks occur at ``factory.leak_sites`` (110 cells in the default hall). Neighbouring sites
are merged into at most ``n_zones`` zones (TabPFN-3.5 handles up to 160 classes), so the
classifier's ``predict_proba`` is a probability map over zones. Readings get the same
noise model as ``sim.readings``. Cost: one batched simulation, ~0.4 s per 100 leaks.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.sim import Factory, Scenario, sample_scenarios, sensor_values, simulate
from tabpfn_spatial.source.features import feature_matrix

T_OBS = 60.0
EVERY = 5.0
_BATCH = 500  # leaks per simulate call: memory O(batch * t * ny * nx)


@dataclass(frozen=True)
class Zones:
    """Leak sites grouped into zones. ``site_zone[i]`` is the zone of ``sites[i]``."""

    sites: np.ndarray  # (n, 2) metres
    site_zone: np.ndarray  # (n,) int
    centres: np.ndarray  # (z, 2) metres

    @property
    def n_zones(self) -> int:
        """Number of zones."""
        return len(self.centres)

    def zone_of(self, xy: np.ndarray) -> np.ndarray:
        """Zone of each point (m, 2): the zone of its nearest leak site."""
        d2 = ((np.asarray(xy, dtype=float)[:, None, :] - self.sites[None]) ** 2).sum(-1)
        return self.site_zone[d2.argmin(axis=1)]


def leak_zones(factory: Factory, n_zones: int = 100) -> Zones:
    """Merge the closest pair of zones until ``n_zones`` remain (deterministic).

    Agglomerative, centroid linkage; O(n^3) for n sites, instant for n ~ 100.
    """
    sites = factory.leak_sites
    members = [[i] for i in range(len(sites))]
    centres = [sites[i].copy() for i in range(len(sites))]
    while len(members) > n_zones:
        c = np.array(centres)
        d2 = ((c[:, None] - c[None]) ** 2).sum(-1) + np.diag(np.full(len(c), np.inf))
        a, b = sorted(np.unravel_index(int(d2.argmin()), d2.shape))
        members[a] += members.pop(b)
        centres.pop(b)
        centres[a] = sites[members[a]].mean(axis=0)
    site_zone = np.empty(len(sites), dtype=int)
    for z, idx in enumerate(members):
        site_zone[idx] = z
    return Zones(sites, site_zone, np.array(centres))


def noisy_series(
    factory: Factory,
    scenarios: Sequence[Scenario],
    rng: np.random.Generator,
    t_obs: float = T_OBS,
    every: float = EVERY,
    noise: float = 0.1,
    floor: float = 0.001,
    wind_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``times`` (t,), sensor ``values`` (B, k, t), measured ``wind`` (B, 2) for many leaks.

    Same sample times and noise model as ``sim.readings``, simulated in batches.
    """
    times = np.arange(0.0, t_obs + 1e-9, every)
    clean = np.concatenate(
        [
            sensor_values(factory, simulate(factory, scenarios[i : i + _BATCH], times))
            for i in range(0, len(scenarios), _BATCH)
        ]
    ).transpose(0, 2, 1)  # (B, k, t)
    noisy = clean * (1 + noise * rng.standard_normal(clean.shape))
    noisy += floor * rng.standard_normal(clean.shape)
    wind = np.array([s.wind for s in scenarios], dtype=float).reshape(-1, 2)
    wind = wind + wind_noise * rng.standard_normal(wind.shape)
    return times, np.maximum(noisy, 0.0), wind


def library_from_scenarios(
    factory: Factory,
    scenarios: Sequence[Scenario],
    rng: np.random.Generator,
    zones: Zones,
    **noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows ``X`` (B, n_features) and true ``zone`` (B,) for the given leaks."""
    times, values, wind = noisy_series(factory, scenarios, rng, **noise)
    sources = np.array([s.source for s in scenarios], dtype=float).reshape(-1, 2)
    return feature_matrix(times, values, wind), zones.zone_of(sources)


def make_library(
    factory: Factory, n: int, seed: int, n_zones: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` random simulated leaks (seeded) as ``(X, zone)``; never committed, regenerate."""
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(factory, n, rng)
    return library_from_scenarios(factory, scenarios, rng, leak_zones(factory, n_zones))
