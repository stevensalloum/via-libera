"""Training table for simulation-based inference: one row per (simulated leak, moment).

Each leak is simulated once; its noisy sensor series gives an alarm time, and a row is cut
at every moment ``w`` seconds after that alarm (default 0, 5, ..., 30 s). The answer
columns are the leak zone (classification) and the release rate (regression). Leaks that
never alarm within ``ALARM_WINDOW`` are skipped, as a real system would never be asked.

Noise, wind and sample times match ``sim.readings`` (the measured wind is the mean over
the samples so far). Cost: one batched simulation, ~0.4 s per 100 leaks; rows =
leaks x moments (keep it under TabPFN's ~10,000-row context).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.sim import Factory, Scenario, sample_scenarios, sensor_values, simulate, wind_at
from tabpfn_spatial.source.features import alarm_time, moment_row

EVERY = 5.0  # seconds between sensor samples
ALARM_WINDOW = 40.0  # leaks that alarm later than this are left out of the table
DEFAULT_WS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
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
    """Merge the closest pair of zones until at most ``n_zones`` remain (deterministic).

    With fewer sites than ``n_zones`` (24 pipe cells on the default plant) every site is
    its own zone. Agglomerative, centroid linkage; O(n^3) for n sites.
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


@dataclass(frozen=True)
class Library:
    """Training rows. All arrays have one entry per row (leak x moment)."""

    X: np.ndarray  # (rows, 2k + 3) features
    zone: np.ndarray  # (rows,) true zone
    rate: np.ndarray  # (rows,) true release rate
    source: np.ndarray  # (rows, 2) true leak position, metres
    w: np.ndarray  # (rows,) seconds since the alarm
    t_alarm: np.ndarray  # (rows,) alarm time on the simulator clock
    leak: np.ndarray  # (rows,) index of the leak in the input list


def noisy_series(
    factory: Factory,
    scenarios: Sequence[Scenario],
    rng: np.random.Generator,
    t_max: float,
    noise: float = 0.1,
    floor: float = 0.001,
    wind_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``times`` (t,), noisy ``values`` (B, k, t) and running-mean ``wind`` (B, t, 2).

    ``wind[b, j]`` is what ``sim.readings`` reports at ``t_obs = times[j]``.
    """
    times = np.arange(0.0, t_max + 1e-9, EVERY)
    clean = np.concatenate(
        [
            sensor_values(factory, simulate(factory, scenarios[i : i + _BATCH], times))
            for i in range(0, len(scenarios), _BATCH)
        ]
    ).transpose(0, 2, 1)
    noisy = clean * (1 + noise * rng.standard_normal(clean.shape))
    noisy += floor * rng.standard_normal(clean.shape)
    n = np.arange(1, len(times) + 1)[None, :, None]
    mean_wind = np.stack([np.cumsum(wind_at(s, times), axis=0) for s in scenarios]) / n
    mean_wind = mean_wind + wind_noise * rng.standard_normal((len(scenarios), 1, 2))
    return times, np.maximum(noisy, 0.0), mean_wind


def library_from_scenarios(
    factory: Factory,
    scenarios: Sequence[Scenario],
    rng: np.random.Generator,
    ws: Sequence[float] = DEFAULT_WS,
    n_zones: int = 100,
    **noise: float,
) -> Library:
    """Rows for the given leaks at each moment ``ws`` after their alarm."""
    times, values, wind = noisy_series(factory, scenarios, rng, ALARM_WINDOW + max(ws), **noise)
    zones = leak_zones(factory, n_zones)
    cols: dict[str, list] = {k: [] for k in ("X", "zone", "rate", "source", "w", "t_alarm", "leak")}
    for b, sc in enumerate(scenarios):
        t_a = alarm_time(times, values[b])
        if t_a is None or t_a > ALARM_WINDOW:
            continue
        zone = int(zones.zone_of(np.asarray(sc.source)[None])[0])
        for w in ws:
            j = int(np.searchsorted(times, t_a + w + 1e-9))  # samples up to t_a + w
            cols["X"].append(moment_row(times[:j], values[b, :, :j], wind[b, j - 1], t_a))
            for key, val in (("zone", zone), ("rate", sc.rate), ("source", sc.source),
                             ("w", float(w)), ("t_alarm", t_a), ("leak", b)):  # fmt: skip
                cols[key].append(val)
    return Library(**{k: np.array(v) for k, v in cols.items()})


def build_library(
    factory: Factory,
    n: int,
    seed: int,
    ws: Sequence[float] = DEFAULT_WS,
    n_zones: int = 100,
    drift: float = 0.0,
) -> Library:
    """``n`` random simulated leaks (seeded) as a table; regenerate, never commit it."""
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(factory, n, rng, drift=drift)
    return library_from_scenarios(factory, scenarios, rng, ws, n_zones)
