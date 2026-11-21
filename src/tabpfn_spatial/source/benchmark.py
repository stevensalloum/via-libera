"""Benchmark leak locators at each moment after the alarm, on held-out simulated leaks.

    uv run python -m tabpfn_spatial.source.benchmark --backend baseline   # offline k-NN
    uv run python -m tabpfn_spatial.source.benchmark --backend cloud      # + TabPFN

Every method sees the same leaks at the same moments (0, 5, ..., 30 s after the alarm):
``error_m`` (most likely cell to the true leak), ``p_true_zone``, ``top1`` / ``top3`` zone
accuracy and ``ece``. Classifier methods locate all (leak, moment) rows in **one**
batched call. Baselines: the weighted sensor centroid and the prior alone (the likeliest
fitting, no sensors).
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.metrics import ece
from tabpfn_spatial.env import load_env
from tabpfn_spatial.sim import Factory, sample_scenarios
from tabpfn_spatial.source.baseline import WeightedCentroidLocator, localisation_error
from tabpfn_spatial.source.library import (
    ALARM_WINDOW,
    DEFAULT_WS,
    Zones,
    leak_zones,
    library_from_scenarios,
    noisy_series,
)
from tabpfn_spatial.source.tabpfn_locator import TabPFNLocator


@dataclass(frozen=True)
class HeldOut:
    """Test readings, one per (leak, moment after its alarm)."""

    readings: list[SensorReadings]
    sources: np.ndarray  # (n, 2)
    w: np.ndarray  # (n,) seconds since alarm
    t_alarm: np.ndarray  # (n,)


def held_out(factory: Factory, n: int, seed: int, ws: Sequence[float] = DEFAULT_WS) -> HeldOut:
    """Readings for ``n`` seeded test leaks at each moment ``ws`` after their alarm."""
    scenarios = sample_scenarios(factory, n, np.random.default_rng(seed))
    lib = library_from_scenarios(factory, scenarios, np.random.default_rng(seed + 1), ws)
    times, values, wind = noisy_series(
        factory, scenarios, np.random.default_rng(seed + 1), ALARM_WINDOW + max(ws)
    )
    rs = []
    for b, t_a, w in zip(lib.leak, lib.t_alarm, lib.w, strict=True):
        j = int(np.searchsorted(times, t_a + w + 1e-9))
        rs.append(
            SensorReadings(factory.sensors.copy(), times[:j], values[b, :, :j], wind[b, j - 1])
        )
    return HeldOut(rs, lib.source, lib.w, lib.t_alarm)


def map_to_zones(factory: Factory, zones: Zones, maps: Sequence[SourceEstimate]) -> np.ndarray:
    """(n, n_zones) probability mass of each map on each zone's cells (renormalised)."""
    member = np.zeros((factory.ny * factory.nx, zones.n_zones))
    for xy, z in zip(zones.sites, zones.site_zone, strict=True):
        iy, ix = factory.cell_of(xy)
        member[iy * factory.nx + ix, z] = 1.0
    p = np.stack([m.prob.ravel() for m in maps]) @ member
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def source_metrics(
    factory: Factory, zones: Zones, maps: Sequence[SourceEstimate], sources: np.ndarray
) -> dict[str, float]:
    """Mean error (m), P(true zone), top-1 / top-3 zone accuracy and ECE."""
    p = map_to_zones(factory, zones, maps)
    true = zones.zone_of(sources)
    top3 = np.argsort(p, axis=1)[:, ::-1][:, :3]
    errs = [localisation_error(factory, m, s) for m, s in zip(maps, sources, strict=True)]
    return {
        "error_m": float(np.mean(errs)),
        "p_true_zone": float(p[np.arange(len(true)), true].mean()),
        "top1": float(np.mean(p.argmax(axis=1) == true)),
        "top3": float(np.mean((top3 == true[:, None]).any(axis=1))),
        "ece": ece(p, true),
    }


def method_maps(
    factory: Factory, h: HeldOut, backend: str, n_train: int, seed: int
) -> tuple[dict[str, list[SourceEstimate]], dict[str, float]]:
    """Maps per method for every held-out row, and seconds per batched call."""
    prior = SourceEstimate(factory.leak_prior)
    maps = {
        "prior_only": [prior] * len(h.readings),
        "weighted_centroid": [WeightedCentroidLocator().locate(factory, r) for r in h.readings],
    }
    seconds = {}
    methods = [("knn_same_table", "baseline")] + (
        [(f"tabpfn_{backend}", backend)] if backend != "baseline" else []
    )
    for name, b in methods:
        loc = TabPFNLocator(b, n_train=n_train, seed=seed).fit(factory)
        t0 = time.perf_counter()
        maps[name] = loc.locate_many(factory, h.readings)
        seconds[name] = time.perf_counter() - t0
    return maps, seconds


def by_moment(factory: Factory, h: HeldOut, maps: dict[str, list[SourceEstimate]]) -> dict:
    """{method: {w: metrics}} for each moment after the alarm."""
    zones = leak_zones(factory)
    out: dict[str, dict[float, dict[str, float]]] = {}
    for name, ms in maps.items():
        out[name] = {}
        for w in np.unique(h.w):
            sel = np.flatnonzero(h.w == w)
            out[name][float(w)] = source_metrics(
                factory, zones, [ms[i] for i in sel], h.sources[sel]
            )
    return out


def main() -> int:
    """CLI; see the module docstring."""
    p = argparse.ArgumentParser(description="Benchmark leak locators after the alarm.")
    p.add_argument("--backend", default="baseline", choices=["baseline", "cloud", "local"])
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--train-seed", type=int, default=12_345)
    p.add_argument("--test-seed", type=int, default=777)
    a = p.parse_args()
    load_env()
    f = Factory.default()
    h = held_out(f, a.n_test, a.test_seed)
    maps, seconds = method_maps(f, h, a.backend, a.n_train, a.train_seed)
    for name, rows in by_moment(f, h, maps).items():
        for w, m in rows.items():
            print(f"{name:>20} w={w:4.0f}s: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    print("seconds per batched locate_many:", {k: round(v, 2) for k, v in seconds.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
