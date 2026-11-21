"""Benchmark leak locators on held-out simulated leaks (other seeds than training).

    uv run python -m tabpfn_spatial.source.benchmark --backend baseline   # offline k-NN
    uv run python -m tabpfn_spatial.source.benchmark --backend cloud --evaluate 30

Every method is scored on the same leaks: ``error_m`` (most likely cell to the true leak),
``p_true_zone`` (mean probability on the true zone), ``top1`` / ``top3`` zone accuracy and
``ece`` (calibration of the top zone). Classifier methods locate all test leaks in **one**
batched call. ``--evaluate N`` also runs ``pipeline.evaluate`` (N cloud calls, one per
episode, because the pipeline locates one emergency at a time).
"""

import argparse
from collections.abc import Sequence

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.metrics import ece
from tabpfn_spatial.env import load_env
from tabpfn_spatial.sim import Factory, sample_scenarios
from tabpfn_spatial.source.baseline import WeightedCentroidLocator, localisation_error
from tabpfn_spatial.source.library import Zones, leak_zones, noisy_series
from tabpfn_spatial.source.tabpfn_locator import TabPFNLocator


def held_out(factory: Factory, n: int, seed: int) -> tuple[np.ndarray, list[SensorReadings]]:
    """True sources (n, 2) and noisy readings for ``n`` seeded test leaks."""
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(factory, n, rng)
    times, values, wind = noisy_series(factory, scenarios, rng)
    rs = [
        SensorReadings(factory.sensors.copy(), times, v, w)
        for v, w in zip(values, wind, strict=True)
    ]
    return np.array([s.source for s in scenarios]), rs


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
    return {
        "error_m": float(
            np.mean([localisation_error(factory, m, s) for m, s in zip(maps, sources, strict=True)])
        ),
        "p_true_zone": float(p[np.arange(len(true)), true].mean()),
        "top1": float(np.mean(p.argmax(axis=1) == true)),
        "top3": float(np.mean((top3 == true[:, None]).any(axis=1))),
        "ece": ece(p, true),
    }


def _usage(backend: str) -> str:
    if backend != "cloud":
        return "n/a"
    load_env()
    from tabpfn_client import get_api_usage

    return str(get_api_usage())


def run(
    backend: str, n_train: int, n_test: int, n_zones: int, train_seed: int, test_seed: int
) -> dict[str, dict[str, float]]:
    """Scores per method on the same held-out leaks; prints token usage for the cloud."""
    f = Factory.default()
    zones = leak_zones(f, n_zones)
    sources, rs = held_out(f, n_test, test_seed)
    centroid = WeightedCentroidLocator()
    out = {
        "weighted_centroid": source_metrics(f, zones, [centroid.locate(f, r) for r in rs], sources)
    }
    methods = [("knn_same_library", "baseline")]
    if backend != "baseline":
        methods.append((f"tabpfn_{backend}", backend))
    for name, b in methods:
        loc = TabPFNLocator(b, n_train=n_train, seed=train_seed, n_zones=n_zones).fit(f)
        out[name] = source_metrics(f, zones, loc.locate_many(f, rs), sources)
        guard = getattr(loc.model, "guard", None)
        if guard is not None:
            out[name]["tokens_quoted"] = float(guard.spent)
    return out


def main() -> int:
    """CLI; see the module docstring."""
    p = argparse.ArgumentParser(description="Benchmark leak locators on held-out leaks.")
    p.add_argument("--backend", default="baseline", choices=["baseline", "cloud", "local"])
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-zones", type=int, default=100)
    p.add_argument("--train-seed", type=int, default=12_345)
    p.add_argument("--test-seed", type=int, default=777)
    p.add_argument("--evaluate", type=int, default=0, help="also run pipeline.evaluate(N)")
    a = p.parse_args()
    print("usage before:", _usage(a.backend))
    res = run(a.backend, a.n_train, a.n_test, a.n_zones, a.train_seed, a.test_seed)
    for name, m in res.items():
        print(f"{name:>20}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    if a.evaluate:
        from tabpfn_spatial.pipeline import evaluate

        f = Factory.default()
        loc = TabPFNLocator(a.backend, n_train=a.n_train, seed=a.train_seed, n_zones=a.n_zones)
        print("pipeline baseline:", evaluate(f, a.evaluate, seed=0))
        print("pipeline tabpfn  :", evaluate(f, a.evaluate, seed=0, locator=loc))
    print("usage after: ", _usage(a.backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
