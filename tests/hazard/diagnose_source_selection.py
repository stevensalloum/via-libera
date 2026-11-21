"""Offline comparison of source truncation using simulator mixtures, not TabPFN.

Run with ``uv run python tests/hazard/diagnose_source_selection.py``.
"""

import time

import numpy as np

from tabpfn_spatial.contracts import TOXIC_LIMIT
from tabpfn_spatial.sim import (
    Factory,
    Scenario,
    readings,
    sample_people,
    sample_scenarios,
    simulate,
)
from tabpfn_spatial.source import WeightedCentroidLocator

start = time.perf_counter()
f = Factory.default()
rng = np.random.default_rng(0)
scenarios = sample_scenarios(f, 30, rng)
times = np.arange(70.0, 181.0, 10.0)
truth_times = np.r_[60.0, times]
open_ = ~f.walls
score3, score90, masses, counts, hit3, hit90 = [], [], [], [], [], []
for i, sc in enumerate(scenarios):
    sample_people(f, 30, rng)
    r = readings(f, sc, 60.0, rng)
    p = WeightedCentroidLocator().locate(f, r).prob.ravel()
    order = np.argsort(p)[::-1]
    k = int(np.searchsorted(np.cumsum(p[order]), 0.9) + 1)
    selected = order[:k]
    wind = r.wind
    sources = [
        Scenario(f.centre(*np.unravel_index(index, f.walls.shape)), 1.0, wind) for index in selected
    ]
    truth = simulate(f, [sc], truth_times)[0, 1:, :, :] > TOXIC_LIMIT
    predicted = simulate(f, sources, times) > TOXIC_LIMIT
    weights3 = p[selected[:3]] / p[selected[:3]].sum()
    weights90 = p[selected] / p[selected].sum()
    prob3 = np.tensordot(weights3, predicted[:3], axes=1)
    prob90 = np.tensordot(weights90, predicted, axes=1)
    score3.append(float(np.mean((prob3[:, open_] - truth[:, open_]) ** 2)))
    score90.append(float(np.mean((prob90[:, open_] - truth[:, open_]) ** 2)))
    true_index = np.ravel_multi_index(f.cell_of(sc.source), f.walls.shape)
    hit3.append(true_index in selected[:3])
    hit90.append(true_index in selected)
    masses.append(float(p[selected[:3]].sum()))
    counts.append(k)
    if (i + 1) % 10 == 0:
        print("completed episodes:", i + 1, flush=True)
diff = np.asarray(score90) - np.asarray(score3)
print("episodes:", len(scenarios), "seed: 0; same source maps, measured wind, rate 1.0")
print("top3 mean mass:", np.mean(masses), "mean cells:", 3, "source hits:", sum(hit3))
print("90% mean cells:", np.mean(counts), "source hits:", sum(hit90))
print("toxic Brier open floor, forward-sim top3:", np.mean(score3))
print("toxic Brier open floor, forward-sim 90%:", np.mean(score90))
print("paired Brier difference (90% - top3):", np.mean(diff))
print(
    "episodes 90% wins / ties / loses:",
    int((diff < -1e-12).sum()),
    int((abs(diff) <= 1e-12).sum()),
    int((diff > 1e-12).sum()),
)
print("total seconds both methods and setup:", round(time.perf_counter() - start, 2))
print("cloud credits: 0")
