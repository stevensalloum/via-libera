"""End-to-end example: a full map from sparse readings, in one TabPFN call.

A hidden field over a 10 m x 10 m room (two obstacles and a wall gradient) is read at
``n_readings`` random spots; the model predicts every grid cell with uncertainty.
Swap ``hidden_field`` for real sensor rows to reuse the pipeline.

    uv run spatial-example --backend baseline   # offline, instant
    uv run spatial-example --backend cloud      # TabPFN-3.5, about 10-30k tokens
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabpfn_spatial.core.grid import Grid2D
from tabpfn_spatial.core.metrics import coverage, crps
from tabpfn_spatial.core.models import make_regressor, pick_backend
from tabpfn_spatial.core.viz import plot_map

ROOM = Grid2D((0.0, 10.0), (0.0, 10.0), 2, 2)  # only ranges are used for sampling


def hidden_field(xy: np.ndarray) -> np.ndarray:
    """Occupancy-like field in [0, 1]: two soft obstacles plus a rising wall. O(n)."""
    x, y = xy[:, 0], xy[:, 1]
    blob = lambda cx, cy, r: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2))  # noqa: E731
    field = 0.9 * blob(3.0, 7.0, 1.2) + 0.7 * blob(7.0, 3.0, 0.8) + 0.3 * (x / 10) ** 3
    return np.clip(field, 0.0, 1.0)


@dataclass(frozen=True)
class Report:
    """Scores of one run (lower CRPS is better; coverage90 should be near 0.9)."""

    backend: str
    crps: float
    coverage90: float
    figure: Path


def run_example(
    backend: str = "auto",
    n_readings: int = 200,
    resolution: int = 60,
    noise: float = 0.03,
    seed: int = 0,
    out_dir: Path = Path("outputs"),
) -> Report:
    """Sample readings, predict the whole grid in one call, score and plot."""
    rng = np.random.default_rng(seed)
    readings = rng.uniform(0, 10, size=(n_readings, 2))
    values = hidden_field(readings) + rng.normal(scale=noise, size=n_readings)
    grid = Grid2D(ROOM.x_range, ROOM.y_range, resolution, resolution)
    chosen = pick_backend(backend)
    dist = make_regressor(chosen).fit(readings, values).predict_dist(grid.points())
    truth = hidden_field(grid.points())
    figure = plot_map(
        grid,
        dist,
        out_dir / f"example_map_{chosen}.png",
        truth=truth,
        readings=readings,
        title=f"{n_readings} readings -> {grid.size:,} cells ({chosen})",
    )
    return Report(chosen, crps(dist, truth), coverage(dist, truth, 0.9), figure)


def main() -> int:
    """CLI: ``uv run spatial-example [--backend ...] [--readings N] [--resolution R]``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="auto", choices=["auto", "cloud", "local", "baseline"])
    p.add_argument("--readings", type=int, default=200)
    p.add_argument("--resolution", type=int, default=60)
    args = p.parse_args()
    r = run_example(args.backend, args.readings, args.resolution)
    print(f"{r.backend}: CRPS {r.crps:.4f}, 90 % coverage {r.coverage90:.0%} -> {r.figure}")
    return 0
