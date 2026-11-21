"""Generate reproducible training examples from the gas simulator."""

import numpy as np

from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.sim import Factory, sample_scenarios, simulate


def training_rows(
    factory: Factory,
    n_scenarios: int = 100,
    cells_per_scenario: int = 20,
    times: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate features and concentrations; O(scenarios * times * cells) plus simulation."""
    times = np.asarray(times if times is not None else [70.0, 100.0, 140.0, 180.0], dtype=float)
    if n_scenarios < 1 or cells_per_scenario < 1 or len(times) == 0:
        raise ValueError("training requires scenarios, cells and times")
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be non-negative and strictly increasing")
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(factory, n_scenarios, rng)
    fields = simulate(factory, scenarios, times)
    open_cells = np.argwhere(~factory.walls)
    features, targets = [], []
    for i, sc in enumerate(scenarios):
        chosen = open_cells[rng.integers(len(open_cells), size=cells_per_scenario)]
        xy = factory.centres()[chosen[:, 0], chosen[:, 1]]
        for j, t in enumerate(times):
            features.append(cell_features(factory, sc.source, sc.wind, sc.rate, float(t), xy))
            targets.append(fields[i, j, chosen[:, 0], chosen[:, 1]])
    return np.concatenate(features), np.concatenate(targets)
