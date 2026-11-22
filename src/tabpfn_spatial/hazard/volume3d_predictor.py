"""Experimental height-aware hazard models for explicit 3D source hypotheses.

This is not the team's shared 2D hazard interface and does not invent source
height. Rayan's eventual 3D output needs an agreed adapter before integration.
"""

from collections.abc import Iterator

import numpy as np

from tabpfn_spatial.contracts import LEL, TOXIC_LIMIT, UEL
from tabpfn_spatial.core.models import Regressor, make_regressor
from tabpfn_spatial.hazard.tabpfn_predictor import _cdf
from tabpfn_spatial.hazard.volume3d import HazardVolume, VolumeGrid, simulate_volume, validate_times


def features_3d(
    grid: VolumeGrid, source: np.ndarray, wind: np.ndarray, rate: float, t: float, xyz: np.ndarray
) -> np.ndarray:
    """Features per 3D point; LOS samples supplied voxels, O(points * grid diameter)."""
    source, xyz, wind = np.asarray(source, float), np.asarray(xyz, float), np.asarray(wind, float)
    grid.indices(source[None])
    grid.indices(xyz)
    if wind.shape != (2,) or np.any(~np.isfinite(wind)):
        raise ValueError("wind must be a finite horizontal (u, v) vector")
    speed = float(np.linalg.norm(wind))
    direction = wind / speed if speed > 1e-12 else np.array([1.0, 0.0])
    relative = xyz - source
    downwind = relative[:, :2] @ direction
    crosswind = relative[:, :2] @ np.array([-direction[1], direction[0]])
    steps = max(2, int(np.ceil(np.linalg.norm(grid.solid.shape))) + 1)
    path = source + np.linspace(0.0, 1.0, steps)[None, :, None] * relative[:, None, :]
    idx = grid.indices(path.reshape(-1, 3)).reshape(len(xyz), steps, 3)
    open_fraction = (~grid.solid[idx[..., 0], idx[..., 1], idx[..., 2]]).mean(axis=1)
    return np.column_stack([
        downwind, crosswind, np.linalg.norm(relative, axis=1), xyz[:, 2], relative[:, 2],
        np.full(len(xyz), t), np.full(len(xyz), speed), np.full(len(xyz), rate), open_fraction,
    ])  # fmt: skip


def training_rows_3d(
    grid: VolumeGrid,
    source_candidates: np.ndarray,
    n_scenarios: int = 100,
    cells_per_scenario: int = 20,
    times: np.ndarray | None = None,
    seed: int = 1001,
    diffusivity_range: tuple[float, float] = (0.8, 0.8),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labelled 3D examples from synthetic, explicitly located source voxels.

    Each scenario has a distinct synthetic rate and horizontal wind. It uses
    the true synthetic source, not a simulated estimate from Rayan. The source
    mask is mandatory; no pipe or leak heights are inferred. Caller must use
    different scenarios for evaluation.
    """
    source_candidates = np.asarray(source_candidates, dtype=bool)
    if source_candidates.shape != grid.solid.shape or np.any(source_candidates & grid.solid):
        raise ValueError("source candidates must be open voxels matching the grid")
    candidates = grid.centres()[source_candidates]
    if not len(candidates) or n_scenarios < 1 or cells_per_scenario < 1:
        raise ValueError("training requires source candidates, scenarios and cells")
    lo, hi = diffusivity_range
    if not np.isfinite([lo, hi]).all() or lo <= 0 or hi < lo:
        raise ValueError("diffusivity range must be positive, finite and ordered")
    times = validate_times(np.array([5.0, 15.0, 30.0, 60.0]) if times is None else times)
    rng = np.random.default_rng(seed)
    open_xyz = grid.centres()[~grid.solid]
    rows, targets = [], []
    for _ in range(n_scenarios):
        source = candidates[rng.integers(len(candidates))]
        rate = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        speed, angle = rng.uniform(0.2, 1.5), rng.uniform(0, 2 * np.pi)
        wind = speed * np.array([np.cos(angle), np.sin(angle)])
        diffusion = lo if lo == hi else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        cells = open_xyz[rng.integers(len(open_xyz), size=cells_per_scenario)]
        ids = grid.indices(cells)
        fields = simulate_volume(grid, source, rate, wind, times, diffusion)
        for i, t in enumerate(times):
            rows.append(features_3d(grid, source, wind, rate, float(t), cells))
            targets.append(fields[i, ids[:, 0], ids[:, 1], ids[:, 2]])
    return np.concatenate(rows), np.concatenate(targets)


def _inputs(
    grid: VolumeGrid,
    sources: np.ndarray,
    weights: np.ndarray,
    wind: np.ndarray,
    rate: float,
    times: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Validate explicit 3D source hypotheses and select their highest weights."""
    sources, weights, wind = (
        np.asarray(sources, float),
        np.asarray(weights, float),
        np.asarray(wind, float),
    )
    ids = grid.indices(sources)
    if not len(sources) or weights.shape != (len(sources),):
        raise ValueError("provide one weight for each explicit 3D source")
    if np.any(grid.solid[ids[:, 0], ids[:, 1], ids[:, 2]]):
        raise ValueError("source positions must be in open voxels")
    if (np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or not np.isfinite(weights.sum()) or weights.sum() <= 0):  # fmt: skip
        raise ValueError("source weights must be finite, nonnegative and have positive mass")
    if wind.shape != (2,) or np.any(~np.isfinite(wind)):
        raise ValueError("wind must be a finite horizontal (u, v) vector")
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("rate must be finite and nonnegative")
    times = validate_times(times)
    weights = weights / weights.sum()
    order = np.argsort(weights)[::-1][:top_k]
    order = order[weights[order] > 0]
    retained = float(weights[order].sum())
    return sources[order], weights[order] / retained, wind, times, retained


def _feature_batches(
    grid: VolumeGrid,
    sources: np.ndarray,
    wind: np.ndarray,
    rate: float,
    times: np.ndarray,
    xyz: np.ndarray,
    chunk: int,
) -> Iterator[np.ndarray]:
    """Yield bounded batches in source/time/voxel order, without a full feature matrix."""
    pending: list[np.ndarray] = []
    size = 0
    for source in sources:
        for t in times:
            start = 0
            while start < len(xyz):
                count = min(chunk - size, len(xyz) - start)
                pending.append(
                    features_3d(grid, source, wind, rate, float(t), xyz[start : start + count])
                )
                start += count
                size += count
                if size == chunk:
                    yield np.concatenate(pending)
                    pending, size = [], 0
    if pending:
        yield np.concatenate(pending)


class ForwardVolumePredictor:
    """Baseline: simulate the most probable supplied 3D source with fixed diffusivity."""

    def __init__(self, diffusivity: float = 0.8) -> None:
        """Select the simulation's fixed spreading rate."""
        self.diffusivity = diffusivity

    def predict(
        self,
        grid: VolumeGrid,
        sources: np.ndarray,
        weights: np.ndarray,
        wind: np.ndarray,
        rate: float,
        times: np.ndarray,
    ) -> HazardVolume:
        """Return deterministic 3D toxic/fire maps for the best source hypothesis."""
        sources, _, wind, times, _ = _inputs(grid, sources, weights, wind, rate, times, 1)
        fields = simulate_volume(grid, sources[0], rate, wind, times, self.diffusivity)
        toxic = (fields > TOXIC_LIMIT).astype(float)
        fire = ((fields >= LEL) & (fields <= UEL) & grid.ignition).astype(float)
        return HazardVolume(times, toxic, fire)


class TabPFNVolumePredictor:
    """Fit on simulated 3D points and combine supplied 3D source hypotheses."""

    def __init__(self, backend: str = "cloud", top_k: int = 3, batch_size: int = 10000) -> None:
        """Choose inference backend, maximum source hypotheses and query batch size."""
        if (
            not isinstance(top_k, int)
            or not isinstance(batch_size, int)
            or top_k < 1
            or batch_size < 1
        ):
            raise ValueError("top_k and batch_size must be positive")
        self.backend, self.top_k, self.batch_size = backend, top_k, batch_size
        self.model: Regressor | None = None
        self._training_grid: VolumeGrid | None = None
        self.last_retained_mass: float | None = None

    def fit(
        self,
        grid: VolumeGrid,
        source_candidates: np.ndarray,
        n_scenarios: int = 100,
        cells_per_scenario: int = 20,
        times: np.ndarray | None = None,
        seed: int = 1001,
        *,
        diffusivity_range: tuple[float, float] = (0.8, 0.8),
    ) -> "TabPFNVolumePredictor":
        """Prepare simulated training examples; model weights are not retrained."""
        x, y = training_rows_3d(
            grid, source_candidates, n_scenarios, cells_per_scenario, times, seed, diffusivity_range
        )
        return self.fit_examples(grid, x, y)

    def fit_examples(
        self, grid: VolumeGrid, features: np.ndarray, concentrations: np.ndarray
    ) -> "TabPFNVolumePredictor":
        """Fit from labelled nine-feature rows, for example from Steven's 3D simulator."""
        x, y = np.asarray(features, float), np.asarray(concentrations, float)
        if (x.ndim != 2 or x.shape[1] != 9 or not len(x) or y.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(y < 0)):  # fmt: skip
            raise ValueError("training requires finite nine-feature rows and concentrations")
        self.model = make_regressor(self.backend).fit(x, y)
        self._training_grid = VolumeGrid(grid.solid, grid.ignition, grid.cell, grid.dz)
        return self

    def predict(
        self,
        grid: VolumeGrid,
        sources: np.ndarray,
        weights: np.ndarray,
        wind: np.ndarray,
        rate: float,
        times: np.ndarray,
        keep: np.ndarray | None = None,
    ) -> HazardVolume:
        """Predict (time, z, y, x) probabilities from explicit 3D source hypotheses.

        ``keep``: optional (z, y, x) bool mask of the voxels to predict (e.g. breathing
        height for the planner); the others stay 0. Features are unchanged.
        """
        if self.model is None:
            raise ValueError("fit the predictor before predicting")
        trained = self._training_grid
        if (trained is None or grid.cell != trained.cell or grid.dz != trained.dz
                or not np.array_equal(grid.solid, trained.solid)
                or not np.array_equal(grid.ignition, trained.ignition)):  # fmt: skip
            raise ValueError("prediction layout must match the training layout")
        sources, weights, wind, times, self.last_retained_mass = _inputs(
            grid, sources, weights, wind, rate, times, self.top_k
        )
        open_ = ~grid.solid if keep is None else ~grid.solid & np.asarray(keep, bool)
        xyz = grid.centres()[open_]
        chunk = min(self.batch_size, 64) if self.backend == "baseline" else self.batch_size
        toxic, fire = [], []
        for rows in _feature_batches(grid, sources, wind, rate, times, xyz, chunk):
            dist = self.model.predict_dist(rows)
            toxic.append(1 - _cdf(dist.values, dist.levels, TOXIC_LIMIT))
            fire.append(_cdf(dist.values, dist.levels, UEL) - _cdf(dist.values, dist.levels, LEL))
        shape = (len(sources), len(times), len(xyz))
        p_toxic = np.zeros((len(times), *grid.solid.shape))
        p_fire = np.zeros_like(p_toxic)
        p_toxic[:, open_] = np.tensordot(weights, np.concatenate(toxic).reshape(shape), axes=1)
        p_fire[:, open_] = np.tensordot(weights, np.concatenate(fire).reshape(shape), axes=1)
        p_fire *= grid.ignition
        return HazardVolume(times, p_toxic, p_fire)
