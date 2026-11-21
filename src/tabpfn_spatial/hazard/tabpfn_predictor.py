"""Probabilistic hazard forecasts from simulated training rows and source uncertainty."""

import numpy as np

from tabpfn_spatial.contracts import (
    LEL,
    TOXIC_LIMIT,
    UEL,
    HazardMap,
    SensorReadings,
    SourceEstimate,
)
from tabpfn_spatial.core.models import Regressor, make_regressor
from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.hazard.library import training_rows
from tabpfn_spatial.sim import Factory


def _cdf(values: np.ndarray, levels: np.ndarray, threshold: float) -> np.ndarray:
    """Interpolate a CDF from quantiles, with zero and one outside their range."""
    idx = np.sum(values <= threshold, axis=1)
    lo = np.clip(idx - 1, 0, len(levels) - 1)
    hi = np.clip(idx, 0, len(levels) - 1)
    rows = np.arange(len(values))
    low, high = values[rows, lo], values[rows, hi]
    frac = np.divide(threshold - low, high - low, out=np.zeros(len(values)), where=high > low)
    cdf = levels[lo] + frac * (levels[hi] - levels[lo])
    return np.where(idx == 0, 0.0, np.where(idx == len(levels), 1.0, cdf))


class TabPFNHazardPredictor:
    """Mix source-conditioned concentration distributions in one prediction per episode."""

    def __init__(self, backend: str = "cloud", top_k: int = 3, model: Regressor | None = None):
        """Choose the backend and maximum number of source hypotheses."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.backend, self.top_k, self.model = backend, top_k, model

    def fit(
        self,
        factory: Factory,
        n_scenarios: int = 100,
        cells_per_scenario: int = 20,
        times: np.ndarray | None = None,
        seed: int = 42,
    ) -> "TabPFNHazardPredictor":
        """Fit one regressor on simulated concentrations; generated rows stay in memory."""
        x, y = training_rows(factory, n_scenarios, cells_per_scenario, times, seed)
        self.model = make_regressor(self.backend).fit(x, y)
        return self

    def predict(
        self,
        factory: Factory,
        readings: SensorReadings,
        source: SourceEstimate,
        times: np.ndarray,
    ) -> HazardMap:
        """Forecast all times and source hypotheses in a single batched model call."""
        if self.model is None:
            raise ValueError("fit the predictor before predicting")
        times = np.asarray(times, dtype=float)
        probs = np.asarray(source.prob, dtype=float)
        if probs.shape != factory.walls.shape or np.any(~np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("source probabilities must be finite, nonnegative and match the grid")
        valid = np.where(factory.walls, 0.0, probs)
        if valid.sum() <= 0 or not len(times):
            raise ValueError("source needs positive open-cell probability and nonempty times")
        indices = np.argsort(valid.ravel())[::-1][: self.top_k]
        indices = indices[valid.ravel()[indices] > 0]
        weights = valid.ravel()[indices]
        weights /= weights.sum()
        rate = (
            1.0
            if source.rate is None or source.rate_levels is None
            else float(np.interp(0.5, source.rate_levels, source.rate))
        )
        xy = factory.centres().reshape(-1, 2)
        features = [
            cell_features(factory, xy[idx], readings.wind, rate, float(t), xy)
            for idx in indices
            for t in times
        ]
        dist = self.model.predict_dist(np.concatenate(features))
        toxic = 1 - _cdf(dist.values, dist.levels, TOXIC_LIMIT)
        fire = _cdf(dist.values, dist.levels, UEL) - _cdf(dist.values, dist.levels, LEL)
        shape = (len(indices), len(times), factory.ny, factory.nx)
        p_toxic = np.tensordot(weights, toxic.reshape(shape), axes=1) * ~factory.walls
        p_fire = np.tensordot(weights, fire.reshape(shape), axes=1) * factory.ignition
        return HazardMap(times, p_toxic, p_fire)
