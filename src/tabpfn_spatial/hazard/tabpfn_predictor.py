"""Probabilistic hazard forecasts from simulated training rows and source uncertainty."""

import warnings

import numpy as np

from tabpfn_spatial.contracts import (
    LEL,
    TOXIC_LIMIT,
    UEL,
    HazardMap,
    SensorReadings,
    SourceEstimate,
    SourceLocator,
)
from tabpfn_spatial.core.models import Regressor, make_regressor
from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.hazard.geometry import path_features
from tabpfn_spatial.hazard.library import DEFAULT_CONTEXT_SEED, training_rows
from tabpfn_spatial.hazard.sensor_history import sensor_history_features, time_context_features
from tabpfn_spatial.hazard.source_selection import SourceSelection, select_source_cells
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

    def __init__(
        self,
        backend: str = "cloud",
        top_k: int = 3,
        model: Regressor | None = None,
        *,
        include_geometry: bool = False,
        source_mass: float | None = None,
        open_cells_only: bool = False,
        include_time_context: bool = False,
        include_sensor_history: bool = False,
    ) -> None:
        """Choose backend, source selection, optional geometry and open-cell queries."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if source_mass is not None and (not np.isfinite(source_mass) or not 0 < source_mass <= 1):
            raise ValueError("source_mass must be in (0, 1]")
        self.backend, self.top_k, self.model = backend, top_k, model
        self.include_geometry = include_geometry
        self.source_mass = source_mass
        self.open_cells_only = open_cells_only
        self.include_time_context = include_time_context or include_sensor_history
        self.include_sensor_history = include_sensor_history
        self.last_selection: SourceSelection | None = None

    def fit(
        self,
        factory: Factory,
        n_scenarios: int = 100,
        cells_per_scenario: int = 20,
        times: np.ndarray | None = None,
        seed: int = DEFAULT_CONTEXT_SEED,
        *,
        source_locator: SourceLocator | None = None,
        observation_times: np.ndarray | None = None,
        diffusivity_range: tuple[float, float] | None = None,
    ) -> "TabPFNHazardPredictor":
        """Fit on simulated concentrations; optionally use the live source locator.

        The supplied locator must already be ready to predict for this factory.
        A locator using cloud inference may spend credits once per training scenario.
        ``diffusivity_range`` optionally varies spreading per simulated scenario;
        the model does not receive the unknown diffusivity as a feature.
        """
        x, y = training_rows(
            factory,
            n_scenarios,
            cells_per_scenario,
            times,
            seed,
            include_geometry=self.include_geometry,
            locator=source_locator,
            include_time_context=self.include_time_context,
            include_sensor_history=self.include_sensor_history,
            observation_times=observation_times,
            diffusivity_range=diffusivity_range,
        )
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
        if not len(times):
            raise ValueError("times must be nonempty")
        self.last_selection = None
        selection = select_source_cells(source.prob, factory.walls, self.top_k, self.source_mass)
        self.last_selection = selection
        if selection.target_reached is False:
            warnings.warn(
                f"Selected cells retained {selection.retained_mass:.1%} of source probability, "
                f"below the requested {self.source_mass:.1%} target.",
                RuntimeWarning,
                stacklevel=2,
            )
        indices, weights = selection.indices, selection.weights
        rate = (
            1.0
            if source.rate is None or source.rate_levels is None
            else float(np.interp(0.5, source.rate_levels, source.rate))
        )
        all_xy = factory.centres().reshape(-1, 2)
        queried = (~factory.walls).ravel() if self.open_cells_only else np.ones(len(all_xy), bool)
        xy = all_xy[queried]
        features = []
        for idx in indices:
            geometry = path_features(factory, all_xy[idx], xy) if self.include_geometry else None
            history = (
                sensor_history_features(readings, all_xy[idx], xy, readings.t_obs)
                if self.include_sensor_history
                else None
            )
            for t in times:
                row = cell_features(factory, all_xy[idx], readings.wind, rate, float(t), xy)
                columns = [row]
                if geometry is not None:
                    columns.append(geometry)
                if self.include_time_context:
                    columns.append(time_context_features(readings.t_obs, float(t), len(xy)))
                if history is not None:
                    columns.append(history)
                features.append(np.column_stack(columns) if len(columns) > 1 else row)
        dist = self.model.predict_dist(np.concatenate(features))
        toxic = 1 - _cdf(dist.values, dist.levels, TOXIC_LIMIT)
        fire = _cdf(dist.values, dist.levels, UEL) - _cdf(dist.values, dist.levels, LEL)
        shape = (len(indices), len(times), len(xy))
        full_toxic = np.zeros((len(times), len(all_xy)))
        full_fire = np.zeros_like(full_toxic)
        full_toxic[:, queried] = np.tensordot(weights, toxic.reshape(shape), axes=1)
        full_fire[:, queried] = np.tensordot(weights, fire.reshape(shape), axes=1)
        p_toxic = full_toxic.reshape(-1, factory.ny, factory.nx) * ~factory.walls
        p_fire = full_fire.reshape(-1, factory.ny, factory.nx) * factory.ignition
        return HazardMap(times, p_toxic, p_fire)
