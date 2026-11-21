"""Generate reproducible training examples from the gas simulator."""

import numpy as np

from tabpfn_spatial.contracts import SourceLocator
from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.hazard.geometry import path_features
from tabpfn_spatial.hazard.sensor_history import sensor_history_features, time_context_features
from tabpfn_spatial.hazard.source_selection import select_source_cells
from tabpfn_spatial.sim import Factory, readings, sample_scenarios, simulate
from tabpfn_spatial.source import WeightedCentroidLocator

DEFAULT_CONTEXT_SEED = 1001


def training_rows(
    factory: Factory,
    n_scenarios: int = 100,
    cells_per_scenario: int = 20,
    times: np.ndarray | None = None,
    seed: int = DEFAULT_CONTEXT_SEED,
    *,
    include_geometry: bool = False,
    locator: SourceLocator | None = None,
    include_time_context: bool = False,
    include_sensor_history: bool = False,
    observation_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate examples using estimated sources and measured, noisy wind.

    Sample a top-three source hypothesis using its weight and label it with the
    true scenario's concentration. Forecasts never precede their observations.
    Defaults cover decision times 5-60 s and horizons 0-120 s. The default
    locator is a proxy until the live source model is available.

    Cost: simulator time plus O(scenarios * times * cells * grid diameter).
    """
    if n_scenarios < 1 or cells_per_scenario < 1:
        raise ValueError("training requires scenarios, cells and times")
    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not len(times) or np.any(~np.isfinite(times)):
            raise ValueError("times must be a nonempty finite vector")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("times must be non-negative and strictly increasing")
    if observation_times is not None:
        observation_times = np.asarray(observation_times, dtype=float)
        if (
            observation_times.ndim != 1
            or not len(observation_times)
            or np.any(~np.isfinite(observation_times))
            or np.any(observation_times < 0)
            or np.any(np.diff(observation_times) <= 0)
        ):
            raise ValueError("observation times must be finite, non-negative and increasing")
        if times is not None and observation_times[0] > times[0]:
            raise ValueError("an observation time must not follow the first forecast")
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(factory, n_scenarios, rng)
    locator = locator if locator is not None else WeightedCentroidLocator()
    open_cells = np.argwhere(~factory.walls)
    features, targets = [], []
    for sc in scenarios:
        if times is None:
            choices = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 60.0]
            t_obs = float(rng.choice(choices if observation_times is None else observation_times))
            horizons = np.sort(np.r_[0.0, rng.choice(np.arange(10.0, 121.0, 10.0), 3, False)])
            forecast_times = t_obs + horizons
        else:
            available = (
                np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 60.0])
                if observation_times is None
                else observation_times
            )
            t_obs = float(rng.choice(available[available <= times[0]]))
            forecast_times = times
        obs = readings(factory, sc, t_obs, rng)
        estimate = locator.locate(factory, obs)
        selection = select_source_cells(estimate.prob, factory.walls, 3, None)
        source_index = int(rng.choice(selection.indices, p=selection.weights))
        source_xy = factory.centres().reshape(-1, 2)[source_index]
        rate = (
            1.0
            if estimate.rate is None or estimate.rate_levels is None
            else float(np.interp(0.5, estimate.rate_levels, estimate.rate))
        )
        fields = simulate(factory, [sc], forecast_times)[0]
        chosen = open_cells[rng.integers(len(open_cells), size=cells_per_scenario)]
        xy = factory.centres()[chosen[:, 0], chosen[:, 1]]
        geometry = path_features(factory, source_xy, xy) if include_geometry else None
        history = (
            sensor_history_features(obs, source_xy, xy, obs.t_obs)
            if include_sensor_history
            else None
        )
        for j, t in enumerate(forecast_times):
            row = cell_features(factory, source_xy, obs.wind, rate, float(t), xy)
            columns = [row]
            if geometry is not None:
                columns.append(geometry)
            if include_time_context or include_sensor_history:
                columns.append(time_context_features(obs.t_obs, float(t), len(xy)))
            if history is not None:
                columns.append(history)
            features.append(np.column_stack(columns) if len(columns) > 1 else row)
            targets.append(fields[j, chosen[:, 0], chosen[:, 1]])
    return np.concatenate(features), np.concatenate(targets)
