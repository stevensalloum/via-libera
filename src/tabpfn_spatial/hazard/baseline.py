"""Baseline hazard predictor and the shared hazard score.

Baseline: take the single most likely source cell and the median release rate, rerun the
simulator forward with the measured wind, threshold the result. Probabilities are 0 or 1
(no uncertainty), which is what TabPFN must beat. Cost: one simulation.
"""

import numpy as np

from tabpfn_spatial.contracts import (
    LEL,
    TOXIC_LIMIT,
    UEL,
    HazardMap,
    SensorReadings,
    SourceEstimate,
)
from tabpfn_spatial.sim import Factory, Scenario, simulate


def hazard_from_fields(factory: Factory, fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean toxic and fire masks from concentration fields (..., ny, nx)."""
    toxic = fields > TOXIC_LIMIT
    fire = (fields >= LEL) & (fields <= UEL) & factory.ignition
    return toxic, fire


class ForwardSimPredictor:
    """Most likely source + median rate -> deterministic forward simulation."""

    def predict(
        self,
        factory: Factory,
        readings: SensorReadings,
        source: SourceEstimate,
        times: np.ndarray,
    ) -> HazardMap:
        """0/1 probability maps at ``times``."""
        iy, ix = source.top_cells(1)[0]
        rate = 1.0
        if source.rate is not None and source.rate_levels is not None:
            rate = float(np.interp(0.5, source.rate_levels, source.rate))
        sc = Scenario(factory.centre(int(iy), int(ix)), rate, readings.wind)
        fields = simulate(factory, [sc], np.asarray(times, dtype=float))[0]
        toxic, fire = hazard_from_fields(factory, fields)
        return HazardMap(np.asarray(times, dtype=float), toxic.astype(float), fire.astype(float))


def brier(prob: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and the true 0/1 outcome."""
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(truth, dtype=float)) ** 2))
