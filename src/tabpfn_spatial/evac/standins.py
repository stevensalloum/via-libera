"""Simulated stand-ins for the source and hazard models, until the real ones are linked.

They cheat on purpose: both are built from the true ``Scenario``. ``OracleLocator`` plays
"Rayan's model with an error of about ``sigma`` metres"; ``OracleHazard`` plays "a perfect
hazard model" (the upper bound). Swap in any ``SourceLocator`` / ``HazardPredictor`` from
``contracts`` (e.g. the baselines) without changing the planner.
"""

import numpy as np

from tabpfn_spatial.contracts import HazardMap, SensorReadings, SourceEstimate
from tabpfn_spatial.hazard import hazard_from_fields
from tabpfn_spatial.sim import Factory, Scenario, simulate


class OracleLocator:
    """The true leak, moved by a fresh N(0, sigma) error each call and blurred, on the pipes."""

    def __init__(
        self, scenario: Scenario, sigma: float = 3.0, rng: np.random.Generator | None = None
    ) -> None:
        """``sigma``: typical localisation error in metres (0 = exact)."""
        self.scenario, self.sigma = scenario, sigma
        self.rng = rng or np.random.default_rng()

    def locate(self, factory: Factory, readings: SensorReadings | None) -> SourceEstimate:
        """Gaussian of width max(sigma, cell) times the leak prior; true rate as quantiles."""
        centre = self.scenario.source + self.rng.normal(0.0, self.sigma, 2) * (self.sigma > 0)
        width = max(self.sigma, factory.cell)
        d2 = ((factory.centres() - centre) ** 2).sum(axis=-1)
        prob = (np.exp(-d2 / (2 * width**2)) + 1e-12) * factory.leak_prior
        levels = np.array([0.1, 0.5, 0.9])
        return SourceEstimate(prob / prob.sum(), np.full(3, self.scenario.rate), levels)


class OracleHazard:
    """The true future toxic / fire masks from the simulator (a perfect hazard model)."""

    def __init__(self, scenario: Scenario) -> None:
        """Bound to one leak."""
        self.scenario = scenario

    def predict(
        self,
        factory: Factory,
        readings: SensorReadings | None,
        source: SourceEstimate | None,
        times: np.ndarray,
    ) -> HazardMap:
        """0/1 maps at ``times``. Cost: one simulation up to ``max(times)``."""
        times = np.asarray(times, dtype=float)
        toxic, fire = hazard_from_fields(factory, simulate(factory, [self.scenario], times)[0])
        return HazardMap(times, toxic.astype(float), fire.astype(float))
