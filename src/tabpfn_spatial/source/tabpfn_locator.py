"""TabPFN leak locator: simulation-based inference in one forward pass.

``fit`` simulates a library of seeded leaks (features -> zone) and fits a classifier on
it; ``locate`` turns ``predict_proba`` for the new readings into a probability map, each
zone's probability spread evenly over its leak-site cells. The map is naturally
multimodal ("valve A *or* B"). Use ``locate_many`` to batch many emergencies into one call
(cloud: one call costs >= 10,000 tokens, so batch benchmarks).

Cost: fit = one simulation of ``n_train`` leaks (~0.4 s per 100) + upload;
locate_many = one ``predict_proba`` call for all rows. TabPFN context: n_train <= 10,000.
"""

from collections.abc import Sequence

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.models import Classifier, make_classifier
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.source.features import feature_matrix
from tabpfn_spatial.source.library import Zones, leak_zones, make_library


class TabPFNLocator:
    """``contracts.SourceLocator`` backed by a TabPFN (or any) classifier over leak zones."""

    def __init__(
        self,
        backend: str = "cloud",
        n_train: int = 3000,
        seed: int = 12_345,
        n_zones: int = 100,
        classifier: Classifier | None = None,
    ) -> None:
        """``seed`` drives the training leaks; evaluate on other seeds.

        ``classifier`` overrides ``make_classifier(backend)`` (tests inject fakes).
        """
        self.backend, self.n_train, self.seed, self.n_zones = backend, n_train, seed, n_zones
        self.classifier = classifier
        self._factory: Factory | None = None

    def fit(self, factory: Factory) -> "TabPFNLocator":
        """Simulate the training library for ``factory`` and fit the classifier."""
        X, zone = make_library(factory, self.n_train, self.seed, self.n_zones)
        self.zones: Zones = leak_zones(factory, self.n_zones)
        self.model = self.classifier or make_classifier(self.backend)
        self.model.fit(X, zone)
        self._spread = self._zone_to_grid(factory)
        self._factory = factory
        return self

    def _zone_to_grid(self, factory: Factory) -> np.ndarray:
        """(n_zones, ny * nx) matrix spreading each zone evenly over its site cells."""
        m = np.zeros((self.zones.n_zones, factory.ny * factory.nx))
        for xy, z in zip(self.zones.sites, self.zones.site_zone, strict=True):
            iy, ix = factory.cell_of(xy)
            m[z, iy * factory.nx + ix] = 1.0
        return m / m.sum(axis=1, keepdims=True)

    def zone_proba(self, factory: Factory, readings: Sequence[SensorReadings]) -> np.ndarray:
        """(n, n_zones) zone probabilities in zone order, one batched model call."""
        if self._factory is not factory:
            self.fit(factory)
        X = np.vstack([feature_matrix(r.times, r.values[None], r.wind[None]) for r in readings])
        p = np.asarray(self.model.predict_proba(X), dtype=float)
        full = np.zeros((len(X), self.zones.n_zones))
        full[:, np.asarray(self.model.classes_, dtype=int)] = p  # zones unseen in training: 0
        return full / full.sum(axis=1, keepdims=True)

    def locate_many(
        self, factory: Factory, readings: Sequence[SensorReadings]
    ) -> list[SourceEstimate]:
        """Probability maps for many emergencies from one batched call."""
        grids = self.zone_proba(factory, readings) @ self._spread
        return [SourceEstimate(g.reshape(factory.ny, factory.nx)) for g in grids]

    def locate(self, factory: Factory, readings: SensorReadings) -> SourceEstimate:
        """Probability map of the leak location (fits on first use)."""
        return self.locate_many(factory, [readings])[0]
