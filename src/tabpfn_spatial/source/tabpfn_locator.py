"""TabPFN leak locator, updated every sensor sample after the first alarm.

``fit`` simulates a table of seeded leaks, one row per (leak, moment after its alarm),
and fits a classifier on it (answer = leak zone). ``locate`` finds the alarm in the
readings, builds one 31-column row for "now" and turns ``predict_proba`` into a map,
each zone's probability spread evenly over its leak-site cells. Before any alarm the map
is the plant's prior (``factory.leak_prior``). Call it every 5 s: each call is
independent, the row's arrival times carry the history.

Cost: fit = one simulation of ``n_train`` leaks + upload; ``locate_many`` = one
``predict_proba`` call for all rows (cloud: >= 10,000 tokens per call, batch benchmarks).
Table rows = n_train x len(ws), kept under TabPFN's ~10,000-row context.
"""

from collections.abc import Sequence

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.models import BaselineClassifier, Classifier, make_classifier
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.source.features import alarm_time, moment_row
from tabpfn_spatial.source.library import DEFAULT_WS, Zones, build_library, leak_zones

_NAN_FILL = -1.0  # the k-NN stand-in cannot read NaN (dead sensors); TabPFN can


class TabPFNLocator:
    """``contracts.SourceLocator`` backed by a TabPFN (or any) classifier over leak zones."""

    def __init__(
        self,
        backend: str = "cloud",
        n_train: int = 1200,
        seed: int = 12_345,
        ws: Sequence[float] = DEFAULT_WS,
        n_zones: int = 100,
        classifier: Classifier | None = None,
    ) -> None:
        """Configure the table and the model; nothing is simulated until ``fit``.

        ``seed`` drives the training leaks (evaluate on other seeds); ``ws`` = moments
        after the alarm (s) put in the table. ``classifier`` overrides ``backend`` (tests).
        """
        self.backend, self.n_train, self.seed = backend, n_train, seed
        self.ws, self.n_zones, self.classifier = tuple(ws), n_zones, classifier
        self._factory: Factory | None = None

    def fit(self, factory: Factory) -> "TabPFNLocator":
        """Simulate the table for ``factory`` and fit the classifier on it."""
        lib = build_library(factory, self.n_train, self.seed, self.ws, self.n_zones)
        self.zones: Zones = leak_zones(factory, self.n_zones)
        self.model = self.classifier or make_classifier(self.backend)
        self.model.fit(self._ready(lib.X), lib.zone)
        self._spread = self._zone_to_grid(factory)
        self._factory = factory
        return self

    def _ready(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN only for the k-NN stand-in; TabPFN handles missing values itself."""
        return np.nan_to_num(X, nan=_NAN_FILL) if isinstance(self.model, BaselineClassifier) else X

    def _zone_to_grid(self, factory: Factory) -> np.ndarray:
        """(n_zones, ny * nx) matrix spreading each zone evenly over its site cells."""
        m = np.zeros((self.zones.n_zones, factory.ny * factory.nx))
        for xy, z in zip(self.zones.sites, self.zones.site_zone, strict=True):
            iy, ix = factory.cell_of(xy)
            m[z, iy * factory.nx + ix] = 1.0
        return m / m.sum(axis=1, keepdims=True)

    def rows(self, readings: Sequence[SensorReadings]) -> tuple[np.ndarray, list[int]]:
        """Feature rows for the readings that have alarmed, and their positions."""
        X, which = [], []
        for i, r in enumerate(readings):
            t_a = alarm_time(r.times, r.values)
            if t_a is not None:
                X.append(moment_row(r.times, r.values, r.wind, t_a))
                which.append(i)
        return np.array(X), which

    def zone_proba(self, factory: Factory, readings: Sequence[SensorReadings]) -> np.ndarray:
        """(n, n_zones) zone probabilities; prior mass per zone where no alarm yet."""
        if self._factory is not factory:
            self.fit(factory)
        prior = self._spread @ factory.leak_prior.ravel()
        out = np.tile(prior / prior.sum(), (len(readings), 1))
        X, which = self.rows(readings)
        if which:
            p = np.asarray(self.model.predict_proba(self._ready(X)), dtype=float)
            full = np.zeros((len(which), self.zones.n_zones))
            full[:, np.asarray(self.model.classes_, dtype=int)] = p
            out[which] = full / full.sum(axis=1, keepdims=True)
        return out

    def locate_many(
        self, factory: Factory, readings: Sequence[SensorReadings]
    ) -> list[SourceEstimate]:
        """Maps for many moments or emergencies from one batched model call."""
        probs = self.zone_proba(factory, readings)
        out = []
        for r, p in zip(readings, probs, strict=True):
            alarmed = alarm_time(r.times, r.values) is not None
            grid = (p @ self._spread).reshape(factory.ny, factory.nx)
            out.append(SourceEstimate(grid if alarmed else factory.leak_prior.copy()))
        return out

    def locate(self, factory: Factory, readings: SensorReadings) -> SourceEstimate:
        """Probability map of the leak location now (fits on first use)."""
        return self.locate_many(factory, [readings])[0]
