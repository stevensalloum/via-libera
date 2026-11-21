"""TabPFN leak locator, updated every sensor sample after the first alarm.

``fit`` simulates a table of seeded leaks, one row per (leak, moment after its alarm),
and fits two models on the same 31 columns: a classifier (answer = leak zone) and a
regressor (answer = log release rate). ``locate`` finds the alarm in the readings, builds
one row for "now", turns ``predict_proba`` into a map (each zone's probability spread
evenly over its leak-site cells) and the regressor's quantiles into ``rate``. Before any
alarm both are the plant's prior. Call it every 5 s: each call is independent, the
row's arrival times carry the history.

Cost: fit = one simulation of ``n_train`` leaks + upload; ``locate_many`` = one
``predict_proba`` and one ``predict`` call for all rows, sent in parallel (cloud: >=
10,000 tokens per call). Table rows = n_train x len(ws), kept under TabPFN's ~10,000.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.dist import DEFAULT_LEVELS
from tabpfn_spatial.core.models import (
    BaselineClassifier,
    BaselineRegressor,
    Classifier,
    Regressor,
    make_classifier,
    make_regressor,
)
from tabpfn_spatial.sim import Factory
from tabpfn_spatial.source.features import alarm_time, moment_row
from tabpfn_spatial.source.library import DEFAULT_WS, Zones, build_library, leak_zones

_NAN_FILL = -1.0  # the k-NN stand-ins cannot read NaN (dead sensors); TabPFN can
RATE_RANGE = (0.5, 2.0)  # the simulator draws release rates log-uniformly in this range


def prior_rate_quantiles(
    levels: Sequence[float] | None = DEFAULT_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """``(levels, quantiles)`` of the release rate before any reading (log-uniform)."""
    lv = np.asarray(DEFAULT_LEVELS if levels is None else levels, dtype=float)
    lo, hi = np.log(RATE_RANGE)
    return lv, np.exp(lo + lv * (hi - lo))


class TabPFNLocator:
    """``contracts.SourceLocator`` backed by TabPFN (or any) models over leak zones."""

    def __init__(
        self,
        backend: str = "cloud",
        n_train: int = 1200,
        seed: int = 12_345,
        ws: Sequence[float] = DEFAULT_WS,
        n_zones: int = 100,
        estimate_rate: bool = True,
        classifier: Classifier | None = None,
        regressor: Regressor | None = None,
    ) -> None:
        """Configure the table and the models; nothing is simulated until ``fit``.

        ``seed`` drives the training leaks (evaluate on other seeds); ``ws`` = moments
        after the alarm (s) put in the table. ``classifier`` / ``regressor`` override
        ``backend`` (tests). ``estimate_rate=False`` skips the size model.
        """
        self.backend, self.n_train, self.seed = backend, n_train, seed
        self.ws, self.n_zones, self.estimate_rate = tuple(ws), n_zones, estimate_rate
        self.classifier, self.regressor = classifier, regressor
        self._factory: Factory | None = None

    def fit(self, factory: Factory) -> "TabPFNLocator":
        """Simulate the table for ``factory`` and fit the zone (and size) model on it."""
        lib = build_library(factory, self.n_train, self.seed, self.ws, self.n_zones)
        self.zones: Zones = leak_zones(factory, self.n_zones)
        self.model = self.classifier or make_classifier(self.backend)
        self.model.fit(_ready(lib.X, self.model), lib.zone)
        self.rate_model = None
        if self.estimate_rate:
            self.rate_model = self.regressor or make_regressor(self.backend)
            self.rate_model.fit(_ready(lib.X, self.rate_model), np.log(lib.rate))
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

    def rows(self, readings: Sequence[SensorReadings]) -> tuple[np.ndarray, list[int]]:
        """Feature rows for the readings that have alarmed, and their positions."""
        X, which = [], []
        for i, r in enumerate(readings):
            t_a = alarm_time(r.times, r.values)
            if t_a is not None:
                X.append(moment_row(r.times, r.values, r.wind, t_a))
                which.append(i)
        return np.array(X), which

    def _zone_rows(self, X: np.ndarray) -> np.ndarray:
        """(n, n_zones) zone probabilities for feature rows (one model call)."""
        p = np.asarray(self.model.predict_proba(_ready(X, self.model)), dtype=float)
        full = np.zeros((len(X), self.zones.n_zones))
        full[:, np.asarray(self.model.classes_, dtype=int)] = p
        return full / full.sum(axis=1, keepdims=True)

    def _rate_rows(self, X: np.ndarray) -> np.ndarray:
        """(n, q) release-rate quantiles at ``DEFAULT_LEVELS`` (one model call)."""
        dist = self.rate_model.predict_dist(_ready(X, self.rate_model))
        return np.exp(np.stack([dist.quantile(lv) for lv in DEFAULT_LEVELS], axis=1))

    def predict_all(
        self, factory: Factory, readings: Sequence[SensorReadings]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Zone probabilities (n, n_zones) and rate quantiles (n, q) or None.

        Rows without an alarm get the prior. The two model calls run in parallel.
        """
        if self._factory is not factory:
            self.fit(factory)
        prior = self._spread @ factory.leak_prior.ravel()
        zones = np.tile(prior / prior.sum(), (len(readings), 1))
        rates = np.tile(prior_rate_quantiles()[1], (len(readings), 1)) if self.rate_model else None
        X, which = self.rows(readings)
        if which:
            with ThreadPoolExecutor(max_workers=2) as pool:
                z = pool.submit(self._zone_rows, X)
                r = pool.submit(self._rate_rows, X) if self.rate_model else None
                zones[which] = z.result()
                if r is not None:
                    rates[which] = r.result()
        return zones, rates

    def zone_proba(self, factory: Factory, readings: Sequence[SensorReadings]) -> np.ndarray:
        """(n, n_zones) zone probabilities; prior mass per zone where no alarm yet."""
        return self.predict_all(factory, readings)[0]

    def locate_many(
        self, factory: Factory, readings: Sequence[SensorReadings]
    ) -> list[SourceEstimate]:
        """Maps (and leak sizes) for many moments or emergencies, one call per model."""
        zones, rates = self.predict_all(factory, readings)
        levels = np.asarray(DEFAULT_LEVELS) if rates is not None else None
        out = []
        for i, r in enumerate(readings):
            alarmed = alarm_time(r.times, r.values) is not None
            grid = (zones[i] @ self._spread).reshape(factory.ny, factory.nx)
            prob = grid if alarmed else factory.leak_prior.copy()
            out.append(SourceEstimate(prob, None if rates is None else rates[i], levels))
        return out

    def locate(self, factory: Factory, readings: SensorReadings) -> SourceEstimate:
        """Probability map and leak size now (fits on first use)."""
        return self.locate_many(factory, [readings])[0]


def _ready(X: np.ndarray, model: object) -> np.ndarray:
    """Fill NaN only for the k-NN stand-ins; TabPFN handles missing values itself."""
    baseline = isinstance(model, BaselineClassifier | BaselineRegressor)
    return np.nan_to_num(X, nan=_NAN_FILL) if baseline else X
