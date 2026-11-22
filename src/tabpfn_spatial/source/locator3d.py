"""Horizontal source and release-rate locator trained on 3D detector readings."""

import numpy as np

from tabpfn_spatial.contracts import SensorReadings
from tabpfn_spatial.core.models import make_classifier, make_regressor
from tabpfn_spatial.sim import Factory, sample_scenarios
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.source.library import leak_zones
from tabpfn_spatial.source.tabpfn_locator import TabPFNLocator, _ready
from tabpfn_spatial.source.vertical3d import PipeLeak, clean_readings, height_rows


class TabPFNLocator3D(TabPFNLocator):
    """Return horizontal and rate estimates from a 3D plant's full detector layout.

    Fit explicitly on a Plant3D before calling ``locate(plant.f, readings)``.
    Pair this with HeightLocator for the separate height quantiles. Simulated
    3D training is slower than the old 2D library; fit once, before an alarm.
    """

    def fit(self, plant: Plant3D) -> "TabPFNLocator3D":
        """Build the source library using all the plant's 3D detector readings."""
        if not isinstance(plant, Plant3D):
            raise ValueError("fit the 3D locator with a Plant3D")
        rng = np.random.default_rng(self.seed)
        scenarios = sample_scenarios(plant.f, self.n_train, rng)
        sites, weights = plant.leak_sites
        idx = plant.grid.index(sites)
        valid = ~plant.solid[tuple(idx.T)]
        sites, weights = sites[valid], weights[valid]
        if not len(sites) or weights.sum() <= 0:
            raise ValueError("3D plant has no open leak sites")
        picks = rng.choice(len(sites), self.n_train, p=weights / weights.sum())
        leaks = [
            PipeLeak(tuple(sites[i]), sc.rate, tuple(sc.wind), 0)
            for sc, i in zip(scenarios, picks, strict=True)
        ]
        detectors, _ = plant.detectors
        times, clean = clean_readings(plant, leaks, {"det": detectors})
        lib = height_rows(times, clean["det"], leaks, np.random.default_rng(self.seed + 1), self.ws)
        if not len(lib.X):
            raise ValueError("3D training leaks did not trigger an alarm")
        self.zones = leak_zones(plant.f, self.n_zones)
        xy = np.asarray([leaks[i].xyz[:2] for i in lib.leak])
        zones = self.zones.zone_of(xy)
        rates = np.asarray([leaks[i].rate for i in lib.leak])
        self.model = self.classifier or make_classifier(self.backend)
        self.model.fit(_ready(lib.X, self.model), zones)
        self.rate_model = None
        if self.estimate_rate:
            self.rate_model = self.regressor or make_regressor(self.backend)
            self.rate_model.fit(_ready(lib.X, self.rate_model), np.log(rates))
        self._spread = self._zone_to_grid(plant.f)
        self._factory = plant.f
        self._detectors = detectors.copy()
        return self

    def _validate_3d(self, factory: Factory, readings: SensorReadings) -> None:
        if self._factory is not factory:
            raise ValueError("fit the 3D locator on this plant first")
        if (readings.heights is None or readings.positions.shape != self._detectors[:, :2].shape
                or readings.heights.shape != self._detectors[:, 2].shape
                or len(readings.values) != len(self._detectors)
                or not np.allclose(readings.positions, self._detectors[:, :2])
                or not np.allclose(readings.heights, self._detectors[:, 2])):  # fmt: skip
            raise ValueError("readings must match the fitted plant's 3D detectors in order")

    def predict_all(self, factory: Factory, readings):
        """Predict horizontal positions and rates from all 3D detectors."""
        for r in readings:
            self._validate_3d(factory, r)
        return super().predict_all(factory, readings)

    def locate(self, factory: Factory, readings: SensorReadings):
        """Predict from the full 3D sensor stream, including a mast-triggered alarm."""
        self._validate_3d(factory, readings)
        return super().locate(factory, readings)
