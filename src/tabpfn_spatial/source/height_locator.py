"""Leak height on a 3D plant: the source model's height part for ``SourceEstimate.height``.

Trains, like ``TabPFNLocator``, on a simulated table of the plant: ``n_train`` leaks on the
plant's vertical pipes (``source.vertical3d``), simulated with Steven's 3D gas
(``sim.gas3d``), read by the plant's own detectors (``Plant3D.detectors``, the mast rule),
one row per leak and moment after its alarm (the locator's row). A regressor (TabPFN by
default) answers with 19 height quantiles (``DEFAULT_LEVELS``), clipped to the pipes'
height range. Before any alarm: the plant's prior (uniform over its vertical leak sites).

Use it next to ``TabPFNLocator`` (x, y, size): ``est = locator.locate(...)`` then
``est = heights.attach(est, readings)``. Readings must come from the same plant's detectors
(``Run3D.readings``), in the same order.

Results (``source.height_study3d``, default plant, 600 training leaks, cloud): mean height
error 0.17-0.26 m from the alarm on (masts 1 m beside the pipe up to masts on it), against
0.76 m for a guess and 0.17 m for the best the 1 m voxels allow.

Cost: ``fit`` = ``n_train`` 3D simulations (~0.6 s each on the default plant; ``workers``
processes in parallel; ``cache`` keeps them) + one upload; ``height`` = one ``predict``
call for all readings (cloud: >= 10,000 tokens).
"""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.core.dist import DEFAULT_LEVELS
from tabpfn_spatial.core.models import Regressor, make_regressor
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.source.features import alarm_time, moment_row
from tabpfn_spatial.source.library import DEFAULT_WS
from tabpfn_spatial.source.tabpfn_locator import _ready
from tabpfn_spatial.source.vertical3d import (
    clean_readings,
    height_rows,
    sample_pipe_leaks,
    vertical_sites,
)


class HeightLocator:
    """Height quantiles (m) of a leak on a 3D plant's vertical pipes."""

    def __init__(
        self,
        backend: str = "cloud",
        n_train: int = 600,
        seed: int = 100,
        ws: Sequence[float] = DEFAULT_WS,
        regressor: Regressor | None = None,
        workers: int = 1,
        cache: Path | None = None,
    ) -> None:
        """Configure the table and the model; nothing is simulated until ``fit``.

        ``regressor`` overrides ``backend`` (tests); ``cache`` is an ``.npz`` for the
        simulated readings of the training leaks.
        """
        self.backend, self.n_train, self.seed, self.ws = backend, n_train, seed, tuple(ws)
        self.regressor, self.workers, self.cache = regressor, workers, cache

    def fit(self, plant: Plant3D) -> "HeightLocator":
        """Simulate the training leaks of ``plant`` and fit the height model."""
        det, _ = plant.detectors
        leaks = sample_pipe_leaks(plant, self.n_train, np.random.default_rng(self.seed))
        times, clean = clean_readings(
            plant, leaks, {"det": det}, workers=self.workers, cache=self.cache
        )
        lib = height_rows(times, clean["det"], leaks, np.random.default_rng(self.seed + 1), self.ws)
        self.model = self.regressor or make_regressor(self.backend)
        self.model.fit(_ready(lib.X, self.model), lib.z)
        self.sites = vertical_sites(plant)[0][:, 2]
        self.z_range = (float(self.sites.min()), float(self.sites.max()))
        self.n_detectors = len(det)
        return self

    def prior(self) -> np.ndarray:
        """(q,) height quantiles before any reading: uniform over the vertical leak sites."""
        return np.quantile(self.sites, DEFAULT_LEVELS)

    def height(self, readings: Sequence[SensorReadings]) -> np.ndarray:
        """(n, q) height quantiles at ``DEFAULT_LEVELS``; the prior where nothing alarmed."""
        out = np.tile(self.prior(), (len(readings), 1))
        X, which = [], []
        for i, r in enumerate(readings):
            if len(r.values) != self.n_detectors:
                raise ValueError(
                    f"{len(r.values)} readings, the plant has {self.n_detectors} detectors"
                )
            t_a = alarm_time(r.times, r.values)
            if t_a is not None:
                X.append(moment_row(r.times, r.values, r.wind, t_a))
                which.append(i)
        if which:
            dist = self.model.predict_dist(_ready(np.array(X), self.model))
            q = np.stack([dist.quantile(lv) for lv in DEFAULT_LEVELS], axis=1)
            out[which] = np.sort(np.clip(q, *self.z_range), axis=1)
        return out

    def attach(self, est: SourceEstimate, readings: SensorReadings) -> SourceEstimate:
        """``est`` with its ``height`` and ``height_levels`` filled in."""
        h = self.height([readings])[0]
        return replace(est, height=h, height_levels=np.asarray(DEFAULT_LEVELS, dtype=float))
