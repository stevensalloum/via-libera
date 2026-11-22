"""One interface over TabPFN cloud, TabPFN local and numpy baselines.

``make_regressor(backend)`` returns an object with ``fit(X, y)`` and
``predict_dist(X) -> QuantileDist``; ``make_classifier`` one with ``fit``,
``predict_proba`` and ``predict``. Backends:

- ``"cloud"``: ``tabpfn_client`` (TabPFN-3.5 on Prior Labs GPUs, spends tokens, guarded
  by a ``CostGuard``).
- ``"local"``: ``tabpfn`` + PyTorch on this machine (``uv sync --extra local``); the key
  only unlocks the weights; device is picked automatically (CUDA, MPS, CPU).
- ``"baseline"``: k-nearest-neighbour models in numpy; offline, instant, and the honest
  comparison for every pitch number.
- ``"auto"``: local if installed with a GPU, else cloud if a token is set, else baseline.
"""

import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, Protocol

import numpy as np

from tabpfn_spatial.core.budget import CostGuard
from tabpfn_spatial.core.dist import DEFAULT_LEVELS, QuantileDist
from tabpfn_spatial.env import TOKEN_VAR, load_env

Backend = Literal["cloud", "local", "baseline"]
_CHUNK = 4096  # baseline test rows per distance block: memory O(chunk * n_train)


class Regressor(Protocol):
    """Probabilistic regressor."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regressor":
        """Store or learn from the training rows."""
        ...

    def predict_dist(self, X: np.ndarray) -> QuantileDist:
        """Predictive distribution for each row of ``X``."""
        ...


class Classifier(Protocol):
    """Probabilistic classifier."""

    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        """Store or learn from the training rows."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, k) class probabilities in the order of ``classes_``."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely class per row."""
        ...


# ------------------------------------------------------------------ TabPFN adapters --
RETRIES = 8  # rate-limited calls are retried this many times before giving up


def _retry_s(err: Exception) -> float | None:
    """Seconds to wait if ``err`` is the cloud's rate limit (HTTP 429), else None."""
    msg = str(err)
    if "429" not in msg and "Rate limit" not in msg:
        return None
    import re

    m = re.search(r"Retry in (\d+(?:\.\d+)?)\s*s", msg)
    return float(m.group(1)) if m else 10.0


def with_retry(fn, *args, **kwargs):
    """Call ``fn``; on a rate-limit reply wait as told (plus jitter) and try again.

    The cloud allows 60 predict requests per minute per account; parallel studies can
    exceed it. Other errors are raised at once.
    """
    import random
    import time

    for attempt in range(RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as err:
            wait = _retry_s(err)
            if wait is None or attempt == RETRIES:
                raise
            time.sleep(wait + random.uniform(0.5, 3.0) * (attempt + 1))
    raise AssertionError("unreachable")


class TabPFNRegressorAdapter:
    """Wraps ``tabpfn.TabPFNRegressor`` or ``tabpfn_client.TabPFNRegressor``."""

    def __init__(
        self,
        estimator_cls: type,
        levels: Sequence[float] = DEFAULT_LEVELS,
        guard: CostGuard | None = None,
        **kwargs: Any,
    ) -> None:
        """``kwargs`` go to the TabPFN constructor (e.g. ``n_estimators``)."""
        self.estimator = estimator_cls(**kwargs)
        self.levels = tuple(levels)
        self.guard = guard
        self._train_rows = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNRegressorAdapter":
        """Fit (cloud: uploads the training rows)."""
        self._train_rows = len(X)
        with_retry(self.estimator.fit, X, y)
        return self

    def predict_dist(self, X: np.ndarray) -> QuantileDist:
        """One call for all rows of ``X``; the guard quotes it first (cloud only)."""
        X = np.asarray(X)
        if self.guard is not None:
            self.guard.check(self._train_rows, len(X), X.shape[1])
        out = with_retry(self.estimator.predict, X, output_type="main", quantiles=list(self.levels))
        return QuantileDist.from_main(out, self.levels)


class TabPFNClassifierAdapter:
    """Wraps ``tabpfn.TabPFNClassifier`` or ``tabpfn_client.TabPFNClassifier``."""

    def __init__(self, estimator_cls: type, guard: CostGuard | None = None, **kwargs: Any):
        """``kwargs`` go to the TabPFN constructor."""
        self.estimator = estimator_cls(**kwargs)
        self.guard = guard
        self._train_rows = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNClassifierAdapter":
        """Fit (cloud: uploads the training rows)."""
        self._train_rows = len(X)
        with_retry(self.estimator.fit, X, y)
        self.classes_ = np.asarray(self.estimator.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """One call for all rows; the guard quotes it first (cloud only)."""
        X = np.asarray(X)
        if self.guard is not None:
            self.guard.check(self._train_rows, len(X), X.shape[1])
        return np.asarray(with_retry(self.estimator.predict_proba, X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely class (reuses ``predict_proba``: one call, not two)."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ------------------------------------------------------------------------ baselines --
def _knn(X_train: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Indices (n, k) of the k nearest training rows (standardised features).

    Time O(n * n_train * d), memory O(chunk * n_train).
    """
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    A, B = (X_train - mu) / sd, (np.asarray(X, dtype=float) - mu) / sd
    blocks = []
    for start in range(0, len(B), _CHUNK):
        d2 = ((B[start : start + _CHUNK, None, :] - A[None]) ** 2).sum(-1)
        blocks.append(np.argpartition(d2, k - 1, axis=1)[:, :k])
    return np.concatenate(blocks) if blocks else np.empty((0, k), dtype=int)


class BaselineRegressor:
    """k-NN mean with a Gaussian spread from the neighbours' values."""

    def __init__(self, k: int = 10, levels: Sequence[float] = DEFAULT_LEVELS) -> None:
        """``k`` neighbours (capped at the training size)."""
        self.k, self.levels = k, tuple(levels)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineRegressor":
        """Store the rows; O(1)."""
        self._X, self._y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        return self

    def predict_dist(self, X: np.ndarray) -> QuantileDist:
        """Neighbour mean +- neighbour std times normal quantiles."""
        k = min(self.k, len(self._y))
        ys = self._y[_knn(self._X, X, k)]
        mu = ys.mean(axis=1)
        sd = np.maximum(ys.std(axis=1), 1e-3 * (self._y.std() + 1e-12))
        z = np.array([NormalDist().inv_cdf(p) for p in self.levels])
        return QuantileDist(np.asarray(self.levels), mu[:, None] + sd[:, None] * z, mean=mu)


class BaselineClassifier:
    """k-NN class frequencies with add-one smoothing."""

    def __init__(self, k: int = 10) -> None:
        """``k`` neighbours (capped at the training size)."""
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        """Store the rows; O(1)."""
        self._X = np.asarray(X, dtype=float)
        self.classes_, self._codes = np.unique(np.asarray(y), return_inverse=True)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed neighbour class frequencies, rows sum to 1."""
        k = min(self.k, len(self._codes))
        codes = self._codes[_knn(self._X, X, k)]
        n_cls = len(self.classes_)
        counts = np.stack([(codes == c).sum(axis=1) for c in range(n_cls)], axis=1) + 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most frequent neighbour class."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ------------------------------------------------------------------------ factories --
def _local_available() -> bool:
    """True if ``tabpfn`` is installed and a CUDA or Apple MPS GPU is usable."""
    if importlib.util.find_spec("tabpfn") is None:
        return False
    import torch

    return torch.cuda.is_available() or torch.backends.mps.is_available()


def pick_backend(requested: str = "auto", env_file: Path | str | None = "auto") -> Backend:
    """Resolve ``"auto"``; ``env_file=None`` skips reading ``.env`` (tests)."""
    if requested in ("cloud", "local", "baseline"):
        return requested  # type: ignore[return-value]
    if requested != "auto":
        raise ValueError(f"unknown backend {requested!r}")
    if env_file is not None:
        load_env(None if env_file == "auto" else Path(env_file))
    if _local_available():
        return "local"
    return "cloud" if os.environ.get(TOKEN_VAR, "").strip() else "baseline"


_DEFAULT = object()


def _guard(backend: Backend, guard: object) -> CostGuard | None:
    if guard is _DEFAULT:
        return CostGuard() if backend == "cloud" else None
    return guard  # type: ignore[return-value]


def _tabpfn_module(backend: Backend) -> Any:
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # never open a login page silently
    load_env()
    return importlib.import_module("tabpfn_client" if backend == "cloud" else "tabpfn")


def make_regressor(backend: str = "auto", guard: object = _DEFAULT, **kwargs: Any) -> Regressor:
    """Probabilistic regressor for ``backend``; ``kwargs`` go to the estimator."""
    chosen = pick_backend(backend)
    if chosen == "baseline":
        return BaselineRegressor(**kwargs)
    cls = _tabpfn_module(chosen).TabPFNRegressor
    return TabPFNRegressorAdapter(cls, guard=_guard(chosen, guard), **kwargs)


def make_classifier(backend: str = "auto", guard: object = _DEFAULT, **kwargs: Any) -> Classifier:
    """Probabilistic classifier for ``backend``; ``kwargs`` go to the estimator."""
    chosen = pick_backend(backend)
    if chosen == "baseline":
        return BaselineClassifier(**kwargs)
    cls = _tabpfn_module(chosen).TabPFNClassifier
    return TabPFNClassifierAdapter(cls, guard=_guard(chosen, guard), **kwargs)
