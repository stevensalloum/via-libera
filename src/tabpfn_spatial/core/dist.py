"""Predictive distributions as quantiles, the common output of every regressor."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# 5 %, 10 %, ..., 95 %: symmetric pairs give 10 % ... 90 % central intervals.
DEFAULT_LEVELS: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))


@dataclass(frozen=True)
class QuantileDist:
    """``n`` predictive distributions, each described by ``q`` quantiles.

    ``levels``: (q,) increasing probabilities in (0, 1). ``values``: (n, q) quantiles,
    sorted per row on construction (fixes rare quantile crossing). ``mean``: (n,).
    ``mode``: (n,) or None. Memory O(n q).
    """

    levels: np.ndarray
    values: np.ndarray
    mean: np.ndarray
    mode: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate shapes and sort quantiles per row."""
        levels = np.asarray(self.levels, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if levels.ndim != 1 or np.any(np.diff(levels) <= 0):
            raise ValueError("levels must be strictly increasing")
        if values.ndim != 2 or values.shape[1] != len(levels):
            raise ValueError(f"values must be (n, {len(levels)}), got {values.shape}")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "values", np.sort(values, axis=1))
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=float))

    @classmethod
    def from_main(cls, out: Mapping[str, object], levels: Sequence[float]) -> "QuantileDist":
        """Build from TabPFN's ``predict(output_type="main", quantiles=levels)`` dict."""
        values = np.stack([np.asarray(q, dtype=float) for q in out["quantiles"]], axis=1)
        mode = out.get("mode")
        return cls(
            np.asarray(levels),
            values,
            np.asarray(out["mean"]),
            None if mode is None else np.asarray(mode, dtype=float),
        )

    def __len__(self) -> int:
        """Number of distributions (rows)."""
        return len(self.values)

    def quantile(self, level: float) -> np.ndarray:
        """Values at one of ``levels`` (exact match, else ValueError). O(n)."""
        hits = np.flatnonzero(np.isclose(self.levels, level))
        if not len(hits):
            raise ValueError(f"level {level} not in {self.levels.tolist()}")
        return self.values[:, hits[0]]

    @property
    def median(self) -> np.ndarray:
        """The 50 % quantile."""
        return self.quantile(0.5)

    def interval(self, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        """Central interval holding ``coverage`` mass, e.g. 0.9 -> (5 %, 95 %)."""
        tail = (1 - coverage) / 2
        return self.quantile(tail), self.quantile(1 - tail)

    def width(self, coverage: float = 0.9) -> np.ndarray:
        """Width of the central interval: a per-row uncertainty map."""
        lo, hi = self.interval(coverage)
        return hi - lo
