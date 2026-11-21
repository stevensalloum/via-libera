"""Optional bounded batches for regressors with per-request row limits."""

import numpy as np

from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.core.models import Regressor


class ChunkedRegressor:
    """Split a prediction into ordered requests; each request may incur cloud cost."""

    def __init__(self, model: Regressor, max_rows: int = 10_000) -> None:
        """Wrap a regressor with a positive per-request row cap."""
        if not isinstance(max_rows, (int, np.integer)) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        self.model, self.max_rows = model, int(max_rows)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChunkedRegressor":
        """Fit the underlying model once."""
        self.model.fit(X, y)
        return self

    def predict_dist(self, X: np.ndarray) -> QuantileDist:
        """Concatenate distributions in input order. O(rows * quantiles) output memory."""
        rows = np.asarray(X)
        if rows.ndim != 2 or len(rows) == 0:
            raise ValueError("prediction input must be a nonempty two-dimensional array")
        parts = [
            self.model.predict_dist(rows[i : i + self.max_rows])
            for i in range(0, len(rows), self.max_rows)
        ]
        levels = parts[0].levels
        if any(not np.array_equal(part.levels, levels) for part in parts):
            raise ValueError("prediction batches returned inconsistent quantile levels")
        has_mode = parts[0].mode is not None
        if any((part.mode is not None) != has_mode for part in parts):
            raise ValueError("prediction batches returned inconsistent modes")
        mode = np.concatenate([part.mode for part in parts]) if has_mode else None
        return QuantileDist(
            levels,
            np.concatenate([part.values for part in parts]),
            np.concatenate([part.mean for part in parts]),
            mode,
        )
