"""``tabpfn-check``: verify your TabPFN API key and see what experiments cost.

Run ``uv run tabpfn-check`` once per laptop. It

1. loads ``.env`` and checks that ``TABPFN_TOKEN`` is set (never prints it),
2. prints your token usage,
3. asks the server for free cost quotes (only shapes are sent, nothing is spent),
4. runs one tiny real classification (60 train x 4 features, 20 test rows),
5. prints your usage again, so the difference is the cost of step 4.

The network-facing part sits behind ``Backend`` so tests use a fake.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from tabpfn_spatial.env import MissingTokenError, load_env, require_token


@dataclass(frozen=True)
class Shape:
    """Dataset size used for a cost quote."""

    train_rows: int
    test_rows: int
    columns: int

    def __str__(self) -> str:
        """Human-readable size, e.g. ``1,000 train + 1,000 test x 10 cols``."""
        return f"{self.train_rows:,} train + {self.test_rows:,} test x {self.columns} cols"


# Small call (hits the per-call minimum), mid-size call, and a large batch of queries.
QUOTE_SHAPES = (Shape(100, 100, 5), Shape(10_000, 10_000, 20), Shape(1_000, 100_000, 10))


@dataclass(frozen=True)
class Quote:
    """Server-side token estimate for one shape."""

    shape: Shape
    tokens: int


@dataclass(frozen=True)
class CheckResult:
    """Outcome of ``run_check``."""

    accuracy: float
    quotes: tuple[Quote, ...]


class Backend(Protocol):
    """What the check needs from TabPFN; ``ClientBackend`` is the real one."""

    def usage(self) -> str:
        """Human-readable token usage."""
        ...

    def quote(self, shape: Shape) -> int:
        """Estimated tokens for one predict call of this shape (free)."""
        ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """Fit a classifier and return predicted labels for ``X_test``."""
        ...


class ClientBackend:
    """``tabpfn-client`` cloud backend. Imported lazily so tests never touch it."""

    def usage(self) -> str:
        """Usage string from the server."""
        from tabpfn_client import get_api_usage

        return get_api_usage()

    def quote(self, shape: Shape) -> int:
        """Ask the server for a predict quote; only the array shapes are sent."""
        from tabpfn_client import estimate_cost

        zeros = np.float32(0)  # broadcast views: O(1) memory whatever the shape
        train = np.broadcast_to(zeros, (shape.train_rows, shape.columns))
        test = np.broadcast_to(zeros, (shape.test_rows, shape.columns))
        return estimate_cost(train, test).estimated_cost

    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """One real fit + predict on the TabPFN cloud (spends a few tokens)."""
        from tabpfn_client import TabPFNClassifier

        return np.asarray(TabPFNClassifier().fit(X, y).predict(X_test))


def toy_task(
    seed: int = 0, n_train: int = 60, n_test: int = 20, n_features: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two well-separated Gaussian blobs; returns ``X, y, X_test, y_test``.

    Both classes always appear in the training set (labels alternate).
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    labels = np.arange(n) % 2
    X = rng.normal(size=(n, n_features)) + 3.0 * labels[:, None]
    order = np.concatenate([np.arange(n_train), n_train + rng.permutation(n_test)])
    X, labels = X[order], labels[order]
    return X[:n_train], labels[:n_train], X[n_train:], labels[n_train:]


def run_check(backend: Backend, say: Callable[[str], None] = print) -> CheckResult:
    """Run usage, quotes, one tiny fit, usage again; report each step via ``say``."""
    say(f"Usage before: {backend.usage()}")
    quotes = tuple(Quote(s, backend.quote(s)) for s in QUOTE_SHAPES)
    for q in quotes:
        say(f"Quote (free): {q.shape} -> {q.tokens:,} tokens per predict call")
    X, y, X_test, y_test = toy_task()
    accuracy = float((backend.fit_predict(X, y, X_test) == y_test).mean())
    say(f"Tiny fit on the cloud: accuracy {accuracy:.0%} on {len(y_test)} test rows")
    say(f"Usage after:  {backend.usage()}")
    return CheckResult(accuracy, quotes)


def main(env_file: Path | None = None, backend: Backend | None = None) -> int:
    """Entry point for ``uv run tabpfn-check``; returns the process exit code."""
    load_env(env_file)
    try:
        require_token()
    except MissingTokenError as err:
        print(err)
        return 1
    print("TABPFN_TOKEN found (not shown).")
    result = run_check(backend or ClientBackend())
    print("OK: key works." if result.accuracy >= 0.9 else "WARNING: unexpectedly low accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
