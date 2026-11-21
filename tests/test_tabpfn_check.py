"""Tests for the TabPFN key check, with a fake backend (no network, no tokens spent)."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tabpfn_spatial.env import TOKEN_VAR
from tabpfn_spatial.tabpfn_check import QUOTE_SHAPES, Shape, main, run_check, toy_task


class FakeBackend:
    """Records calls; predicts the true toy labels via a nearest-centroid rule."""

    def __init__(self) -> None:
        self.usage_calls = 0
        self.quoted: list[Shape] = []

    def usage(self) -> str:
        self.usage_calls += 1
        return f"used {self.usage_calls}"

    def quote(self, shape: Shape) -> int:
        self.quoted.append(shape)
        return shape.train_rows + shape.test_rows

    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        centroids = np.stack([X[y == c].mean(axis=0) for c in (0, 1)])
        return np.argmin(((X_test[:, None] - centroids[None]) ** 2).sum(-1), axis=1)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_toy_task_shapes_and_labels(seed: int) -> None:
    X, y, X_test, y_test = toy_task(seed)
    assert X.shape == (60, 4) and X_test.shape == (20, 4)
    assert set(np.unique(y)) == {0, 1} and set(np.unique(y_test)) <= {0, 1}


def test_run_check_reports_everything() -> None:
    backend = FakeBackend()
    lines: list[str] = []
    result = run_check(backend, say=lines.append)
    assert backend.usage_calls == 2
    assert backend.quoted == list(QUOTE_SHAPES)
    assert result.accuracy >= 0.9
    assert [q.tokens for q in result.quotes] == [s.train_rows + s.test_rows for s in QUOTE_SHAPES]
    assert any("used 1" in line for line in lines) and any("used 2" in line for line in lines)


def test_main_without_token_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    assert main(env_file=tmp_path / ".env") == 1
    assert TOKEN_VAR in capsys.readouterr().out


def test_main_never_prints_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-very-secret-value"
    monkeypatch.setenv(TOKEN_VAR, secret)
    assert main(env_file=tmp_path / ".env", backend=FakeBackend()) == 0
    assert secret not in capsys.readouterr().out
