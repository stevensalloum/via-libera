"""Tests for .env loading and the TabPFN token lookup."""

from pathlib import Path

import pytest

from tabpfn_spatial.env import TOKEN_VAR, MissingTokenError, load_env, require_token


def test_load_env_reads_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"{TOKEN_VAR}=abc123\n")
    assert load_env(env) is True
    assert require_token() == "abc123"


def test_load_env_keeps_existing_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_VAR, "from-shell")
    env = tmp_path / ".env"
    env.write_text(f"{TOKEN_VAR}=from-file\n")
    load_env(env)
    assert require_token() == "from-shell"


def test_load_env_missing_file_is_false(tmp_path: Path) -> None:
    assert load_env(tmp_path / "nope.env") is False


@pytest.mark.parametrize("value", [None, "", "   "])
def test_require_token_missing(value: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    if value is None:
        monkeypatch.delenv(TOKEN_VAR, raising=False)
    else:
        monkeypatch.setenv(TOKEN_VAR, value)
    with pytest.raises(MissingTokenError, match=TOKEN_VAR):
        require_token()


def test_require_token_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_VAR, "  tok \n")
    assert require_token() == "tok"
