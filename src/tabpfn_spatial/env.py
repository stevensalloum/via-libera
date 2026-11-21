"""Environment loading: read the gitignored ``.env`` and expose the TabPFN API token.

Both ``tabpfn-client`` (cloud inference, spends tokens) and the local ``tabpfn`` package
(uses the key to unlock the newer model weights) read the token from ``TABPFN_TOKEN``.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

TOKEN_VAR = "TABPFN_TOKEN"


class MissingTokenError(RuntimeError):
    """Raised when ``TABPFN_TOKEN`` is unset or blank."""


def load_env(path: Path | None = None) -> bool:
    """Load ``path`` (default: nearest ``.env`` from the cwd up) into ``os.environ``.

    Values already set in the shell win over the file. Returns True if a file was read.
    """
    target = path if path is not None else find_dotenv(usecwd=True)
    if not target or not Path(target).is_file():
        return False
    return load_dotenv(target, override=False)


def require_token() -> str:
    """Return the stripped TabPFN token, or raise ``MissingTokenError``."""
    token = os.environ.get(TOKEN_VAR, "").strip()
    if not token:
        raise MissingTokenError(
            f"{TOKEN_VAR} is not set. Copy .env.example to .env and add your API key."
        )
    return token
