"""Environment-variable parsing helpers (Q2-5/Q2-6).

Single implementation of tolerant env parsing shared by ``core.config``,
``core.llm_adapters`` and ``tools.builtin.bash`` — previously three
near-identical copies existed. The benchmark package keeps its own copy
(``benchmark/_utils.py``) to stay importable in direct-script-execution
mode without depending on the ``hello_agents`` package root.
"""

from __future__ import annotations

import os

_TRUE_TOKENS = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var; unset/empty/invalid falls back to *default*."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in _TRUE_TOKENS


def env_int(name: str, default: int) -> int:
    """Parse a non-negative int env var; unset/empty/invalid/negative → *default*.

    Negative values are treated as configuration errors and fall back to
    *default* rather than being clamped to 0 — for resource limits 0 means
    "no limit", so silently turning a bad negative value into 0 would
    disable the limit entirely.
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def env_float(name: str, default: float) -> float:
    """Parse a non-negative float env var; unset/empty/invalid/negative → *default*."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default
