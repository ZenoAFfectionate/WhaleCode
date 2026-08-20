"""Agent-logging facade (建议-7: print → injectable sink).

All library code that historically called ``print`` directly should route
through ``agent_print`` / ``agent_eprint`` so integrators can redirect output
without monkey-patching ``sys.stdout``. The default sinks are plain ``print``,
preserving backward compatibility.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

_AgentPrintFn = Callable[..., None]

_agent_print_fn: _AgentPrintFn = lambda *a, **kw: print(*a, file=sys.stdout, flush=kw.get("flush", False) or kw.pop("flush", False))
_agent_eprint_fn: _AgentPrintFn = lambda *a, **kw: print(*a, file=sys.stderr, flush=kw.get("flush", False) or kw.pop("flush", False))


def agent_print(*args: Any, **kwargs: Any) -> None:
    """Drop-in ``print`` replacement for library code."""
    _agent_print_fn(*args, **kwargs)


def agent_eprint(*args: Any, **kwargs: Any) -> None:
    """Drop-in ``print(..., file=sys.stderr)`` replacement."""
    _agent_eprint_fn(*args, **kwargs)
