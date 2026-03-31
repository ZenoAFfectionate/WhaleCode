"""Context engineering primitives exposed by the package.

This module intentionally avoids eager imports so lightweight consumers such as
``HistoryManager`` do not also trigger tokenizer initialization or prompt file
loading performed by heavier context components.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "ContextBuilder",
    "ContextCompactor",
    "ContextConfig",
    "ContextPacket",
    "HistoryManager",
    "ObservationTruncator",
    "TokenCounter",
]

_EXPORTS = {
    "ContextBuilder": (".builder", "ContextBuilder"),
    "ContextCompactor": (".compactor", "ContextCompactor"),
    "ContextConfig": (".builder", "ContextConfig"),
    "ContextPacket": (".builder", "ContextPacket"),
    "HistoryManager": (".history", "HistoryManager"),
    "ObservationTruncator": (".truncator", "ObservationTruncator"),
    "TokenCounter": (".token_counter", "TokenCounter"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
