"""Context Engineering Module

Provides context engineering capabilities for the HelloAgents framework:
- ContextBuilder: GSSC pipeline (Gather-Select-Structure-Compress)
- HistoryManager: History management and compression
- ObservationTruncator: Tool output truncation
- TokenCounter: Token counter (caching + incremental calculation)
- Compactor: Conversation compression and integration
- NotesManager: Structured notes management
- ContextObserver: Observability and metrics tracking
"""

from .builder import ContextBuilder, ContextConfig, ContextPacket
from .compactor import ContextCompactor
from .history import HistoryManager
from .truncator import ObservationTruncator
from .token_counter import TokenCounter

__all__ = [
    "ContextBuilder",
    "ContextCompactor",
    "ContextConfig",
    "ContextPacket",
    "HistoryManager",
    "ObservationTruncator",
    "TokenCounter",
]