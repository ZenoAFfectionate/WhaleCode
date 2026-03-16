"""Observability Module

Provides TraceLogger for recording Agent execution traces:
- JSONL format: Machine-readable, supports streaming analysis
- HTML format: Human-readable, visual audit interface
"""

from .trace_logger import TraceLogger

__all__ = ["TraceLogger"]