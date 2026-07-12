"""Common request/result objects for benchmark execution environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol, Sequence


EvalStatus = Literal["passed", "failed", "timeout", "security_blocked", "error"]
EvalVisibility = Literal["public", "private", "hidden"]


@dataclass(frozen=True)
class EvalRequest:
    """A normalized request for one benchmark evaluator run."""

    cwd: Path
    timeout_s: int
    code: Optional[str] = None
    command: Optional[Sequence[str]] = None
    stdin: str = ""
    env: Optional[Dict[str, str]] = None
    limits: Any = None
    artifact_store: Any = None
    artifact_stem: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    visibility: EvalVisibility = "hidden"

    def __post_init__(self) -> None:
        if self.code is None and self.command is None:
            raise ValueError("EvalRequest requires either code or command.")
        if self.timeout_s <= 0:
            raise ValueError("EvalRequest.timeout_s must be positive.")
        object.__setattr__(self, "cwd", Path(self.cwd))


@dataclass(frozen=True)
class EvalResult:
    """A normalized result from one benchmark evaluator run."""

    passed: bool
    status: EvalStatus
    stdout: str = ""
    stderr: str = ""
    feedback: str = ""
    returncode: int = 0
    elapsed_s: float = 0.0
    timeout_s: int = 0
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()


class BenchmarkExecutionEnvironment(Protocol):
    """Protocol implemented by benchmark execution backends."""

    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Run a benchmark evaluation request and return a normalized result."""
