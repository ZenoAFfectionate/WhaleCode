"""Mockable Docker benchmark execution environment."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .artifacts import BenchmarkArtifactStore
from .base import BenchmarkExecutionEnvironment, EvalRequest, EvalResult
from .config import BenchmarkRuntimeConfig


@dataclass(frozen=True)
class DockerCommandRunner:
    """Small adapter around subprocess.run for easy unit-test faking."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout: int,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )


class DockerExecutionEnvironment(BenchmarkExecutionEnvironment):
    """Execute already-formed Docker/Compose commands with EvalResult semantics."""

    def __init__(
        self,
        *,
        runner: Optional[DockerCommandRunner] = None,
        artifact_store: Optional[BenchmarkArtifactStore] = None,
        config: Optional[BenchmarkRuntimeConfig] = None,
        cleanup: Optional[Callable[[], None]] = None,
    ):
        self.runner = runner or DockerCommandRunner()
        self.artifact_store = artifact_store
        self.config = config or BenchmarkRuntimeConfig.from_env(profile="repo_docker")
        self.cleanup = cleanup

    def evaluate(self, request: EvalRequest) -> EvalResult:
        if not request.command:
            raise ValueError("DockerExecutionEnvironment requires request.command.")
        timeout = max(1, int(request.timeout_s))
        start = time.time()
        command = [str(part) for part in request.command]
        try:
            proc = self.runner.run(command, cwd=Path(request.cwd), timeout=timeout)
            result = EvalResult(
                passed=proc.returncode == 0,
                status="passed" if proc.returncode == 0 else "failed",
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                feedback=(proc.stdout or "") + (proc.stderr or ""),
                returncode=int(proc.returncode),
                elapsed_s=round(time.time() - start, 3),
                timeout_s=timeout,
                metrics=self._metrics(command, request),
            )
        except subprocess.TimeoutExpired as exc:
            result = EvalResult(
                passed=False,
                status="timeout",
                stdout=_decode_timeout_text(exc.stdout),
                stderr=_decode_timeout_text(exc.stderr),
                feedback=f"TIMEOUT: docker evaluator exceeded {timeout}s.",
                returncode=-9,
                elapsed_s=round(time.time() - start, 3),
                timeout_s=timeout,
                metrics=self._metrics(command, request),
            )
        except Exception as exc:
            result = EvalResult(
                passed=False,
                status="error",
                stderr=str(exc),
                feedback=str(exc),
                returncode=1,
                elapsed_s=round(time.time() - start, 3),
                timeout_s=timeout,
                metrics=self._metrics(command, request),
            )
        finally:
            if self.cleanup is not None:
                try:
                    self.cleanup()
                except Exception:
                    pass
        return self._with_artifacts(request, result)

    def _metrics(self, command: Sequence[str], request: EvalRequest) -> dict:
        return {
            "command": list(command),
            "visibility": request.visibility,
            "runtime_config": self.config.to_metadata(),
        }

    def _with_artifacts(self, request: EvalRequest, result: EvalResult) -> EvalResult:
        store = request.artifact_store or self.artifact_store
        if store is None:
            return result
        artifacts = store.record_eval(
            stem=request.artifact_stem or "docker-eval",
            code=None,
            stdout=result.stdout,
            stderr=result.stderr,
            metadata={
                "status": result.status,
                "passed": result.passed,
                "returncode": result.returncode,
                "timeout_s": result.timeout_s,
                "elapsed_s": result.elapsed_s,
                "request": {
                    "command": list(request.command or []),
                    "cwd": str(request.cwd),
                    "visibility": request.visibility,
                    "metadata": request.metadata,
                },
                "runtime_config": self.config.to_metadata(),
            },
        )
        return EvalResult(
            passed=result.passed,
            status=result.status,
            stdout=result.stdout,
            stderr=result.stderr,
            feedback=result.feedback,
            returncode=result.returncode,
            elapsed_s=result.elapsed_s,
            timeout_s=result.timeout_s,
            artifacts={**result.artifacts, **artifacts},
            metrics=result.metrics,
        )


def _decode_timeout_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value or ""


__all__ = ["DockerCommandRunner", "DockerExecutionEnvironment"]
