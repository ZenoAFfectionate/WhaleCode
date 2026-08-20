"""Python subprocess benchmark execution environment."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .artifacts import BenchmarkArtifactStore
from .base import BenchmarkExecutionEnvironment, EvalRequest, EvalResult, EvalStatus
from .config import BenchmarkRuntimeConfig
from .feedback import append_artifact_hint

try:
    from ..safe_exec import (
        BenchmarkSandboxLimits,
        benchmark_child_env,
        make_benchmark_preexec,
    )
except ImportError:  # pragma: no cover - direct script execution
    from safe_exec import (  # type: ignore
        BenchmarkSandboxLimits,
        benchmark_child_env,
        make_benchmark_preexec,
    )


class PythonSubprocessEnvironment(BenchmarkExecutionEnvironment):
    """Run Python benchmark evaluators in a bounded subprocess group."""

    def __init__(
        self,
        *,
        artifact_store: Optional[BenchmarkArtifactStore] = None,
        config: Optional[BenchmarkRuntimeConfig] = None,
    ):
        self.artifact_store = artifact_store
        self.config = config or BenchmarkRuntimeConfig.from_env(profile="python_strict")

    def evaluate(self, request: EvalRequest) -> EvalResult:
        timeout = max(1, int(request.timeout_s))
        cwd = Path(request.cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        limits = request.limits
        if limits is None:
            limits = self.config.to_sandbox_limits(timeout)

        tmp_path: Optional[Path] = None
        command_args = list(request.command or [])
        code_for_artifact = request.code
        try:
            # Dispatch execution mode:
            #   command=None  → stdin  mode: code piped to ``python -``
            #   command=()    → file  mode: code in temp .py, stdin passed to file
            #   command=[...] → command mode: no code, just run command
            if request.code is not None and request.command is None:
                command_args = ["-"]
                input_text = request.code
            elif request.code is not None:
                tmp_path = self._write_temp_code(cwd, request.code)
                command_args = [str(tmp_path), *command_args]
                input_text = request.stdin
            else:
                input_text = request.stdin

            command = [sys.executable, *command_args]
            base_metrics = {
                "command": command,
                "visibility": request.visibility,
                "runtime_config": self.config.to_metadata(),
            }
            start = time.time()
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd),
                env=request.env if request.env is not None else benchmark_child_env(),
                start_new_session=True,
                preexec_fn=make_benchmark_preexec(limits),
            )
            try:
                stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
                returncode = int(proc.returncode or 0)
                elapsed = round(time.time() - start, 3)
                status: EvalStatus = "passed" if returncode == 0 else "failed"
                result = EvalResult(
                    passed=returncode == 0,
                    status=status,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    feedback=(stdout or "") + (stderr or ""),
                    returncode=returncode,
                    elapsed_s=elapsed,
                    timeout_s=timeout,
                    metrics=base_metrics,
                )
                return self._with_artifacts(request, result, code_for_artifact, limits)
            except subprocess.TimeoutExpired:
                self._kill_process_group(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=2)
                except Exception:
                    stdout, stderr = "", ""
                result = EvalResult(
                    passed=False,
                    status="timeout",
                    stdout=stdout or "",
                    stderr=stderr or "",
                    feedback=f"TIMEOUT: evaluator exceeded {timeout}s.",
                    returncode=-9,
                    elapsed_s=round(time.time() - start, 3),
                    timeout_s=timeout,
                    metrics=base_metrics,
                )
                return self._with_artifacts(request, result, code_for_artifact, limits)
        except Exception as exc:
            result = EvalResult(
                passed=False,
                status="error",
                stderr=str(exc),
                feedback=str(exc),
                returncode=1,
                timeout_s=timeout,
                metrics={
                    "visibility": request.visibility,
                    "runtime_config": self.config.to_metadata(),
                },
            )
            return self._with_artifacts(request, result, code_for_artifact, limits)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _with_artifacts(
        self,
        request: EvalRequest,
        result: EvalResult,
        code: Optional[str],
        limits,
    ) -> EvalResult:
        store = request.artifact_store or self.artifact_store
        if store is None:
            return result
        artifacts = store.record_eval(
            stem=request.artifact_stem or "python-eval",
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            metadata={
                "status": result.status,
                "passed": result.passed,
                "returncode": result.returncode,
                "timeout_s": result.timeout_s,
                "elapsed_s": result.elapsed_s,
                "request": {
                    "cwd": str(request.cwd),
                    "command": list(request.command or []),
                    "stdin_length": len(request.stdin or ""),
                    "visibility": request.visibility,
                    "metadata": request.metadata,
                },
                "limits": getattr(limits, "__dict__", repr(limits)),
                "runtime_config": self.config.to_metadata(),
            },
        )
        feedback = append_artifact_hint(result.feedback, artifacts)
        return EvalResult(
            passed=result.passed,
            status=result.status,
            stdout=result.stdout,
            stderr=result.stderr,
            feedback=feedback,
            returncode=result.returncode,
            elapsed_s=result.elapsed_s,
            timeout_s=result.timeout_s,
            artifacts={**result.artifacts, **artifacts},
            metrics=result.metrics,
        )

    @staticmethod
    def _write_temp_code(cwd: Path, code: str) -> Path:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".py",
            prefix="._bench_eval_",
            dir=str(cwd),
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(code)
            return Path(handle.name)

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        pid = getattr(proc, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            return
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass


__all__ = ["PythonSubprocessEnvironment", "BenchmarkSandboxLimits"]
