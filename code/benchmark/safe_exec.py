"""Safer subprocess helpers for benchmark-side Python evaluation.

Benchmark evaluators intentionally execute model-written Python code. This
module keeps that execution out of the runner process, strips credentials from
the environment, applies resource limits where the OS supports them, and kills
the whole child process group on timeout.
"""

from __future__ import annotations

import os
import sys
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

try:  # POSIX-only. Windows/macOS support varies by limit.
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent
    _resource = None

try:
    from ._utils import build_minimal_child_env
except ImportError:  # pragma: no cover - direct script execution
    from _utils import build_minimal_child_env


@dataclass(frozen=True)
class BenchmarkSandboxLimits:
    """Resource limits for one benchmark evaluator subprocess."""

    cpu_seconds: int = 0
    address_space_bytes: int = 0
    max_processes: int = 128
    file_size_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class SafePythonResult:
    """Completed or timed-out Python evaluator result."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_s: float
    timeout_s: int
    feedback: str = ""
    artifacts: Optional[Dict[str, str]] = None

    @property
    def output(self) -> str:
        if self.feedback:
            return self.feedback.strip()
        return (self.stdout + self.stderr).strip()


class UnsafeBenchmarkCodeError(ValueError):
    """Raised when submitted benchmark code uses obviously unsafe primitives."""


_BLOCKED_IMPORT_ROOTS = {
    "ftplib",
    "http",
    "paramiko",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
_BLOCKED_BUILTIN_CALLS = {"__import__", "compile", "eval", "exec", "open"}
_BLOCKED_ATTR_CALLS = {
    "os": {
        "chmod",
        "chown",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "kill",
        "killpg",
        "link",
        "makedirs",
        "mkdir",
        "popen",
        "remove",
        "removedirs",
        "rename",
        "replace",
        "rmdir",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "symlink",
        "system",
        "unlink",
    },
    "pathlib.Path": {
        "chmod",
        "hardlink_to",
        "link_to",
        "mkdir",
        "open",
        "read_bytes",
        "read_text",
        "rename",
        "replace",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    },
    "Path": {
        "chmod",
        "hardlink_to",
        "link_to",
        "mkdir",
        "open",
        "read_bytes",
        "read_text",
        "rename",
        "replace",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    },
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def validate_python_source_safe(source: str, *, filename: str = "solution.py") -> None:
    """Reject submitted code that uses high-risk primitives before execution.

    This is a coarse preflight check, not the sandbox itself. It is deliberately
    limited to APIs that benchmark solutions should not need: process spawning,
    network clients, host file access, and dynamic code execution.
    """

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in _BLOCKED_IMPORT_ROOTS:
                    raise UnsafeBenchmarkCodeError(
                        f"Unsafe benchmark submission: import of {root!r} is not allowed in {filename}."
                    )

        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _BLOCKED_BUILTIN_CALLS:
                raise UnsafeBenchmarkCodeError(
                    f"Unsafe benchmark submission: call to {name!r} is not allowed in {filename}."
                )
            for owner, methods in _BLOCKED_ATTR_CALLS.items():
                prefix = owner + "."
                if name.startswith(prefix) and name[len(prefix):].split(".", 1)[0] in methods:
                    raise UnsafeBenchmarkCodeError(
                        f"Unsafe benchmark submission: call to {name!r} is not allowed in {filename}."
                    )

        if isinstance(node, ast.Attribute):
            name = _call_name(node)
            if name == "os.environ":
                raise UnsafeBenchmarkCodeError(
                    f"Unsafe benchmark submission: reading os.environ is not allowed in {filename}."
                )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def default_benchmark_limits(timeout_s: int) -> BenchmarkSandboxLimits:
    """Return conservative evaluator limits, overridable via environment."""

    cpu_default = max(1, int(timeout_s) + 2)
    return BenchmarkSandboxLimits(
        cpu_seconds=_env_int("WHALE_BENCH_EVAL_CPU_SECONDS", cpu_default),
        address_space_bytes=_env_int("WHALE_BENCH_EVAL_MEMORY_BYTES", 0),
        max_processes=_env_int("WHALE_BENCH_EVAL_MAX_PROCESSES", 128),
        file_size_bytes=_env_int(
            "WHALE_BENCH_EVAL_FILE_SIZE_BYTES",
            256 * 1024 * 1024,
        ),
    )


def benchmark_child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return the sanitized environment used by benchmark evaluator children."""

    env = build_minimal_child_env()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Numeric packages such as NumPy/OpenBLAS may otherwise try to create many
    # worker threads and collide with the evaluator's process/thread limits.
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def make_benchmark_preexec(
    limits: Optional[BenchmarkSandboxLimits],
) -> Optional[Callable[[], None]]:
    """Build a child-only preexec hook that applies POSIX resource limits."""

    if limits is None or _resource is None:
        return None

    pairs: list[tuple[int, int]] = []
    if limits.cpu_seconds > 0 and hasattr(_resource, "RLIMIT_CPU"):
        pairs.append((_resource.RLIMIT_CPU, limits.cpu_seconds))
    if limits.address_space_bytes > 0 and hasattr(_resource, "RLIMIT_AS"):
        pairs.append((_resource.RLIMIT_AS, limits.address_space_bytes))
    if limits.max_processes > 0 and hasattr(_resource, "RLIMIT_NPROC"):
        pairs.append((_resource.RLIMIT_NPROC, limits.max_processes))
    if limits.file_size_bytes > 0 and hasattr(_resource, "RLIMIT_FSIZE"):
        pairs.append((_resource.RLIMIT_FSIZE, limits.file_size_bytes))

    if not pairs:
        return None

    def _apply() -> None:  # pragma: no cover - runs after fork in child
        for resource_id, value in pairs:
            try:
                _resource.setrlimit(resource_id, (value, value))
            except (OSError, ValueError):
                continue

    return _apply


def _result_from_eval(result) -> SafePythonResult:
    return SafePythonResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.status == "timeout",
        elapsed_s=result.elapsed_s,
        timeout_s=result.timeout_s,
        feedback=result.feedback,
        artifacts=dict(result.artifacts or {}),
    )


def _python_environment():
    try:
        from .runtime.base import EvalRequest
        from .runtime.artifacts import BenchmarkArtifactStore
        from .runtime.config import BenchmarkRuntimeConfig
        from .runtime.python_env import PythonSubprocessEnvironment
    except ImportError:  # pragma: no cover - direct script execution
        from runtime.base import EvalRequest  # type: ignore
        from runtime.artifacts import BenchmarkArtifactStore  # type: ignore
        from runtime.config import BenchmarkRuntimeConfig  # type: ignore
        from runtime.python_env import PythonSubprocessEnvironment  # type: ignore
    config = BenchmarkRuntimeConfig.from_env(profile="python_strict")
    store = None
    if os.getenv("WHALE_BENCH_EVAL_ARTIFACTS", "1").strip().lower() not in {"0", "false", "off", "no"}:
        default_root = Path(__file__).resolve().parents[2] / "memory" / "benchmark_artifacts" / "python"
        root = Path(os.getenv("WHALE_BENCH_EVAL_ARTIFACT_DIR", str(default_root))).expanduser()
        store = BenchmarkArtifactStore(root=root, retention=config.artifact_retention)
    return EvalRequest, PythonSubprocessEnvironment(artifact_store=store, config=config)


def run_python_command(
    args: Sequence[str],
    *,
    cwd: Path,
    input_text: str = "",
    timeout: int,
    env: Optional[Dict[str, str]] = None,
    limits: Optional[BenchmarkSandboxLimits] = None,
    artifact_stem: Optional[str] = None,
) -> SafePythonResult:
    """Run ``python`` with *args* in a bounded subprocess group."""

    timeout = max(1, int(timeout))
    EvalRequest, environment = _python_environment()
    request = EvalRequest(
        command=list(args),
        cwd=Path(cwd),
        stdin=input_text,
        timeout_s=timeout,
        env=env,
        limits=limits,
        artifact_stem=artifact_stem or "python-command",
    )
    return _result_from_eval(environment.evaluate(request))


def run_python_code(
    code: str,
    *,
    cwd: Path,
    timeout: int,
    env: Optional[Dict[str, str]] = None,
    limits: Optional[BenchmarkSandboxLimits] = None,
    artifact_stem: Optional[str] = None,
) -> SafePythonResult:
    """Execute Python source from stdin in a bounded evaluator subprocess."""

    timeout = max(1, int(timeout))
    EvalRequest, environment = _python_environment()
    request = EvalRequest(
        code=code,
        cwd=Path(cwd),
        timeout_s=timeout,
        env=env,
        limits=limits,
        artifact_stem=artifact_stem or "python-code",
    )
    return _result_from_eval(environment.evaluate(request))


def run_python_code_with_stdin(
    code: str,
    stdin_text: str,
    *,
    cwd: Path,
    timeout: int,
    env: Optional[Dict[str, str]] = None,
    limits: Optional[BenchmarkSandboxLimits] = None,
    artifact_stem: Optional[str] = None,
) -> SafePythonResult:
    """Write *code* to a temporary file and pass *stdin_text* to that file."""

    timeout = max(1, int(timeout))
    EvalRequest, environment = _python_environment()
    request = EvalRequest(
        code=code,
        command=(),
        cwd=Path(cwd),
        stdin=stdin_text,
        timeout_s=timeout,
        env=env,
        limits=limits,
        artifact_stem=artifact_stem or "python-stdin",
    )
    return _result_from_eval(environment.evaluate(request))


def format_safe_python_failure(prefix: str, result: SafePythonResult) -> str:
    """Return a concise, user-facing failure line for a sandboxed evaluator."""

    if result.timed_out:
        return f"TIMEOUT: {prefix} exceeded {result.timeout_s}s."
    output = result.output
    if output:
        return output
    return f"{prefix} failed with exit code {result.returncode}."
