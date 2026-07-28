"""SWE-bench Verified benchmark runner for Whale Code agent.

Two-phase evaluation:
  Phase 1 (this file): Run the agent on each instance, collect diffs,
      and output a predictions JSONL compatible with the official harness.
  Phase 2 (scripts/run_swev_eval.sh): Feed predictions to
      ``swebench.harness.run_evaluation`` for Docker-based grading.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

try:
    from .base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _DEFAULT_RESULTS_DIR,
        _DEFAULT_TRAJECTORY_DIR,
        _PROJECT_ROOT,
    )
except ImportError:
    from base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _DEFAULT_RESULTS_DIR,
        _DEFAULT_TRAJECTORY_DIR,
        _PROJECT_ROOT,
    )

_SWEV_ADDENDUM = """\

---

## SWE-bench Override: Autonomous Issue Resolution

You are an autonomous software engineer. Your sole job is to resolve
the GitHub issue by editing source code in the local repository.
There is no human in the loop.

### Workflow (follow strictly)

1. Locate relevant code with targeted searches.
2. Read only the minimal functions/classes needed.
3. Diagnose root cause, then apply the minimal correct fix.
4. Run fast, targeted verification only when useful.
5. Call `Finish` with a concise summary of what changed and why.

### Critical Rules

- Prefer minimal, correct changes; multi-file edits are OK if required.
- Do NOT modify test files (tests/, test_*.py, *_test.py).
- Do NOT add dependencies unless the fix truly needs them.
- Avoid sweeping refactors or formatting-only changes.
- Avoid writing to .git or leaving build artifacts in the repo.

### Efficiency Rules (save tokens and steps)

- Do NOT read the whole repository. Search targeted: use Grep/Glob/Read.
- Once you identify the fix location, edit promptly.
- If a search returns too many results, narrow the query.
- Avoid shell commands for searching when Grep/Glob/Read are available.
- Use TodoWrite only if the issue truly needs multi-step planning.
- `Finish` must be the last tool you call for the task.
"""

_SWEV_SYSTEM_PROMPT = BENCHMARK_BASE_SYSTEM_PROMPT + _SWEV_ADDENDUM

_SWEV_ARTIFACT_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".eggs",
    ".idea",
    ".vscode",
}
_SWEV_ARTIFACT_FILES = {
    ".coverage",
    ".DS_Store",
}
_SWEV_ARTIFACT_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
)

_CONTAINER_WORKDIR = PurePosixPath("/testbed")
_PROCESS_ERROR_CLIP = 1200


def _clip_output(text: Optional[str], *, limit: int = _PROCESS_ERROR_CLIP) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _format_subprocess_error(
    *,
    step: str,
    command: List[str],
    cwd: Optional[Path] = None,
    returncode: Optional[int] = None,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> str:
    cmd_text = shlex.join(command)
    lines = [f"{step} failed"]
    lines.append(f"command: {cmd_text}")
    if cwd is not None:
        lines.append(f"cwd: {cwd}")
    if timeout_s is not None:
        lines.append(f"timeout_s: {timeout_s}")
    if returncode is not None:
        lines.append(f"returncode: {returncode}")
    if stdout:
        lines.append(f"stdout: {_clip_output(stdout)}")
    if stderr:
        lines.append(f"stderr: {_clip_output(stderr)}")
    return "\n".join(lines)


def _parse_slice_spec(slice_spec: str) -> Optional[slice]:
    text = (slice_spec or "").strip()
    if not text:
        return None
    if ":" not in text:
        raise ValueError("Invalid --slice format, expected `start:end[:step]`")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError("Invalid --slice format, expected `start:end[:step]`")
    values: List[Optional[int]] = []
    for part in parts:
        part = part.strip()
        if not part:
            values.append(None)
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid --slice component: {part!r}") from exc
    while len(values) < 3:
        values.append(None)
    return slice(values[0], values[1], values[2])


class DockerizedWorkspace:
    """Container lifecycle wrapper for one SWE-bench instance."""

    def __init__(
        self,
        *,
        image: str,
        workspace: Path,
        executable: str = "docker",
        container_timeout: str = "2h",
        pull_timeout: int = 600,
        container_workdir: PurePosixPath = _CONTAINER_WORKDIR,
    ):
        self.image = image
        self.workspace = workspace.expanduser().resolve()
        self.executable = executable
        self.container_timeout = container_timeout
        self.pull_timeout = pull_timeout
        self.container_workdir = container_workdir
        self.container_name = f"whale-swev-{uuid.uuid4().hex[:8]}"
        self.container_id: Optional[str] = None

    def start(self) -> None:
        cmd = [
            self.executable,
            "run",
            "-d",
            "--name",
            self.container_name,
            "-w",
            str(self.container_workdir),
            "-v",
            f"{self.workspace}:{self.container_workdir}",
            "--rm",
            self.image,
            "sleep",
            self.container_timeout,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.pull_timeout,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Container executable not found: {self.executable}. "
                "Install Docker/Podman or pass --docker-executable."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                _format_subprocess_error(
                    step="docker run (container startup)",
                    command=cmd,
                    cwd=self.workspace,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    timeout_s=self.pull_timeout,
                )
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                _format_subprocess_error(
                    step="docker run (container startup)",
                    command=cmd,
                    cwd=self.workspace,
                    returncode=exc.returncode,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                )
            ) from exc

        container_id = (result.stdout or "").strip()
        if not container_id:
            raise RuntimeError(
                _format_subprocess_error(
                    step="docker run (container startup)",
                    command=cmd,
                    cwd=self.workspace,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
        self.container_id = container_id

    def popen(
        self,
        *,
        command: str,
        container_directory: PurePosixPath,
    ) -> subprocess.Popen:
        if not self.container_id:
            raise RuntimeError("Docker container is not running")
        cmd = [
            self.executable,
            "exec",
            "-w",
            str(container_directory),
            self.container_id,
            "bash",
            "-lc",
            command,
        ]
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Container executable not found: {self.executable}") from exc

    def cleanup(self) -> None:
        if not self.container_id:
            return
        try:
            subprocess.run(
                [self.executable, "stop", self.container_id],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:
            try:
                subprocess.run(
                    [self.executable, "rm", "-f", self.container_id],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except Exception:
                pass
        finally:
            self.container_id = None


class DockerBashTool:
    """Benchmark-only Bash tool that executes commands inside a Docker container."""

    def __init__(self, *, docker_workspace: DockerizedWorkspace, local_bash_tool):
        self._docker_workspace = docker_workspace
        self._delegate = local_bash_tool
        self.name = local_bash_tool.name
        self.description = local_bash_tool.description
        self.expandable = getattr(local_bash_tool, "expandable", False)
        self.project_root = local_bash_tool.project_root
        self.working_dir = local_bash_tool.working_dir
        self.DEFAULT_BLOCK_UNTIL_MS = local_bash_tool.DEFAULT_BLOCK_UNTIL_MS
        self.MAX_BLOCK_UNTIL_MS = local_bash_tool.MAX_BLOCK_UNTIL_MS

    def get_parameters(self):
        return self._delegate.get_parameters()

    def run_with_timing(self, parameters):
        return self._delegate.run_with_timing.__func__(self, parameters)  # type: ignore[attr-defined]

    def _validate_command(self, command: str):
        return self._delegate._validate_command(command)

    def _background_response(self, **kwargs):
        return self._delegate._background_response(**kwargs)

    def _format_response(self, **kwargs):
        return self._delegate._format_response(**kwargs)

    def run(self, parameters):
        from hello_agents.tools.builtin._code_utils import relative_display, resolve_path
        from hello_agents.tools.errors import ToolErrorCode
        from hello_agents.tools.response import ToolResponse

        command = parameters.get("command")

        description_raw = parameters.get("description", "")
        if description_raw is None:
            description = ""
        elif isinstance(description_raw, str):
            description = description_raw.strip()
        else:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    "Invalid parameter `description`: expected string when provided, "
                    f"got {type(description_raw).__name__}."
                ),
            )

        working_directory = parameters.get("working_directory")
        if working_directory is None:
            working_directory = parameters.get("directory", ".")

        block_until_ms = parameters.get("block_until_ms")
        if block_until_ms is None:
            timeout_alias = parameters.get("timeout_ms")
            block_until_ms = timeout_alias if timeout_alias is not None else self.DEFAULT_BLOCK_UNTIL_MS

        if not isinstance(command, str) or not command.strip():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    "Invalid parameter `command`: expected non-empty string, "
                    f"got {type(command).__name__}."
                ),
            )
        command = command.strip()

        if not isinstance(block_until_ms, int) or block_until_ms < 0 or block_until_ms > self.MAX_BLOCK_UNTIL_MS:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"Invalid parameter `block_until_ms`: expected integer between 0 and {self.MAX_BLOCK_UNTIL_MS}, "
                    f"got value={block_until_ms!r} ({type(block_until_ms).__name__})."
                ),
            )

        try:
            target_dir = resolve_path(self.project_root, self.working_dir, working_directory)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=(
                    "Invalid `working_directory`: path escapes workspace root.\n"
                    f"working_directory={working_directory!r}"
                ),
            )

        if not target_dir.exists() or not target_dir.is_dir():
            return ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"Working directory not found: {working_directory}",
            )

        policy_error = self._validate_command(command)
        if policy_error:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=(
                    f"Command blocked by Bash policy: {policy_error}\n"
                    f"Command: {command}\n"
                    f"Directory: {working_directory}"
                ),
            )

        rel_dir = relative_display(self.project_root, target_dir)
        container_dir = self._docker_workspace.container_workdir
        if rel_dir != ".":
            container_dir = container_dir / rel_dir

        try:
            process = self._docker_workspace.popen(command=command, container_directory=container_dir)
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=(
                    f"Failed to execute shell command inside Docker: {exc}\n"
                    f"Command: {command}\n"
                    f"Directory: {working_directory}"
                ),
            )

        event_stream = self._delegate._create_event_stream()
        event_stream.start(process.stdout)

        if block_until_ms == 0:
            return self._background_response(
                process=process,
                event_stream=event_stream,
                command=command,
                description=description,
                directory=target_dir,
                block_until_ms=block_until_ms,
                reason="immediate_background",
            )

        try:
            process.wait(timeout=block_until_ms / 1000)
            event_stream.wait_closed(timeout=2.0)
        except subprocess.TimeoutExpired:
            return self._background_response(
                process=process,
                event_stream=event_stream,
                command=command,
                description=description,
                directory=target_dir,
                block_until_ms=block_until_ms,
                reason="exceeded_block_until",
            )
        except Exception as exc:
            try:
                process.kill()
            except Exception:
                pass
            event_stream.wait_closed(timeout=1.0)
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed while waiting for Docker command: {exc}",
            )

        return self._format_response(
            command=command,
            description=description,
            directory=target_dir,
            exit_code=process.returncode,
            event_stream=event_stream,
        )


class SWEBenchVerifiedBenchmark(BenchmarkRunner):
    """Evaluate the agent on SWE-bench Verified (500 real GitHub issue instances).

    Workflow per instance:
    1. Clone (or use cached) the repository at ``base_commit``.
    2. Set the cloned repo as the agent workspace.
    3. Present the ``problem_statement`` to the agent.
    4. After the agent finishes, capture ``git diff`` of all changes.
    5. Record the diff for offline Docker evaluation.

    Use ``--repo-cache-dir`` to persist cloned repos across runs.
    """

    benchmark_name = "swev"
    runtime_profile = "repo_docker"

    def __init__(
        self,
        *args,
        repo_cache_dir: Optional[str] = None,
        model_name: str = "whale-code",
        task_timeout: int = 1200,
        resume_file: Optional[str] = None,
        trajectory_dir: Optional[str] = None,
        docker_executable: str = "docker",
        docker_container_timeout: str = "2h",
        docker_pull_timeout: int = 600,
        filter_spec: str = "",
        slice_spec: str = "",
        shuffle: bool = False,
        seed: int = 42,
        preds_path: Optional[str] = None,
        redo_existing: bool = False,
        workers: int = 1,
        **kwargs,
    ):
        super().__init__(*args, trajectory_dir=trajectory_dir, task_timeout=task_timeout, **kwargs)
        self.repo_cache_dir = Path(repo_cache_dir) if repo_cache_dir else None
        if self.repo_cache_dir:
            self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.resume_file = resume_file
        self.docker_executable = docker_executable
        self.docker_container_timeout = docker_container_timeout
        self.docker_pull_timeout = docker_pull_timeout
        self.filter_spec = filter_spec.strip()
        self.slice_spec = slice_spec.strip()
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.preds_path = Path(preds_path).expanduser() if preds_path else None
        self.redo_existing = bool(redo_existing)
        self.workers = max(1, int(workers))
        if self.filter_spec:
            try:
                self._filter_regex = re.compile(self.filter_spec)
            except re.error as exc:
                raise ValueError(f"Invalid --filter regex {self.filter_spec!r}: {exc}") from exc
        else:
            self._filter_regex = None
        self._slice = _parse_slice_spec(self.slice_spec) if self.slice_spec else None
        self._preds_completed_ids: Set[str] = set()
        if self.preds_path and not self.redo_existing:
            self._preds_completed_ids = self._load_existing_prediction_ids(self.preds_path)
        self._repo_lock_guard = threading.Lock()
        self._repo_locks: Dict[str, threading.Lock] = {}
        self._thread_state = threading.local()

    def _get_system_prompt(self) -> Optional[str]:
        """Use the SWE-bench-specific system prompt."""
        return _SWEV_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks = self._load_jsonl_tasks(
            task_transform=lambda task: {
                **task,
                "task_id": task.get("instance_id", task.get("task_id")),
            }
        )
        if self._filter_regex is not None:
            tasks = [task for task in tasks if self._filter_regex.search(str(task.get("task_id", "")))]
        if self.shuffle:
            tasks = sorted(tasks, key=lambda item: str(item.get("task_id", "")))
            rng = random.Random(self.seed)
            rng.shuffle(tasks)
        if self._slice is not None:
            tasks = tasks[self._slice]
        if self._preds_completed_ids:
            tasks = [task for task in tasks if str(task.get("task_id", "")) not in self._preds_completed_ids]
        return tasks

    @staticmethod
    def _load_existing_prediction_ids(preds_path: Path) -> Set[str]:
        if not preds_path.exists():
            return set()

        completed: Set[str] = set()
        suffix = preds_path.suffix.lower()
        if suffix == ".json":
            try:
                payload = json.loads(preds_path.read_text(encoding="utf-8"))
            except Exception:
                return completed
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, dict):
                        instance_id = value.get("instance_id", key)
                    else:
                        instance_id = key
                    if instance_id:
                        completed.add(str(instance_id))
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and item.get("instance_id"):
                        completed.add(str(item["instance_id"]))
            return completed

        if suffix == ".jsonl":
            try:
                with preds_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(item, dict) and item.get("instance_id"):
                            completed.add(str(item["instance_id"]))
            except Exception:
                return completed
        return completed

    @staticmethod
    def _load_completed_ids(resume_file: Path) -> Set[str]:
        """Treat finished inference records as resumable completions.

        SWE-bench phase 1 results usually keep ``passed=None`` until Docker
        grading, so the base implementation would never skip already-finished
        inference tasks. Here we consider a task complete only if phase 1
        really finished: it either produced a patch or explicitly finished
        with no patch. Transient failures are left resumable.
        """
        completed: Set[str] = set()
        if not resume_file.exists():
            return completed

        records = BenchmarkRunner._load_result_records(resume_file)
        for record in BenchmarkRunner._latest_result_records(records):
            task_id = record.get("task_id")
            if task_id is None:
                continue
            has_patch = bool(record.get("agent_diff"))
            no_patch_finished = record.get("error") == "Agent produced no changes"
            if has_patch or no_patch_finished:
                completed.add(str(task_id))
        return completed

    def _docker_preflight(self) -> None:
        cmd = [self.docker_executable, "info"]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=min(60, self.docker_pull_timeout),
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Docker executable not found: {self.docker_executable}. "
                "Install Docker/Podman or pass --docker-executable."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                _format_subprocess_error(
                    step="docker preflight (`docker info`)",
                    command=cmd,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    timeout_s=float(exc.timeout or 0),
                )
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                _format_subprocess_error(
                    step="docker preflight (`docker info`)",
                    command=cmd,
                    returncode=exc.returncode,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                )
            ) from exc

    # ------------------------------------------------------------------
    # Repository management
    # ------------------------------------------------------------------

    def _set_clone_error(self, message: Optional[str]) -> None:
        self._thread_state.clone_error = message

    def _get_clone_error(self) -> Optional[str]:
        return getattr(self._thread_state, "clone_error", None)

    def _repo_lock_for(self, repo_slug: str) -> threading.Lock:
        with self._repo_lock_guard:
            lock = self._repo_locks.get(repo_slug)
            if lock is None:
                lock = threading.Lock()
                self._repo_locks[repo_slug] = lock
            return lock

    @staticmethod
    def _remove_git_lock_files(repo_path: Path) -> None:
        """Remove stale git lock files that prevent checkout/reset."""
        for lock_file in ["index.lock", "HEAD.lock", "refs/heads/*.lock"]:
            for p in repo_path.glob(f".git/{lock_file}"):
                try:
                    p.unlink()
                except OSError:
                    pass

    @staticmethod
    def _is_prunable_untracked(path: Path) -> bool:
        """Return True if an untracked path looks like a build/test artifact."""
        parts = set(path.parts)
        if parts & _SWEV_ARTIFACT_DIRS:
            return True
        name = path.name
        if name in _SWEV_ARTIFACT_FILES:
            return True
        if name.startswith(".coverage."):
            return True
        if name.endswith(_SWEV_ARTIFACT_SUFFIXES):
            return True
        if name.endswith(".egg-info"):
            return True
        return False

    @staticmethod
    def _prune_untracked_artifacts(repo_path: Path) -> None:
        """Remove common untracked artifacts to avoid patch pollution."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "-uall", "-z"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=20,
            )
            if not result.stdout:
                return
            for entry in result.stdout.split("\0"):
                if not entry or not entry.startswith("?? "):
                    continue
                rel = entry[3:]
                if not rel or rel.startswith(".git/"):
                    continue
                path = repo_path / rel
                if not SWEBenchVerifiedBenchmark._is_prunable_untracked(path):
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        path.unlink()
                    except OSError:
                        pass
        except Exception:
            return

    @staticmethod
    def _normalize_failed_command(raw_cmd: Any, *, fallback: List[str]) -> List[str]:
        if isinstance(raw_cmd, (list, tuple)):
            return [str(part) for part in raw_cmd]
        if isinstance(raw_cmd, str):
            try:
                parsed = shlex.split(raw_cmd)
            except ValueError:
                parsed = []
            if parsed:
                return parsed
        return list(fallback)

    @staticmethod
    def _run_checked_command(
        *,
        step: str,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout_s: int,
    ) -> None:
        try:
            subprocess.run(
                command,
                cwd=str(cwd) if cwd is not None else None,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            failed_cmd = SWEBenchVerifiedBenchmark._normalize_failed_command(exc.cmd, fallback=command)
            raise RuntimeError(
                _format_subprocess_error(
                    step=step,
                    command=failed_cmd,
                    cwd=cwd,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    timeout_s=float(exc.timeout or timeout_s),
                )
            ) from exc
        except subprocess.CalledProcessError as exc:
            failed_cmd = SWEBenchVerifiedBenchmark._normalize_failed_command(exc.cmd, fallback=command)
            raise RuntimeError(
                _format_subprocess_error(
                    step=step,
                    command=failed_cmd,
                    cwd=cwd,
                    returncode=exc.returncode,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                )
            ) from exc

    def _run_clone_sequence(
        self,
        *,
        step: str,
        target: Path,
        commands: List[Tuple[List[str], Optional[Path], int]],
    ) -> bool:
        try:
            for command, cwd, timeout_s in commands:
                self._run_checked_command(
                    step=step,
                    command=command,
                    cwd=cwd,
                    timeout_s=timeout_s,
                )
            return True
        except RuntimeError as exc:
            message = str(exc)
        except Exception as exc:
            message = f"{step} failed: {type(exc).__name__}: {exc}"

        self._set_clone_error(message)
        print(f"\n  [WARN] {message}")
        shutil.rmtree(target, ignore_errors=True)
        return False

    def _reset_cached_repo(self, cached: Path, base_commit: str) -> bool:
        """Reset a cached repo to a specific commit. Returns True on success."""
        ws = str(cached)
        try:
            # Clean up any stale lock files from previous crashed runs
            self._remove_git_lock_files(cached)

            # Hard reset to discard any staged/unstaged changes from previous task
            subprocess.run(
                ["git", "reset", "--hard"],
                cwd=ws, capture_output=True, timeout=120,
            )
            # Remove all untracked files and directories
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=ws, capture_output=True, timeout=120,
            )
            # Checkout the target commit
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=ws, capture_output=True, timeout=300, check=True,
            )
            # Clean again after checkout (in case checkout brought changes)
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=ws, capture_output=True, timeout=120,
            )
            return True
        except Exception:
            return False

    def _clone_repo_to_target(self, repo: str, base_commit: str, target: Path) -> Optional[Path]:
        """Clone a GitHub repo at a specific commit into *target*."""
        url = f"https://github.com/{repo}.git"
        clone_cmd = ["git", "clone", "--quiet", "--filter=blob:none", url, str(target)]
        checkout_cmd = ["git", "checkout", "-f", base_commit]
        if self._run_clone_sequence(
            step=f"clone {repo}@{base_commit[:10]}",
            target=target,
            commands=[
                (clone_cmd, None, 600),
                (checkout_cmd, target, 120),
            ],
        ):
            return target
        return None

    def _clone_repo_from_cache(self, cached: Path, base_commit: str, repo_slug: str) -> Optional[Path]:
        """Materialize an isolated temp workspace from the cached repo."""
        target = self._make_workspace(f"swev_{repo_slug}_")
        clone_cmd = ["git", "clone", "--quiet", "--shared", "--no-checkout", str(cached), str(target)]
        checkout_cmd = ["git", "checkout", "-f", base_commit]
        if self._run_clone_sequence(
            step=f"materialize cache {cached.name}@{base_commit[:10]}",
            target=target,
            commands=[
                (clone_cmd, None, 180),
                (checkout_cmd, target, 120),
            ],
        ):
            return target
        return None

    def _clone_repo(self, repo: str, base_commit: str) -> Optional[Path]:
        """Clone a GitHub repo at a specific commit.

        Uses ``--filter=blob:none`` for a blobless clone (much faster for
        large repos like astropy/django) — blobs are fetched on demand.
        When ``repo_cache_dir`` is set, the cache repo is used only as the
        source of truth and each task gets an isolated temp workspace derived
        from it. Returns the workspace path, or None on failure.
        """
        self._set_clone_error(None)
        repo_slug = repo.replace("/", "__")
        repo_lock = self._repo_lock_for(repo_slug)

        # Check cache first
        if self.repo_cache_dir:
            with repo_lock:
                cached = self.repo_cache_dir / repo_slug
                if cached.exists():
                    # Try direct checkout
                    if self._reset_cached_repo(cached, base_commit):
                        isolated = self._clone_repo_from_cache(cached, base_commit, repo_slug)
                        if isolated:
                            return isolated

                    # First attempt failed — try fetching latest refs then retry
                    print(f"\n  [WARN] Cache checkout failed for {repo}@{base_commit[:10]}, fetching...")
                    try:
                        subprocess.run(
                            ["git", "fetch", "--all"],
                            cwd=str(cached), capture_output=True, timeout=300,
                        )
                    except Exception:
                        pass

                    if self._reset_cached_repo(cached, base_commit):
                        isolated = self._clone_repo_from_cache(cached, base_commit, repo_slug)
                        if isolated:
                            return isolated

                    print(f"\n  [WARN] Retry checkout also failed for {repo}@{base_commit[:10]}")
                    shutil.rmtree(cached, ignore_errors=True)

                # Re-clone into cache path
                fresh = self._clone_repo_to_target(repo, base_commit, cached)
                if fresh:
                    isolated = self._clone_repo_from_cache(fresh, base_commit, repo_slug)
                    if isolated:
                        return isolated

            # Fallback to isolated temp clone if cache path failed
            temp_target = self._make_workspace(f"swev_{repo_slug}_")
            return self._clone_repo_to_target(repo, base_commit, temp_target)

        # Clone fresh (no cache)
        temp_target = self._make_workspace(f"swev_{repo_slug}_")
        return self._clone_repo_to_target(repo, base_commit, temp_target)

    def _get_agent_diff(self, workspace: Path) -> str:
        """Capture the diff of all changes the agent made, including new files."""
        ws = str(workspace)
        try:
            self._prune_untracked_artifacts(workspace)
            # Stage all new (untracked) files so they appear in the diff
            subprocess.run(
                ["git", "add", "-A"],
                cwd=ws, capture_output=True, timeout=30,
            )
            # Diff between base_commit (HEAD) and current staged state
            result = subprocess.run(
                ["git", "diff", "--no-color", "HEAD"],
                cwd=ws, capture_output=True, text=True, timeout=60,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _get_docker_image_name(task: Dict[str, Any]) -> str:
        image_name = task.get("image_name") or task.get("docker_image")
        if image_name:
            return str(image_name)
        instance_id = task["task_id"]
        docker_safe_id = instance_id.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{docker_safe_id}:latest".lower()

    def _save_trajectory(
        self,
        *,
        task: Dict[str, Any],
        workspace: Optional[Path],
        docker_workspace: Optional[DockerizedWorkspace],
        agent,
        agent_prompt: str,
        exit_status: str,
        error: Optional[str],
        elapsed_s: float,
        agent_response: str,
        agent_diff: str,
    ) -> str:
        return self._save_task_trajectory(
            task=task,
            workspace=workspace,
            agent=agent,
            prompt_texts=[agent_prompt] if agent_prompt else [],
            result={
                "task_id": task.get("task_id"),
                "repo": task.get("repo"),
                "passed": None,
                "error": error,
                "agent_diff": agent_diff,
                "agent_response": (agent_response or "")[:500],
                "exit_status": exit_status,
                "docker_image": docker_workspace.image if docker_workspace else self._get_docker_image_name(task),
                "elapsed_s": round(elapsed_s, 2),
                "has_diff": bool(agent_diff),
            },
            extra={
                "repo": task.get("repo"),
                "base_commit": task.get("base_commit"),
                "docker": {
                    "executable": self.docker_executable,
                    "image": docker_workspace.image if docker_workspace else self._get_docker_image_name(task),
                    "container_timeout": self.docker_container_timeout,
                    "pull_timeout": self.docker_pull_timeout,
                    "container_id": docker_workspace.container_id if docker_workspace else None,
                    "workspace_mount": str(workspace) if workspace else None,
                    "container_workdir": str(_CONTAINER_WORKDIR),
                },
                "submission": agent_diff,
                "traceback": traceback.format_exc() if error and sys.exc_info()[0] else "",
            },
        )

    def _create_agent(self, workspace: Path, docker_workspace: Optional[DockerizedWorkspace] = None):
        agent = super()._create_agent(workspace)
        if docker_workspace is None:
            return agent
        local_bash = agent.tool_registry.get_tool("Bash")
        if local_bash is None:
            raise RuntimeError("Benchmark agent is missing the Bash tool")
        agent.tool_registry.unregister("Bash")
        agent.tool_registry.register_tool(
            DockerBashTool(docker_workspace=docker_workspace, local_bash_tool=local_bash)
        )
        return agent

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        repo = task["repo"]
        base_commit = task["base_commit"]
        problem_statement = task["problem_statement"]

        start = time.time()
        cache_root = self.repo_cache_dir.resolve() if self.repo_cache_dir else None
        workspace: Optional[Path] = None
        docker_workspace: Optional[DockerizedWorkspace] = None
        agent = None
        agent_response = ""
        agent_diff = ""
        agent_prompt = ""
        error: Optional[str] = None
        exit_status = "Unknown"
        is_temp = False

        # Step 1: Clone repo
        try:
            workspace = self._clone_repo(repo, base_commit)
            if workspace is None:
                detailed_clone_error = self._get_clone_error()
                if detailed_clone_error:
                    error = detailed_clone_error
                else:
                    error = f"Failed to clone {repo}@{base_commit}"
                exit_status = "CloneFailed"
                return {
                    "task_id": task_id,
                    "repo": repo,
                    "passed": None,
                    "error": error,
                    "agent_diff": "",
                    "exit_status": exit_status,
                    "docker_image": self._get_docker_image_name(task),
                    "elapsed_s": round(time.time() - start, 2),
                }

            is_temp = cache_root is None or workspace.resolve().parent != cache_root

            docker_workspace = DockerizedWorkspace(
                image=self._get_docker_image_name(task),
                workspace=workspace,
                executable=self.docker_executable,
                container_timeout=self.docker_container_timeout,
                pull_timeout=self.docker_pull_timeout,
            )
            docker_workspace.start()

            # Step 2: Run agent
            agent = self._create_agent(workspace, docker_workspace=docker_workspace)
            hints = task.get("hints_text", "").strip()
            hints_block = f"\n\n## Hints\n\n{hints}" if hints else ""

            agent_prompt = (
                f"Fix the following issue in the `{repo}` repository.\n\n"
                f"## Issue\n\n{problem_statement}\n"
                f"{hints_block}\n\n"
                f"The repository is already checked out at the correct commit. "
                f"Your working directory is the repo root. "
                f"Shell commands execute inside the official SWE-bench Docker image, "
                f"while file tools edit the mounted repository workspace.\n\n"
                f"## Strategy\n\n"
                f"1. Locate the bug with targeted search (symbols/errors from the issue).\n"
                f"2. Read only relevant code blocks; diagnose root cause before editing.\n"
                f"3. Apply the minimal correct fix; avoid unrelated refactors.\n"
                f"4. Run quick targeted validation when low-cost; otherwise skip.\n"
                f"5. Call `Finish` with a concise change summary.\n"
            )

            try:
                agent_response = self._run_agent_with_timeout(agent, agent_prompt)
            except _TaskTimeout:
                agent_diff = self._get_agent_diff(workspace)
                error = f"Task timed out after {self.task_timeout}s"
                exit_status = "TaskTimeout"
                return {
                    "task_id": task_id,
                    "repo": repo,
                    "passed": None,
                    "error": error,
                    "agent_diff": agent_diff,
                    "agent_response": "",
                    "exit_status": exit_status,
                    "docker_image": docker_workspace.image,
                    "elapsed_s": round(time.time() - start, 2),
                }
            except Exception as exc:
                error = f"Agent error: {exc}"
                exit_status = type(exc).__name__
                return {
                    "task_id": task_id,
                    "repo": repo,
                    "passed": None,
                    "error": error,
                    "agent_diff": "",
                    "agent_response": "",
                    "exit_status": exit_status,
                    "docker_image": docker_workspace.image,
                    "elapsed_s": round(time.time() - start, 2),
                }

            # Step 3: Capture diff (full, not truncated)
            agent_diff = self._get_agent_diff(workspace)
            exit_status = "Completed" if agent_diff else "NoDiff"
            error = "Agent produced no changes" if not agent_diff else None

            return {
                "task_id": task_id,
                "repo": repo,
                "passed": None,  # Determined by Docker eval
                "has_diff": bool(agent_diff),
                "error": error,
                "agent_diff": agent_diff,
                "agent_response": (agent_response or "")[:500],
                "exit_status": exit_status,
                "docker_image": docker_workspace.image,
                "elapsed_s": round(time.time() - start, 2),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            exit_status = type(exc).__name__
            agent_diff = self._get_agent_diff(workspace) if workspace else ""
            return {
                "task_id": task_id,
                "repo": repo,
                "passed": None,
                "error": error,
                "agent_diff": agent_diff,
                "agent_response": "",
                "exit_status": exit_status,
                "docker_image": self._get_docker_image_name(task),
                "elapsed_s": round(time.time() - start, 2),
            }
        finally:
            self._save_trajectory(
                task=task,
                workspace=workspace,
                docker_workspace=docker_workspace,
                agent=agent,
                agent_prompt=agent_prompt,
                exit_status=exit_status,
                error=error,
                elapsed_s=time.time() - start,
                agent_response=agent_response,
                agent_diff=agent_diff,
            )
            if docker_workspace is not None:
                docker_workspace.cleanup()
            if is_temp and workspace and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    def _use_subprocess_task_timeout(self) -> bool:
        """SWE-bench uses its own agent timeout so partial diffs can be recovered."""
        return False

    def _run_agent_with_timeout(self, agent, prompt: str) -> str:
        """Run ``agent.run()`` with a wall-clock timeout and recover partial diffs."""
        run_kwargs = self._benchmark_agent_run_kwargs()
        if self.task_timeout <= 0:
            result = agent.run(prompt, **run_kwargs)
            return "" if result is None else str(result).strip()

        # Use a threading-based timeout on all platforms
        import threading

        result_holder: Dict[str, Any] = {}
        exception_holder: List[BaseException] = []

        def target():
            try:
                result_holder["value"] = agent.run(prompt, **run_kwargs)
            except Exception as exc:
                exception_holder.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.task_timeout)

        if thread.is_alive():
            # Thread is still running — we can't forcefully kill it,
            # but we signal timeout and move on. The daemon thread will
            # be cleaned up when the process exits or on next task.
            raise _TaskTimeout(f"Agent did not finish within {self.task_timeout}s")

        if exception_holder:
            raise exception_holder[0]

        result = result_holder.get("value", "")
        return "" if result is None else str(result).strip()

    # ------------------------------------------------------------------
    # Override run() to add resume support and predictions export
    # ------------------------------------------------------------------

    def _export_predictions(
        self,
        *,
        results_file: Path,
        timestamp: str,
    ) -> Tuple[Path, Path, int, int]:
        predictions_file = self.output_dir / f"swev_predictions_{timestamp}.jsonl"
        preds_json_file = self.output_dir / f"swev_preds_{timestamp}.json"
        latest_preds_file = self.output_dir / "preds.json"

        diff_count = 0
        final_results = self._latest_result_records(self._load_result_records(results_file))
        preds_map: Dict[str, Dict[str, str]] = {}
        with predictions_file.open("w", encoding="utf-8") as fout:
            for result in final_results:
                instance_id = str(result.get("task_id", "") or "")
                agent_diff = str(result.get("agent_diff", "") or "")
                if agent_diff:
                    diff_count += 1
                prediction = {
                    "instance_id": instance_id,
                    "model_name_or_path": self.model_name,
                    "model_patch": agent_diff,
                }
                preds_map[instance_id] = prediction
                fout.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        preds_json_file.write_text(
            json.dumps(preds_map, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        latest_preds_file.write_text(
            json.dumps(preds_map, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return predictions_file, preds_json_file, diff_count, len(final_results)

    def _run_parallel(
        self,
        *,
        limit: Optional[int],
        task_ids: Optional[List[str]],
        dry_run: bool,
        resume: Optional[str],
        fresh: bool = False,
    ) -> Dict[str, Any]:
        tasks = self._load_tasks()
        if task_ids:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.get("task_id") in id_set]
        if limit and limit > 0:
            tasks = tasks[:limit]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        completed_ids: Set[str] = set()
        resume_path: Optional[Path] = None
        persisted_records: List[Dict[str, Any]] = []
        record_index: Dict[str, int] = {}

        if resume:
            resume_path = Path(resume)
            if not resume_path.exists():
                print(f"  ▶ Resume target does not exist yet: {resume_path}")
                print("    A new results file will be created at this path.\n")
            else:
                raw_records = self._load_result_records(resume_path)
                persisted_records = self._latest_result_records(raw_records)
                if len(persisted_records) != len(raw_records):
                    duplicate_count = len(raw_records) - len(persisted_records)
                    self._write_result_records(resume_path, persisted_records)
                    print(f"  ▶ Cleaned {duplicate_count} duplicate result record(s) before resuming")
                completed_ids = self._load_completed_ids(resume_path)
                print(f"  ▶ Resuming from: {resume_path}")
                print(f"    Already completed: {len(completed_ids)} tasks")
        else:
            # Canonical per-dataset file (e.g. ``swev.jsonl``).
            canonical = self.output_dir / f"{self.benchmark_name}.jsonl"

            if fresh and canonical.exists():
                canonical.unlink()
                print(f"  ▶ Fresh run requested — removed previous results: {canonical}\n")

            if canonical.exists():
                resume_path = canonical
                raw_records = self._load_result_records(resume_path)
                persisted_records = self._latest_result_records(raw_records)
                if len(persisted_records) != len(raw_records):
                    duplicate_count = len(raw_records) - len(persisted_records)
                    self._write_result_records(resume_path, persisted_records)
                    print(f"  ▶ Cleaned {duplicate_count} duplicate result record(s)")
                completed_ids = self._load_completed_ids(resume_path)
                print(f"  ▶ Auto-resuming from: {resume_path}")
                print(f"    Already completed: {len(completed_ids)} tasks\n")

        if resume_path is not None:
            results_file = resume_path
            results_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            results_file = self.output_dir / f"{self.benchmark_name}.jsonl"

        if not persisted_records and results_file.exists():
            persisted_records = self._latest_result_records(self._load_result_records(results_file))
        for idx, record in enumerate(persisted_records):
            task_id = record.get("task_id")
            if task_id is not None:
                record_index[str(task_id)] = idx

        print(f"\n{'=' * 60}")
        print(f"  Benchmark: {self.benchmark_name}")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Workers: {self.workers}")
        model_label = self.model or os.getenv("LLM_MODEL_ID") or "(from env)"
        print(f"  Model: {model_label}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Task timeout: {self.task_timeout}s")
        if self._preds_completed_ids and not self.redo_existing:
            print(f"  Skip existing preds: {len(self._preds_completed_ids)} IDs from {self.preds_path}")
        if completed_ids:
            remaining = sum(1 for t in tasks if str(t.get("task_id", "")) not in completed_ids)
            print(f"  Resume: {len(completed_ids)} done, {remaining} remaining")
        print(f"{'=' * 60}\n")

        if dry_run:
            for task in tasks:
                task_id = str(task.get("task_id", ""))
                tag = " [SKIP]" if task_id in completed_ids else ""
                print(f"  [dry-run] {task_id}{tag}")
            return {"benchmark": self.benchmark_name, "total": len(tasks), "dry_run": True}

        pending_tasks = [task for task in tasks if str(task.get("task_id", "")) not in completed_ids]
        skipped = len(tasks) - len(pending_tasks)
        if pending_tasks:
            self._docker_preflight()

        results: List[Dict[str, Any]] = []
        passed_count = 0
        total_time = 0.0
        lock = threading.Lock()
        completed = 0
        total_pending = len(pending_tasks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_task = {
                executor.submit(
                    self.evaluate,
                    task,
                    task_id=str(task.get("task_id", f"task_{idx}")),
                ): task
                for idx, task in enumerate(pending_tasks)
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                task_id = str(task.get("task_id", ""))
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_id": task_id,
                        "passed": False,
                        "error": f"Runner exception: {type(exc).__name__}: {exc}",
                        "elapsed_s": 0.0,
                        "agent_diff": "",
                    }

                with lock:
                    completed += 1
                    results.append(result)
                    if result.get("passed") is True:
                        passed_count += 1
                    total_time += float(result.get("elapsed_s", 0.0) or 0.0)

                    self._upsert_result_record(persisted_records, record_index, result)
                    self._write_result_records(results_file, persisted_records)

                    status = "PASS" if result.get("passed") is True else "FAIL"
                    if result.get("passed") is None:
                        status = "UNFIN"
                    has_diff = bool(result.get("agent_diff"))
                    print(
                        f"  [{completed:>4}/{total_pending}] {status:<5} "
                        f"{task_id} diff={str(has_diff):<5} "
                        f"time={float(result.get('elapsed_s', 0.0) or 0.0):.1f}s"
                    )

        evaluated = len(results)
        new_pass_rate = (passed_count / evaluated * 100) if evaluated > 0 else 0.0
        combined = self._summarize_result_records(persisted_records)
        summary = {
            "benchmark": self.benchmark_name,
            "model": self.model or "(from env)",
            "total": len(tasks),
            "evaluated": combined["tasks"],
            "new_evaluated": evaluated,
            "skipped": skipped,
            "passed": combined["passed"],
            "failed": combined["failed"],
            "unfinished": combined["unfinished"],
            "pass_rate": combined["pass_rate"],
            "total_time_s": combined["total_time_s"],
            "avg_time_s": combined["avg_time_s"],
            "records_in_file": combined["records_in_file"],
            "new_passed": passed_count,
            "new_failed": sum(1 for r in results if r.get("passed") is False),
            "new_unfinished": sum(1 for r in results if r.get("passed") is None),
            "new_pass_rate": round(new_pass_rate, 2),
            "new_total_time_s": round(total_time, 2),
            "new_avg_time_s": round(total_time / evaluated, 2) if evaluated > 0 else 0,
            "timestamp": timestamp,
            "results_file": str(results_file),
            "trajectory_dir": str(self.trajectory_dir),
            "resumed_from": resume if resume else None,
            "workers": self.workers,
        }

        summary_file = self.output_dir / f"{self.benchmark_name}_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n{'=' * 60}")
        if skipped:
            print(f"  Resumed/Skipped: {skipped} tasks")
        print(
            f"  Combined results: {combined['passed']}/{combined['tasks']} passed "
            f"({combined['pass_rate']:.1f}%)"
        )
        if combined["unfinished"]:
            print(f"  Combined unfinished: {combined['unfinished']}")
        print(
            f"  New results: {passed_count}/{evaluated} passed "
            f"({new_pass_rate:.1f}%)"
        )
        print(f"  Output: {results_file}")
        print(f"  Summary: {summary_file}")
        print(f"{'=' * 60}\n")
        return summary

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: Optional[str] = None,
        fresh: bool = False,
    ) -> Dict[str, Any]:
        """Run the benchmark and export predictions for Docker evaluation."""
        # Use the constructor's resume_file (swev has custom clone-failure logic)
        effective_resume = resume or self.resume_file
        summary = self._run_parallel(
            limit=limit,
            task_ids=task_ids,
            dry_run=dry_run,
            resume=effective_resume,
            fresh=fresh,
        )

        if dry_run:
            return summary

        # Read back the results JSONL written by the base class and produce
        # a predictions file in the official SWE-bench format.
        results_file = Path(summary.get("results_file", ""))
        if not results_file.exists():
            return summary

        timestamp = summary.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        predictions_file, preds_json_file, diff_count, total = self._export_predictions(
            results_file=results_file,
            timestamp=timestamp,
        )
        print(f"\n{'=' * 60}")
        print(f"  Predictions: {predictions_file}")
        print(f"  Predictions (json): {preds_json_file}")
        print(f"  Predictions (latest): {self.output_dir / 'preds.json'}")
        print(f"  Diffs produced: {diff_count}/{total}")
        print("\n  To evaluate with Docker:")
        print(f"  bash scripts/run_swev_eval.sh {predictions_file}")
        print(f"{'=' * 60}\n")

        summary["predictions_file"] = str(predictions_file)
        summary["predictions_json_file"] = str(preds_json_file)
        summary["diff_count"] = diff_count
        summary["trajectory_dir"] = str(self.trajectory_dir)
        return summary


class _TaskTimeout(Exception):
    """Raised when a single task exceeds its wall-clock timeout."""


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run SWE-bench Verified benchmark (Phase 1: agent inference)")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "SWEV" / "test.jsonl"),
        help="Path to SWE-bench Verified JSONL file",
    )
    parser.add_argument("--output-dir", default=str(_DEFAULT_RESULTS_DIR))
    parser.add_argument("--trajectory-dir", default=str(_DEFAULT_TRAJECTORY_DIR))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument(
        "--filter",
        default="",
        help="Regex filter for instance_id/task_id (applied before --limit)",
    )
    parser.add_argument(
        "--slice",
        default="",
        help="Slice expression like `0:50` or `10:200:2` (applied after --filter/shuffle)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle tasks deterministically before slicing/limit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled",
    )
    parser.add_argument(
        "--repo-cache-dir",
        default=None,
        help="Directory to cache cloned repos between runs",
    )
    parser.add_argument(
        "--preds-path",
        default=None,
        help=(
            "Optional existing predictions file (.json/.jsonl). "
            "When provided, completed IDs are skipped unless --redo-existing is set."
        ),
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Do not skip IDs found in --preds-path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker threads for phase-1 inference (default: 1)",
    )
    parser.add_argument(
        "--model-name",
        default="whale-code",
        help="Model name for predictions file (default: whale-code)",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=3600,
        help="Per-instance agent wall-clock timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65536,
        help="Max output tokens per LLM call (reasoning+content). 0 disables the cap.",
    )
    parser.add_argument(
        "--docker-executable",
        default=os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker"),
        help="Container runtime executable to use for SWE-bench images",
    )
    parser.add_argument(
        "--docker-pull-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for pulling/starting one SWE-bench Docker image",
    )
    parser.add_argument(
        "--docker-container-timeout",
        default="2h",
        help="How long to keep each benchmark container alive (sleep duration)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RESULTS_FILE",
        help="Resume from a previous results JSONL file, skipping completed tasks",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing results file and start a fresh run",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only run Docker preflight checks and exit",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = SWEBenchVerifiedBenchmark(
        data_path=args.data_path,
        repo_cache_dir=args.repo_cache_dir,
        model_name=args.model_name,
        resume_file=args.resume,
        docker_executable=args.docker_executable,
        docker_container_timeout=args.docker_container_timeout,
        docker_pull_timeout=args.docker_pull_timeout,
        filter_spec=args.filter,
        slice_spec=args.slice,
        shuffle=args.shuffle,
        seed=args.seed,
        preds_path=args.preds_path,
        redo_existing=args.redo_existing,
        workers=args.workers,
        **BenchmarkRunner.runner_kwargs_from_args(args, include_task_timeout=True),
    )
    if args.preflight_only:
        bench._docker_preflight()
        print("SWEV preflight check passed.")
        return
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume, fresh=args.fresh)


if __name__ == "__main__":
    main()
