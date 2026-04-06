"""SWE-bench Verified benchmark runner for Whale Code agent.

Two-phase evaluation:
  Phase 1 (this file): Run the agent on each instance, collect diffs,
      and output a predictions JSONL compatible with the official harness.
  Phase 2 (scripts/run_swev_eval.sh): Feed predictions to
      ``swebench.harness.run_evaluation`` for Docker-based grading.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Set

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
2. Read only the functions/classes you need to understand the issue.
3. Diagnose the root cause and decide the fix.
4. Edit the minimal set of files required for a correct fix.
5. Verify lightly if it is fast and low-risk; otherwise skip local testing.
6. Call `Finish` alone with a brief summary of what changed and why.

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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.pull_timeout,
            check=True,
        )
        self.container_id = result.stdout.strip()

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
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

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

    def _get_system_prompt(self) -> Optional[str]:
        """Use the SWE-bench-specific system prompt."""
        return _SWEV_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        return self._load_jsonl_tasks(
            task_transform=lambda task: {
                **task,
                "task_id": task.get("instance_id", task.get("task_id")),
            }
        )

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

    # ------------------------------------------------------------------
    # Repository management
    # ------------------------------------------------------------------

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
        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--filter=blob:none", url, str(target)],
                capture_output=True, text=True, timeout=600, check=True,
            )
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=str(target), capture_output=True, text=True, timeout=120, check=True,
            )
            return target
        except subprocess.TimeoutExpired:
            print(f"\n  [WARN] Clone timed out for {repo}")
            shutil.rmtree(target, ignore_errors=True)
            return None
        except subprocess.CalledProcessError as exc:
            print(f"\n  [WARN] Clone failed for {repo}: {(exc.stderr or '')[:200]}")
            shutil.rmtree(target, ignore_errors=True)
            return None
        except Exception as exc:
            print(f"\n  [WARN] Clone error for {repo}: {exc}")
            shutil.rmtree(target, ignore_errors=True)
            return None

    def _clone_repo_from_cache(self, cached: Path, base_commit: str, repo_slug: str) -> Optional[Path]:
        """Materialize an isolated temp workspace from the cached repo."""
        target = self._make_workspace(f"swev_{repo_slug}_")
        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--shared", "--no-checkout", str(cached), str(target)],
                capture_output=True, text=True, timeout=180, check=True,
            )
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=str(target), capture_output=True, text=True, timeout=120, check=True,
            )
            return target
        except subprocess.TimeoutExpired:
            print(f"\n  [WARN] Local cache materialization timed out for {cached.name}")
            shutil.rmtree(target, ignore_errors=True)
            return None
        except subprocess.CalledProcessError as exc:
            print(f"\n  [WARN] Local cache materialization failed for {cached.name}: {(exc.stderr or '')[:200]}")
            shutil.rmtree(target, ignore_errors=True)
            return None
        except Exception as exc:
            print(f"\n  [WARN] Local cache materialization error for {cached.name}: {exc}")
            shutil.rmtree(target, ignore_errors=True)
            return None

    def _clone_repo(self, repo: str, base_commit: str) -> Optional[Path]:
        """Clone a GitHub repo at a specific commit.

        Uses ``--filter=blob:none`` for a blobless clone (much faster for
        large repos like astropy/django) — blobs are fetched on demand.
        When ``repo_cache_dir`` is set, the cache repo is used only as the
        source of truth and each task gets an isolated temp workspace derived
        from it. Returns the workspace path, or None on failure.
        """
        repo_slug = repo.replace("/", "__")

        # Check cache first
        if self.repo_cache_dir:
            cached = self.repo_cache_dir / repo_slug
            if cached.exists():
                # Try direct checkout
                if self._reset_cached_repo(cached, base_commit):
                    isolated = self._clone_repo_from_cache(cached, base_commit, repo_slug)
                    if isolated:
                        return isolated
                    return cached

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
                    return cached

                print(f"\n  [WARN] Retry checkout also failed for {repo}@{base_commit[:10]}")
                shutil.rmtree(cached, ignore_errors=True)

            # Re-clone into cache path
            fresh = self._clone_repo_to_target(repo, base_commit, cached)
            if fresh:
                isolated = self._clone_repo_from_cache(fresh, base_commit, repo_slug)
                if isolated:
                    return isolated
                return fresh

            # Fallback to temp if cache clone failed
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

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
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
                f"1. **Identify the bug location**: Grep for key symbols from the "
                f"issue (class names, method names, error messages). Narrow down "
                f"to the smallest relevant set of files.\n"
                f"2. **Read the relevant code**: Read the specific function or "
                f"method (not the whole file). Understand the logic.\n"
                f"3. **Diagnose**: Reason from the code before editing: "
                f"(a) what the current code does wrong, (b) what it should do instead.\n"
                f"4. **Edit**: Make the minimal fix needed for correctness.\n"
                f"5. **Verify**: Re-read the edited lines. If there is an obvious "
                f"fast reproducer or targeted test, run it; otherwise skip local testing.\n"
                f"6. **Respond**: Call `Finish` alone with a concise summary of what changed and why.\n"
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

    def _evaluate_task_with_timeout(self, task: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """SWE-bench uses its own agent timeout so partial diffs can be recovered."""
        return self._evaluate_task(task)

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

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark and export predictions for Docker evaluation."""
        # Use the constructor's resume_file (swev has custom clone-failure logic)
        effective_resume = resume or self.resume_file

        summary = super().run(
            limit=limit, task_ids=task_ids, dry_run=dry_run,
            resume=effective_resume,
        )

        if dry_run:
            return summary

        # Read back the results JSONL written by the base class and produce
        # a predictions file in the official SWE-bench format.
        results_file = Path(summary.get("results_file", ""))
        if not results_file.exists():
            return summary

        timestamp = summary.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        predictions_file = self.output_dir / f"swev_predictions_{timestamp}.jsonl"

        diff_count = 0
        final_results = self._latest_result_records(self._load_result_records(results_file))
        with open(predictions_file, "w", encoding="utf-8") as fout:
            for result in final_results:
                instance_id = result.get("task_id", "")
                agent_diff = result.get("agent_diff", "")
                if agent_diff:
                    diff_count += 1
                prediction = {
                    "instance_id": instance_id,
                    "model_name_or_path": self.model_name,
                    "model_patch": agent_diff,
                }
                fout.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        total = len(final_results)
        print(f"\n{'=' * 60}")
        print(f"  Predictions: {predictions_file}")
        print(f"  Diffs produced: {diff_count}/{total}")
        print("\n  To evaluate with Docker:")
        print(f"  bash scripts/run_swev_eval.sh {predictions_file}")
        print(f"{'=' * 60}\n")

        summary["predictions_file"] = str(predictions_file)
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
        "--repo-cache-dir",
        default=None,
        help="Directory to cache cloned repos between runs",
    )
    parser.add_argument(
        "--model-name",
        default="whale-code",
        help="Model name for predictions file (default: whale-code)",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=1200,
        help="Per-instance agent wall-clock timeout in seconds (default: 1200)",
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = SWEBenchVerifiedBenchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_steps=args.max_steps,
        timeout=args.timeout,
        repo_cache_dir=args.repo_cache_dir,
        model_name=args.model_name,
        task_timeout=args.task_timeout,
        resume_file=args.resume,
        trajectory_dir=args.trajectory_dir,
        docker_executable=args.docker_executable,
        docker_container_timeout=args.docker_container_timeout,
        docker_pull_timeout=args.docker_pull_timeout,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
