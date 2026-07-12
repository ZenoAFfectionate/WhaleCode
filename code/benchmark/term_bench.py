"""Terminal Bench 2.0 benchmark runner for Whale Code agent.

This runner evaluates tasks from ``data/TERM/test.jsonl``. Each task points to
its local task bundle (instruction/tests/task.toml-derived metadata) and the
official Docker image.

Execution model:
1. Hydrate a host workspace from the image's ``/app`` directory.
2. Mount that workspace back to ``/app`` inside a running container.
3. Run the Whale Code agent with file tools on host workspace + Bash in Docker.
4. Execute verifier command (default ``bash /tests/test.sh``) in Docker.
5. Parse ``/logs/verifier/reward.txt`` for pass/fail and persist benchmark stats.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from dotenv import load_dotenv

try:
    from .base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
        truncate_feedback,
    )
    from .swev_bench import DockerBashTool
except ImportError:
    from base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
        truncate_feedback,
    )
    from swev_bench import DockerBashTool


_TERM_ADDENDUM = """\
You are solving Terminal Bench 2.0 tasks inside a Linux container.

Your writable project directory is `/app`.
Use tools to inspect files, edit code/config/scripts, and run commands.

Guidelines:
- Focus only on task-required outputs/behavior.
- Prefer targeted reads/edits over broad changes.
- Keep fixes minimal, correct, and reproducible.
- Validate with lightweight checks when useful.
- Avoid destructive cleanup that can hide root-cause evidence.
- Call `Finish` with a brief summary when done.
"""

_TERM_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## Terminal Bench 2.0 Benchmark Override\n\n"
    + _TERM_ADDENDUM
)

_TERM_CONTAINER_WORKDIR = PurePosixPath("/app")


def _parse_positive_seconds(value: Any, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _normalize_memory(task: Dict[str, Any]) -> Optional[str]:
    memory = task.get("memory")
    if isinstance(memory, str) and memory.strip():
        return memory.strip()
    memory_mb = task.get("memory_mb")
    try:
        mb = int(float(memory_mb))
    except (TypeError, ValueError):
        return None
    if mb <= 0:
        return None
    return f"{mb}m"


def _normalize_cpus(task: Dict[str, Any]) -> Optional[str]:
    cpus = task.get("cpus")
    if cpus is None:
        return None
    try:
        value = float(cpus)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value.is_integer():
        return str(int(value))
    return str(value)


class TerminalBenchDockerWorkspace:
    """Container lifecycle wrapper for one Terminal Bench task."""

    def __init__(
        self,
        *,
        image: str,
        workspace: Path,
        tests_dir: Path,
        logs_dir: Path,
        executable: str = "docker",
        container_timeout: str = "3h",
        pull_timeout: int = 600,
        container_workdir: PurePosixPath = _TERM_CONTAINER_WORKDIR,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
    ):
        self.image = image
        self.workspace = workspace.expanduser().resolve()
        self.tests_dir = tests_dir.expanduser().resolve()
        self.logs_dir = logs_dir.expanduser().resolve()
        self.executable = executable
        self.container_timeout = container_timeout
        self.pull_timeout = pull_timeout
        self.container_workdir = container_workdir
        self.cpus = cpus
        self.memory = memory
        self.container_name = f"whale-term-{uuid.uuid4().hex[:8]}"
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
            "-v",
            f"{self.tests_dir}:/tests",
            "-v",
            f"{self.logs_dir}:/logs",
            "--rm",
        ]
        if self.cpus:
            cmd.extend(["--cpus", self.cpus])
        if self.memory:
            cmd.extend(["--memory", self.memory])
        cmd.extend([self.image, "sleep", self.container_timeout])
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

    def run(
        self,
        *,
        command: str,
        container_directory: PurePosixPath,
        timeout: int,
    ) -> subprocess.CompletedProcess:
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
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
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


class TerminalBenchComposeWorkspace:
    """Docker Compose lifecycle wrapper for one Terminal Bench task."""

    def __init__(
        self,
        *,
        image: str,
        compose_executable: str,
        compose_files: Sequence[Path],
        project_name: str,
        service_name: str,
        project_dir: Path,
        container_workdir: PurePosixPath = _TERM_CONTAINER_WORKDIR,
        pull_timeout: int = 600,
    ):
        self.image = image
        self.compose_executable = compose_executable
        self.compose_files = [path.expanduser().resolve() for path in compose_files]
        self.project_name = project_name
        self.service_name = service_name
        self.project_dir = project_dir.expanduser().resolve()
        self.container_workdir = container_workdir
        self.pull_timeout = pull_timeout
        self.container_id: Optional[str] = None

    def _base_cmd(self) -> List[str]:
        cmd = [self.compose_executable, "compose", "-p", self.project_name]
        for compose_file in self.compose_files:
            cmd.extend(["-f", str(compose_file)])
        return cmd

    def start(self) -> None:
        cmd = self._base_cmd() + [
            "up",
            "-d",
            "--force-recreate",
            "--remove-orphans",
            self.service_name,
        ]
        subprocess.run(
            cmd,
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            timeout=self.pull_timeout,
            check=True,
        )
        ps_cmd = self._base_cmd() + ["ps", "-q", self.service_name]
        result = subprocess.run(
            ps_cmd,
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        container_id = result.stdout.strip()
        if container_id:
            self.container_id = container_id

    def popen(
        self,
        *,
        command: str,
        container_directory: PurePosixPath,
    ) -> subprocess.Popen:
        cmd = self._base_cmd() + [
            "exec",
            "-T",
            "-w",
            str(container_directory),
            self.service_name,
            "bash",
            "-lc",
            command,
        ]
        return subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    def run(
        self,
        *,
        command: str,
        container_directory: PurePosixPath,
        timeout: int,
    ) -> subprocess.CompletedProcess:
        cmd = self._base_cmd() + [
            "exec",
            "-T",
            "-w",
            str(container_directory),
            self.service_name,
            "bash",
            "-lc",
            command,
        ]
        return subprocess.run(
            cmd,
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )

    def cleanup(self) -> None:
        cmd = self._base_cmd() + ["down", "--remove-orphans"]
        try:
            subprocess.run(
                cmd,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
        finally:
            self.container_id = None


class TerminalBench2Benchmark(BenchmarkRunner):
    """Evaluate Whale Code agent on Terminal Bench 2.0 tasks."""

    benchmark_name = "term_bench_2"

    def __init__(
        self,
        *args,
        docker_executable: str = "docker",
        docker_pull_timeout: int = 600,
        docker_container_timeout: str = "3h",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.docker_executable = docker_executable
        self.docker_pull_timeout = docker_pull_timeout
        self.docker_container_timeout = docker_container_timeout

    def _get_system_prompt(self) -> Optional[str]:
        return _TERM_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        return self._load_jsonl_tasks(
            task_transform=lambda task: {
                **task,
                "task_id": task.get("task_id") or task.get("id") or task.get("name"),
            }
        )

    def _create_agent(self, workspace: Path, docker_workspace: Optional[Any] = None):
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

    def _task_bundle_dir(self, task: Dict[str, Any]) -> Path:
        candidate = task.get("task_dir")
        if isinstance(candidate, str) and candidate.strip():
            task_dir = Path(candidate)
            if not task_dir.is_absolute():
                task_dir = (self.data_path.parent / task_dir).resolve()
            return task_dir
        return (self.data_path.parent / "tasks" / str(task.get("task_id"))).resolve()

    @staticmethod
    def _resolve_compose_file(task_dir: Path) -> Optional[Path]:
        candidates = [
            task_dir / "docker-compose.yml",
            task_dir / "docker-compose.yaml",
            task_dir / "compose.yml",
            task_dir / "compose.yaml",
            task_dir / "environment" / "docker-compose.yml",
            task_dir / "environment" / "docker-compose.yaml",
            task_dir / "environment" / "compose.yml",
            task_dir / "environment" / "compose.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _detect_compose_service(
        self,
        *,
        compose_files: Sequence[Path],
        project_dir: Path,
        task: Dict[str, Any],
    ) -> str:
        if isinstance(task.get("compose_service"), str) and task["compose_service"].strip():
            return task["compose_service"].strip()

        cmd = [self.docker_executable, "compose"]
        for compose_file in compose_files:
            cmd.extend(["-f", str(compose_file)])
        cmd.extend(["config", "--services"])

        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        services = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if not services:
            return "app"
        for candidate in ("app", "agent", "workspace", "main"):
            if candidate in services:
                return candidate
        return services[0]

    def _build_compose_plan(
        self,
        *,
        task: Dict[str, Any],
        task_dir: Path,
        workspace: Path,
        tests_dir: Path,
        logs_dir: Path,
        docker_image: str,
    ) -> Tuple[List[Path], str, str]:
        compose_mode = "generated"
        provided = self._resolve_compose_file(task_dir)

        compose_dir = workspace / ".term_bench" / "compose"
        compose_dir.mkdir(parents=True, exist_ok=True)

        if provided is not None:
            base_compose_file = provided
            project_dir = provided.parent
            compose_mode = "provided"
        else:
            generated = compose_dir / "docker-compose.generated.yml"
            base_payload: Dict[str, Any] = {
                "services": {
                    "app": {
                        "image": docker_image,
                    }
                }
            }
            generated.write_text(yaml.safe_dump(base_payload, sort_keys=False), encoding="utf-8")
            base_compose_file = generated
            project_dir = compose_dir

        service_name = self._detect_compose_service(
            compose_files=[base_compose_file],
            project_dir=project_dir,
            task=task,
        )

        override_file = compose_dir / "docker-compose.override.yml"
        service_override: Dict[str, Any] = {
            "working_dir": str(_TERM_CONTAINER_WORKDIR),
            "command": ["sleep", self.docker_container_timeout],
            "volumes": [
                f"{workspace.resolve()}:{_TERM_CONTAINER_WORKDIR}",
                f"{tests_dir.resolve()}:/tests",
                f"{logs_dir.resolve()}:/logs",
            ],
        }
        memory = _normalize_memory(task)
        cpus = _normalize_cpus(task)
        if memory:
            service_override["mem_limit"] = memory
        if cpus:
            service_override["cpus"] = cpus

        override_payload: Dict[str, Any] = {
            "services": {
                service_name: service_override,
            }
        }
        override_file.write_text(yaml.safe_dump(override_payload, sort_keys=False), encoding="utf-8")
        return [base_compose_file, override_file], service_name, compose_mode

    def _resolve_instruction(self, task: Dict[str, Any], task_dir: Path) -> str:
        text = task.get("instruction")
        if isinstance(text, str) and text.strip():
            return text.strip()
        instruction_file = task_dir / "instruction.md"
        if instruction_file.exists():
            return instruction_file.read_text(encoding="utf-8").strip()
        return "Follow the task requirements and modify files in /app."

    def _hydrate_workspace_from_image(self, *, image: str, workspace: Path) -> str:
        """Copy initial task files from image into host workspace.

        Returns the source directory copied from inside the container.
        """
        create = subprocess.run(
            [self.docker_executable, "create", image],
            capture_output=True,
            text=True,
            timeout=self.docker_pull_timeout,
            check=True,
        )
        container_id = create.stdout.strip()
        copied_from = ""
        try:
            inspect = subprocess.run(
                [self.docker_executable, "inspect", container_id, "--format", "{{.Config.WorkingDir}}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            workdir = inspect.stdout.strip() if inspect.returncode == 0 else ""
            candidates: List[str] = []
            if workdir and workdir != "/":
                candidates.append(workdir)
            candidates.extend(["/app", "/workspace"])

            seen: set[str] = set()
            unique_candidates: List[str] = []
            for path in candidates:
                if path not in seen:
                    seen.add(path)
                    unique_candidates.append(path)

            for path in unique_candidates:
                copy_proc = subprocess.run(
                    [self.docker_executable, "cp", f"{container_id}:{path}/.", str(workspace)],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if copy_proc.returncode == 0:
                    copied_from = path
                    break
        finally:
            subprocess.run(
                [self.docker_executable, "rm", "-f", container_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
        return copied_from

    def _run_agent_with_timeout(self, agent: Any, prompt: str, timeout_s: int) -> str:
        run_kwargs = self._benchmark_agent_run_kwargs()
        if timeout_s <= 0:
            result = agent.run(prompt, **run_kwargs)
            return "" if result is None else str(result).strip()

        result_holder: Dict[str, Any] = {}
        exception_holder: List[BaseException] = []

        def target() -> None:
            try:
                result_holder["value"] = agent.run(prompt, **run_kwargs)
            except BaseException as exc:
                exception_holder.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout_s)

        if thread.is_alive():
            raise _TaskTimeout(f"Agent did not finish within {timeout_s}s")
        if exception_holder:
            raise exception_holder[0]
        result = result_holder.get("value", "")
        return "" if result is None else str(result).strip()

    def _run_verifier(
        self,
        *,
        docker_workspace: Any,
        command: str,
        timeout_s: int,
        reward_file: Path,
    ) -> Dict[str, Any]:
        timed_out = False
        exit_code: Optional[int] = None
        output = ""
        start = time.time()
        try:
            proc = docker_workspace.run(
                command=command,
                container_directory=_TERM_CONTAINER_WORKDIR,
                timeout=timeout_s,
            )
            exit_code = proc.returncode
            output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            output = (stdout + stderr).strip()
            if output:
                output += "\n"
            output += f"Verifier timeout after {timeout_s}s."
        except Exception as exc:
            output = f"Verifier runner error: {type(exc).__name__}: {exc}"

        reward_text = reward_file.read_text(encoding="utf-8").strip() if reward_file.exists() else ""
        reward_value: Optional[int] = None
        if reward_text:
            if reward_text.startswith("1"):
                reward_value = 1
            elif reward_text.startswith("0"):
                reward_value = 0

        if reward_value is not None:
            passed = reward_value == 1
        else:
            passed = (not timed_out) and (exit_code == 0)

        return {
            "passed": passed,
            "timed_out": timed_out,
            "exit_code": exit_code,
            "reward_text": reward_text,
            "output": output,
            "elapsed_s": round(time.time() - start, 2),
        }

    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(task["task_id"])
        task_dir = self._task_bundle_dir(task)
        docker_image = str(task.get("docker_image") or "").strip()
        verifier_command = str(task.get("verifier_command") or "bash /tests/test.sh").strip()
        agent_timeout_s = _parse_positive_seconds(task.get("agent_timeout_sec"), self.task_timeout or 1800)
        verifier_timeout_s = _parse_positive_seconds(task.get("verifier_timeout_sec"), max(self.timeout, 1800))

        workspace = self._make_workspace(f"term_{task_id.replace('/', '_')}_")
        agent = None
        docker_workspace: Optional[Any] = None
        agent_response = ""
        prompt_history: List[str] = []
        result: Optional[Dict[str, Any]] = None
        compose_mode = "disabled"
        compose_service = ""

        logs_dir = workspace / ".term_bench" / "logs"
        tests_mount = workspace / ".term_bench" / "tests"
        reward_file = logs_dir / "verifier" / "reward.txt"
        verifier_output_file = logs_dir / "verifier_output.txt"
        logs_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        copied_from = ""
        try:
            if not task_dir.exists():
                return self._build_result(
                    task_id,
                    passed=False,
                    error=f"Task bundle directory not found: {task_dir}",
                    start_time=start,
                )
            if not docker_image:
                return self._build_result(
                    task_id,
                    passed=False,
                    error="docker_image is missing in task definition",
                    start_time=start,
                )

            tests_src = task_dir / "tests"
            if not tests_src.exists():
                return self._build_result(
                    task_id,
                    passed=False,
                    error=f"Task tests directory not found: {tests_src}",
                    start_time=start,
                )
            shutil.copytree(tests_src, tests_mount, dirs_exist_ok=True)

            copied_from = self._hydrate_workspace_from_image(image=docker_image, workspace=workspace)
            if not copied_from:
                return self._build_result(
                    task_id,
                    passed=False,
                    error=(
                        "Failed to hydrate workspace from Docker image. "
                        "Tried image working directory, /app, and /workspace."
                    ),
                    start_time=start,
                    extra={
                        "docker_image": docker_image,
                        "task_dir": str(task_dir),
                    },
                )

            instruction = self._resolve_instruction(task, task_dir)
            prompt = (
                f"Solve Terminal Bench 2.0 task `{task_id}`.\n\n"
                f"Task instruction:\n{instruction}\n\n"
                f"Constraints:\n"
                f"- Work only under `/app`.\n"
                f"- Runner executes verifier after `Finish`: `{verifier_command}`.\n"
                f"- Agent timeout: {agent_timeout_s}s; verifier timeout: {verifier_timeout_s}s.\n"
                f"- Make minimal edits and prioritize passing verifier criteria.\n"
            )
            prompt_history.append(prompt)

            if bool(task.get("custom_docker_compose", False)):
                compose_files, compose_service, compose_mode = self._build_compose_plan(
                    task=task,
                    task_dir=task_dir,
                    workspace=workspace,
                    tests_dir=tests_mount,
                    logs_dir=logs_dir,
                    docker_image=docker_image,
                )
                docker_workspace = TerminalBenchComposeWorkspace(
                    image=docker_image,
                    compose_executable=self.docker_executable,
                    compose_files=compose_files,
                    project_name=f"whale-term-{uuid.uuid4().hex[:8]}",
                    service_name=compose_service,
                    project_dir=compose_files[0].parent,
                    pull_timeout=self.docker_pull_timeout,
                )
            else:
                docker_workspace = TerminalBenchDockerWorkspace(
                    image=docker_image,
                    workspace=workspace,
                    tests_dir=tests_mount,
                    logs_dir=logs_dir,
                    executable=self.docker_executable,
                    container_timeout=self.docker_container_timeout,
                    pull_timeout=self.docker_pull_timeout,
                    cpus=_normalize_cpus(task),
                    memory=_normalize_memory(task),
                )
                compose_mode = "disabled"
            docker_workspace.start()

            verifier_script = tests_mount / "test.sh"
            if not verifier_script.exists():
                result = self._build_result(
                    task_id,
                    passed=False,
                    error=f"Verifier script not found: {verifier_script}",
                    start_time=start,
                    extra={
                        "docker_image": docker_image,
                        "task_dir": str(task_dir),
                        "compose_mode": compose_mode,
                        "compose_service": compose_service,
                    },
                )
                return result
            try:
                verifier_script.chmod(0o755)
            except Exception:
                pass

            agent = self._create_agent(workspace, docker_workspace=docker_workspace)
            try:
                agent_response = self._run_agent_with_timeout(agent, prompt, agent_timeout_s)
            except _TaskTimeout as exc:
                result = self._build_result(
                    task_id,
                    passed=False,
                    error=f"Agent timeout: {exc}",
                    agent_response=agent_response,
                    start_time=start,
                    extra={
                        "docker_image": docker_image,
                        "agent_timeout_sec": agent_timeout_s,
                        "verifier_timeout_sec": verifier_timeout_s,
                        "task_dir": str(task_dir),
                        "compose_mode": compose_mode,
                        "compose_service": compose_service,
                    },
                )
                return result
            except Exception as exc:
                result = self._build_result(
                    task_id,
                    passed=False,
                    error=f"Agent error: {type(exc).__name__}: {exc}",
                    agent_response=agent_response,
                    start_time=start,
                    extra={
                        "docker_image": docker_image,
                        "agent_timeout_sec": agent_timeout_s,
                        "verifier_timeout_sec": verifier_timeout_s,
                        "task_dir": str(task_dir),
                        "compose_mode": compose_mode,
                        "compose_service": compose_service,
                    },
                )
                return result

            verification = self._run_verifier(
                docker_workspace=docker_workspace,
                command=verifier_command,
                timeout_s=verifier_timeout_s,
                reward_file=reward_file,
            )
            verifier_output_file.write_text(verification["output"], encoding="utf-8")
            feedback = truncate_feedback(verification["output"], max_lines=140, max_chars=18000)

            result = self._build_result(
                task_id,
                passed=bool(verification["passed"]),
                error=None if verification["passed"] else (feedback or "Terminal Bench verifier failed."),
                agent_response=agent_response,
                start_time=start,
                extra={
                    "docker_image": docker_image,
                    "task_dir": str(task_dir),
                    "task_category": task.get("category"),
                    "task_difficulty": task.get("difficulty"),
                    "task_tags": task.get("tags", []),
                    "copied_from": copied_from,
                    "agent_timeout_sec": agent_timeout_s,
                    "verifier_timeout_sec": verifier_timeout_s,
                    "verifier_command": verifier_command,
                    "verifier_exit_code": verification["exit_code"],
                    "verifier_timed_out": verification["timed_out"],
                    "verifier_reward_text": verification["reward_text"],
                    "verifier_elapsed_s": verification["elapsed_s"],
                    "compose_mode": compose_mode,
                    "compose_service": compose_service,
                },
            )
            return result
        except Exception as exc:
            result = self._build_result(
                task_id,
                passed=False,
                error=f"Runner exception: {type(exc).__name__}: {exc}",
                agent_response=agent_response,
                start_time=start,
                extra={
                    "docker_image": docker_image,
                    "task_dir": str(task_dir),
                    "copied_from": copied_from,
                    "compose_mode": compose_mode,
                    "compose_service": compose_service,
                },
            )
            return result
        finally:
            self._finalize_workspace_task(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=prompt_history,
                result=result,
                artifact_paths=[
                    ".term_bench/logs/verifier/reward.txt",
                    ".term_bench/logs/verifier/ctrf.json",
                    ".term_bench/logs/verifier_output.txt",
                ],
                extra={
                    "task_dir": str(task_dir),
                    "docker_image": docker_image,
                    "copied_from": copied_from,
                    "compose_mode": compose_mode,
                    "compose_service": compose_service,
                },
            )
            if docker_workspace is not None:
                docker_workspace.cleanup()

    def _use_subprocess_task_timeout(self) -> bool:
        """Use in-task timeout controls to preserve Docker cleanup guarantees."""
        return False

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: Optional[str] = None,
    ) -> Dict[str, Any]:
        summary = super().run(limit=limit, task_ids=task_ids, dry_run=dry_run, resume=resume)
        if dry_run:
            return summary

        timestamp = str(summary.get("timestamp", time.strftime("%Y%m%d_%H%M%S")))
        results_file = Path(str(summary.get("results_file", "")))
        if not results_file.exists():
            return summary

        records = self._latest_result_records(self._load_result_records(results_file))
        task_meta = {str(task["task_id"]): task for task in self._load_tasks() if task.get("task_id")}

        def add_group_breakdown(field: str) -> Dict[str, Dict[str, Any]]:
            grouped: Dict[str, Dict[str, Any]] = {}
            for record in records:
                task_id = str(record.get("task_id", ""))
                meta = task_meta.get(task_id, {})
                key = str(meta.get(field) or "unknown")
                bucket = grouped.setdefault(
                    key,
                    {"tasks": 0, "passed": 0, "failed": 0, "unfinished": 0, "pass_rate": 0.0},
                )
                bucket["tasks"] += 1
                passed = record.get("passed")
                if passed is True:
                    bucket["passed"] += 1
                elif passed is False:
                    bucket["failed"] += 1
                else:
                    bucket["unfinished"] += 1
            for bucket in grouped.values():
                tasks = bucket["tasks"]
                bucket["pass_rate"] = round((bucket["passed"] / tasks * 100) if tasks else 0.0, 2)
            return dict(sorted(grouped.items(), key=lambda item: item[0]))

        breakdown = {
            "benchmark": self.benchmark_name,
            "timestamp": timestamp,
            "records": len(records),
            "by_category": add_group_breakdown("category"),
            "by_difficulty": add_group_breakdown("difficulty"),
        }

        breakdown_file = self.output_dir / f"{self.benchmark_name}_{timestamp}_breakdown.json"
        with breakdown_file.open("w", encoding="utf-8") as f:
            json.dump(breakdown, f, indent=2, ensure_ascii=False)

        summary["breakdown_file"] = str(breakdown_file)

        summary_file = self.output_dir / f"{self.benchmark_name}_{timestamp}_summary.json"
        if summary_file.exists():
            try:
                payload = json.loads(summary_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = dict(summary)
            payload["breakdown_file"] = str(breakdown_file)
            payload["category_breakdown"] = breakdown["by_category"]
            payload["difficulty_breakdown"] = breakdown["by_difficulty"]
            summary_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n{'=' * 60}")
        print("  Terminal Bench breakdown by category:")
        for key, value in breakdown["by_category"].items():
            print(
                f"    - {key}: {value['passed']}/{value['tasks']} "
                f"({value['pass_rate']:.1f}%), unfinished={value['unfinished']}"
            )
        print("  Terminal Bench breakdown by difficulty:")
        for key, value in breakdown["by_difficulty"].items():
            print(
                f"    - {key}: {value['passed']}/{value['tasks']} "
                f"({value['pass_rate']:.1f}%), unfinished={value['unfinished']}"
            )
        print(f"  Breakdown: {breakdown_file}")
        print(f"{'=' * 60}\n")

        return summary


class _TaskTimeout(Exception):
    """Raised when a task agent run exceeds its timeout."""


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run Terminal Bench 2.0 benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "TERM" / "test.jsonl"),
        help="Path to Terminal Bench 2.0 JSONL file",
    )
    BenchmarkRunner.add_shared_run_args(
        parser,
        default_temperature=1.0,
        default_max_steps=128,
        default_timeout=120,
        include_task_timeout=True,
        default_task_timeout=0,
    )
    parser.add_argument(
        "--docker-executable",
        default=os.getenv("TERM_DOCKER_EXECUTABLE", "docker"),
        help="Container runtime executable to use for Terminal Bench images",
    )
    parser.add_argument(
        "--docker-pull-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for pulling/starting one Terminal Bench Docker image",
    )
    parser.add_argument(
        "--docker-container-timeout",
        default="3h",
        help="How long to keep each benchmark container alive (sleep duration)",
    )
    args = parser.parse_args()

    bench = TerminalBench2Benchmark(
        data_path=args.data_path,
        docker_executable=args.docker_executable,
        docker_pull_timeout=args.docker_pull_timeout,
        docker_container_timeout=args.docker_container_timeout,
        **BenchmarkRunner.runner_kwargs_from_args(args, include_task_timeout=True),
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
