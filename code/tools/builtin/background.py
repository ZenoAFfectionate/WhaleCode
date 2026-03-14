"""Background execution tools for long-running workspace commands."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolResponse, tool_action
from ..errors import ToolErrorCode
from ._code_utils import (
    apply_line_limit,
    atomic_write,
    ensure_working_dir,
    relative_display,
    resolve_path,
    safe_decode_output,
)
from .bash import BashTool
from .task_tool import TaskManager


VALID_BACKGROUND_STATUSES = {
    "running",
    "completed",
    "failed",
    "timeout",
    "cancelled",
    "error",
    "interrupted",
}


@dataclass
class BackgroundTaskRecord:
    id: str
    command: str
    directory: str
    status: str = "running"
    timeout_ms: int = 300000
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str = ""
    exit_code: Optional[int] = None
    pid: Optional[int] = None
    linked_task_id: Optional[int] = None
    complete_task_on_success: bool = False
    mark_task_in_progress: bool = False
    cancel_requested: bool = False
    stdout_preview: str = ""
    stderr_preview: str = ""
    output_truncated: bool = False
    error: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BackgroundTaskRecord":
        return cls(**payload)


class BackgroundManager:
    """Persistent background-process manager with a completion notification queue."""

    DEFAULT_TIMEOUT_MS = 300000
    MAX_TIMEOUT_MS = BashTool.MAX_TIMEOUT_MS

    def __init__(
        self,
        project_root: str | Path,
        persistence_dir: str = "memory/background",
        tasks_dir: str = "memory/tasks",
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.persistence_dir = (self.project_root / persistence_dir).resolve()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.task_manager = TaskManager(self.project_root / tasks_dir)
        self._lock = threading.RLock()
        self._tasks: Dict[str, BackgroundTaskRecord] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._notification_queue: List[Dict[str, Any]] = []
        self._policy = BashTool(project_root=str(self.project_root), working_dir=str(self.project_root))
        self._load_existing_tasks()

    def start_task(
        self,
        command: str,
        *,
        working_dir: str | Path | None,
        directory: str = ".",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        linked_task_id: Optional[int] = None,
        mark_task_in_progress: bool = False,
        complete_task_on_success: bool = False,
    ) -> BackgroundTaskRecord:
        command = (command or "").strip()
        if not command:
            raise ValueError("command must be a non-empty string")
        if not isinstance(timeout_ms, int) or timeout_ms < 1 or timeout_ms > self.MAX_TIMEOUT_MS:
            raise ValueError(
                f"timeout_ms must be an integer between 1 and {self.MAX_TIMEOUT_MS}"
            )

        try:
            target_dir = resolve_path(self.project_root, working_dir, directory)
        except ValueError as exc:
            raise PermissionError("directory escapes the workspace root") from exc
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Working directory not found: {directory}")

        policy_error = self._policy.validate_command_policy(command)
        if policy_error:
            raise PermissionError(policy_error)

        if linked_task_id is not None:
            self.task_manager.get(linked_task_id)
            if mark_task_in_progress:
                self.task_manager.update(linked_task_id, status="in_progress")

        record = BackgroundTaskRecord(
            id=uuid.uuid4().hex[:8],
            command=command,
            directory=relative_display(self.project_root, target_dir),
            timeout_ms=timeout_ms,
            linked_task_id=linked_task_id,
            complete_task_on_success=complete_task_on_success,
            mark_task_in_progress=mark_task_in_progress,
        )

        with self._lock:
            self._tasks[record.id] = record
            self._save(record)
            thread = threading.Thread(
                target=self._execute_task,
                args=(record.id, target_dir),
                daemon=True,
                name=f"background-task-{record.id}",
            )
            self._threads[record.id] = thread
            thread.start()

        return record

    def get_task(self, background_id: str) -> BackgroundTaskRecord:
        with self._lock:
            task = self._tasks.get(background_id)
        if not task:
            raise ValueError(f"Background task {background_id} not found")
        return task

    def list_tasks(self, status: Optional[str] = None) -> List[BackgroundTaskRecord]:
        with self._lock:
            records = list(self._tasks.values())
        if status:
            if status not in VALID_BACKGROUND_STATUSES and status != "all":
                raise ValueError(
                    "status must be one of: all, running, completed, failed, timeout, cancelled, error, interrupted"
                )
            if status != "all":
                records = [record for record in records if record.status == status]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def cancel_task(self, background_id: str) -> tuple[BackgroundTaskRecord, str]:
        with self._lock:
            record = self._tasks.get(background_id)
            if not record:
                raise ValueError(f"Background task {background_id} not found")

            if record.status != "running":
                return record, f"Background task {background_id} is already {record.status}."

            record.cancel_requested = True
            self._save(record)
            process = self._processes.get(background_id)

        if process is None:
            return record, (
                f"Cancellation requested for background task {background_id}. "
                "The subprocess has not started yet; final status will update shortly."
            )

        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except Exception as exc:
            return record, f"Cancellation requested but terminate raised: {exc}"

        return record, f"Cancellation signal sent to background task {background_id}."

    def drain_notifications(self) -> List[Dict[str, Any]]:
        with self._lock:
            notifications = list(self._notification_queue)
            self._notification_queue.clear()
        return notifications

    def format_notifications(self, notifications: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for notification in notifications:
            status = notification["status"]
            bg_id = notification["id"]
            directory = notification["directory"]
            command = notification["command"]
            exit_suffix = ""
            if notification.get("exit_code") is not None:
                exit_suffix = f" exit_code={notification['exit_code']}"
            linked_suffix = ""
            if notification.get("linked_task_id") is not None:
                linked_suffix = f" linked_task=#{notification['linked_task_id']}"
            lines.append(
                f"[bg:{bg_id}] {status}{exit_suffix} dir={directory}{linked_suffix} cmd={command}"
            )
            preview = notification.get("preview") or notification.get("error") or "(no output)"
            preview = preview.strip() or "(no output)"
            lines.append(preview)
        return "\n".join(lines)

    def _load_existing_tasks(self) -> None:
        for file_path in sorted(self.persistence_dir.glob("bg_*.json")):
            try:
                record = BackgroundTaskRecord.from_dict(
                    json.loads(file_path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue

            if record.status == "running":
                record.status = "interrupted"
                record.note = (
                    "Recovered from disk without a live subprocess attachment. "
                    "This background task was interrupted by a process restart."
                )
                record.finished_at = record.finished_at or datetime.now().isoformat()
                self._save(record)

            self._tasks[record.id] = record

    def _task_path(self, background_id: str) -> Path:
        return self.persistence_dir / f"bg_{background_id}.json"

    def _save(self, record: BackgroundTaskRecord) -> None:
        atomic_write(
            self._task_path(record.id),
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )

    def _execute_task(self, background_id: str, target_dir: Path) -> None:
        with self._lock:
            record = self._tasks[background_id]
            command = record.command
            timeout_ms = record.timeout_ms
            cancel_requested = record.cancel_requested

        if cancel_requested:
            with self._lock:
                record = self._tasks[background_id]
                record.status = "cancelled"
                record.finished_at = datetime.now().isoformat()
                record.note = "Background task was cancelled before the subprocess started."
                self._save(record)
                self._notification_queue.append(self._make_notification(record))
            return

        stdout = b""
        stderr = b""
        error_message = ""
        exit_code: Optional[int] = None
        status = "error"

        try:
            process = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PROJECT_ROOT": str(self.project_root)},
            )
            with self._lock:
                self._processes[background_id] = process
                record = self._tasks[background_id]
                record.pid = process.pid
                self._save(record)

            try:
                stdout, stderr = process.communicate(timeout=timeout_ms / 1000)
                exit_code = process.returncode
                with self._lock:
                    cancel_requested = self._tasks[background_id].cancel_requested
                if cancel_requested:
                    status = "cancelled"
                else:
                    status = "completed" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                status = "timeout"
                error_message = f"Command exceeded timeout of {timeout_ms} ms."

        except Exception as exc:
            error_message = str(exc)
            status = "error"
        finally:
            with self._lock:
                self._processes.pop(background_id, None)

        stdout_text, stdout_bytes_truncated = safe_decode_output(stdout)
        stderr_text, stderr_bytes_truncated = safe_decode_output(stderr)
        stdout_text, stdout_lines_truncated = apply_line_limit(stdout_text)
        stderr_text, stderr_lines_truncated = apply_line_limit(stderr_text)
        truncated = any(
            [
                stdout_bytes_truncated,
                stderr_bytes_truncated,
                stdout_lines_truncated,
                stderr_lines_truncated,
            ]
        )

        with self._lock:
            record = self._tasks[background_id]
            record.status = status
            record.finished_at = datetime.now().isoformat()
            record.exit_code = exit_code
            record.stdout_preview = stdout_text
            record.stderr_preview = stderr_text
            record.output_truncated = truncated
            record.error = error_message

            if status == "cancelled" and not record.note:
                record.note = "Background task was cancelled by the agent."
            elif status == "timeout" and not record.note:
                record.note = error_message
            elif status == "failed" and exit_code is not None:
                record.note = f"Command exited with code {exit_code}."
            elif status == "completed" and record.complete_task_on_success and record.linked_task_id is not None:
                try:
                    self.task_manager.update(record.linked_task_id, status="completed")
                    record.note = f"Linked task #{record.linked_task_id} marked completed."
                except Exception as exc:
                    record.note = (
                        f"Command completed, but failed to mark linked task #{record.linked_task_id} completed: {exc}"
                    )

            self._save(record)
            self._notification_queue.append(self._make_notification(record))

    @staticmethod
    def _make_notification(record: BackgroundTaskRecord) -> Dict[str, Any]:
        preview = record.stdout_preview or record.stderr_preview or record.note or record.error or "(no output)"
        return {
            "id": record.id,
            "status": record.status,
            "directory": record.directory,
            "command": record.command,
            "exit_code": record.exit_code,
            "linked_task_id": record.linked_task_id,
            "preview": preview,
            "error": record.error,
            "output_truncated": record.output_truncated,
        }


_BACKGROUND_MANAGERS: Dict[str, BackgroundManager] = {}
_BACKGROUND_MANAGERS_LOCK = threading.Lock()


def get_background_manager(
    project_root: str | Path,
    persistence_dir: str = "memory/background",
    tasks_dir: str = "memory/tasks",
) -> BackgroundManager:
    root = str(Path(project_root).expanduser().resolve())
    with _BACKGROUND_MANAGERS_LOCK:
        manager = _BACKGROUND_MANAGERS.get(root)
        if manager is None:
            manager = BackgroundManager(
                project_root=root,
                persistence_dir=persistence_dir,
                tasks_dir=tasks_dir,
            )
            _BACKGROUND_MANAGERS[root] = manager
        return manager


class BackgroundTool(Tool):
    """Expandable toolset for long-running background workspace commands."""

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        persistence_dir: str = "memory/background",
        tasks_dir: str = "memory/tasks",
    ):
        super().__init__(
            name="BackgroundSystem",
            description=(
                "Run long workspace commands in the background and keep the agent moving. "
                "Use `background_run` for slow builds, tests, installs, or scripts that should not block the current reasoning loop. "
                "Use `background_check` to inspect status and `background_cancel` to stop a running task. Completed results are injected before the next model call."
            ),
            expandable=True,
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = ensure_working_dir(self.project_root, working_dir)
        self.manager = get_background_manager(
            self.project_root,
            persistence_dir=persistence_dir,
            tasks_dir=tasks_dir,
        )

    def get_parameters(self) -> List[Any]:
        return []

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message=(
                "BackgroundSystem is an expandable tool. Use `background_run`, `background_check`, or `background_cancel`."
            ),
        )

    @tool_action(
        name="background_run",
        description=(
            "Start a long-running shell command in the background. Use this instead of blocking `Bash` when a build, test suite, install, or script may take a while and you can keep working meanwhile."
        ),
    )
    def background_run(
        self,
        command: str,
        directory: str = ".",
        timeout_ms: int = BackgroundManager.DEFAULT_TIMEOUT_MS,
        task_id: int = None,
        mark_task_in_progress: bool = False,
        complete_task_on_success: bool = False,
    ) -> ToolResponse:
        """Start a background command and return immediately.

        Args:
            command: Shell command to execute asynchronously inside the workspace.
            directory: Working directory relative to the workspace root.
            timeout_ms: Maximum runtime in milliseconds before the command is terminated.
            task_id: Optional persistent task ID to associate with this background run.
            mark_task_in_progress: If true and `task_id` is provided, mark that task as `in_progress` before launching.
            complete_task_on_success: If true and `task_id` is provided, mark that task as `completed` when the command exits successfully.
        """
        try:
            record = self.manager.start_task(
                command,
                working_dir=self.working_dir,
                directory=directory,
                timeout_ms=timeout_ms,
                linked_task_id=task_id,
                mark_task_in_progress=mark_task_in_progress,
                complete_task_on_success=complete_task_on_success,
            )
        except PermissionError as exc:
            return ToolResponse.error(code=ToolErrorCode.ACCESS_DENIED, message=str(exc))
        except FileNotFoundError as exc:
            return ToolResponse.error(code=ToolErrorCode.NOT_FOUND, message=str(exc))
        except ValueError as exc:
            message = str(exc)
            code = ToolErrorCode.NOT_FOUND if "not found" in message.lower() else ToolErrorCode.INVALID_PARAM
            return ToolResponse.error(code=code, message=message)
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to start background task: {exc}",
            )

        linked_text = (
            f"\nLinked task: #{record.linked_task_id}"
            if record.linked_task_id is not None
            else ""
        )

        return ToolResponse.success(
            text=(
                f"Started background task {record.id}\n"
                f"Command: {record.command}\n"
                f"Directory: {record.directory}\n"
                f"Timeout: {record.timeout_ms} ms\n"
                f"Status: {record.status}{linked_text}\n"
                "Completion updates will be injected automatically before the next model call."
            ),
            data={
                "background_task": record.to_dict(),
            },
        )

    @tool_action(
        name="background_check",
        description=(
            "Inspect one background task or list all background tasks. Use this when you need explicit status, exit code, or captured output previews."
        ),
    )
    def background_check(
        self,
        background_id: str = None,
        status: str = "all",
        include_output: bool = False,
    ) -> ToolResponse:
        """Inspect background task status.

        Args:
            background_id: Optional background task ID. Omit it to list all known background tasks.
            status: Optional filter when listing all tasks.
            include_output: When checking a single task, include captured stdout/stderr previews.
        """
        try:
            if background_id:
                record = self.manager.get_task(background_id)
                lines = [
                    f"Background task {record.id}",
                    f"Status: {record.status}",
                    f"Command: {record.command}",
                    f"Directory: {record.directory}",
                    f"Timeout: {record.timeout_ms} ms",
                    f"Exit code: {record.exit_code if record.exit_code is not None else '[running]'}",
                    f"PID: {record.pid if record.pid is not None else '[not started]'}",
                    f"Linked task: #{record.linked_task_id}" if record.linked_task_id is not None else "Linked task: none",
                    f"Created: {record.created_at}",
                    f"Finished: {record.finished_at or '[still running]'}",
                ]
                if record.note:
                    lines.append(f"Note: {record.note}")
                if record.error:
                    lines.append(f"Error: {record.error}")
                if include_output:
                    stdout_text = record.stdout_preview or "[empty]"
                    stderr_text = record.stderr_preview or "[empty]"
                    lines.extend(["", "[stdout]", stdout_text, "", "[stderr]", stderr_text])
                    if record.output_truncated:
                        lines.append("[captured output truncated]")
                return ToolResponse.success(
                    text="\n".join(lines),
                    data={"background_task": record.to_dict()},
                )

            records = self.manager.list_tasks(status=status)
        except ValueError as exc:
            message = str(exc)
            code = ToolErrorCode.NOT_FOUND if "not found" in message.lower() else ToolErrorCode.INVALID_PARAM
            return ToolResponse.error(code=code, message=message)
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to inspect background tasks: {exc}",
            )

        if not records:
            return ToolResponse.success(
                text=f"No background tasks (status={status}).",
                data={"background_tasks": []},
            )

        lines = [f"Background tasks (status={status})"]
        for record in records:
            linked_suffix = f" linked_task=#{record.linked_task_id}" if record.linked_task_id is not None else ""
            exit_suffix = f" exit_code={record.exit_code}" if record.exit_code is not None else ""
            lines.append(
                f"- {record.id}: [{record.status}] dir={record.directory}{exit_suffix}{linked_suffix} cmd={record.command}"
            )

        return ToolResponse.success(
            text="\n".join(lines),
            data={"background_tasks": [record.to_dict() for record in records]},
        )

    @tool_action(
        name="background_cancel",
        description=(
            "Cancel a running background task. Use this when a background build or test run is no longer useful or is clearly stuck."
        ),
    )
    def background_cancel(self, background_id: str) -> ToolResponse:
        """Request cancellation for one running background task.

        Args:
            background_id: Background task ID returned by `background_run`.
        """
        if not background_id or not isinstance(background_id, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="background_id must be a non-empty string.",
            )

        try:
            record, message = self.manager.cancel_task(background_id)
        except ValueError as exc:
            return ToolResponse.error(code=ToolErrorCode.NOT_FOUND, message=str(exc))
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to cancel background task: {exc}",
            )

        response_factory = ToolResponse.partial if record.status != "running" else ToolResponse.success
        return response_factory(
            text=(
                f"{message}\n"
                f"Task: {record.id}\n"
                f"Current status: {record.status}\n"
                f"Command: {record.command}"
            ),
            data={"background_task": record.to_dict()},
        )
