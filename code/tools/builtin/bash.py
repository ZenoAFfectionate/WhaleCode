"""Shell execution tool for build, test, and developer commands."""

from __future__ import annotations

import codecs
import json
import os
import re
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...context.truncator import ObservationTruncator
from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import (
    atomic_write,
    ensure_working_dir,
    relative_display,
    resolve_path,
)


class _CommandEventStream:
    """Capture merged command output as a timestamped event stream."""

    READ_CHUNK_SIZE = 4096

    def __init__(self, output_prefix: bytes = b""):
        self._lock = threading.RLock()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._events: List[Dict[str, Any]] = []
        self._reader: Optional[threading.Thread] = None
        self._closed = threading.Event()
        self._sequence = 0

        if output_prefix:
            self._append_bytes(output_prefix, source="buffered")

    def start(self, stream) -> None:
        if stream is None:
            self._closed.set()
            return
        self._reader = threading.Thread(
            target=self._drain,
            args=(stream,),
            daemon=True,
            name="bash-output-reader",
        )
        self._reader.start()

    def _drain(self, stream) -> None:
        try:
            while True:
                reader = getattr(stream, "read1", None)
                chunk = reader(self.READ_CHUNK_SIZE) if reader else stream.read(self.READ_CHUNK_SIZE)
                if not chunk:
                    break
                self._append_bytes(chunk)
        finally:
            self._append_bytes(b"", final=True)
            try:
                stream.close()
            except Exception:
                pass
            self._closed.set()

    def _append_bytes(self, payload: bytes, *, source: str = "live", final: bool = False) -> None:
        text = self._decoder.decode(payload, final=final)
        if not text:
            return
        with self._lock:
            self._sequence += 1
            self._events.append(
                {
                    "seq": self._sequence,
                    "ts": datetime.now().isoformat(),
                    "stream": "output",
                    "source": source,
                    "text": text,
                }
            )

    def add_system_event(self, text: str, *, source: str = "system") -> None:
        if not text:
            return
        with self._lock:
            self._sequence += 1
            self._events.append(
                {
                    "seq": self._sequence,
                    "ts": datetime.now().isoformat(),
                    "stream": "system",
                    "source": source,
                    "text": text,
                }
            )

    def wait_closed(self, timeout: Optional[float] = None) -> bool:
        if self._reader:
            self._reader.join(timeout)
        return self._closed.is_set()

    def text(self) -> str:
        with self._lock:
            return "".join(event["text"] for event in self._events if event.get("stream") == "output")

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)


class _TerminalBackgroundManager:
    """Persist background command execution into terminal files."""

    SNAPSHOT_INTERVAL_SECONDS = 0.25
    RETENTION_DAYS = 7

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.terminals_dir = (self.project_root / "memory" / "terminals").resolve()
        self.terminals_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.terminals_dir / "index.json"
        self._lock = threading.RLock()
        self._next_id = self._discover_next_id()
        self._records: Dict[int, Dict[str, Any]] = {}
        self._load_index()
        self._reconcile_records()
        self._cleanup_records()
        self._save_index()

    def track_process(
        self,
        *,
        process: subprocess.Popen,
        command: str,
        working_directory: str,
        description: str,
        block_until_ms: int,
        event_stream: _CommandEventStream,
    ) -> Dict[str, Any]:
        started_at = datetime.now().isoformat()
        started_ts = time.time()
        with self._lock:
            terminal_id = self._next_id
            self._next_id += 1
            terminal_file = self.terminals_dir / f"{terminal_id}.txt"
            event_file = self.terminals_dir / f"{terminal_id}.events.jsonl"
            record = {
                "id": terminal_id,
                "terminal_file": str(terminal_file),
                "event_file": str(event_file),
                "status": "running",
                "pid": process.pid,
                "command": command,
                "working_directory": working_directory,
                "description": description,
                "block_until_ms": block_until_ms,
                "started_at": started_at,
                "started_ts": started_ts,
                "finished_at": "",
                "exit_code": None,
                "elapsed_ms": None,
                "status_reason": "",
            }
            self._records[terminal_id] = record
            self._write_snapshot(record, event_stream.snapshot(), running=True)
            self._save_index()

            waiter = threading.Thread(
                target=self._wait_and_finalize,
                args=(
                    terminal_id,
                    process,
                    event_stream,
                ),
                daemon=True,
                name=f"bash-terminal-{terminal_id}",
            )
            waiter.start()

            return {
                "terminal_id": terminal_id,
                "terminal_file": str(terminal_file),
                "event_file": str(event_file),
                "pid": process.pid,
                "status": "running",
            }

    def _wait_and_finalize(
        self,
        terminal_id: int,
        process: subprocess.Popen,
        event_stream: _CommandEventStream,
    ) -> None:
        last_event_count = -1
        last_snapshot_ts = 0.0
        while True:
            exit_code = process.poll()
            snapshot = event_stream.snapshot()
            if exit_code is None:
                now = time.time()
                if len(snapshot) != last_event_count or now - last_snapshot_ts >= self.SNAPSHOT_INTERVAL_SECONDS:
                    with self._lock:
                        record = self._records.get(terminal_id)
                        if record:
                            self._write_snapshot(record, snapshot, running=True)
                    last_event_count = len(snapshot)
                    last_snapshot_ts = now
                time.sleep(self.SNAPSHOT_INTERVAL_SECONDS)
                continue

            event_stream.wait_closed(timeout=2.0)
            snapshot = event_stream.snapshot()
            with self._lock:
                record = self._records.get(terminal_id)
                if not record:
                    return

                finished_at = datetime.now().isoformat()
                elapsed_ms = int((time.time() - record["started_ts"]) * 1000)
                record["finished_at"] = finished_at
                record["exit_code"] = exit_code
                record["elapsed_ms"] = elapsed_ms
                record["status"] = "completed" if exit_code == 0 else "failed"
                record["status_reason"] = ""
                self._write_snapshot(record, snapshot, running=False)
                self._save_index()
            return

    def _render_running(self, record: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
        running_for_seconds = max(int(time.time() - record["started_ts"]), 0)
        lines = [
            f"[terminal:{record['id']}]",
            f"status: {record['status']}",
            f"pid: {record['pid']}",
            f"running_for_seconds: {running_for_seconds}",
            f"started_at: {record['started_at']}",
            f"working_directory: {record['working_directory']}",
            f"command: {record['command']}",
            f"description: {record['description'] or '[none]'}",
            f"block_until_ms: {record['block_until_ms']}",
            f"event_file: {record['event_file']}",
            f"event_count: {len(events)}",
            "",
            "--- event_stream ---",
            self._render_event_stream(events) or "[no output yet]",
            "",
            "--- output ---",
            (self._events_to_output_text(events) or "[empty]").rstrip(),
            "",
            "[running]",
        ]
        return "\n".join(lines) + "\n"

    def _render_finished(self, record: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
        running_for_seconds = max(int((record.get("elapsed_ms") or 0) / 1000), 0)
        lines = [
            f"[terminal:{record['id']}]",
            f"status: {record['status']}",
            f"pid: {record['pid']}",
            f"running_for_seconds: {running_for_seconds}",
            f"started_at: {record['started_at']}",
            f"finished_at: {record['finished_at']}",
            f"working_directory: {record['working_directory']}",
            f"command: {record['command']}",
            f"description: {record['description'] or '[none]'}",
            f"event_file: {record['event_file']}",
            f"event_count: {len(events)}",
            *( [f"status_reason: {record['status_reason']}"] if record.get("status_reason") else [] ),
            "",
            "--- event_stream ---",
            self._render_event_stream(events) or "[no output]",
            "",
            "--- output ---",
            (self._events_to_output_text(events) or "[empty]").rstrip(),
            "",
            f"exit_code: {record['exit_code']}",
            f"elapsed_ms: {record['elapsed_ms']}",
        ]
        return "\n".join(lines) + "\n"

    def _discover_next_id(self) -> int:
        ids: List[int] = []
        for item in self.terminals_dir.glob("*.txt"):
            try:
                ids.append(int(item.stem))
            except ValueError:
                continue
        return (max(ids) + 1) if ids else 1

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            return
        for item in records:
            if not isinstance(item, dict) or "id" not in item:
                continue
            record = dict(item)
            record.setdefault("event_file", str(self.terminals_dir / f"{record['id']}.events.jsonl"))
            record.setdefault("status_reason", "")
            started_at = record.get("started_at")
            if started_at:
                try:
                    dt = datetime.fromisoformat(started_at)
                    record.setdefault("started_ts", dt.timestamp())
                except Exception:
                    record.setdefault("started_ts", time.time())
            else:
                record.setdefault("started_ts", time.time())
            self._records[int(record["id"])] = record

    def _save_index(self) -> None:
        records: List[Dict[str, Any]] = []
        for record in sorted(self._records.values(), key=lambda item: item["id"]):
            serializable = {
                key: value
                for key, value in record.items()
                if key != "started_ts"
            }
            records.append(serializable)
        atomic_write(
            self.index_path,
            json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
        )

    def _reconcile_records(self) -> None:
        now_ts = time.time()
        for record in self._records.values():
            status = record.get("status")
            pid = record.get("pid")
            if status in {"running", "detached"}:
                if self._pid_alive(pid):
                    if status == "detached":
                        continue
                    record["status"] = "detached"
                    record["status_reason"] = "manager_restarted"
                    record["finished_at"] = ""
                    self._append_system_note(
                        record,
                        "Background process is still alive, but the previous manager restarted. Live capture is detached.",
                    )
                else:
                    record["status"] = "terminated"
                    record["status_reason"] = "stale_running_record"
                    record["finished_at"] = datetime.now().isoformat()
                    record["exit_code"] = None
                    record["elapsed_ms"] = max(int((now_ts - record.get("started_ts", now_ts)) * 1000), 0)
                    self._append_system_note(
                        record,
                        "Recovered stale running record: process is no longer alive.",
                    )

    def _cleanup_records(self) -> None:
        cutoff = time.time() - self.RETENTION_DAYS * 24 * 60 * 60
        active_files: Set[str] = set()
        expired_ids: List[int] = []

        for record_id, record in self._records.items():
            active_files.add(record["terminal_file"])
            active_files.add(record["event_file"])

            finished_at = record.get("finished_at")
            if record.get("status") in {"running", "detached"} and self._pid_alive(record.get("pid")):
                continue
            record_ts = self._record_timestamp(record, finished_at)
            if record_ts >= cutoff:
                continue
            expired_ids.append(record_id)

        for record_id in expired_ids:
            record = self._records.pop(record_id, None)
            if not record:
                continue
            for key in ("terminal_file", "event_file"):
                path = Path(record[key])
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    continue

        for pattern in ("*.txt", "*.events.jsonl"):
            for item in self.terminals_dir.glob(pattern):
                if str(item) in active_files:
                    continue
                try:
                    item.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    continue

    def _append_system_note(self, record: Dict[str, Any], note: str) -> None:
        events = self._load_events(record)
        events.append(
            {
                "seq": len(events) + 1,
                "ts": datetime.now().isoformat(),
                "stream": "system",
                "source": "reconcile",
                "text": note,
            }
        )
        self._write_snapshot(record, events, running=False)

    def _load_events(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = Path(record["event_file"])
        if not path.exists():
            return []
        events: List[Dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    events.append(payload)
        except Exception:
            return []
        return events

    def _write_snapshot(self, record: Dict[str, Any], events: List[Dict[str, Any]], *, running: bool) -> None:
        terminal_path = Path(record["terminal_file"])
        event_path = Path(record["event_file"])
        rendered = self._render_running(record, events) if running else self._render_finished(record, events)
        atomic_write(terminal_path, rendered)
        atomic_write(event_path, self._render_event_jsonl(events))

    @staticmethod
    def _render_event_jsonl(events: List[Dict[str, Any]]) -> str:
        if not events:
            return ""
        return "".join(json.dumps(event, ensure_ascii=False, default=str) + "\n" for event in events)

    @staticmethod
    def _events_to_output_text(events: List[Dict[str, Any]]) -> str:
        return "".join(event.get("text", "") for event in events if event.get("stream") == "output")

    @staticmethod
    def _render_event_stream(events: List[Dict[str, Any]]) -> str:
        rendered: List[str] = []
        for event in events:
            ts = event.get("ts", "")
            stream = event.get("stream", "output")
            source = event.get("source")
            prefix = f"{ts} [{stream}"
            if source and source not in {"live", "system"}:
                prefix += f"/{source}"
            prefix += "] "
            text = event.get("text", "")
            lines = text.splitlines() or [text]
            if not lines:
                rendered.append(prefix.rstrip())
                continue
            for line in lines:
                rendered.append(prefix + line)
        return "\n".join(line.rstrip() for line in rendered if line is not None).strip()

    @staticmethod
    def _record_timestamp(record: Dict[str, Any], timestamp: Optional[str]) -> float:
        if timestamp:
            try:
                return datetime.fromisoformat(timestamp).timestamp()
            except ValueError:
                pass
        return float(record.get("started_ts") or time.time())

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True


_TERMINAL_MANAGERS: Dict[str, _TerminalBackgroundManager] = {}
_TERMINAL_MANAGERS_LOCK = threading.Lock()


def _get_terminal_manager(project_root: Path) -> _TerminalBackgroundManager:
    root_key = str(project_root)
    with _TERMINAL_MANAGERS_LOCK:
        manager = _TERMINAL_MANAGERS.get(root_key)
        if manager is None:
            manager = _TerminalBackgroundManager(project_root)
            _TERMINAL_MANAGERS[root_key] = manager
        return manager


class BashTool(Tool):
    """Run non-interactive shell commands inside the workspace."""

    DEFAULT_BLOCK_UNTIL_MS = 30000
    MAX_BLOCK_UNTIL_MS = 600000
    OUTPUT_PREVIEW_MAX_LINES = 900
    OUTPUT_PREVIEW_MAX_BYTES = 24_000
    MAX_POLICY_PARSE_DEPTH = 3

    INTERACTIVE_COMMANDS: Set[str] = {
        "vim",
        "vi",
        "nano",
        "less",
        "more",
        "top",
        "htop",
        "watch",
        "tmux",
        "screen",
    }
    PRIVILEGED_COMMANDS: Set[str] = {"sudo", "su", "doas"}
    DESTRUCTIVE_COMMANDS: Set[str] = {
        "mkfs",
        "fdisk",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
    }
    DELETE_COMMANDS: Set[str] = {
        "rm",
        "rmdir",
        "unlink",
        "shred",
        "srm",
        "del",
        "erase",
    }
    DELETE_PATTERNS: List[str] = [
        r"\bfind\b[^\n]*\s-delete\b",
        r"\bgit\s+clean\b",
    ]
    PREFER_SPECIALIZED_TOOLS: Set[str] = {
        "ls",
        "grep", "rg",
        "cat",
        "sed", "awk",
    }
    NETWORK_COMMANDS: Set[str] = {
        "npm", "pnpm", "yarn",
        "pip", "pip3",
        "apt", "apt-get", "brew",
        "curl", "wget",
    }
    COMMAND_SEPARATORS: Set[str] = {"&&", "||", ";", "|", "|&", "&", "(", ")"}
    SHELL_WRAPPERS: Set[str] = {"bash", "sh", "zsh", "dash", "ksh"}
    SHELL_COMMAND_FLAGS: Set[str] = {"-c", "-lc", "-ic", "-ec", "-exc", "-lec"}
    ASSIGNMENT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
    PREFERRED_TOOL_MESSAGES: Dict[str, str] = {
        "ls": "Use the LS tool instead of `ls` in Bash for directory listing.",
        "grep": "Use the Grep tool instead of `grep` in Bash for code and text search.",
        "rg": "Use the Grep tool instead of `rg` in Bash for code and text search.",
        "cat": "Use the Read tool instead of `cat` in Bash for file reading.",
        "sed": "Use the Edit tool instead of `sed` in Bash for file editing.",
        "awk": "Use the Edit or Grep tools instead of `awk` in Bash for repository inspection.",
    }

    def __init__(
        self,
        name: str = "Bash",
        project_root: str = ".",
        working_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Run a non-interactive shell command inside the workspace. "
                "Use this for builds, tests, formatters, linters, package scripts, git, and other "
                "developer commands. Prefer dedicated tools instead of Bash for directory listing "
                "(LS), code search (Grep), and file editing (Edit/Write/Delete) when they fit the task. "
                "Commands that exceed block_until_ms are moved to background and tracked by terminal files."
            ),
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = ensure_working_dir(self.project_root, working_dir)
        self.output_truncator = ObservationTruncator(
            max_lines=self.OUTPUT_PREVIEW_MAX_LINES,
            max_bytes=self.OUTPUT_PREVIEW_MAX_BYTES,
            truncate_direction="head",
            output_dir=str(self.project_root / "memory" / "tool-output"),
        )
        self.allow_network = os.getenv("BASH_ALLOW_NETWORK", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description=(
                    "Shell command to execute. Use Bash for terminal workflows like builds, tests, "
                    "git, and package scripts. Prefer dedicated tools over Bash for `ls`, `grep`, "
                    "`cat`, `sed`, or `awk` when they fit the task."
                ),
                required=True,
            ),
            ToolParameter(
                name="working_directory",
                type="string",
                description="Working directory relative to the workspace root. Use this instead of `cd`.",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="block_until_ms",
                type="integer",
                description=(
                    "Milliseconds to wait before returning. Default 30000. "
                    "Set to 0 to run immediately in background."
                ),
                required=False,
                default=self.DEFAULT_BLOCK_UNTIL_MS,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="Optional short summary of what this command does.",
                required=False,
                default="",
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
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

        try:
            process = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PROJECT_ROOT": str(self.project_root)},
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=(
                    f"Failed to execute shell command: {exc}\n"
                    f"Command: {command}\n"
                    f"Directory: {working_directory}"
                ),
            )

        event_stream = self._create_event_stream()
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
                message=f"Failed while waiting for command: {exc}",
            )

        return self._format_response(
            command=command,
            description=description,
            directory=target_dir,
            exit_code=process.returncode,
            event_stream=event_stream,
        )

    def _background_response(
        self,
        *,
        process: subprocess.Popen,
        event_stream: _CommandEventStream,
        command: str,
        description: str,
        directory: Path,
        block_until_ms: int,
        reason: str,
    ) -> ToolResponse:
        rel_dir = relative_display(self.project_root, directory)
        manager = _get_terminal_manager(self.project_root)
        task = manager.track_process(
            process=process,
            command=command,
            working_directory=rel_dir,
            description=description,
            block_until_ms=block_until_ms,
            event_stream=event_stream,
        )

        parts: List[str] = []
        if description:
            parts.append(f"Description: {description}")
        parts.extend(
            [
                f"Command: {command}",
                f"Directory: {rel_dir}",
                f"Status: running in background (pid={task['pid']})",
                f"Terminal file: {task['terminal_file']}",
                f"Event file: {task['event_file']}",
            ]
        )
        if reason == "exceeded_block_until":
            parts.append(f"Reason: exceeded block_until_ms={block_until_ms}")
        else:
            parts.append("Reason: block_until_ms=0")

        return ToolResponse.success(
            text="\n".join(parts),
            data={
                "backgrounded": True,
                "command": command,
                "description": description,
                "working_directory": rel_dir,
                "block_until_ms": block_until_ms,
                "terminal_id": task["terminal_id"],
                "terminal_file": task["terminal_file"],
                "event_file": task["event_file"],
                "pid": task["pid"],
            },
        )

    def _validate_command(self, command: str) -> Optional[str]:
        lowered = command.lower()
        invocations = self._extract_command_invocations(command)

        if re.search(r"(^|[;&|]\s*)rm\s+-rf\s+/", lowered):
            return "Refusing to run a destructive command."

        for tokens in invocations:
            if not tokens:
                continue
            leader = tokens[0]
            if self._is_rm_root(tokens):
                return "Refusing to run a destructive command."
            if leader in self.PRIVILEGED_COMMANDS:
                return f"Privileged commands are not allowed (detected '{leader}')."
            if leader in self.INTERACTIVE_COMMANDS:
                return f"Interactive terminal commands are not allowed (detected '{leader}')."
            if leader in self.DESTRUCTIVE_COMMANDS:
                return f"Destructive system commands are not allowed (detected '{leader}')."
            if leader in self.DELETE_COMMANDS or self._is_delete_pattern(tokens):
                return "Delete-related shell commands are blocked in Bash; use the Delete tool instead."
            if not self.allow_network and leader in self.NETWORK_COMMANDS:
                return f"Network-related commands are disabled for Bash by default (detected '{leader}')."
            if leader in self.PREFER_SPECIALIZED_TOOLS:
                return self.PREFERRED_TOOL_MESSAGES.get(
                    leader,
                    (
                        f"Use the dedicated tool instead of `{leader}` in Bash. "
                        "LS → directory listing, Glob → file discovery, Grep → code search, "
                        "Read → file reading, Edit → file editing, Delete → file removal."
                    ),
                )

        for pattern in self.DELETE_PATTERNS:
            if re.search(pattern, lowered):
                return "Delete-related shell commands are blocked in Bash; use the Delete tool instead."

        return None

    @staticmethod
    def _extract_segment_leaders(command: str) -> List[str]:
        return [tokens[0] for tokens in BashTool._extract_command_invocations(command)]

    def validate_command_policy(self, command: str) -> Optional[str]:
        return self._validate_command(command)

    @classmethod
    def _extract_command_invocations(cls, command: str, depth: int = 0) -> List[List[str]]:
        if depth >= cls.MAX_POLICY_PARSE_DEPTH:
            return []

        try:
            tokens = cls._tokenize_command(command)
        except ValueError:
            return cls._fallback_extract_invocations(command)

        invocations: List[List[str]] = []
        current: List[str] = []

        for token in tokens:
            if token in cls.COMMAND_SEPARATORS:
                invocation = cls._normalize_invocation_tokens(current)
                if invocation:
                    nested = cls._extract_nested_shell_invocations(invocation, depth)
                    invocations.extend(nested or [invocation])
                current = []
                continue
            current.append(token)

        invocation = cls._normalize_invocation_tokens(current)
        if invocation:
            nested = cls._extract_nested_shell_invocations(invocation, depth)
            invocations.extend(nested or [invocation])

        return invocations

    @classmethod
    def _tokenize_command(cls, command: str) -> List[str]:
        lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars="|&;()<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)

    @classmethod
    def _fallback_extract_invocations(cls, command: str) -> List[List[str]]:
        invocations: List[List[str]] = []
        for segment in re.split(r"\|\||\|&|&&|;|\|", command):
            segment = segment.strip()
            if not segment:
                continue
            try:
                tokens = shlex.split(segment, posix=True)
            except ValueError:
                tokens = re.findall(r"[A-Za-z0-9_./:+-]+", segment)
            invocation = cls._normalize_invocation_tokens(tokens)
            if invocation:
                invocations.append(invocation)
        return invocations

    @classmethod
    def _normalize_invocation_tokens(cls, tokens: List[str]) -> Optional[List[str]]:
        if not tokens:
            return None

        idx = 0
        while idx < len(tokens) and cls.ASSIGNMENT_PATTERN.fullmatch(tokens[idx]):
            idx += 1
        tokens = tokens[idx:]
        if not tokens:
            return None

        tokens = cls._strip_command_wrappers(tokens)
        if not tokens:
            return None

        leader = Path(tokens[0]).name.lower()
        return [leader, *tokens[1:]]

    @classmethod
    def _strip_command_wrappers(cls, tokens: List[str]) -> List[str]:
        current = list(tokens)

        while current:
            leader = Path(current[0]).name.lower()

            if leader == "env":
                idx = 1
                while idx < len(current):
                    token = current[idx]
                    if token == "--":
                        idx += 1
                        break
                    if token.startswith("-") or cls.ASSIGNMENT_PATTERN.fullmatch(token):
                        idx += 1
                        continue
                    break
                current = current[idx:]
                continue

            if leader in {"command", "builtin", "nohup", "stdbuf"}:
                idx = 1
                while idx < len(current) and current[idx].startswith("-"):
                    idx += 1
                current = current[idx:]
                continue

            if leader == "time":
                idx = 1
                while idx < len(current) and current[idx].startswith("-"):
                    idx += 1
                current = current[idx:]
                continue

            if leader == "timeout":
                idx = 1
                while idx < len(current):
                    token = current[idx]
                    if token.startswith("-") or cls._looks_like_duration(token):
                        idx += 1
                        continue
                    break
                current = current[idx:]
                continue

            if leader == "nice":
                idx = 1
                while idx < len(current):
                    token = current[idx]
                    if token.startswith("-") or re.fullmatch(r"[+-]?\d+", token):
                        idx += 1
                        continue
                    break
                current = current[idx:]
                continue

            break

        return current

    @classmethod
    def _extract_nested_shell_invocations(
        cls,
        invocation: List[str],
        depth: int,
    ) -> List[List[str]]:
        if not invocation:
            return []

        leader = invocation[0]
        if leader not in cls.SHELL_WRAPPERS:
            return []

        for index, token in enumerate(invocation[1:], start=1):
            if token not in cls.SHELL_COMMAND_FLAGS:
                continue
            if index + 1 >= len(invocation):
                return []
            nested_command = invocation[index + 1]
            return cls._extract_command_invocations(nested_command, depth + 1)

        return []

    @staticmethod
    def _looks_like_duration(token: str) -> bool:
        return bool(re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", token))

    @staticmethod
    def _is_rm_root(tokens: List[str]) -> bool:
        if not tokens or tokens[0] != "rm":
            return False
        has_recursive_force = any(
            token.startswith("-") and "r" in token and "f" in token
            for token in tokens[1:]
        )
        touches_root = any(token == "/" for token in tokens[1:] if token != "--")
        return has_recursive_force and touches_root

    @staticmethod
    def _is_delete_pattern(tokens: List[str]) -> bool:
        if not tokens:
            return False
        leader = tokens[0]
        if leader == "find" and any(token == "-delete" for token in tokens[1:]):
            return True
        if leader == "git" and len(tokens) > 1 and tokens[1].lower() == "clean":
            return True
        return False

    def _format_response(
        self,
        command: str,
        description: str,
        directory: Path,
        exit_code: Optional[int],
        event_stream: _CommandEventStream,
    ) -> ToolResponse:
        rel_dir = relative_display(self.project_root, directory)
        output_text = event_stream.text()
        truncation = self.output_truncator.truncate(
            tool_name="bash",
            output=output_text,
            metadata={
                "command": command,
                "description": description,
                "working_directory": rel_dir,
                "exit_code": exit_code,
                "event_count": event_stream.event_count(),
            },
        )
        preview_text = truncation.get("display_preview", truncation.get("preview", output_text))
        truncated = truncation.get("truncated", False)
        full_output_path = truncation.get("full_output_path")

        parts: List[str] = []
        if description:
            parts.append(f"Description: {description}")
        parts.append(f"Command: {command}")
        parts.append(f"Directory: {rel_dir}")
        if exit_code == 0:
            parts.append(f"Exit code: {exit_code} (success)")
        else:
            parts.append(f"Exit code: {exit_code} (failure)")

        if preview_text:
            parts.extend(["", "--- output ---", preview_text])
        else:
            parts.extend(["", "[no output]"])

        text = "\n".join(parts)
        data = {
            "backgrounded": False,
            "command": command,
            "description": description,
            "working_directory": rel_dir,
            "exit_code": exit_code,
            "output": preview_text,
            "stdout": preview_text,
            "stderr": "",
            "event_count": event_stream.event_count(),
            "truncated": truncated,
            "full_output_path": full_output_path,
        }

        if truncated:
            return ToolResponse.partial(text=text, data=data)

        return ToolResponse.success(text=text, data=data)

    @staticmethod
    def _create_event_stream(output_prefix: bytes = b"") -> _CommandEventStream:
        return _CommandEventStream(output_prefix=output_prefix)
