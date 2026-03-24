"""Shell execution tool for build, test, and developer commands."""

from __future__ import annotations

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

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import (
    apply_line_limit,
    atomic_write,
    ensure_working_dir,
    relative_display,
    resolve_path,
    safe_decode_output,
)


class _TerminalBackgroundManager:
    """Persist background command execution into terminal files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.terminals_dir = (self.project_root / "memory" / "terminals").resolve()
        self.terminals_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.terminals_dir / "index.json"
        self._lock = threading.RLock()
        self._next_id = self._discover_next_id()
        self._records: Dict[int, Dict[str, Any]] = {}
        self._load_index()

    def track_process(
        self,
        *,
        process: subprocess.Popen,
        command: str,
        working_directory: str,
        description: str,
        block_until_ms: int,
        stdout_prefix: bytes = b"",
        stderr_prefix: bytes = b"",
    ) -> Dict[str, Any]:
        started_at = datetime.now().isoformat()
        started_ts = time.time()
        with self._lock:
            terminal_id = self._next_id
            self._next_id += 1
            terminal_file = self.terminals_dir / f"{terminal_id}.txt"
            record = {
                "id": terminal_id,
                "terminal_file": str(terminal_file),
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
            }
            self._records[terminal_id] = record
            atomic_write(terminal_file, self._render_running(record))
            self._save_index()

            waiter = threading.Thread(
                target=self._wait_and_finalize,
                args=(
                    terminal_id,
                    process,
                    stdout_prefix,
                    stderr_prefix,
                ),
                daemon=True,
                name=f"bash-terminal-{terminal_id}",
            )
            waiter.start()

            return {
                "terminal_id": terminal_id,
                "terminal_file": str(terminal_file),
                "pid": process.pid,
                "status": "running",
            }

    def _wait_and_finalize(
        self,
        terminal_id: int,
        process: subprocess.Popen,
        stdout_prefix: bytes,
        stderr_prefix: bytes,
    ) -> None:
        try:
            stdout, stderr = process.communicate()
        except Exception as exc:
            stdout = b""
            stderr = str(exc).encode("utf-8", errors="replace")

        stdout = stdout or b""
        stderr = stderr or b""

        if stdout_prefix and not stdout.startswith(stdout_prefix):
            stdout = stdout_prefix + stdout
        if stderr_prefix and not stderr.startswith(stderr_prefix):
            stderr = stderr_prefix + stderr

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        exit_code = process.returncode

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

            terminal_file = Path(record["terminal_file"])
            atomic_write(terminal_file, self._render_finished(record, stdout_text, stderr_text))
            self._save_index()

    def _render_running(self, record: Dict[str, Any]) -> str:
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
            "",
            "--- output ---",
            "[running]",
        ]
        return "\n".join(lines) + "\n"

    def _render_finished(self, record: Dict[str, Any], stdout_text: str, stderr_text: str) -> str:
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
            "",
            "--- stdout ---",
            stdout_text.strip() or "[empty]",
            "",
            "--- stderr ---",
            stderr_text.strip() or "[empty]",
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
        "ls", "find",
        "grep", "rg",
        "sed", "awk",
    }
    NETWORK_COMMANDS: Set[str] = {
        "npm", "pnpm", "yarn",
        "pip", "pip3",
        "apt", "apt-get", "brew",
        "curl", "wget",
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
                "Use this for builds, tests, formatters, linters, package scripts, developer commands, "
                "and quick file inspection (cat, head, tail). "
                "Commands that exceed block_until_ms are moved to background and tracked by terminal files."
            ),
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = ensure_working_dir(self.project_root, working_dir)
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
                description="Shell command to execute.",
                required=True,
            ),
            ToolParameter(
                name="working_directory",
                type="string",
                description="Working directory relative to the workspace root.",
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
                description="Optional summary of this command.",
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
                stderr=subprocess.PIPE,
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

        if block_until_ms == 0:
            return self._background_response(
                process=process,
                command=command,
                description=description,
                directory=target_dir,
                block_until_ms=block_until_ms,
                reason="immediate_background",
            )

        try:
            stdout, stderr = process.communicate(timeout=block_until_ms / 1000)
        except subprocess.TimeoutExpired as exc:
            return self._background_response(
                process=process,
                command=command,
                description=description,
                directory=target_dir,
                block_until_ms=block_until_ms,
                reason="exceeded_block_until",
                stdout_prefix=exc.stdout or b"",
                stderr_prefix=exc.stderr or b"",
            )
        except Exception as exc:
            try:
                process.kill()
            except Exception:
                pass
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed while waiting for command: {exc}",
            )

        return self._format_response(
            command=command,
            description=description,
            directory=target_dir,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _background_response(
        self,
        *,
        process: subprocess.Popen,
        command: str,
        description: str,
        directory: Path,
        block_until_ms: int,
        reason: str,
        stdout_prefix: bytes = b"",
        stderr_prefix: bytes = b"",
    ) -> ToolResponse:
        rel_dir = relative_display(self.project_root, directory)
        manager = _get_terminal_manager(self.project_root)
        task = manager.track_process(
            process=process,
            command=command,
            working_directory=rel_dir,
            description=description,
            block_until_ms=block_until_ms,
            stdout_prefix=stdout_prefix,
            stderr_prefix=stderr_prefix,
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
                "pid": task["pid"],
            },
        )

    def _validate_command(self, command: str) -> Optional[str]:
        lowered = command.lower()

        if re.search(r"(^|[;&|]\s*)rm\s+-rf\s+/", lowered):
            return "Refusing to run a destructive command."

        categories: List[tuple[str, Set[str]]] = [
            ("Privileged commands are not allowed", self.PRIVILEGED_COMMANDS),
            ("Interactive terminal commands are not allowed", self.INTERACTIVE_COMMANDS),
            ("Destructive system commands are not allowed", self.DESTRUCTIVE_COMMANDS),
            (
                "Delete-related shell commands are blocked in Bash; use the Delete tool instead",
                self.DELETE_COMMANDS,
            ),
        ]
        if not self.allow_network:
            categories.append(
                ("Network-related commands are disabled for Bash by default", self.NETWORK_COMMANDS),
            )
        for message, blocklist in categories:
            for blocked in blocklist:
                if re.search(rf"\b{re.escape(blocked)}\b", lowered):
                    return f"{message} (detected '{blocked}')."

        for pattern in self.DELETE_PATTERNS:
            if re.search(pattern, lowered):
                return "Delete-related shell commands are blocked in Bash; use the Delete tool instead."

        for leader in self._extract_segment_leaders(command):
            if leader in self.PREFER_SPECIALIZED_TOOLS:
                return (
                    f"Use the dedicated tool instead of `{leader}` in Bash. "
                    "Read → file reading, Grep → code search, Glob → file finding, "
                    "LS → directory listing, Edit → file editing, Delete → file removal."
                )

        return None

    @staticmethod
    def _extract_segment_leaders(command: str) -> List[str]:
        chain_segments = re.split(r"\|\||&&|;", command)
        leaders: List[str] = []
        for segment in chain_segments:
            segment = segment.strip()
            if not segment:
                continue
            first_pipe = segment.split("|")[0].strip()
            if not first_pipe:
                continue
            try:
                seg_tokens = shlex.split(first_pipe, posix=True)
            except ValueError:
                seg_tokens = re.findall(r"[a-zA-Z0-9_./+-]+", first_pipe)
            if seg_tokens:
                leaders.append(Path(seg_tokens[0]).name)
        return leaders

    def validate_command_policy(self, command: str) -> Optional[str]:
        return self._validate_command(command)

    def _format_response(
        self,
        command: str,
        description: str,
        directory: Path,
        exit_code: Optional[int],
        stdout: bytes,
        stderr: bytes,
    ) -> ToolResponse:
        stdout_text, stdout_bytes_truncated = safe_decode_output(stdout)
        stderr_text, stderr_bytes_truncated = safe_decode_output(stderr)
        stdout_text, stdout_lines_truncated = apply_line_limit(stdout_text)
        stderr_text, stderr_lines_truncated = apply_line_limit(stderr_text)

        rel_dir = relative_display(self.project_root, directory)

        parts: List[str] = []
        if description:
            parts.append(f"Description: {description}")
        parts.append(f"Command: {command}")
        parts.append(f"Directory: {rel_dir}")
        if exit_code == 0:
            parts.append(f"Exit code: {exit_code} (success)")
        else:
            parts.append(f"Exit code: {exit_code} (failure)")

        if stdout_text:
            parts.extend(["", "--- stdout ---", stdout_text])
            if stdout_bytes_truncated or stdout_lines_truncated:
                parts.append("[stdout truncated]")

        if stderr_text:
            parts.extend(["", "--- stderr ---", stderr_text])
            if stderr_bytes_truncated or stderr_lines_truncated:
                parts.append("[stderr truncated]")

        if not stdout_text and not stderr_text:
            parts.extend(["", "[no output]"])

        text = "\n".join(parts)
        data = {
            "backgrounded": False,
            "command": command,
            "description": description,
            "working_directory": rel_dir,
            "exit_code": exit_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

        if exit_code != 0:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=text,
                context={"command": command, "working_directory": rel_dir, "exit_code": exit_code},
            )

        return ToolResponse.success(text=text, data=data)
