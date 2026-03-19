"""Shell execution tool for build, test, and developer commands."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import (
    apply_line_limit,
    ensure_working_dir,
    relative_display,
    resolve_path,
    safe_decode_output,
)


class BashTool(Tool):
    """Run non-interactive shell commands inside the workspace."""

    DEFAULT_TIMEOUT_MS = 120000
    MAX_TIMEOUT_MS = 600000

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
    PREFER_SPECIALIZED_TOOLS: Set[str] = {"ls", "grep", "rg", "find"}
    NETWORK_COMMANDS: Set[str] = {"pnpm", "yarn", "apt", "brew"}

    def __init__(
        self,
        name: str = "Bash",
        project_root: str = ".",
        working_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Run a non-interactive shell command inside the workspace. Prefer this for tests, "
                "builds, formatters, linters, package scripts, and one-off developer commands. "
                "Do not use Bash for simple file listing, file reading, or code search when `LS`, `Read`, or `Grep` can do the job more safely."
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
                description=(
                    "The shell command to execute. Best for commands like `pytest`, `python -m`, "
                    "`npm test`, `ruff check`, `make`, or project scripts. Avoid using this for file browsing or text search."
                ),
                required=True,
            ),
            ToolParameter(
                name="directory",
                type="string",
                description=(
                    "Working directory relative to the workspace root. Use this when the command should run from a subdirectory such as `frontend` or `backend`."
                ),
                required=False,
                default=".",
            ),
            ToolParameter(
                name="timeout_ms",
                type="integer",
                description=(
                    f"Execution timeout in milliseconds. Increase this for slow test suites or builds. "
                    f"Default {self.DEFAULT_TIMEOUT_MS}, hard max {self.MAX_TIMEOUT_MS}."
                ),
                required=False,
                default=self.DEFAULT_TIMEOUT_MS,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        command = parameters.get("command")
        directory = parameters.get("directory", ".")
        timeout_ms = parameters.get("timeout_ms", self.DEFAULT_TIMEOUT_MS)

        if not command or not isinstance(command, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="command must be a non-empty string.",
            )
        if not isinstance(timeout_ms, int) or timeout_ms < 1 or timeout_ms > self.MAX_TIMEOUT_MS:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"timeout_ms must be an integer between 1 and {self.MAX_TIMEOUT_MS}."
                ),
            )

        try:
            target_dir = resolve_path(self.project_root, self.working_dir, directory)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message="directory escapes the workspace root.",
            )

        if not target_dir.exists() or not target_dir.is_dir():
            return ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"Working directory not found: {directory}",
            )

        policy_error = self._validate_command(command)
        if policy_error:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=policy_error,
            )

        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=target_dir,
                capture_output=True,
                timeout=timeout_ms / 1000,
                env={**os.environ, "PROJECT_ROOT": str(self.project_root)},
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            return self._format_response(
                command=command,
                directory=target_dir,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to execute shell command: {exc}",
            )

        response = self._format_response(
            command=command,
            directory=target_dir,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )
        if result.returncode != 0:
            return ToolResponse.partial(
                text=response.text,
                data=response.data,
                context=response.context,
            )
        return response

    def _validate_command(self, command: str) -> Optional[str]:
        """Validate *command* against security policies.

        Three-layer validation:
          1. Catastrophic pattern detection (``rm -rf /``).
          2. Raw-string scan for blocked commands — catches bypass vectors
             such as ``env sudo``, ``$(sudo …)``, `` `sudo` ``, and
             ``bash -c "sudo …"``.
          3. Token-based preference check for segment leaders (soft).
        """
        lowered = command.lower()

        # --- Layer 1: catastrophic patterns ---
        if re.search(r"(^|[;&|]\s*)rm\s+-rf\s+/", lowered):
            return "Refusing to run a destructive command."

        # --- Layer 2: raw-string scan for security-critical commands ---
        # Searches the entire command string (including inside $(), ``,
        # quotes, etc.) so that token-level bypass tricks are caught.
        _categories: List[tuple[str, Set[str]]] = [
            ("Privileged commands are not allowed", self.PRIVILEGED_COMMANDS),
            ("Interactive terminal commands are not allowed", self.INTERACTIVE_COMMANDS),
            ("Destructive system commands are not allowed", self.DESTRUCTIVE_COMMANDS),
        ]
        if not self.allow_network:
            _categories.append(
                ("Network-related commands are disabled for Bash by default", self.NETWORK_COMMANDS),
            )
        for message, blocklist in _categories:
            for cmd in blocklist:
                if re.search(rf"\b{re.escape(cmd)}\b", lowered):
                    return f"{message} (detected '{cmd}')."

        # --- Layer 3: preference blocklist (token-based, segment leaders only) ---
        # Only the leading command of each pipeline/chain segment is checked,
        # so piped usage like `git log | grep fix` is still allowed.
        for leader in self._extract_segment_leaders(command):
            if leader in self.PREFER_SPECIALIZED_TOOLS:
                return (
                    "Use the dedicated tools instead of shell for listing, reading, "
                    "or searching files."
                )

        return None

    @staticmethod
    def _extract_segment_leaders(command: str) -> List[str]:
        """Return the leading command of each chain segment (ignoring piped commands).

        Split by ``&&``, ``||``, ``;`` to get chain segments, then take only
        the first command in each segment (before any ``|``).  This means:

        - ``grep pattern *.py``       → ``['grep']``  (blocked)
        - ``git log | grep fix``      → ``['git']``   (allowed)
        - ``pytest | head -100``      → ``['pytest']`` (allowed)
        - ``cat f.py && head f.py``   → ``['cat', 'head']`` (both blocked)
        """
        # Step 1: split by chain operators (&&, ||, ;) — NOT by pipe |
        chain_segments = re.split(r"\|\||&&|;", command)
        leaders: List[str] = []
        for segment in chain_segments:
            segment = segment.strip()
            if not segment:
                continue
            # Step 2: split by pipe | and take only the FIRST part
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
        """Public wrapper so other tools can share Bash command policy."""
        return self._validate_command(command)

    def _format_response(
        self,
        command: str,
        directory: Path,
        exit_code: Optional[int],
        stdout: bytes,
        stderr: bytes,
        timed_out: bool,
    ) -> ToolResponse:
        stdout_text, stdout_bytes_truncated = safe_decode_output(stdout)
        stderr_text, stderr_bytes_truncated = safe_decode_output(stderr)
        stdout_text, stdout_lines_truncated = apply_line_limit(stdout_text)
        stderr_text, stderr_lines_truncated = apply_line_limit(stderr_text)

        parts = [
            f"Command: {command}",
            f"Directory: {relative_display(self.project_root, directory)}",
        ]
        if timed_out:
            parts.append("Status: timed out")
        else:
            parts.append(f"Exit code: {exit_code}")

        if stdout_text:
            parts.extend(["", "[stdout]", stdout_text])
            if stdout_bytes_truncated or stdout_lines_truncated:
                parts.append("[stdout truncated]")

        if stderr_text:
            parts.extend(["", "[stderr]", stderr_text])
            if stderr_bytes_truncated or stderr_lines_truncated:
                parts.append("[stderr truncated]")

        if not stdout_text and not stderr_text:
            parts.extend(["", "[no output]"])

        if timed_out:
            return ToolResponse.error(
                code=ToolErrorCode.TIMEOUT,
                message="\n".join(parts),
                context={
                    "command": command,
                    "directory": relative_display(self.project_root, directory),
                },
            )

        return ToolResponse.success(
            text="\n".join(parts),
            data={
                "command": command,
                "directory": relative_display(self.project_root, directory),
                "exit_code": exit_code,
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
        )
