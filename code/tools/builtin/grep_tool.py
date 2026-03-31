"""Code-search tool for the coding agent."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import ensure_working_dir, relative_display, resolve_path


@dataclass
class GrepMatch:
    """Structured single-line search match."""

    path: str
    line: int
    text: str
    mtime_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "line": self.line, "text": self.text}


class GrepTool(Tool):
    """Search code with ripgrep."""

    MAX_RESULTS = 100
    MAX_LINE_LENGTH = 2000
    DEFAULT_EXCLUDE_GLOBS = (
        "!.git/**",
        "!**/.git/**",
        "!.backups/**",
        "!**/.backups/**",
        "!.delete_trash/**",
        "!**/.delete_trash/**",
    )
    INTERNAL_ARTIFACT_DIRS = {".backups", ".delete_trash"}

    def __init__(
        self,
        name: str = "Grep",
        project_root: str = ".",
        working_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Search code with a regex pattern using ripgrep. Results are limited to 100 matches. "
                "Use `include` to narrow the search when you already know the relevant file types."
            ),
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = ensure_working_dir(self.project_root, working_dir)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="The regex pattern to search for in file contents.",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description=(
                    "The directory to search in, relative to the project root. "
                    "If omitted, the project root is used."
                ),
                required=False,
                default=".",
            ),
            ToolParameter(
                name="include",
                type="string",
                description='Optional file pattern to include in the search, such as "*.py" or "*.{ts,tsx}".',
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        pattern = parameters.get("pattern")
        raw_path = parameters.get("path", ".")
        include = parameters.get("include")

        if not pattern or not isinstance(pattern, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="pattern must be a non-empty regex string.",
            )

        if include is not None and not isinstance(include, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="include must be a string when provided.",
            )

        try:
            root = resolve_path(self.project_root, self.working_dir, raw_path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message="Path escapes the workspace root.",
            )

        if not root.exists():
            return ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"Search root not found: {raw_path}",
            )

        if not root.is_dir():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Search root is not a directory: {raw_path}",
            )

        try:
            matches, had_inaccessible_paths = self._run_rg(root, pattern, include)
        except FileNotFoundError:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message="ripgrep is required for GrepTool but was not found.",
            )
        except re.error as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid regex pattern: {exc}",
            )
        except subprocess.SubprocessError as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Search failed: {exc}",
            )

        matches.sort(key=lambda match: match.mtime_ms, reverse=True)
        total_matches = len(matches)
        truncated = total_matches > self.MAX_RESULTS
        final_matches = matches[: self.MAX_RESULTS]

        rel_root = relative_display(self.project_root, root)
        data = {
            "pattern": pattern,
            "path": rel_root,
            "matches": [match.to_dict() for match in final_matches],
            "total_matches": total_matches,
            "truncated": truncated,
        }
        if include:
            data["include"] = include

        if not final_matches:
            return ToolResponse.success(text="No files found", data=data)

        lines = [
            f"Found {total_matches} matches{f' (showing first {self.MAX_RESULTS})' if truncated else ''}"
        ]
        current_file: Optional[str] = None

        for match in final_matches:
            if match.path != current_file:
                if current_file is not None:
                    lines.append("")
                current_file = match.path
                lines.append(f"{match.path}:")

            display_text = match.text
            if len(display_text) > self.MAX_LINE_LENGTH:
                display_text = display_text[: self.MAX_LINE_LENGTH] + "..."
            lines.append(f"  Line {match.line}: {display_text}")

        if truncated:
            lines.extend(
                [
                    "",
                    f"(Results truncated: showing {self.MAX_RESULTS} of {total_matches} matches ({total_matches - self.MAX_RESULTS} hidden). Consider using a more specific path or pattern.)",
                ]
            )

        if had_inaccessible_paths:
            lines.extend(["", "(Some paths were inaccessible and skipped)"])

        factory = ToolResponse.partial if truncated or had_inaccessible_paths else ToolResponse.success
        return factory(text="\n".join(lines), data=data)

    def _run_rg(self, root: Path, pattern: str, include: Optional[str]) -> tuple[List[GrepMatch], bool]:
        command = [
            "rg",
            "--json",
            "--hidden",
            "--no-messages",
            "--regexp",
            pattern,
            str(root),
        ]
        for glob in reversed(self.DEFAULT_EXCLUDE_GLOBS):
            command[4:4] = ["--glob", glob]
        if include:
            command[4:4] = ["--glob", include]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        matches: List[GrepMatch] = []
        mtime_cache: Dict[str, int] = {}

        assert process.stdout is not None
        for raw_line in process.stdout:
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise subprocess.SubprocessError(f"Invalid ripgrep JSON output: {exc}") from exc

            if payload.get("type") != "match":
                continue

            match = self._parse_match_event(payload.get("data", {}), mtime_cache)
            if match is not None:
                matches.append(match)

        _, stderr_text = process.communicate()
        return_code = process.returncode
        stderr_text = stderr_text.strip()

        if return_code not in (0, 1, 2):
            raise subprocess.SubprocessError(stderr_text or "ripgrep failed")

        if return_code == 2 and not matches:
            return [], False

        if return_code == 2 and self._looks_like_regex_error(stderr_text):
            raise re.error(stderr_text or "ripgrep regex parse error")

        had_inaccessible_paths = return_code == 2
        return matches, had_inaccessible_paths

    def _parse_match_event(
        self,
        data: Dict[str, Any],
        mtime_cache: Dict[str, int],
    ) -> Optional[GrepMatch]:
        path_text = data.get("path", {}).get("text")
        if not path_text:
            return None

        normalized_path = self._normalize_rg_path(path_text)
        if any(part in self.INTERNAL_ARTIFACT_DIRS for part in Path(normalized_path).parts):
            return None
        if normalized_path not in mtime_cache:
            try:
                mtime_cache[normalized_path] = int(
                    (self.project_root / normalized_path).resolve().stat().st_mtime * 1000
                )
            except OSError:
                mtime_cache[normalized_path] = 0

        line_number = int(data.get("line_number") or 1)
        line_text = (data.get("lines", {}).get("text") or "").rstrip("\r\n")
        return GrepMatch(
            path=normalized_path,
            line=line_number,
            text=line_text,
            mtime_ms=mtime_cache[normalized_path],
        )

    def _normalize_rg_path(self, path_text: str) -> str:
        path_obj = Path(path_text)
        if not path_obj.is_absolute():
            path_obj = path_obj.resolve()
        return relative_display(self.project_root, path_obj)

    @staticmethod
    def _looks_like_regex_error(stderr_text: str) -> bool:
        lowered = stderr_text.lower()
        return "regex parse error" in lowered or "error parsing regex" in lowered
