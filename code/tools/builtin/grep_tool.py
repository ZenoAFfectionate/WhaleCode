"""Code-search tool for the coding agent."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import (
    DEFAULT_IGNORES,
    ensure_working_dir,
    is_binary_file,
    normalize_ignore_patterns,
    prune_walk_dirs,
    relative_display,
    resolve_path,
)


class GrepTool(Tool):
    """Search code using ripgrep when available, with Python fallback."""

    MAX_RESULTS = 100

    def __init__(
        self,
        name: str = "Grep",
        project_root: str = ".",
        working_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Search code with a regex pattern. Prefer this when you need to find symbols, APIs, "
                "config keys, error messages, or code patterns across the repository. Use `include` to narrow the search to the most relevant file types."
            ),
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = ensure_working_dir(self.project_root, working_dir)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description=(
                    "Regex pattern to search for, such as `class\\s+User`, `def run`, or `TODO`. "
                    "Use a precise pattern when possible to reduce noise."
                ),
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description=(
                    "Directory or file to search under, relative to the project root. Narrow this when you already know the likely subsystem."
                ),
                required=False,
                default=".",
            ),
            ToolParameter(
                name="include",
                type="string",
                description=(
                    "Optional file glob filter such as `*.py`, `*.ts`, or `src/**/*.tsx`. Strongly recommended for large repositories."
                ),
                required=False,
            ),
            ToolParameter(
                name="case_sensitive",
                type="boolean",
                description="Set to true only when letter case matters for the search.",
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        pattern = parameters.get("pattern")
        raw_path = parameters.get("path", ".")
        include = parameters.get("include")
        case_sensitive = bool(parameters.get("case_sensitive", False))

        if not pattern or not isinstance(pattern, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="pattern must be a non-empty regex string.",
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

        include_patterns = normalize_ignore_patterns(include)

        try:
            results, engine = self._run_search(root, pattern, include_patterns, case_sensitive)
        except re.error as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid regex pattern: {exc}",
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Search failed: {exc}",
            )

        truncated = len(results) > self.MAX_RESULTS
        results = results[: self.MAX_RESULTS]

        rel_root = relative_display(self.project_root, root)
        lines = [
            f"Pattern: {pattern}",
            f"Search root: {rel_root}",
            f"Engine: {engine}",
        ]
        if include_patterns:
            lines.append(f"Include globs: {', '.join(include_patterns)}")
        lines.append("")

        if results:
            lines.extend(results)
        else:
            lines.append("[no matches]")
        if truncated:
            lines.extend(["", f"[showing first {self.MAX_RESULTS} matches only]"])

        response_factory = ToolResponse.partial if truncated else ToolResponse.success
        return response_factory(
            text="\n".join(lines),
            data={
                "pattern": pattern,
                "path": rel_root,
                "matches": results,
                "engine": engine,
                "truncated": truncated,
            },
        )

    def _run_search(
        self,
        root: Path,
        pattern: str,
        include_patterns: List[str],
        case_sensitive: bool,
    ) -> tuple[List[str], str]:
        if shutil.which("rg"):
            try:
                return self._run_rg(root, pattern, include_patterns, case_sensitive), "ripgrep"
            except subprocess.SubprocessError:
                pass
        return self._run_python(root, pattern, include_patterns, case_sensitive), "python"

    def _run_rg(
        self,
        root: Path,
        pattern: str,
        include_patterns: List[str],
        case_sensitive: bool,
    ) -> List[str]:
        command = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-count",
            str(self.MAX_RESULTS),
        ]
        if not case_sensitive:
            command.append("-i")
        for include_pattern in include_patterns:
            command.extend(["--glob", include_pattern])
        command.extend([pattern, str(root)])

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        if completed.returncode not in (0, 1):
            raise subprocess.SubprocessError(completed.stderr.strip() or "ripgrep failed")

        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        results = []
        for line in output_lines:
            if len(results) >= self.MAX_RESULTS:
                break
            results.append(self._normalize_rg_line(line))
        return results

    def _run_python(
        self,
        root: Path,
        pattern: str,
        include_patterns: List[str],
        case_sensitive: bool,
    ) -> List[str]:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
        matches: List[str] = []

        if root.is_file():
            files = [root]
        else:
            files = []
            for current_root, dirnames, filenames in os.walk(root):
                prune_walk_dirs(dirnames, include_hidden=False)
                for filename in filenames:
                    if filename in DEFAULT_IGNORES or filename.startswith("."):
                        continue
                    file_path = Path(current_root) / filename
                    rel_path = relative_display(self.project_root, file_path)
                    if include_patterns and not any(
                        fnmatch.fnmatch(rel_path, pattern_item)
                        or fnmatch.fnmatch(filename, pattern_item)
                        for pattern_item in include_patterns
                    ):
                        continue
                    files.append(file_path)

        for file_path in files:
            if len(matches) >= self.MAX_RESULTS:
                break
            if file_path.is_dir() or is_binary_file(file_path):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            rel_path = relative_display(self.project_root, file_path)
            for line_number, line in enumerate(content.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(f"{rel_path}:{line_number}: {line}")
                    if len(matches) >= self.MAX_RESULTS:
                        break

        return matches

    def _normalize_rg_line(self, line: str) -> str:
        path_part, line_number, rest = line.split(":", 2)
        path_obj = Path(path_part)
        if not path_obj.is_absolute():
            path_obj = (Path.cwd() / path_obj).resolve()
        return f"{relative_display(self.project_root, path_obj)}:{line_number}: {rest}"
