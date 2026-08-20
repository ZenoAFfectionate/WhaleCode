"""Glob file-search tool for the coding agent."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import INTERNAL_ARTIFACT_DIRS, SEARCH_EXCLUDE_GLOBS
from .file_tools import _WorkspaceFileTool


class GlobTool(_WorkspaceFileTool):
    """Find files via ripgrep-backed file enumeration."""

    MAX_RESULTS = 100
    # Q2-8: 单一事实来源在 _code_utils（与 GrepTool 共享，防止漂移）
    DEFAULT_EXCLUDE_GLOBS = SEARCH_EXCLUDE_GLOBS
    INTERNAL_ARTIFACT_DIRS = set(INTERNAL_ARTIFACT_DIRS)

    def __init__(
        self,
        name: str = "Glob",
        project_root: str = ".",
        working_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Find files by glob pattern. Prefer this when you need to locate files by extension, "
                "naming convention, or directory structure. Results are limited to 100 files and "
                "sorted by modification time (newest first)."
            ),
            project_root=project_root,
            working_dir=working_dir,
            category="readonly",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="The glob pattern to match files against.",
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
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        pattern = parameters.get("pattern")
        raw_path = parameters.get("path", ".")

        if not pattern or not isinstance(pattern, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="pattern must be a non-empty glob string.",
            )

        try:
            root = self._resolve_path(raw_path)
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
            rel_paths, truncated = self._run_rg_files(root, pattern)
        except FileNotFoundError:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message="ripgrep is required for GlobTool but was not found.",
            )
        except subprocess.SubprocessError as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Glob search failed: {exc}",
            )

        rel_root = self._display_path(root)
        lines = []
        if rel_paths:
            lines.extend(rel_paths)
            if truncated:
                lines.extend(
                    [
                        "",
                        f"(Results are truncated: showing first {self.MAX_RESULTS} results. Consider using a more specific path or pattern.)",
                    ]
                )
        else:
            lines.append("No files found")

        data = {
            "pattern": pattern,
            "path": rel_root,
            "matches": [{"path": path} for path in rel_paths],
            "count": len(rel_paths),
            "truncated": truncated,
        }

        factory = ToolResponse.partial if truncated else ToolResponse.success
        return factory(text="\n".join(lines), data=data)

    def _run_rg_files(self, root: Path, pattern: str) -> tuple[List[str], bool]:
        command = [
            "rg",
            "--files",
            "--hidden",
        ]
        command.extend(f"--glob={glob}" for glob in self.DEFAULT_EXCLUDE_GLOBS)
        command.append(f"--glob={pattern}")

        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        matches: List[tuple[str, float]] = []
        truncated = False

        assert process.stdout is not None
        for raw_line in process.stdout:
            rel_line = raw_line.rstrip("\r\n")
            if not rel_line:
                continue
            if any(part in self.INTERNAL_ARTIFACT_DIRS for part in Path(rel_line).parts):
                continue

            if len(matches) >= self.MAX_RESULTS:
                truncated = True
                process.terminate()
                break

            full_path = (root / rel_line).resolve()
            try:
                mtime = full_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            matches.append((self._display_path(full_path), mtime))

        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

        if process.returncode not in (0, None) and not truncated:
            raise subprocess.SubprocessError("ripgrep file enumeration failed")

        matches.sort(key=lambda item: item[1], reverse=True)
        return [path for path, _ in matches], truncated
