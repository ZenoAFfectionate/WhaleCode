"""Glob file-search tool for the coding agent."""

from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import (
    DEFAULT_IGNORES,
    ensure_working_dir,
    prune_walk_dirs,
    relative_display,
    resolve_path,
)


class GlobTool(Tool):
    """Fast file pattern matching using glob patterns.

    Uses ``os.walk`` with directory pruning for performance and ``fnmatch``
    with a custom glob-to-fnmatch conversion so that single ``*`` never
    crosses directory boundaries while ``**`` does.
    """

    MAX_RESULTS = 200
    MAX_VISITED_ENTRIES = 20_000
    MAX_DURATION_MS = 2_000

    def __init__(
        self,
        name: str = "Glob",
        project_root: str = ".",
        working_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Find files by name using glob patterns. Prefer this when you need to locate files "
                "by extension, naming convention, or directory structure. Use patterns like "
                "`**/*.py`, `src/**/*.ts`, or `**/test_*.py`. Results are sorted by modification time (newest first)."
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
                    "Glob pattern to match files against, such as `**/*.py`, `src/**/*.tsx`, "
                    "`**/config.*`, or `test_*.py`. Use `**` to match across directories."
                ),
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description=(
                    "Directory to search in, relative to the project root. "
                    "Defaults to the project root."
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
            matches, aborted_reason = self._search(root, pattern)
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Glob search failed: {exc}",
            )

        # Sort by modification time (newest first).
        matches.sort(key=lambda p: self._safe_mtime(p), reverse=True)

        truncated = len(matches) > self.MAX_RESULTS
        matches = matches[: self.MAX_RESULTS]

        rel_root = relative_display(self.project_root, root)
        rel_paths = [relative_display(self.project_root, m) for m in matches]

        lines = [
            f"Pattern: {pattern}",
            f"Search root: {rel_root}",
            f"Found: {len(rel_paths)} file(s)",
        ]
        if aborted_reason:
            lines.append(f"Warning: search stopped early ({aborted_reason})")
        lines.append("")

        if rel_paths:
            lines.extend(rel_paths)
        else:
            lines.append("[no matches]")
        if truncated:
            lines.extend(["", f"[showing first {self.MAX_RESULTS} matches only]"])

        data = {
            "pattern": pattern,
            "path": rel_root,
            "matches": rel_paths,
            "truncated": truncated,
        }

        if aborted_reason and not rel_paths:
            return ToolResponse.error(
                code=ToolErrorCode.TIMEOUT,
                message="\n".join(lines),
            )
        if truncated or aborted_reason:
            return ToolResponse.partial(text="\n".join(lines), data=data)
        return ToolResponse.success(text="\n".join(lines), data=data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(
        self, root: Path, pattern: str
    ) -> tuple[List[Path], Optional[str]]:
        """Walk the directory tree and collect matching files.

        Returns ``(matches, aborted_reason)`` where *aborted_reason* is
        ``None`` when the search completed normally or a short label such as
        ``"count_limit"`` / ``"time_limit"`` when it was stopped early.
        """
        start = time.monotonic()
        visited = 0
        matches: List[Path] = []
        aborted_reason: Optional[str] = None

        normalized = self._strip_relative_prefix(pattern.replace("\\", "/").strip())

        for current_root, dirnames, filenames in os.walk(root, topdown=True):
            dirnames.sort()
            filenames.sort()
            prune_walk_dirs(dirnames, include_hidden=False)

            visited += 1
            aborted_reason = self._check_limits(start, visited)
            if aborted_reason:
                break

            for filename in filenames:
                visited += 1
                aborted_reason = self._check_limits(start, visited)
                if aborted_reason:
                    break

                if filename.startswith("."):
                    continue

                file_path = Path(current_root) / filename
                try:
                    rel_posix = file_path.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue

                if self._match_pattern(rel_posix, normalized):
                    matches.append(file_path)
                    if len(matches) > self.MAX_RESULTS:
                        break

            if aborted_reason or len(matches) > self.MAX_RESULTS:
                break

        return matches, aborted_reason

    def _check_limits(self, start: float, visited: int) -> Optional[str]:
        if visited > self.MAX_VISITED_ENTRIES:
            return "count_limit"
        if (time.monotonic() - start) * 1000 > self.MAX_DURATION_MS:
            return "time_limit"
        return None

    # ------------------------------------------------------------------
    # Pattern matching (fnmatch with proper glob semantics)
    # ------------------------------------------------------------------

    @staticmethod
    def _match_pattern(rel_posix: str, pattern: str) -> bool:
        """Match *rel_posix* against a normalised glob *pattern*.

        ``fnmatch.fnmatch`` treats ``*`` as matching everything including
        ``/``, which breaks glob semantics.  This method converts the
        pattern so that only ``**`` crosses directory boundaries.
        """
        converted = GlobTool._convert_glob_to_fnmatch(pattern)
        if fnmatch.fnmatch(rel_posix, converted):
            return True
        # ``**/`` should also match zero directory levels.
        if pattern.startswith("**/"):
            zero = GlobTool._convert_glob_to_fnmatch(pattern[3:])
            if fnmatch.fnmatch(rel_posix, zero):
                return True
        return False

    @staticmethod
    def _convert_glob_to_fnmatch(pattern: str) -> str:
        """Convert a glob pattern to an fnmatch-compatible pattern.

        * ``**`` -> ``*``  (fnmatch ``*`` matches everything incl. ``/``)
        * lone ``*`` -> ``[!/]*``  (does **not** match ``/``)
        * lone ``?`` -> ``[!/]``
        """
        result: List[str] = []
        i, n = 0, len(pattern)
        while i < n:
            ch = pattern[i]
            if ch == "*" and i + 1 < n and pattern[i + 1] == "*":
                result.append("*")
                i += 2
            elif ch == "*":
                result.append("[!/]*")
                i += 1
            elif ch == "?":
                result.append("[!/]")
                i += 1
            else:
                result.append(ch)
                i += 1
        return "".join(result)

    @staticmethod
    def _strip_relative_prefix(pattern: str) -> str:
        while pattern.startswith("./"):
            pattern = pattern[2:]
        while pattern.startswith("/"):
            pattern = pattern[1:]
        return pattern

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0
