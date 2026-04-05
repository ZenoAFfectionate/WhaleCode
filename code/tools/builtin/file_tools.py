"""Workspace file tools with bounded reads and safer optimistic locking."""

from __future__ import annotations

import os
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import (
    DiagnosticsResult,
    EditAmbiguousError,
    EditMatchError,
    EditNotFoundError,
    FormatterResult,
    atomic_write,
    detect_line_ending,
    ensure_working_dir,
    format_numbered_lines,
    is_binary_file,
    make_diff_preview,
    normalize_line_endings,
    read_text_file,
    read_text_window,
    relative_display,
    replace_with_flexible_match,
    resolve_path,
    run_diagnostics,
    run_formatter,
)

if TYPE_CHECKING:
    from ..registry import ToolRegistry


DEFAULT_READ_LIMIT = 2000
DEFAULT_LIST_LIMIT = 200
MAX_READ_BYTES = 50 * 1024
MAX_READ_LINE_LENGTH = 2000

_FILE_LOCKS: dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class FileState:
    """Filesystem snapshot used for optimistic locking."""

    mtime_ms: int
    ctime_ms: int
    size_bytes: int
    encoding: Optional[str] = None


def _format_diff_section(diff_preview: str, diff_truncated: bool) -> str:
    lines = ["", "Unified diff preview:", diff_preview]
    if diff_truncated:
        lines.append("[diff preview truncated]")
    return "\n".join(lines)


def _snapshot_file(path: Path, encoding: Optional[str] = None) -> FileState:
    stat = path.stat()
    return FileState(
        mtime_ms=int(stat.st_mtime * 1000),
        ctime_ms=int(stat.st_ctime * 1000),
        size_bytes=stat.st_size,
        encoding=encoding,
    )


def _metadata_payload(state: FileState) -> Dict[str, Any]:
    return {
        "file_mtime_ms": state.mtime_ms,
        "file_ctime_ms": state.ctime_ms,
        "file_size_bytes": state.size_bytes,
        "expected_mtime_ms": state.mtime_ms,
        "expected_ctime_ms": state.ctime_ms,
        "expected_size_bytes": state.size_bytes,
    }


def _metadata_text(state: FileState) -> str:
    return (
        f"Metadata: file_mtime_ms={state.mtime_ms}, "
        f"file_ctime_ms={state.ctime_ms}, "
        f"file_size_bytes={state.size_bytes}, "
        f"expected_mtime_ms={state.mtime_ms}, "
        f"expected_ctime_ms={state.ctime_ms}, "
        f"expected_size_bytes={state.size_bytes}"
    )


def _no_change_response(action_text: str, rel_path: str, state: FileState) -> ToolResponse:
    return ToolResponse.partial(
        text=(
            f"{action_text}: {rel_path}\n"
            "No actual textual changes.\n"
            f"{_metadata_text(state)}\n"
            "Unified diff preview:\n[no textual diff]"
        ),
        data={
            "path": rel_path,
            "modified": False,
            "diff_preview": "[no textual diff]",
            "diff_truncated": False,
            **_metadata_payload(state),
        },
    )


def _post_process_failed(formatter: FormatterResult, diagnostics: DiagnosticsResult) -> bool:
    formatter_failed = formatter.attempted and not formatter.success
    diagnostics_failed = diagnostics.attempted and not diagnostics.success
    return formatter_failed or diagnostics_failed


def _formatter_summary(result: FormatterResult) -> Optional[str]:
    if not result.attempted:
        return None
    if result.success:
        change_text = "changed file" if result.changed else "no changes"
        return f"Formatter: {result.tool} ({change_text})"
    return f"Formatter failed: {result.tool or 'unknown'} ({result.skipped_reason or 'execution failed'})"


def _diagnostics_summary(result: DiagnosticsResult) -> List[str]:
    if not result.attempted:
        return []
    if not result.success:
        return [f"Diagnostics failed: {result.tool or 'unknown'} ({result.skipped_reason or 'execution failed'})"]

    if result.total == 0:
        return [f"Diagnostics: {result.tool} reported no issues."]

    lines = [
        f"Diagnostics: {result.tool} reported {result.total} issue{'s' if result.total != 1 else ''}"
        + (" (showing first 20)" if result.truncated else "")
    ]
    for item in result.diagnostics:
        code = f"[{item.code}] " if item.code else ""
        lines.append(f"  {item.severity.upper()} L{item.line}:C{item.column} {code}{item.message}")
    return lines


def _format_range(offset: int, shown: int, total: int, label: str) -> str:
    if total == 0:
        return f"{label}: empty / 0"
    if shown == 0:
        return f"{label}: no entries at offset {offset + 1} / {total}"
    return f"{label}: {offset + 1}-{offset + shown} / {total}"


@contextmanager
def _path_lock(path: Path) -> Iterator[None]:
    key = str(path)
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.setdefault(key, threading.RLock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


class _WorkspaceFileTool(Tool):
    """Shared path and optimistic-lock helpers."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional["ToolRegistry"] = None,
    ):
        super().__init__(name=name, description=description, expandable=False)
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = ensure_working_dir(self.project_root, working_dir)
        self.registry = registry

    def _resolve_path(self, path: str) -> Path:
        return resolve_path(self.project_root, self.working_dir, path)

    def _display_path(self, path: Path) -> str:
        return relative_display(self.project_root, path)

    def _nearest_existing_ancestor(self, path: Path) -> Path:
        current = path.parent if not path.exists() else path
        while current != self.project_root and not current.exists():
            current = current.parent
        return current if current.exists() else self.project_root

    def _missing_path_response(
        self,
        requested_path: str,
        full_path: Path,
        *,
        detail_line: str = "No content was read from the workspace.",
    ) -> ToolResponse:
        rel_path = self._display_path(full_path)
        parent_path = full_path.parent
        parent_display = self._display_path(parent_path)
        parent_exists = parent_path.exists()
        nearest_existing = self._nearest_existing_ancestor(full_path)
        nearest_display = self._display_path(nearest_existing)
        suggested_check_path = parent_display if parent_exists else nearest_display

        text_lines = [
            f"Path does not exist: {rel_path}",
            detail_line,
        ]
        if requested_path != rel_path:
            text_lines.append(f"Requested path: {requested_path}")
        if not parent_exists:
            text_lines.append(f"Nearest existing ancestor: {nearest_display}")
        text_lines.append(
            f"Please verify the path and inspect '{suggested_check_path}' to confirm the correct location or name."
        )

        return ToolResponse.partial(
            text="\n".join(text_lines),
            data={
                "path": rel_path,
                "requested_path": requested_path,
                "exists": False,
                "missing_path": True,
                "parent_path": parent_display,
                "parent_exists": parent_exists,
                "nearest_existing_path": nearest_display,
                "suggested_check_path": suggested_check_path,
            },
        )

    def _get_cached_metadata(self, rel_path: str) -> Optional[Dict[str, Any]]:
        if self.registry is None:
            return None
        getter = getattr(self.registry, "get_read_metadata", None)
        if callable(getter):
            return getter(rel_path)
        return getattr(self.registry, "read_metadata_cache", {}).get(rel_path)

    def _cache_state(self, rel_path: str, state: FileState) -> None:
        if self.registry is None:
            return
        cache = getattr(self.registry, "cache_read_metadata", None)
        if callable(cache):
            payload = _metadata_payload(state)
            if state.encoding:
                payload["encoding"] = state.encoding
            cache(rel_path, payload)

    def _clear_cached_state(self, rel_path: str, recursive: bool = False) -> None:
        if self.registry is None:
            return
        if not recursive:
            clear = getattr(self.registry, "clear_read_cache", None)
            if callable(clear):
                clear(rel_path)
            return

        cache = getattr(self.registry, "read_metadata_cache", None)
        if not isinstance(cache, dict):
            return
        prefix = f"{rel_path}/"
        for key in list(cache):
            if key == rel_path or key.startswith(prefix):
                cache.pop(key, None)

    def _require_prior_read(self, rel_path: str) -> Optional[ToolResponse]:
        if self.registry is None:
            return None
        if self._get_cached_metadata(rel_path) is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"You must Read '{rel_path}' before editing. "
                    "Call the Read tool first to see current content."
                ),
            )
        return None

    def _expected_state_from_parameters(self, parameters: Dict[str, Any], rel_path: str) -> Dict[str, Any]:
        cached = self._get_cached_metadata(rel_path) or {}
        expected_mtime_ms = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        expected_ctime_ms = parameters.get("expected_ctime_ms", parameters.get("file_ctime_ms"))
        expected_size_bytes = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))

        if expected_mtime_ms is None:
            expected_mtime_ms = cached.get("file_mtime_ms")
        if expected_ctime_ms is None:
            expected_ctime_ms = cached.get("file_ctime_ms")
        if expected_size_bytes is None:
            expected_size_bytes = cached.get("file_size_bytes")

        return {
            "expected_mtime_ms": expected_mtime_ms,
            "expected_ctime_ms": expected_ctime_ms,
            "expected_size_bytes": expected_size_bytes,
        }

    def _check_expected_state(
        self,
        *,
        rel_path: str,
        current_state: FileState,
        expected_state: Dict[str, Any],
    ) -> Optional[ToolResponse]:
        expected_mtime_ms = expected_state.get("expected_mtime_ms")
        expected_ctime_ms = expected_state.get("expected_ctime_ms")
        expected_size_bytes = expected_state.get("expected_size_bytes")

        if expected_mtime_ms is not None and current_state.mtime_ms != expected_mtime_ms:
            return ToolResponse.error(
                code=ToolErrorCode.CONFLICT,
                message=(
                    f"File changed since last read. expected_mtime_ms={expected_mtime_ms}, "
                    f"actual_mtime_ms={current_state.mtime_ms}."
                ),
                context={
                    "path": rel_path,
                    "expected_mtime_ms": expected_mtime_ms,
                    "actual_mtime_ms": current_state.mtime_ms,
                },
            )

        if expected_ctime_ms is not None and current_state.ctime_ms != expected_ctime_ms:
            return ToolResponse.error(
                code=ToolErrorCode.CONFLICT,
                message=(
                    f"File changed since last read. expected_ctime_ms={expected_ctime_ms}, "
                    f"actual_ctime_ms={current_state.ctime_ms}."
                ),
                context={
                    "path": rel_path,
                    "expected_ctime_ms": expected_ctime_ms,
                    "actual_ctime_ms": current_state.ctime_ms,
                },
            )

        if expected_size_bytes is not None and current_state.size_bytes != expected_size_bytes:
            return ToolResponse.error(
                code=ToolErrorCode.CONFLICT,
                message=(
                    f"File size changed since last read. expected_size_bytes={expected_size_bytes}, "
                    f"actual_size_bytes={current_state.size_bytes}."
                ),
                context={
                    "path": rel_path,
                    "expected_size_bytes": expected_size_bytes,
                    "actual_size_bytes": current_state.size_bytes,
                },
            )

        return None


class ReadTool(_WorkspaceFileTool):
    """Read a text file or inspect a directory."""

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional["ToolRegistry"] = None,
    ):
        super().__init__(
            name="Read",
            description=(
                "Read a text file or inspect a directory. Use this before editing so you can "
                "see the current content and capture optimistic-lock metadata for later Write, "
                "Edit, or Delete calls."
            ),
            project_root=project_root,
            working_dir=working_dir,
            registry=registry,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description=(
                    "File or directory path relative to the project root. If the path is a "
                    "directory, the tool returns a directory listing instead of file content."
                ),
                required=True,
            ),
            ToolParameter(
                name="offset",
                type="integer",
                description=(
                    "Zero-based starting offset. For files it is the first line to read. "
                    "For directories it is the first entry to list."
                ),
                required=False,
                default=0,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description=(
                    "Maximum number of lines or directory entries to return. "
                    "Use a focused window for large files or folders."
                ),
                required=False,
                default=DEFAULT_READ_LIMIT,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path")
        offset = parameters.get("offset", 0)
        limit = parameters.get("limit", DEFAULT_READ_LIMIT)

        if not path or not isinstance(path, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path",
            )
        if not isinstance(offset, int) or offset < 0:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Parameter offset must be an integer greater than or equal to 0",
            )
        if not isinstance(limit, int) or limit < 1:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Parameter limit must be an integer greater than or equal to 1",
            )

        try:
            full_path = self._resolve_path(path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied",
            )

        try:
            if not full_path.exists():
                return self._missing_path_response(path, full_path)

            if full_path.is_dir():
                return self._list_directory(full_path=full_path, offset=offset, limit=limit)

            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"File '{path}' is a binary file and cannot be read as text",
                )

            try:
                window = read_text_window(
                    full_path,
                    offset=offset,
                    limit=limit,
                    max_bytes=MAX_READ_BYTES,
                    max_line_length=MAX_READ_LINE_LENGTH,
                )
            except ValueError as exc:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=str(exc),
                )

            state = _snapshot_file(full_path, encoding=window.encoding)
            rel_path = self._display_path(full_path)
            self._cache_state(rel_path, state)

            numbered = format_numbered_lines(window.content, start_line=offset + 1)
            text_lines = [
                f"File: {rel_path}",
                f"Encoding: {window.encoding}",
                _format_range(offset, window.shown_lines, window.total_lines, "Line range"),
                _metadata_text(state),
                "",
                numbered,
            ]

            if window.truncated:
                notes = []
                if window.truncated_by_bytes:
                    notes.append(f"Output capped at {MAX_READ_BYTES // 1024} KB.")
                if window.truncated_long_lines:
                    notes.append(f"Some long lines were truncated to {MAX_READ_LINE_LENGTH} characters.")
                if window.next_offset is not None:
                    notes.append(f"Use offset={window.next_offset} to continue.")
                if notes:
                    text_lines.extend(["", " ".join(notes)])

            data = {
                "path": rel_path,
                "content": window.content,
                "lines": window.shown_lines,
                "total_lines": window.total_lines,
                "offset": offset,
                "limit": limit,
                "encoding": window.encoding,
                "truncated": window.truncated,
                "truncated_by_bytes": window.truncated_by_bytes,
                "truncated_long_lines": window.truncated_long_lines,
                "next_offset": window.next_offset,
                **_metadata_payload(state),
            }
            response_factory = ToolResponse.partial if window.truncated else ToolResponse.success
            return response_factory(text="\n".join(text_lines), data=data)

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to read '{path}'",
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to read file: {exc}",
            )

    def _list_directory(self, *, full_path: Path, offset: int, limit: int) -> ToolResponse:
        try:
            raw_entries = list(full_path.iterdir())
        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"No permission to access directory '{self._display_path(full_path)}'",
            )

        raw_entries.sort(key=lambda item: (not item.is_dir(), item.name.lower()))
        total_entries = len(raw_entries)
        if offset > total_entries and not (offset == 0 and total_entries == 0):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Offset {offset} is out of range for this directory ({total_entries} entries)",
            )

        total_files = 0
        total_dirs = 0
        entries = []
        for entry in raw_entries:
            is_dir = entry.is_dir()
            total_dirs += int(is_dir)
            total_files += int(not is_dir)
            stat_result = None
            if not is_dir:
                try:
                    stat_result = entry.stat()
                except OSError:
                    stat_result = None
            try:
                mtime = entry.stat().st_mtime
                mtime_text = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                mtime_text = "?"

            size_text = "<DIR>" if is_dir else (self._format_size(stat_result.st_size) if stat_result else "?")
            entries.append(
                {
                    "name": entry.name,
                    "type": "directory" if is_dir else "file",
                    "size": size_text,
                    "mtime": mtime_text,
                    "path": self._display_path(entry),
                }
            )

        selected_entries = entries[offset : offset + limit]
        truncated = offset + len(selected_entries) < total_entries
        rel_path = self._display_path(full_path)

        if not selected_entries and total_entries == 0:
            body = "[empty directory]"
        else:
            body_lines = []
            for entry in selected_entries:
                marker = "[DIR]" if entry["type"] == "directory" else "[FILE]"
                body_lines.append(
                    f"{marker:<6} {entry['name']:<40} {entry['size']:>10} {entry['mtime']}"
                )
            body = "\n".join(body_lines) if body_lines else "[no entries in requested window]"

        text_lines = [
            f"Directory: {rel_path}",
            _format_range(offset, len(selected_entries), total_entries, "Entry range"),
            f"Totals: directories={total_dirs}, files={total_files}",
            "",
            body,
        ]
        if truncated:
            text_lines.extend(["", f"Use offset={offset + len(selected_entries)} to continue."])

        data = {
            "path": rel_path,
            "entries": selected_entries,
            "total_entries": total_entries,
            "total_files": total_files,
            "total_dirs": total_dirs,
            "is_directory": True,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": offset + len(selected_entries) if truncated else None,
        }
        response_factory = ToolResponse.partial if truncated else ToolResponse.success
        return response_factory(text="\n".join(text_lines), data=data)

    @staticmethod
    def _format_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024.0:
                return f"{value:.1f}{unit}"
            value /= 1024.0
        return f"{value:.1f}TB"


class ListFilesTool(ReadTool):
    """Directory listing tool using the historical LS name."""

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional["ToolRegistry"] = None,
    ):
        super().__init__(project_root=project_root, working_dir=working_dir, registry=registry)
        self.name = "LS"
        self.description = (
            "List files and folders inside a directory. Prefer this over Read when you need "
            "to understand project structure, discover file names, or decide what to inspect next."
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Directory path relative to the project root. Defaults to the workspace root.",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="offset",
                type="integer",
                description="Zero-based starting directory-entry offset.",
                required=False,
                default=0,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of directory entries to return.",
                required=False,
                default=DEFAULT_LIST_LIMIT,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path", ".")
        offset = parameters.get("offset", 0)
        limit = parameters.get("limit", DEFAULT_LIST_LIMIT)

        if not isinstance(offset, int) or offset < 0:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Parameter offset must be an integer greater than or equal to 0",
            )
        if not isinstance(limit, int) or limit < 1:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Parameter limit must be an integer greater than or equal to 1",
            )

        try:
            full_path = self._resolve_path(path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied",
            )

        if not full_path.exists():
            return self._missing_path_response(path, full_path)
        if not full_path.is_dir():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Path '{path}' is not a directory",
            )
        return self._list_directory(full_path=full_path, offset=offset, limit=limit)


class WriteTool(_WorkspaceFileTool):
    """Create or overwrite a text file."""

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional["ToolRegistry"] = None,
    ):
        super().__init__(
            name="Write",
            description=(
                "Create a new file or replace the full contents of an existing file. "
                "Use this for full-file rewrites or generated files."
            ),
            project_root=project_root,
            working_dir=working_dir,
            registry=registry,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Target file path relative to the project root.",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="The complete new file content to write. This replaces the whole file.",
                required=True,
            ),
            ToolParameter(
                name="file_mtime_ms",
                type="integer",
                description="Legacy optimistic-lock mtime from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="file_ctime_ms",
                type="integer",
                description="Legacy optimistic-lock ctime from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="file_size_bytes",
                type="integer",
                description="Legacy optimistic-lock size from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="expected_mtime_ms",
                type="integer",
                description="Preferred optimistic-lock mtime from Read.",
                required=False,
            ),
            ToolParameter(
                name="expected_ctime_ms",
                type="integer",
                description="Preferred optimistic-lock ctime from Read.",
                required=False,
            ),
            ToolParameter(
                name="expected_size_bytes",
                type="integer",
                description="Optional optimistic-lock file size from Read for stricter conflict detection.",
                required=False,
            ),
            ToolParameter(
                name="dry_run",
                type="boolean",
                description="If true, show the unified diff preview without writing anything to disk.",
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path")
        content = parameters.get("content")
        dry_run = bool(parameters.get("dry_run", False))

        if not path or not isinstance(path, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path",
            )
        if content is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: content",
            )
        if not isinstance(content, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Parameter content must be a string",
            )

        try:
            full_path = self._resolve_path(path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied",
            )

        try:
            rel_path = self._display_path(full_path)
            with _path_lock(full_path):
                old_content = ""
                encoding = "utf-8"
                line_ending = "\n"
                current_state: Optional[FileState] = None
                backup_path: Optional[Path] = None

                if full_path.exists():
                    if full_path.is_dir():
                        return ToolResponse.error(
                            code=ToolErrorCode.IS_DIRECTORY,
                            message=f"Path '{path}' is a directory, cannot write as a file",
                        )
                    if is_binary_file(full_path):
                        return ToolResponse.error(
                            code=ToolErrorCode.BINARY_FILE,
                            message=f"File '{path}' is a binary file, refusing to overwrite as text",
                        )

                    old_content, encoding = read_text_file(full_path)
                    line_ending = detect_line_ending(old_content)
                    current_state = _snapshot_file(full_path, encoding=encoding)
                    expected_state = self._expected_state_from_parameters(parameters, rel_path)
                    conflict = self._check_expected_state(
                        rel_path=rel_path,
                        current_state=current_state,
                        expected_state=expected_state,
                    )
                    if conflict:
                        return conflict

                else:
                    full_path.parent.mkdir(parents=True, exist_ok=True)

                new_content = normalize_line_endings(content, line_ending) if old_content else content
                if current_state and new_content == old_content:
                    self._cache_state(rel_path, current_state)
                    return _no_change_response("Write check complete", rel_path, current_state)

                diff_preview, diff_truncated = make_diff_preview(old_content, new_content, rel_path)
                predicted_size_bytes = len(new_content.encode(encoding))
                formatter_result = FormatterResult(
                    attempted=False,
                    available=False,
                    success=False,
                    skipped_reason="Formatter was not run.",
                )
                diagnostics_result = DiagnosticsResult(
                    attempted=False,
                    available=False,
                    success=False,
                    skipped_reason="Diagnostics were not run.",
                )

                if not dry_run:
                    if full_path.exists():
                        backup_path = self._backup_file(full_path)
                    atomic_write(full_path, new_content, encoding=encoding)
                    formatter_result = run_formatter(full_path, self.project_root)
                    diagnostics_result = run_diagnostics(full_path, self.project_root)
                    final_content, final_encoding = read_text_file(full_path)
                    encoding = final_encoding
                    predicted_size_bytes = len(final_content.encode(encoding))
                    diff_preview, diff_truncated = make_diff_preview(old_content, final_content, rel_path)
                    new_state = _snapshot_file(full_path, encoding=encoding)
                    self._cache_state(rel_path, new_state)
                else:
                    new_state = current_state

                text_lines = [
                    f"{'Dry run: would write' if dry_run else 'Successfully wrote'} {rel_path} ({predicted_size_bytes} bytes)",
                ]
                if new_state is not None:
                    text_lines.append(_metadata_text(new_state))
                else:
                    text_lines.append("Metadata: file will be created on actual execution.")
                if backup_path is not None:
                    text_lines.append(f"Backup: {self._display_path(backup_path)}")
                formatter_summary = _formatter_summary(formatter_result)
                if formatter_summary:
                    text_lines.append(formatter_summary)
                text_lines.extend(_diagnostics_summary(diagnostics_result))
                text_lines.append(_format_diff_section(diff_preview, diff_truncated))

                data = {
                    "path": rel_path,
                    "written": not dry_run,
                    "modified": True,
                    "dry_run": dry_run,
                    "size_bytes": predicted_size_bytes,
                    "diff_preview": diff_preview,
                    "diff_truncated": diff_truncated,
                    "backup_path": self._display_path(backup_path) if backup_path else None,
                    "encoding": encoding,
                    "formatter": formatter_result.to_dict(),
                    "diagnostics": diagnostics_result.to_dict(),
                }
                if new_state is not None:
                    data.update(_metadata_payload(new_state))
                else:
                    data.update(
                        {
                            "file_mtime_ms": None,
                            "file_ctime_ms": None,
                            "file_size_bytes": predicted_size_bytes,
                            "expected_mtime_ms": None,
                            "expected_ctime_ms": None,
                            "expected_size_bytes": predicted_size_bytes,
                        }
                    )

                response_factory = (
                    ToolResponse.partial if dry_run or _post_process_failed(formatter_result, diagnostics_result) else ToolResponse.success
                )
                return response_factory(text="\n".join(text_lines), data=data)

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to write '{path}'",
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to write file: {exc}",
            )

    @staticmethod
    def _backup_file(full_path: Path) -> Path:
        backup_dir = full_path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_dir / f"{full_path.name}.{timestamp}.bak"
        shutil.copy2(full_path, backup_path)
        return backup_path


class DeleteTool(_WorkspaceFileTool):
    """Safe delete for files and directories."""

    PROTECTED_PARTS = {".git", ".hg", ".svn"}
    MAX_DIR_ENTRIES_WITHOUT_FORCE = 2000

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional["ToolRegistry"] = None,
    ):
        super().__init__(
            name="Delete",
            description=(
                "Delete a file or directory safely. Uses an atomic move to an internal trash area "
                "first to reduce accidental data loss risk."
            ),
            project_root=project_root,
            working_dir=working_dir,
            registry=registry,
        )
        self.trash_root = (self.project_root / "memory" / ".delete_trash").resolve()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Target file or directory path relative to the project root.",
                required=True,
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="Required for deleting non-empty directories.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="force",
                type="boolean",
                description="Allow large recursive deletes after safety checks.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="dry_run",
                type="boolean",
                description="If true, only report what would be deleted.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="purge",
                type="boolean",
                description="If true, permanently delete from trash after the atomic move.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="file_mtime_ms",
                type="integer",
                description="Legacy optimistic-lock mtime from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="file_ctime_ms",
                type="integer",
                description="Legacy optimistic-lock ctime from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="file_size_bytes",
                type="integer",
                description="Legacy optimistic-lock size from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="expected_mtime_ms",
                type="integer",
                description="Preferred optimistic-lock mtime from Read.",
                required=False,
            ),
            ToolParameter(
                name="expected_ctime_ms",
                type="integer",
                description="Preferred optimistic-lock ctime from Read.",
                required=False,
            ),
            ToolParameter(
                name="expected_size_bytes",
                type="integer",
                description="Optional optimistic-lock file size from Read.",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path")
        recursive = bool(parameters.get("recursive", False))
        force = bool(parameters.get("force", False))
        dry_run = bool(parameters.get("dry_run", False))
        purge = bool(parameters.get("purge", False))

        if not path or not isinstance(path, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path",
            )

        try:
            full_path = self._resolve_path(path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied",
            )

        try:
            if not full_path.exists():
                return self._missing_path_response(
                    path,
                    full_path,
                    detail_line="Nothing was deleted.",
                )

            protected_error = self._validate_protected_path(full_path)
            if protected_error:
                return ToolResponse.error(
                    code=ToolErrorCode.ACCESS_DENIED,
                    message=protected_error,
                )

            rel_path = self._display_path(full_path)
            with _path_lock(full_path):
                stat = full_path.stat()
                current_state = FileState(
                    mtime_ms=int(stat.st_mtime * 1000),
                    ctime_ms=int(stat.st_ctime * 1000),
                    size_bytes=stat.st_size if full_path.is_file() else 0,
                )

                expected_state = self._expected_state_from_parameters(parameters, rel_path)
                conflict = self._check_expected_state(
                    rel_path=rel_path,
                    current_state=current_state,
                    expected_state=expected_state,
                )
                if conflict:
                    return conflict

                entry_count = 1
                total_file_bytes = current_state.size_bytes
                is_dir = full_path.is_dir()

                if is_dir:
                    entry_count, total_file_bytes = self._scan_directory_stats(full_path)
                    if entry_count > 0 and not recursive:
                        return ToolResponse.error(
                            code=ToolErrorCode.INVALID_PARAM,
                            message=f"Directory '{path}' is not empty. Set recursive=true to delete it.",
                        )
                    if entry_count > self.MAX_DIR_ENTRIES_WITHOUT_FORCE and not force:
                        return ToolResponse.error(
                            code=ToolErrorCode.INVALID_PARAM,
                            message=(
                                f"Directory '{path}' contains {entry_count} entries. "
                                "Set force=true to confirm large recursive deletion."
                            ),
                        )

                if dry_run:
                    return ToolResponse.partial(
                        text=(
                            f"Dry run: would delete {rel_path}\n"
                            f"Type: {'directory' if is_dir else 'file'}\n"
                            f"Recursive: {recursive}\n"
                            f"Estimated entries: {entry_count}\n"
                            f"Estimated bytes: {total_file_bytes}"
                        ),
                        data={
                            "path": rel_path,
                            "deleted": False,
                            "dry_run": True,
                            "is_directory": is_dir,
                            "recursive": recursive,
                            "entry_count": entry_count,
                            "estimated_bytes": total_file_bytes,
                            **_metadata_payload(current_state),
                        },
                    )

                trash_path = self._make_trash_path(full_path)
                self.trash_root.mkdir(parents=True, exist_ok=True)
                os.replace(str(full_path), str(trash_path))

                purged = False
                if purge:
                    if trash_path.is_dir():
                        shutil.rmtree(trash_path)
                    else:
                        trash_path.unlink(missing_ok=True)
                    purged = True

                self._clear_cached_state(rel_path, recursive=is_dir)
                return ToolResponse.success(
                    text=(
                        f"Deleted {rel_path} via atomic trash move\n"
                        f"Type: {'directory' if is_dir else 'file'}\n"
                        f"Recursive: {recursive}\n"
                        f"Purged: {purged}\n"
                        f"Trash path: {self._display_path(trash_path)}"
                    ),
                    data={
                        "path": rel_path,
                        "deleted": True,
                        "is_directory": is_dir,
                        "recursive": recursive,
                        "purged": purged,
                        "entry_count": entry_count,
                        "estimated_bytes": total_file_bytes,
                        "trash_path": self._display_path(trash_path),
                    },
                )

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to delete '{path}'",
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to delete path: {exc}",
            )

    def _validate_protected_path(self, full_path: Path) -> Optional[str]:
        if full_path == self.project_root:
            return "Refusing to delete the project root."
        if full_path == self.working_dir:
            return "Refusing to delete the current working directory."
        if self.working_dir.is_relative_to(full_path):
            return "Refusing to delete an ancestor of the current working directory."
        if self.trash_root == full_path or self.trash_root.is_relative_to(full_path):
            return "Refusing to delete the internal delete trash area."
        for part in full_path.parts:
            if part in self.PROTECTED_PARTS:
                return f"Refusing to delete protected path segment '{part}'."
        return None

    @staticmethod
    def _scan_directory_stats(directory: Path) -> tuple[int, int]:
        entry_count = 0
        total_file_bytes = 0
        for root, dirnames, filenames in os.walk(directory):
            entry_count += len(dirnames) + len(filenames)
            for filename in filenames:
                try:
                    total_file_bytes += (Path(root) / filename).stat().st_size
                except OSError:
                    continue
        return entry_count, total_file_bytes

    def _make_trash_path(self, source_path: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = source_path.name or "root"
        candidate = self.trash_root / f"{timestamp}_{base_name}"
        if candidate.exists():
            candidate = self.trash_root / f"{timestamp}_{os.getpid()}_{base_name}"
        return candidate


class EditTool(_WorkspaceFileTool):
    """Apply precise text replacement inside an existing text file."""

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional["ToolRegistry"] = None,
    ):
        super().__init__(
            name="Edit",
            description=(
                "Apply precise text replacement(s) inside an existing text file. "
                "By default it performs one unique match replacement; set replace_all=true "
                "to replace every occurrence."
            ),
            project_root=project_root,
            working_dir=working_dir,
            registry=registry,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Existing file path relative to the project root.",
                required=True,
            ),
            ToolParameter(
                name="old_string",
                type="string",
                description=(
                    "Exact existing text to replace. It must match exactly one location in the file "
                    "unless replace_all=true."
                ),
                required=True,
            ),
            ToolParameter(
                name="new_string",
                type="string",
                description="Replacement text for the matched old_string.",
                required=True,
            ),
            ToolParameter(
                name="replace_all",
                type="boolean",
                description="If true, replace all occurrences of old_string.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="file_mtime_ms",
                type="integer",
                description="Legacy optimistic-lock mtime from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="file_ctime_ms",
                type="integer",
                description="Legacy optimistic-lock ctime from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="file_size_bytes",
                type="integer",
                description="Legacy optimistic-lock size from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="expected_mtime_ms",
                type="integer",
                description="Preferred optimistic-lock mtime from Read.",
                required=False,
            ),
            ToolParameter(
                name="expected_ctime_ms",
                type="integer",
                description="Preferred optimistic-lock ctime from Read.",
                required=False,
            ),
            ToolParameter(
                name="expected_size_bytes",
                type="integer",
                description="Optional optimistic-lock file size from Read for stricter conflict detection.",
                required=False,
            ),
            ToolParameter(
                name="dry_run",
                type="boolean",
                description="If true, show the unified diff preview without writing anything to disk.",
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path")
        old_string = parameters.get("old_string")
        new_string = parameters.get("new_string")
        replace_all = bool(parameters.get("replace_all", False))
        dry_run = bool(parameters.get("dry_run", False))

        if not path or not isinstance(path, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path",
            )
        if old_string is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: old_string",
            )
        if new_string is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: new_string",
            )
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="old_string and new_string must both be strings",
            )
        if old_string == "":
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="old_string cannot be empty",
            )

        try:
            full_path = self._resolve_path(path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied",
            )

        try:
            if not full_path.exists():
                return self._missing_path_response(
                    path,
                    full_path,
                    detail_line="No edits were applied.",
                )
            if full_path.is_dir():
                return ToolResponse.error(
                    code=ToolErrorCode.IS_DIRECTORY,
                    message=f"Path '{path}' is a directory, cannot edit as a file",
                )
            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"File '{path}' is a binary file, cannot edit as text",
                )

            rel_path = self._display_path(full_path)
            with _path_lock(full_path):
                require_read_error = self._require_prior_read(rel_path)
                if require_read_error:
                    return require_read_error

                content, encoding = read_text_file(full_path)
                line_ending = detect_line_ending(content)
                current_state = _snapshot_file(full_path, encoding=encoding)
                expected_state = self._expected_state_from_parameters(parameters, rel_path)
                conflict = self._check_expected_state(
                    rel_path=rel_path,
                    current_state=current_state,
                    expected_state=expected_state,
                )
                if conflict:
                    return conflict

                normalized_old = normalize_line_endings(old_string, line_ending)
                normalized_new = normalize_line_endings(new_string, line_ending)
                try:
                    replace_result = replace_with_flexible_match(
                        content,
                        normalized_old,
                        normalized_new,
                        replace_all=replace_all,
                    )
                except EditNotFoundError as exc:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=str(exc),
                        context={"matches": 0},
                    )
                except EditAmbiguousError as exc:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=str(exc),
                    )
                except EditMatchError as exc:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=str(exc),
                    )

                replacements_applied = replace_result.replacements
                new_content = replace_result.content

                if new_content == content:
                    self._cache_state(rel_path, current_state)
                    return _no_change_response("Edit check complete", rel_path, current_state)

                diff_preview, diff_truncated = make_diff_preview(content, new_content, rel_path)
                backup_path = None
                formatter_result = FormatterResult(
                    attempted=False,
                    available=False,
                    success=False,
                    skipped_reason="Formatter was not run.",
                )
                diagnostics_result = DiagnosticsResult(
                    attempted=False,
                    available=False,
                    success=False,
                    skipped_reason="Diagnostics were not run.",
                )
                if not dry_run:
                    backup_path = self._backup_file(full_path)
                    atomic_write(full_path, new_content, encoding=encoding)
                    formatter_result = run_formatter(full_path, self.project_root)
                    diagnostics_result = run_diagnostics(full_path, self.project_root)
                    final_content, final_encoding = read_text_file(full_path)
                    encoding = final_encoding
                    new_content = final_content
                    diff_preview, diff_truncated = make_diff_preview(content, new_content, rel_path)
                    new_state = _snapshot_file(full_path, encoding=encoding)
                    self._cache_state(rel_path, new_state)
                else:
                    new_state = current_state

                changed_bytes = len(new_content.encode(encoding)) - len(content.encode(encoding))
                text_lines = [
                    (
                        f"{'Dry run: would edit' if dry_run else 'Successfully edited'} {rel_path} "
                        f"({replacements_applied} replacement{'s' if replacements_applied != 1 else ''}, "
                        f"{changed_bytes:+d} bytes changed, strategy={replace_result.strategy})"
                    ),
                    _metadata_text(new_state),
                ]
                if backup_path is not None:
                    text_lines.append(f"Backup: {self._display_path(backup_path)}")
                formatter_summary = _formatter_summary(formatter_result)
                if formatter_summary:
                    text_lines.append(formatter_summary)
                text_lines.extend(_diagnostics_summary(diagnostics_result))
                text_lines.append(_format_diff_section(diff_preview, diff_truncated))

                data = {
                    "path": rel_path,
                    "modified": not dry_run,
                    "dry_run": dry_run,
                    "replace_all": replace_all,
                    "num_replacements": replacements_applied,
                    "changed_bytes": changed_bytes,
                    "match_strategy": replace_result.strategy,
                    "diff_preview": diff_preview,
                    "diff_truncated": diff_truncated,
                    "backup_path": self._display_path(backup_path) if backup_path else None,
                    "encoding": encoding,
                    "formatter": formatter_result.to_dict(),
                    "diagnostics": diagnostics_result.to_dict(),
                    **_metadata_payload(new_state),
                }
                response_factory = (
                    ToolResponse.partial if dry_run or _post_process_failed(formatter_result, diagnostics_result) else ToolResponse.success
                )
                return response_factory(text="\n".join(text_lines), data=data)

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to edit '{path}'",
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to edit file: {exc}",
            )

    @staticmethod
    def _backup_file(full_path: Path) -> Path:
        backup_dir = full_path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_dir / f"{full_path.name}.{timestamp}.bak"
        shutil.copy2(full_path, backup_path)
        return backup_path
