"""File Operation Tools - Supporting Optimistic Locking

Provides standard file reading, writing, and editing capabilities:
- ReadTool: Read file + Metadata caching
- WriteTool: Write file + Conflict detection + Atomic write
- DeleteTool: Safe delete for files/directories + Atomic move to trash
- EditTool: Precise replacement + Conflict detection + Backup
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ..base import Tool, ToolParameter
from ..response import ToolResponse
from ..errors import ToolErrorCode
from ._code_utils import (
    atomic_write,
    detect_line_ending,
    format_numbered_lines,
    is_binary_file,
    make_diff_preview,
    normalize_line_endings,
    read_text_file,
)

if TYPE_CHECKING:
    from ..registry import ToolRegistry


def _display_path(project_root: Path, full_path: Path) -> str:
    """Returns a display path relative to the project_root."""
    try:
        rel = full_path.relative_to(project_root)
        text = str(rel).replace(os.sep, '/')
        return text or "."
    except ValueError:
        return str(full_path).replace(os.sep, '/')


def _format_diff_section(diff_preview: str, diff_truncated: bool) -> str:
    """Format unified diff preview text."""
    lines = ["", "Unified diff preview:", diff_preview]
    if diff_truncated:
        lines.append("[diff preview truncated]")
    return '\n'.join(lines)


def _require_prior_read(
    registry: Optional['ToolRegistry'],
    rel_path: str,
) -> Optional[ToolResponse]:
    """Return an error response if the file has not been Read yet, else None."""
    if registry is not None and rel_path not in registry.read_metadata_cache:
        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message=(
                f"You must Read '{rel_path}' before editing. "
                f"Call the Read tool first to see current content."
            ),
        )
    return None


def _no_change_response(
    action_text: str,
    rel_path: str,
    mtime_ms: int,
    size_bytes: int,
) -> ToolResponse:
    """Unified no-change response to facilitate agent reasoning."""
    return ToolResponse.partial(
        text=(
            f"{action_text}: {rel_path}\n"
            f"No actual textual changes.\n"
            f"Metadata: file_mtime_ms={mtime_ms}, file_size_bytes={size_bytes}, "
            f"expected_mtime_ms={mtime_ms}, expected_size_bytes={size_bytes}\n"
            f"Unified diff preview:\n[no textual diff]"
        ),
        data={
            "path": rel_path,
            "modified": False,
            "file_mtime_ms": mtime_ms,
            "file_size_bytes": size_bytes,
            "expected_mtime_ms": mtime_ms,
            "expected_size_bytes": size_bytes,
            "diff_preview": "[no textual diff]",
            "diff_truncated": False,
        },
    )


class ReadTool(Tool):
    """File Read Tool

    Features:
    - Read file content (supports offset/limit)
    - List directory contents (when path is a directory)
    - Automatically retrieve file metadata (mtime, size)
    - Cache metadata in ToolRegistry (for optimistic locking)
    - Cross-platform compatibility (Windows/Linux)

    Parameters:
    - path: File or directory path (relative to project_root)
    - offset: Starting line number (optional, default 0, valid for files only)
    - limit: Maximum number of lines (optional, default 2000, valid for files only)
    """
    
    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional['ToolRegistry'] = None
    ):
        super().__init__(
            name="Read",
            description=(
                "Read a text file or inspect a directory. Use this before editing so you can "
                "see the current content and capture optimistic-lock metadata for later Write "
                "or Edit calls."
            ),
            expandable=False
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = Path(working_dir).expanduser().resolve() if working_dir else self.project_root
        self.registry = registry

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description=(
                    "File or directory path relative to the project root. If the path is a "
                    "directory, the tool returns a directory listing instead of file content."
                ),
                required=True
            ),
            ToolParameter(
                name="offset",
                type="integer",
                description=(
                    "Zero-based starting line offset when reading a file. Use this to continue "
                    "reading large files in chunks."
                ),
                required=False,
                default=0
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description=(
                    "Maximum number of lines to return for a file read. Keep this focused so the "
                    "agent sees the most relevant slice."
                ),
                required=False,
                default=2000
            )
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute file read or directory listing"""
        path = parameters.get("path")
        offset = parameters.get("offset", 0)
        limit = parameters.get("limit", 2000)

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path"
            )

        if not isinstance(offset, int) or offset < 0:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Parameter offset must be an integer greater than or equal to 0"
            )

        if not isinstance(limit, int) or limit < 1:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Parameter limit must be an integer greater than or equal to 1"
            )

        try:
            # Resolve path
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Path '{path}' does not exist"
                )

            # If it's a directory, return directory listing
            if full_path.is_dir():
                return self._list_directory(_display_path(self.project_root, full_path), full_path)

            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"File '{path}' is a binary file and cannot be read as text"
                )

            # Read file (with encoding fallback)
            content, encoding = read_text_file(full_path)
            lines = content.splitlines()

            # Apply offset and limit
            total_lines = len(lines)
            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]

            selected_content = '\n'.join(lines)

            # Get file metadata (for optimistic locking)
            mtime = os.path.getmtime(full_path)
            size = os.path.getsize(full_path)
            file_mtime_ms = int(mtime * 1000)
            file_size_bytes = size
            rel_path = _display_path(self.project_root, full_path)

            # Cache metadata to ToolRegistry
            if self.registry:
                self.registry.cache_read_metadata(rel_path, {
                    "file_mtime_ms": file_mtime_ms,
                    "file_size_bytes": file_size_bytes
                })

            if lines:
                numbered = format_numbered_lines(selected_content, start_line=offset + 1)
            else:
                numbered = "[empty file]"

            text_lines = [
                f"File: {rel_path}",
                f"Encoding: {encoding}",
                f"Line range: {offset + 1}-{offset + len(lines)} / {total_lines}",
                (
                    f"Metadata: file_mtime_ms={file_mtime_ms}, "
                    f"file_size_bytes={file_size_bytes}, "
                    f"expected_mtime_ms={file_mtime_ms}, "
                    f"expected_size_bytes={file_size_bytes}"
                ),
                "",
                numbered,
            ]

            return ToolResponse.success(
                text='\n'.join(text_lines),
                data={
                    "path": rel_path,
                    "content": selected_content,
                    "lines": len(lines),
                    "total_lines": total_lines,
                    "file_mtime_ms": file_mtime_ms,
                    "file_size_bytes": file_size_bytes,
                    "expected_mtime_ms": file_mtime_ms,
                    "expected_size_bytes": file_size_bytes,
                    "offset": offset,
                    "limit": limit
                }
            )
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied"
            )
        
        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to read '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to read file: {str(e)}"
            )

    def _list_directory(self, path: str, full_path: Path) -> ToolResponse:
        """List directory contents (Windows and Linux compatible)"""
        try:
            entries = []
            total_files = 0
            total_dirs = 0

            # Get all entries in the directory
            for entry in sorted(full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    # Get entry information
                    is_dir = entry.is_dir()
                    name = entry.name

                    # Get size and modification time
                    if is_dir:
                        size_str = "<DIR>"
                        total_dirs += 1
                    else:
                        try:
                            size = entry.stat().st_size
                            size_str = self._format_size(size)
                            total_files += 1
                        except:
                            size_str = "?"

                    # Get modification time
                    try:
                        mtime = entry.stat().st_mtime
                        mtime_str = self._format_time(mtime)
                    except:
                        mtime_str = "?"

                    # Use forward slash as path separator (cross-platform compatibility)
                    relative_path = str(entry.relative_to(self.project_root)).replace(os.sep, '/')

                    entries.append({
                        "name": name,
                        "type": "directory" if is_dir else "file",
                        "size": size_str,
                        "mtime": mtime_str,
                        "path": relative_path
                    })
                except Exception:
                    # Skip inaccessible entries
                    continue

            # Construct output text
            if not entries:
                text = f"Directory '{path}' is empty"
            else:
                lines = [f"Directory '{path}' contains {total_files} files, {total_dirs} directories:\n"]
                for entry in entries:
                    type_icon = "📁" if entry["type"] == "directory" else "📄"
                    lines.append(f"{type_icon} {entry['name']:<40} {entry['size']:>10} {entry['mtime']}")
                text = "\n".join(lines)

            return ToolResponse.success(
                text=text,
                data={
                    "path": path,
                    "entries": entries,
                    "total_files": total_files,
                    "total_dirs": total_dirs,
                    "is_directory": True
                }
            )
        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"No permission to access directory '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to list directory: {str(e)}"
            )

    def _format_size(self, size: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

    def _format_time(self, timestamp: float) -> str:
        """Format timestamp (Windows and Linux compatible)"""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path (Windows and Linux compatible)"""
        # Unify path separators: convert backslashes to forward slashes
        path = path.replace('\\', '/')

        # If it's an absolute path, use it directly
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            # Otherwise, relative to working_dir
            full_path = (self.working_dir / path).resolve()

        full_path.relative_to(self.project_root)
        return full_path


class ListFilesTool(ReadTool):
    """Directory listing tool (compatibility with legacy LS tool naming)"""

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional['ToolRegistry'] = None
    ):
        super().__init__(project_root=project_root, working_dir=working_dir, registry=registry)
        self.name = "LS"
        self.description = (
            "List files and folders inside a directory. Prefer this over Read when you need to "
            "understand project structure, discover file names, or decide what to inspect next."
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Directory path relative to the project root. Defaults to the current workspace root.",
                required=False,
                default="."
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path", ".")
        try:
            full_path = self._resolve_path(path)
            if not full_path.exists():
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Path '{path}' does not exist"
                )
            if not full_path.is_dir():
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"Path '{path}' is not a directory"
                )
            return self._list_directory(_display_path(self.project_root, full_path), full_path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied"
            )


class WriteTool(Tool):
    """File Write Tool

    Features:
    - Create or overwrite a file
    - Optimistic locking conflict detection (if file exists)
    - Atomic write (temporary file + rename)
    - Automatically back up the original file

    Parameters:
    - path: File path
    - content: File content
    - file_mtime_ms: Cached mtime (optional, for conflict detection)
    """

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional['ToolRegistry'] = None
    ):
        super().__init__(
            name="Write",
            description=(
                "Create a new file or replace the full contents of an existing file. Use this for "
                "full-file rewrites, generated files, or when a change is too broad for Edit."
            ),
            expandable=False
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = Path(working_dir).expanduser().resolve() if working_dir else self.project_root
        self.registry = registry

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Target file path relative to the project root.",
                required=True
            ),
            ToolParameter(
                name="content",
                type="string",
                description="The complete new file content to write. This replaces the whole file.",
                required=True
            ),
            ToolParameter(
                name="file_mtime_ms",
                type="integer",
                description="Legacy optimistic-lock mtime from a previous Read call.",
                required=False
            ),
            ToolParameter(
                name="expected_mtime_ms",
                type="integer",
                description="Preferred optimistic-lock mtime from Read. Pass this when rewriting an existing file.",
                required=False
            ),
            ToolParameter(
                name="expected_size_bytes",
                type="integer",
                description="Optional optimistic-lock file size from Read for stricter conflict detection.",
                required=False
            ),
            ToolParameter(
                name="dry_run",
                type="boolean",
                description="If true, show the unified diff preview without writing anything to disk.",
                required=False,
                default=False
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute file write"""
        path = parameters.get("path")
        content = parameters.get("content")
        cached_mtime = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        cached_size = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))
        dry_run = bool(parameters.get("dry_run", False))

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path"
            )

        if content is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: content"
            )

        try:
            # Resolve path
            full_path = self._resolve_path(path)
            backup_path = None
            old_content = ""
            line_ending = "\n"

            # Check if file exists
            if full_path.exists():
                if full_path.is_dir():
                    return ToolResponse.error(
                        code=ToolErrorCode.IS_DIRECTORY,
                        message=f"Path '{path}' is a directory, cannot write as a file"
                    )
                if is_binary_file(full_path):
                    return ToolResponse.error(
                        code=ToolErrorCode.BINARY_FILE,
                        message=f"File '{path}' is a binary file, refusing to overwrite as text"
                    )

                # Enforce read-before-write (existing files only)
                rel_path_early = _display_path(self.project_root, full_path)
                err = _require_prior_read(self.registry, rel_path_early)
                if err:
                    return err

                # Get current file metadata
                current_mtime = os.path.getmtime(full_path)
                current_mtime_ms = int(current_mtime * 1000)
                current_size = os.path.getsize(full_path)
                old_content, _ = read_text_file(full_path)
                line_ending = detect_line_ending(old_content)

                # Check optimistic locking conflict
                if cached_mtime is not None:
                    if current_mtime_ms != cached_mtime:
                        return ToolResponse.error(
                            code=ToolErrorCode.CONFLICT,
                            message=f"File modified since last read. Current mtime={current_mtime_ms}, cached mtime={cached_mtime}",
                            context={
                                "current_mtime_ms": current_mtime_ms,
                                "cached_mtime_ms": cached_mtime
                            }
                        )
                if cached_size is not None and current_size != cached_size:
                    return ToolResponse.error(
                        code=ToolErrorCode.CONFLICT,
                        message=f"File size changed since last read. Current size={current_size}, cached size={cached_size}",
                        context={
                            "current_size_bytes": current_size,
                            "cached_size_bytes": cached_size
                        }
                    )

                # Back up original file
                if not dry_run:
                    backup_path = self._backup_file(full_path)
            else:
                # Ensure parent directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)

            rel_path = _display_path(self.project_root, full_path)
            normalized_content = normalize_line_endings(content, line_ending) if old_content else content

            if old_content == normalized_content:
                existing_mtime = int(os.getmtime(full_path) * 1000) if full_path.exists() else 0
                existing_size = os.path.getsize(full_path) if full_path.exists() else len(normalized_content.encode('utf-8'))
                return _no_change_response("Write check complete", rel_path, existing_mtime, existing_size)

            diff_preview, diff_truncated = make_diff_preview(old_content, normalized_content, rel_path)

            if not dry_run:
                # Atomic write
                atomic_write(full_path, normalized_content)

            size_bytes = len(normalized_content.encode('utf-8'))
            if dry_run:
                new_mtime_ms = cached_mtime
            else:
                new_mtime_ms = int(os.path.getmtime(full_path) * 1000)

            response_factory = ToolResponse.partial if dry_run else ToolResponse.success
            return response_factory(
                text=(
                    f"{'Dry run: Will write' if dry_run else 'Successfully wrote'} {rel_path} ({size_bytes} bytes)\n"
                    f"Metadata: file_mtime_ms={new_mtime_ms}, file_size_bytes={size_bytes}, "
                    f"expected_mtime_ms={new_mtime_ms}, expected_size_bytes={size_bytes}"
                    + _format_diff_section(diff_preview, diff_truncated)
                ),
                data={
                    "path": rel_path,
                    "written": True,
                    "dry_run": dry_run,
                    "size_bytes": size_bytes,
                    "file_mtime_ms": new_mtime_ms,
                    "file_size_bytes": size_bytes,
                    "expected_mtime_ms": new_mtime_ms,
                    "expected_size_bytes": size_bytes,
                    "diff_preview": diff_preview,
                    "diff_truncated": diff_truncated,
                    "backup_path": str(backup_path.relative_to(self.working_dir)) if backup_path else None
                }
            )

        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied"
            )

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to write '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to write file: {str(e)}"
            )

    def _backup_file(self, full_path: Path) -> Path:
        """Back up file"""
        backup_dir = full_path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{full_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_name

        shutil.copy2(full_path, backup_path)
        return backup_path

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path"""
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            full_path = (self.working_dir / path).resolve()
        full_path.relative_to(self.project_root)
        return full_path


class DeleteTool(Tool):
    """Safe file/directory deletion tool with atomic move-to-trash behavior."""

    PROTECTED_PARTS = {".git", ".hg", ".svn"}
    MAX_DIR_ENTRIES_WITHOUT_FORCE = 2000

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional['ToolRegistry'] = None,
    ):
        super().__init__(
            name="Delete",
            description=(
                "Delete a file or directory safely. Uses an atomic move to an internal trash area first "
                "to reduce accidental data loss risk. Supports dry-run, recursive directory deletion, and "
                "optional permanent purge."
            ),
            expandable=False,
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = Path(working_dir).expanduser().resolve() if working_dir else self.project_root
        self.registry = registry
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
                description=(
                    "If true, permanently delete from trash after atomic move. "
                    "Default false keeps recoverable trash copy."
                ),
                required=False,
                default=False,
            ),
            ToolParameter(
                name="expected_mtime_ms",
                type="integer",
                description="Optional optimistic-lock mtime from a previous Read call.",
                required=False,
            ),
            ToolParameter(
                name="expected_size_bytes",
                type="integer",
                description="Optional optimistic-lock file size from a previous Read call.",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path")
        recursive = bool(parameters.get("recursive", False))
        force = bool(parameters.get("force", False))
        dry_run = bool(parameters.get("dry_run", False))
        purge = bool(parameters.get("purge", False))
        expected_mtime_ms = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        expected_size_bytes = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))

        if not path or not isinstance(path, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path",
            )

        try:
            full_path = self._resolve_path(path)
            if not full_path.exists():
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Path '{path}' does not exist",
                )

            protected_error = self._validate_protected_path(full_path)
            if protected_error:
                return ToolResponse.error(
                    code=ToolErrorCode.ACCESS_DENIED,
                    message=protected_error,
                )

            rel_path = _display_path(self.project_root, full_path)

            if full_path.is_file():
                require_read_error = _require_prior_read(self.registry, rel_path)
                if require_read_error:
                    return require_read_error

            stat = full_path.stat()
            current_mtime_ms = int(stat.st_mtime * 1000)
            current_size_bytes = stat.st_size if full_path.is_file() else 0

            if expected_mtime_ms is not None and current_mtime_ms != expected_mtime_ms:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=(
                        f"File changed since last read. expected_mtime_ms={expected_mtime_ms}, "
                        f"actual_mtime_ms={current_mtime_ms}."
                    ),
                    context={
                        "path": rel_path,
                        "expected_mtime_ms": expected_mtime_ms,
                        "actual_mtime_ms": current_mtime_ms,
                    },
                )

            if full_path.is_file() and expected_size_bytes is not None and current_size_bytes != expected_size_bytes:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=(
                        f"File size changed since last read. expected_size_bytes={expected_size_bytes}, "
                        f"actual_size_bytes={current_size_bytes}."
                    ),
                    context={
                        "path": rel_path,
                        "expected_size_bytes": expected_size_bytes,
                        "actual_size_bytes": current_size_bytes,
                    },
                )

            entry_count = 0
            total_file_bytes = current_size_bytes
            is_dir = full_path.is_dir()

            if is_dir:
                entry_count, total_file_bytes = self._scan_directory_stats(full_path)
                if entry_count > 0 and not recursive:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=(
                            f"Directory '{path}' is not empty. Set recursive=true to delete it."
                        ),
                    )
                if entry_count > self.MAX_DIR_ENTRIES_WITHOUT_FORCE and not force:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=(
                            f"Directory '{path}' contains {entry_count} entries. "
                            f"Set force=true to confirm large recursive deletion."
                        ),
                    )

            if dry_run:
                return ToolResponse.partial(
                    text=(
                        f"Dry run: would delete {rel_path}\n"
                        f"Type: {'directory' if is_dir else 'file'}\n"
                        f"Recursive: {recursive}\n"
                        f"Estimated entries: {entry_count if is_dir else 1}\n"
                        f"Estimated bytes: {total_file_bytes}"
                    ),
                    data={
                        "path": rel_path,
                        "deleted": False,
                        "dry_run": True,
                        "is_directory": is_dir,
                        "recursive": recursive,
                        "entry_count": entry_count if is_dir else 1,
                        "estimated_bytes": total_file_bytes,
                    },
                )

            trash_path = self._make_trash_path(full_path)
            self.trash_root.mkdir(parents=True, exist_ok=True)

            # Atomic move (same filesystem under project root)
            os.replace(str(full_path), str(trash_path))

            purged = False
            if purge:
                if trash_path.is_dir():
                    shutil.rmtree(trash_path)
                else:
                    trash_path.unlink(missing_ok=True)
                purged = True

            return ToolResponse.success(
                text=(
                    f"Deleted {rel_path} via atomic trash move\n"
                    f"Type: {'directory' if is_dir else 'file'}\n"
                    f"Recursive: {recursive}\n"
                    f"Purged: {purged}\n"
                    f"Trash path: { _display_path(self.project_root, trash_path) }"
                ),
                data={
                    "path": rel_path,
                    "deleted": True,
                    "is_directory": is_dir,
                    "recursive": recursive,
                    "purged": purged,
                    "entry_count": entry_count if is_dir else 1,
                    "estimated_bytes": total_file_bytes,
                    "trash_path": _display_path(self.project_root, trash_path),
                },
            )

        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied",
            )
        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to delete '{path}'",
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to delete path: {str(e)}",
            )

    def _resolve_path(self, path: str) -> Path:
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            full_path = (self.working_dir / path).resolve()
        full_path.relative_to(self.project_root)
        return full_path

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

    def _scan_directory_stats(self, directory: Path) -> tuple[int, int]:
        entry_count = 0
        total_file_bytes = 0
        for root, dirnames, filenames in os.walk(directory):
            entry_count += len(dirnames) + len(filenames)
            for filename in filenames:
                file_path = Path(root) / filename
                try:
                    total_file_bytes += file_path.stat().st_size
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


class EditTool(Tool):
    """File Edit Tool

    Features:
    - Precise replacement of file content (single-match by default, optional replace-all mode)
    - Optimistic locking conflict detection
    - Automatically back up the original file

    Parameters:
    - path: File path
    - old_string: Content to be replaced
    - new_string: Replacement content
    - replace_all: Whether to replace all matches (optional, defaults to False)
    - file_mtime_ms: Cached mtime (optional)
    """

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional['ToolRegistry'] = None
    ):
        super().__init__(
            name="Edit",
            description=(
                "Apply precise text replacement(s) inside an existing text file. By default it performs "
                "one unique match replacement; set `replace_all=true` to replace every occurrence."
            ),
            expandable=False
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.working_dir = Path(working_dir).expanduser().resolve() if working_dir else self.project_root
        self.registry = registry

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Existing file path relative to the project root.",
                required=True
            ),
            ToolParameter(
                name="old_string",
                type="string",
                description=(
                    "Exact existing text to replace. It must match exactly one location in the file. "
                    "Include enough surrounding context to make the match unique."
                ),
                required=True
            ),
            ToolParameter(
                name="new_string",
                type="string",
                description="Replacement text for the matched `old_string`.",
                required=True
            ),
            ToolParameter(
                name="replace_all",
                type="boolean",
                description=(
                    "If true, replace all occurrences of `old_string`. If false (default), "
                    "`old_string` must match exactly one location."
                ),
                required=False,
                default=False
            ),
            ToolParameter(
                name="file_mtime_ms",
                type="integer",
                description="Legacy optimistic-lock mtime from a previous Read call.",
                required=False
            ),
            ToolParameter(
                name="expected_mtime_ms",
                type="integer",
                description="Preferred optimistic-lock mtime from Read.",
                required=False
            ),
            ToolParameter(
                name="expected_size_bytes",
                type="integer",
                description="Optional optimistic-lock file size from Read for stricter conflict detection.",
                required=False
            ),
            ToolParameter(
                name="dry_run",
                type="boolean",
                description="If true, show the unified diff preview without writing anything to disk.",
                required=False,
                default=False
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute file edit"""
        path = parameters.get("path")
        old_string = parameters.get("old_string")
        new_string = parameters.get("new_string")
        replace_all = bool(parameters.get("replace_all", False))
        cached_mtime = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        cached_size = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))
        dry_run = bool(parameters.get("dry_run", False))

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path"
            )

        if old_string is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: old_string"
            )

        if new_string is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: new_string"
            )
        if old_string == "":
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="old_string cannot be empty"
            )

        try:
            # Resolve path
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"File '{path}' does not exist"
                )
            if full_path.is_dir():
                return ToolResponse.error(
                    code=ToolErrorCode.IS_DIRECTORY,
                    message=f"Path '{path}' is a directory, cannot edit as a file"
                )
            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"File '{path}' is a binary file, cannot edit as text"
                )

            # Enforce read-before-edit
            rel_path = _display_path(self.project_root, full_path)
            err = _require_prior_read(self.registry, rel_path)
            if err:
                return err

            # Get current file metadata
            current_mtime = os.path.getmtime(full_path)
            current_mtime_ms = int(current_mtime * 1000)
            current_size = os.path.getsize(full_path)

            # Check optimistic locking conflict
            if cached_mtime is not None and current_mtime_ms != cached_mtime:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"File modified since last read. Current mtime={current_mtime_ms}, cached mtime={cached_mtime}",
                    context={
                        "current_mtime_ms": current_mtime_ms,
                        "cached_mtime_ms": cached_mtime
                    }
                )
            if cached_size is not None and current_size != cached_size:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"File size changed since last read. Current size={current_size}, cached size={cached_size}",
                    context={
                        "current_size_bytes": current_size,
                        "cached_size_bytes": cached_size
                    }
                )

            # Read file content
            content, _ = read_text_file(full_path)
            line_ending = detect_line_ending(content)

            # Validate old_string match count
            matches = content.count(old_string)
            if matches == 0:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message="old_string not found in file content.",
                    context={"matches": matches}
                )
            if not replace_all and matches != 1:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"old_string must match file content uniquely. Found {matches} matches.",
                    context={"matches": matches}
                )

            # Execute replacement
            replacements_applied = matches if replace_all else 1
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)
            new_content = normalize_line_endings(new_content, line_ending)

            if new_content == content:
                return _no_change_response(
                    "Edit check complete",
                    rel_path,
                    current_mtime_ms,
                    current_size,
                )

            diff_preview, diff_truncated = make_diff_preview(content, new_content, rel_path)

            # Back up original file
            backup_path = self._backup_file(full_path) if not dry_run else None

            # Write new content
            if not dry_run:
                atomic_write(full_path, new_content)

            changed_bytes = (
                len(new_string.encode('utf-8')) - len(old_string.encode('utf-8'))
            ) * replacements_applied
            new_mtime_ms = cached_mtime if dry_run else int(os.path.getmtime(full_path) * 1000)
            new_size = len(new_content.encode('utf-8')) if dry_run else os.path.getsize(full_path)

            response_factory = ToolResponse.partial if dry_run else ToolResponse.success
            return response_factory(
                text=(
                    f"{'Dry run: Will edit' if dry_run else 'Successfully edited'} {rel_path} "
                    f"({replacements_applied} replacement{'s' if replacements_applied != 1 else ''}, "
                    f"{changed_bytes:+d} bytes changed)\n"
                    f"Metadata: file_mtime_ms={new_mtime_ms}, file_size_bytes={new_size}, "
                    f"expected_mtime_ms={new_mtime_ms}, expected_size_bytes={new_size}"
                    + _format_diff_section(diff_preview, diff_truncated)
                ),
                data={
                    "path": rel_path,
                    "modified": True,
                    "dry_run": dry_run,
                    "replace_all": replace_all,
                    "num_replacements": replacements_applied,
                    "changed_bytes": changed_bytes,
                    "file_mtime_ms": new_mtime_ms,
                    "file_size_bytes": new_size,
                    "expected_mtime_ms": new_mtime_ms,
                    "expected_size_bytes": new_size,
                    "diff_preview": diff_preview,
                    "diff_truncated": diff_truncated,
                    "backup_path": str(backup_path.relative_to(self.working_dir))
                    if backup_path else None,
                }
            )

        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{path}' is outside the project root, access denied"
            )

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"No permission to edit '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to edit file: {str(e)}"
            )

    def _backup_file(self, full_path: Path) -> Path:
        """Back up file"""
        backup_dir = full_path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{full_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_name

        shutil.copy2(full_path, backup_path)
        return backup_path

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path"""
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            full_path = (self.working_dir / path).resolve()
        full_path.relative_to(self.project_root)
        return full_path
