"""File Operation Tools - Supporting Optimistic Locking

Provides standard file reading, writing, and editing capabilities:
- ReadTool: Read file + Metadata caching
- WriteTool: Write file + Conflict detection + Atomic write
- EditTool: Precise replacement + Conflict detection + Backup
- MultiEditTool: Batch replacement + Atomicity guarantee
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
                "see the current content and capture optimistic-lock metadata for later Write, "
                "Edit, or MultiEdit calls."
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
                "full-file rewrites, generated files, or when a change is too broad for Edit or MultiEdit."
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


class EditTool(Tool):
    """File Edit Tool

    Features:
    - Precise replacement of file content (old_string must match uniquely)
    - Optimistic locking conflict detection
    - Automatically back up the original file

    Parameters:
    - path: File path
    - old_string: Content to be replaced
    - new_string: Replacement content
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
                "Apply one precise text replacement inside an existing text file. Prefer this for a "
                "single surgical change where `old_string` identifies exactly one location."
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

            # Check if old_string matches uniquely
            matches = content.count(old_string)
            if matches != 1:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"old_string must match file content uniquely. Found {matches} matches.",
                    context={"matches": matches}
                )

            # Execute replacement
            new_content = content.replace(old_string, new_string)
            new_content = normalize_line_endings(new_content, line_ending)
            rel_path = _display_path(self.project_root, full_path)

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

            changed_bytes = len(new_string.encode('utf-8')) - len(old_string.encode('utf-8'))
            new_mtime_ms = cached_mtime if dry_run else int(os.path.getmtime(full_path) * 1000)
            new_size = len(new_content.encode('utf-8')) if dry_run else os.path.getsize(full_path)

            response_factory = ToolResponse.partial if dry_run else ToolResponse.success
            return response_factory(
                text=(
                    f"{'Dry run: Will edit' if dry_run else 'Successfully edited'} {rel_path} ({changed_bytes:+d} bytes changed)\n"
                    f"Metadata: file_mtime_ms={new_mtime_ms}, file_size_bytes={new_size}, "
                    f"expected_mtime_ms={new_mtime_ms}, expected_size_bytes={new_size}"
                    + _format_diff_section(diff_preview, diff_truncated)
                ),
                data={
                    "path": rel_path,
                    "modified": True,
                    "dry_run": dry_run,
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


class MultiEditTool(Tool):
    """Batch Edit Tool

    Features:
    - Execute multiple replacement operations in batch
    - Atomicity guarantee (all succeed or all fail)
    - Optimistic locking conflict detection (checked once before all replacements)

    Parameters:
    - path: File path
    - edits: List of replacement objects [{"old_string": "...", "new_string": "..."}]
    - file_mtime_ms: Cached mtime (optional)
    """

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional['ToolRegistry'] = None
    ):
        super().__init__(
            name="MultiEdit",
            description=(
                "Apply multiple independent edits to one file atomically. Use this when several "
                "separate replacements should all match against the original file and be written together."
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
                name="edits",
                type="array",
                description=(
                    "List of edit objects, each with `old_string` and `new_string`. Every `old_string` "
                    "is matched against the original file content, not against intermediate edited states."
                ),
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
        """Execute batch edit"""
        path = parameters.get("path")
        edits = parameters.get("edits")
        cached_mtime = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        cached_size = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))
        dry_run = bool(parameters.get("dry_run", False))

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: path"
            )

        if not edits or not isinstance(edits, list):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Missing required parameter: edits (must be a list)"
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
                    message=f"Path '{path}' is a directory, cannot batch edit as a file"
                )
            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"File '{path}' is a binary file, cannot batch edit as text"
                )

            # Get current file metadata
            current_mtime = os.path.getmtime(full_path)
            current_mtime_ms = int(current_mtime * 1000)
            current_size = os.path.getsize(full_path)

            # Check optimistic locking conflict (checked once before all replacements)
            if cached_mtime is not None and current_mtime_ms != cached_mtime:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"File modified since last read. All replacements cancelled. Current mtime={current_mtime_ms}, cached mtime={cached_mtime}",
                    context={
                        "current_mtime_ms": current_mtime_ms,
                        "cached_mtime_ms": cached_mtime
                    }
                )
            if cached_size is not None and current_size != cached_size:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"File size changed since last read. All replacements cancelled. Current size={current_size}, cached size={cached_size}",
                    context={
                        "current_size_bytes": current_size,
                        "cached_size_bytes": cached_size
                    }
                )

            # Read file content
            original_content, _ = read_text_file(full_path)
            line_ending = detect_line_ending(original_content)
            replacements = []

            # All matches are positioned based on original content to avoid previous replacements affecting subsequent anchors
            for i, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Edit item {i} must be an object containing old_string and new_string"
                    )

                old_string = edit.get("old_string")
                new_string = edit.get("new_string")

                if old_string is None or new_string is None:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Edit item {i} is missing old_string or new_string"
                    )
                if old_string == "":
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"old_string of edit item {i} cannot be empty"
                    )

                matches = original_content.count(old_string)
                if matches != 1:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Edit item {i}: old_string must uniquely match original file content. Found {matches} matches.",
                        context={"edit_index": i, "matches": matches}
                    )

                start = original_content.find(old_string)
                end = start + len(old_string)
                replacements.append({
                    "edit_index": i,
                    "start": start,
                    "end": end,
                    "old_string": old_string,
                    "new_string": new_string,
                })

            # Check for replacement area conflicts (relative to original file areas)
            ordered = sorted(replacements, key=lambda item: item["start"])
            for prev, curr in zip(ordered, ordered[1:]):
                if curr["start"] < prev["end"]:
                    return ToolResponse.error(
                        code=ToolErrorCode.CONFLICT,
                        message=(
                            f"The replacement area for edit item {prev['edit_index']} overlaps with edit item {curr['edit_index']}; "
                            "cannot apply batch atomically."
                        ),
                        context={
                            "previous_edit_index": prev["edit_index"],
                            "current_edit_index": curr["edit_index"],
                        }
                    )

            new_content = original_content
            for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
                new_content = (
                    new_content[: replacement["start"]]
                    + replacement["new_string"]
                    + new_content[replacement["end"] :]
                )

            new_content = normalize_line_endings(new_content, line_ending)

            rel_path = _display_path(self.project_root, full_path)

            if new_content == original_content:
                return _no_change_response(
                    "Batch edit check complete",
                    rel_path,
                    current_mtime_ms,
                    current_size,
                )

            diff_preview, diff_truncated = make_diff_preview(original_content, new_content, rel_path)

            # Back up original file
            backup_path = self._backup_file(full_path) if not dry_run else None

            # Atomic write
            if not dry_run:
                atomic_write(full_path, new_content)

            changed_bytes = len(new_content.encode('utf-8')) - len(original_content.encode('utf-8'))
            new_mtime_ms = cached_mtime if dry_run else int(os.path.getmtime(full_path) * 1000)
            new_size = len(new_content.encode('utf-8')) if dry_run else os.path.getsize(full_path)

            response_factory = ToolResponse.partial if dry_run else ToolResponse.success
            return response_factory(
                text=(
                    f"{'Dry run: Will execute' if dry_run else 'Successfully executed'} {len(edits)} independent replacement operations on {rel_path} ({changed_bytes:+d} bytes changed)\n"
                    f"Metadata: file_mtime_ms={new_mtime_ms}, file_size_bytes={new_size}, "
                    f"expected_mtime_ms={new_mtime_ms}, expected_size_bytes={new_size}"
                    + _format_diff_section(diff_preview, diff_truncated)
                ),
                data={
                    "path": rel_path,
                    "modified": True,
                    "num_edits": len(edits),
                    "dry_run": dry_run,
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
                message=f"Failed to batch edit: {str(e)}"
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


class EditFileMultiTool(MultiEditTool):
    """Compatibility alias: EditFileMulti -> MultiEdit."""

    def __init__(
        self,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        registry: Optional['ToolRegistry'] = None
    ):
        super().__init__(project_root=project_root, working_dir=working_dir, registry=registry)
        self.name = "EditFileMulti"
        self.description = (
            "Compatibility alias for MultiEdit. Apply multiple independent edits to one file and "
            "write them atomically in a single step."
        )
