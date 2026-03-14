"""文件操作工具 - 支持乐观锁机制

提供标准的文件读写编辑能力：
- ReadTool: 读取文件 + 元数据缓存
- WriteTool: 写入文件 + 冲突检测 + 原子写入
- EditTool: 精确替换 + 冲突检测 + 备份
- MultiEditTool: 批量替换 + 原子性保证

使用示例：
```python
from hello_agents import ToolRegistry
from hello_agents.tools.builtin import ReadTool, WriteTool, EditTool

registry = ToolRegistry()
registry.register_tool(ReadTool(project_root="./"))
registry.register_tool(WriteTool(project_root="./"))
registry.register_tool(EditTool(project_root="./"))
```
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import os
import shutil
from datetime import datetime

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
    """返回相对 project_root 的显示路径"""
    try:
        rel = full_path.relative_to(project_root)
        text = str(rel).replace(os.sep, '/')
        return text or "."
    except ValueError:
        return str(full_path).replace(os.sep, '/')


def _format_diff_section(diff_preview: str, diff_truncated: bool) -> str:
    """格式化统一 diff 预览文本。"""
    lines = ["", "统一 diff 预览:", diff_preview]
    if diff_truncated:
        lines.append("[diff 预览已截断]")
    return '\n'.join(lines)


def _no_change_response(
    action_text: str,
    rel_path: str,
    mtime_ms: int,
    size_bytes: int,
) -> ToolResponse:
    """统一的无变化响应，便于 agent 继续推理。"""
    return ToolResponse.partial(
        text=(
            f"{action_text}: {rel_path}\n"
            f"无实际文本变化。\n"
            f"元数据: file_mtime_ms={mtime_ms}, file_size_bytes={size_bytes}, "
            f"expected_mtime_ms={mtime_ms}, expected_size_bytes={size_bytes}\n"
            f"统一 diff 预览:\n[no textual diff]"
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
    """文件读取工具

    功能：
    - 读取文件内容（支持 offset/limit）
    - 列出目录内容（当 path 是目录时）
    - 自动获取文件元数据（mtime, size）
    - 缓存元数据到 ToolRegistry（用于乐观锁）
    - 跨平台兼容（Windows/Linux）

    参数：
    - path: 文件或目录路径（相对于 project_root）
    - offset: 起始行号（可选，默认 0，仅文件有效）
    - limit: 最大行数（可选，默认 2000，仅文件有效）
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
        """执行文件读取或目录列表"""
        path = parameters.get("path")
        offset = parameters.get("offset", 0)
        limit = parameters.get("limit", 2000)

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: path"
            )

        if not isinstance(offset, int) or offset < 0:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="参数 offset 必须是大于等于 0 的整数"
            )

        if not isinstance(limit, int) or limit < 1:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="参数 limit 必须是大于等于 1 的整数"
            )

        try:
            # 解析路径
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"路径 '{path}' 不存在"
                )

            # 如果是目录，返回目录列表
            if full_path.is_dir():
                return self._list_directory(_display_path(self.project_root, full_path), full_path)

            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"文件 '{path}' 是二进制文件，无法按文本读取"
                )

            # 读取文件（带编码回退）
            content, encoding = read_text_file(full_path)
            lines = content.splitlines()

            # 应用 offset 和 limit
            total_lines = len(lines)
            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]

            selected_content = '\n'.join(lines)

            # 获取文件元数据（用于乐观锁）
            mtime = os.path.getmtime(full_path)
            size = os.path.getsize(full_path)
            file_mtime_ms = int(mtime * 1000)
            file_size_bytes = size
            rel_path = _display_path(self.project_root, full_path)

            # 缓存元数据到 ToolRegistry
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
                f"文件: {rel_path}",
                f"编码: {encoding}",
                f"行范围: {offset + 1}-{offset + len(lines)} / {total_lines}",
                (
                    f"元数据: file_mtime_ms={file_mtime_ms}, "
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
                message=f"路径 '{path}' 超出项目根目录，访问被拒绝"
            )
        
        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"无权限读取 '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"读取文件失败：{str(e)}"
            )

    def _list_directory(self, path: str, full_path: Path) -> ToolResponse:
        """列出目录内容（兼容 Windows 和 Linux）"""
        try:
            entries = []
            total_files = 0
            total_dirs = 0

            # 获取目录下所有条目
            for entry in sorted(full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    # 获取条目信息
                    is_dir = entry.is_dir()
                    name = entry.name

                    # 获取大小和修改时间
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

                    # 获取修改时间
                    try:
                        mtime = entry.stat().st_mtime
                        mtime_str = self._format_time(mtime)
                    except:
                        mtime_str = "?"

                    # 使用正斜杠作为路径分隔符（跨平台兼容）
                    relative_path = str(entry.relative_to(self.project_root)).replace(os.sep, '/')

                    entries.append({
                        "name": name,
                        "type": "directory" if is_dir else "file",
                        "size": size_str,
                        "mtime": mtime_str,
                        "path": relative_path
                    })
                except Exception as e:
                    # 跳过无法访问的条目
                    continue

            # 构建输出文本
            if not entries:
                text = f"目录 '{path}' 为空"
            else:
                lines = [f"目录 '{path}' 包含 {total_files} 个文件，{total_dirs} 个目录：\n"]
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
                message=f"无权访问目录 '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"列出目录失败：{str(e)}"
            )

    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

    def _format_time(self, timestamp: float) -> str:
        """格式化时间戳（兼容 Windows 和 Linux）"""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _resolve_path(self, path: str) -> Path:
        """解析相对路径（兼容 Windows 和 Linux）"""
        # 统一路径分隔符：将反斜杠转换为正斜杠
        path = path.replace('\\', '/')

        # 如果是绝对路径，直接使用
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            # 否则相对于 working_dir
            full_path = (self.working_dir / path).resolve()

        full_path.relative_to(self.project_root)
        return full_path


class ListFilesTool(ReadTool):
    """目录列表工具（兼容旧的 LS 工具命名）"""

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
                    message=f"路径 '{path}' 不存在"
                )
            if not full_path.is_dir():
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"路径 '{path}' 不是目录"
                )
            return self._list_directory(_display_path(self.project_root, full_path), full_path)
        except ValueError:
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"路径 '{path}' 超出项目根目录，访问被拒绝"
            )


class WriteTool(Tool):
    """文件写入工具

    功能：
    - 创建或覆盖文件
    - 乐观锁冲突检测（如果文件已存在）
    - 原子写入（临时文件 + rename）
    - 自动备份原文件

    参数：
    - path: 文件路径
    - content: 文件内容
    - file_mtime_ms: 缓存的 mtime（可选，用于冲突检测）
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
        """执行文件写入"""
        path = parameters.get("path")
        content = parameters.get("content")
        cached_mtime = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        cached_size = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))
        dry_run = bool(parameters.get("dry_run", False))

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: path"
            )

        if content is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: content"
            )

        try:
            # 解析路径
            full_path = self._resolve_path(path)
            backup_path = None
            old_content = ""
            line_ending = "\n"

            # 检查文件是否存在
            if full_path.exists():
                if full_path.is_dir():
                    return ToolResponse.error(
                        code=ToolErrorCode.IS_DIRECTORY,
                        message=f"路径 '{path}' 是目录，不能按文件写入"
                    )
                if is_binary_file(full_path):
                    return ToolResponse.error(
                        code=ToolErrorCode.BINARY_FILE,
                        message=f"文件 '{path}' 是二进制文件，拒绝按文本覆盖"
                    )

                # 获取当前文件元数据
                current_mtime = os.path.getmtime(full_path)
                current_mtime_ms = int(current_mtime * 1000)
                current_size = os.path.getsize(full_path)
                old_content, _ = read_text_file(full_path)
                line_ending = detect_line_ending(old_content)

                # 检查乐观锁冲突
                if cached_mtime is not None:
                    if current_mtime_ms != cached_mtime:
                        return ToolResponse.error(
                            code=ToolErrorCode.CONFLICT,
                            message=f"文件自上次读取后被修改。当前 mtime={current_mtime_ms}, 缓存 mtime={cached_mtime}",
                            context={
                                "current_mtime_ms": current_mtime_ms,
                                "cached_mtime_ms": cached_mtime
                            }
                        )
                if cached_size is not None and current_size != cached_size:
                    return ToolResponse.error(
                        code=ToolErrorCode.CONFLICT,
                        message=f"文件自上次读取后大小已变化。当前 size={current_size}, 缓存 size={cached_size}",
                        context={
                            "current_size_bytes": current_size,
                            "cached_size_bytes": cached_size
                        }
                    )

                # 备份原文件
                if not dry_run:
                    backup_path = self._backup_file(full_path)
            else:
                # 确保父目录存在
                full_path.parent.mkdir(parents=True, exist_ok=True)

            rel_path = _display_path(self.project_root, full_path)
            normalized_content = normalize_line_endings(content, line_ending) if old_content else content

            if old_content == normalized_content:
                existing_mtime = int(os.path.getmtime(full_path) * 1000) if full_path.exists() else 0
                existing_size = os.path.getsize(full_path) if full_path.exists() else len(normalized_content.encode('utf-8'))
                return _no_change_response("写入检查完成", rel_path, existing_mtime, existing_size)

            diff_preview, diff_truncated = make_diff_preview(old_content, normalized_content, rel_path)

            if not dry_run:
                # 原子写入
                atomic_write(full_path, normalized_content)

            size_bytes = len(normalized_content.encode('utf-8'))
            if dry_run:
                new_mtime_ms = cached_mtime
            else:
                new_mtime_ms = int(os.path.getmtime(full_path) * 1000)

            response_factory = ToolResponse.partial if dry_run else ToolResponse.success
            return response_factory(
                text=(
                    f"{'Dry run：将写入' if dry_run else '成功写入'} {rel_path} ({size_bytes} 字节)\n"
                    f"元数据: file_mtime_ms={new_mtime_ms}, file_size_bytes={size_bytes}, "
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
                message=f"路径 '{path}' 超出项目根目录，访问被拒绝"
            )

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"无权限写入 '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"写入文件失败：{str(e)}"
            )

    def _backup_file(self, full_path: Path) -> Path:
        """备份文件"""
        backup_dir = full_path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{full_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_name

        shutil.copy2(full_path, backup_path)
        return backup_path

    def _resolve_path(self, path: str) -> Path:
        """解析相对路径"""
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            full_path = (self.working_dir / path).resolve()
        full_path.relative_to(self.project_root)
        return full_path


class EditTool(Tool):
    """文件编辑工具

    功能：
    - 精确替换文件内容（old_string 必须唯一匹配）
    - 乐观锁冲突检测
    - 自动备份原文件

    参数：
    - path: 文件路径
    - old_string: 要替换的内容
    - new_string: 替换后的内容
    - file_mtime_ms: 缓存的 mtime（可选）
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
        """执行文件编辑"""
        path = parameters.get("path")
        old_string = parameters.get("old_string")
        new_string = parameters.get("new_string")
        cached_mtime = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        cached_size = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))
        dry_run = bool(parameters.get("dry_run", False))

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: path"
            )

        if old_string is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: old_string"
            )

        if new_string is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: new_string"
            )

        try:
            # 解析路径
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"文件 '{path}' 不存在"
                )
            if full_path.is_dir():
                return ToolResponse.error(
                    code=ToolErrorCode.IS_DIRECTORY,
                    message=f"路径 '{path}' 是目录，不能按文件编辑"
                )
            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"文件 '{path}' 是二进制文件，无法按文本编辑"
                )

            # 获取当前文件元数据
            current_mtime = os.path.getmtime(full_path)
            current_mtime_ms = int(current_mtime * 1000)
            current_size = os.path.getsize(full_path)

            # 检查乐观锁冲突
            if cached_mtime is not None and current_mtime_ms != cached_mtime:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"文件自上次读取后被修改。当前 mtime={current_mtime_ms}, 缓存 mtime={cached_mtime}",
                    context={
                        "current_mtime_ms": current_mtime_ms,
                        "cached_mtime_ms": cached_mtime
                    }
                )
            if cached_size is not None and current_size != cached_size:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"文件自上次读取后大小已变化。当前 size={current_size}, 缓存 size={cached_size}",
                    context={
                        "current_size_bytes": current_size,
                        "cached_size_bytes": cached_size
                    }
                )

            # 读取文件内容
            content, _ = read_text_file(full_path)
            line_ending = detect_line_ending(content)

            # 检查 old_string 是否唯一匹配
            matches = content.count(old_string)
            if matches != 1:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"old_string 必须唯一匹配文件内容。找到 {matches} 处匹配。",
                    context={"matches": matches}
                )

            # 执行替换
            new_content = content.replace(old_string, new_string)
            new_content = normalize_line_endings(new_content, line_ending)
            rel_path = _display_path(self.project_root, full_path)

            if new_content == content:
                return _no_change_response(
                    "编辑检查完成",
                    rel_path,
                    current_mtime_ms,
                    current_size,
                )

            diff_preview, diff_truncated = make_diff_preview(content, new_content, rel_path)

            # 备份原文件
            backup_path = self._backup_file(full_path) if not dry_run else None

            # 写入新内容
            if not dry_run:
                atomic_write(full_path, new_content)

            changed_bytes = len(new_string.encode('utf-8')) - len(old_string.encode('utf-8'))
            new_mtime_ms = cached_mtime if dry_run else int(os.path.getmtime(full_path) * 1000)
            new_size = len(new_content.encode('utf-8')) if dry_run else os.path.getsize(full_path)

            response_factory = ToolResponse.partial if dry_run else ToolResponse.success
            return response_factory(
                text=(
                    f"{'Dry run：将编辑' if dry_run else '成功编辑'} {rel_path} (变化 {changed_bytes:+d} 字节)\n"
                    f"元数据: file_mtime_ms={new_mtime_ms}, file_size_bytes={new_size}, "
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
                message=f"路径 '{path}' 超出项目根目录，访问被拒绝"
            )

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"无权限编辑 '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"编辑文件失败：{str(e)}"
            )

    def _backup_file(self, full_path: Path) -> Path:
        """备份文件"""
        backup_dir = full_path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{full_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_name

        shutil.copy2(full_path, backup_path)
        return backup_path

    def _resolve_path(self, path: str) -> Path:
        """解析相对路径"""
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            full_path = (self.working_dir / path).resolve()
        full_path.relative_to(self.project_root)
        return full_path


class MultiEditTool(Tool):
    """批量编辑工具

    功能：
    - 批量执行多个替换操作
    - 原子性保证（要么全部成功，要么全部失败）
    - 乐观锁冲突检测（所有替换前检查一次）

    参数：
    - path: 文件路径
    - edits: 替换列表 [{"old_string": "...", "new_string": "..."}]
    - file_mtime_ms: 缓存的 mtime（可选）
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
        """执行批量编辑"""
        path = parameters.get("path")
        edits = parameters.get("edits")
        cached_mtime = parameters.get("expected_mtime_ms", parameters.get("file_mtime_ms"))
        cached_size = parameters.get("expected_size_bytes", parameters.get("file_size_bytes"))
        dry_run = bool(parameters.get("dry_run", False))

        if not path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: path"
            )

        if not edits or not isinstance(edits, list):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少必需参数: edits（必须是列表）"
            )

        try:
            # 解析路径
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"文件 '{path}' 不存在"
                )
            if full_path.is_dir():
                return ToolResponse.error(
                    code=ToolErrorCode.IS_DIRECTORY,
                    message=f"路径 '{path}' 是目录，不能按文件批量编辑"
                )
            if is_binary_file(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.BINARY_FILE,
                    message=f"文件 '{path}' 是二进制文件，无法按文本批量编辑"
                )

            # 获取当前文件元数据
            current_mtime = os.path.getmtime(full_path)
            current_mtime_ms = int(current_mtime * 1000)
            current_size = os.path.getsize(full_path)

            # 检查乐观锁冲突（所有替换前检查一次）
            if cached_mtime is not None and current_mtime_ms != cached_mtime:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"文件自上次读取后被修改。所有替换已取消。当前 mtime={current_mtime_ms}, 缓存 mtime={cached_mtime}",
                    context={
                        "current_mtime_ms": current_mtime_ms,
                        "cached_mtime_ms": cached_mtime
                    }
                )
            if cached_size is not None and current_size != cached_size:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=f"文件自上次读取后大小已变化。所有替换已取消。当前 size={current_size}, 缓存 size={cached_size}",
                    context={
                        "current_size_bytes": current_size,
                        "cached_size_bytes": cached_size
                    }
                )

            # 读取文件内容
            original_content, _ = read_text_file(full_path)
            line_ending = detect_line_ending(original_content)
            replacements = []

            # 所有匹配都基于原始内容定位，避免前一次替换影响后一次锚点
            for i, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"编辑项 {i} 必须是对象，包含 old_string 和 new_string"
                    )

                old_string = edit.get("old_string")
                new_string = edit.get("new_string")

                if old_string is None or new_string is None:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"编辑项 {i} 缺少 old_string 或 new_string"
                    )
                if old_string == "":
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"编辑项 {i} 的 old_string 不能为空"
                    )

                matches = original_content.count(old_string)
                if matches != 1:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"编辑项 {i}: old_string 必须唯一匹配原始文件内容。找到 {matches} 处匹配。",
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

            # 检查替换区域冲突（按原始文件区域）
            ordered = sorted(replacements, key=lambda item: item["start"])
            for prev, curr in zip(ordered, ordered[1:]):
                if curr["start"] < prev["end"]:
                    return ToolResponse.error(
                        code=ToolErrorCode.CONFLICT,
                        message=(
                            f"编辑项 {prev['edit_index']} 与编辑项 {curr['edit_index']} 的替换区域重叠，"
                            "无法原子批量应用。"
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
                    "批量编辑检查完成",
                    rel_path,
                    current_mtime_ms,
                    current_size,
                )

            diff_preview, diff_truncated = make_diff_preview(original_content, new_content, rel_path)

            # 备份原文件
            backup_path = self._backup_file(full_path) if not dry_run else None

            # 原子写入
            if not dry_run:
                atomic_write(full_path, new_content)

            changed_bytes = len(new_content.encode('utf-8')) - len(original_content.encode('utf-8'))
            new_mtime_ms = cached_mtime if dry_run else int(os.path.getmtime(full_path) * 1000)
            new_size = len(new_content.encode('utf-8')) if dry_run else os.path.getsize(full_path)

            response_factory = ToolResponse.partial if dry_run else ToolResponse.success
            return response_factory(
                text=(
                    f"{'Dry run：将执行' if dry_run else '成功执行'} {len(edits)} 个独立替换操作于 {rel_path} (变化 {changed_bytes:+d} 字节)\n"
                    f"元数据: file_mtime_ms={new_mtime_ms}, file_size_bytes={new_size}, "
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
                message=f"路径 '{path}' 超出项目根目录，访问被拒绝"
            )

        except PermissionError:
            return ToolResponse.error(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"无权限编辑 '{path}'"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"批量编辑失败：{str(e)}"
            )

    def _backup_file(self, full_path: Path) -> Path:
        """备份文件"""
        backup_dir = full_path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{full_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_name

        shutil.copy2(full_path, backup_path)
        return backup_path

    def _resolve_path(self, path: str) -> Path:
        """解析相对路径"""
        if os.path.isabs(path):
            full_path = Path(path).resolve()
        else:
            full_path = (self.working_dir / path).resolve()
        full_path.relative_to(self.project_root)
        return full_path


class EditFileMultiTool(MultiEditTool):
    """兼容命名：EditFileMulti -> MultiEdit。"""

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
