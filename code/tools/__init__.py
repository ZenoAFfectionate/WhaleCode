"""工具系统"""

from .base import Tool, ToolParameter, tool_action
from .registry import ToolRegistry, global_registry
from .response import ToolResponse, ToolStatus
from .errors import ToolErrorCode

# 内置工具
from .builtin.bash import BashTool
from .builtin.file_tools import ListFilesTool, ReadTool, WriteTool, EditTool, MultiEditTool, EditFileMultiTool
from .builtin.glob_tool import GlobTool
from .builtin.grep_tool import GrepTool
from .builtin.todowrite_tool import TodoWriteTool, TodoItem, TodoList
from .builtin.task_tool import TaskTool
from .builtin.skill_tool import SkillTool
from .builtin.ask_user import AskUserTool
from .builtin.web_tool import WebSearchTool, WebFetchTool
from .builtin.background import BackgroundTool, BackgroundManager, BackgroundTaskRecord, get_background_manager

# 子代理机制
from .tool_filter import ToolFilter, ReadOnlyFilter, FullAccessFilter, CustomFilter

__all__ = [
    # 基础工具系统
    "Tool",
    "ToolParameter",
    "tool_action",
    "ToolRegistry",
    "global_registry",

    # 工具响应协议
    "ToolResponse",
    "ToolStatus",
    "ToolErrorCode",

    # 内置工具
    "BashTool",
    "ListFilesTool",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "MultiEditTool",
    "EditFileMultiTool",
    "GlobTool",
    "GrepTool",
    "TodoWriteTool",
    "TodoItem",
    "TodoList",
    "TaskTool",
    "SkillTool",
    "AskUserTool",
    "WebSearchTool",
    "WebFetchTool",
    "BackgroundTool",
    "BackgroundManager",
    "BackgroundTaskRecord",
    "get_background_manager",

    # 子代理机制
    "ToolFilter",
    "ReadOnlyFilter",
    "FullAccessFilter",
    "CustomFilter",
]
