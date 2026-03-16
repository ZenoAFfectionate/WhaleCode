"""Built-in Tools Module

A collection of built-in tools for the HelloAgents framework, including:
- ReadTool: File read tool (supports optimistic locking)
- WriteTool: File write tool (supports optimistic locking)
- EditTool: File edit tool (supports optimistic locking)
- MultiEditTool: Batch edit tool (supports optimistic locking)
- BashTool: Shell command execution tool
- GlobTool: File name pattern search tool
- GrepTool: Code content search tool
- TodoWriteTool: Task list management tool (progress tracking)
- TaskTool: Persistent task graph tool
- BackgroundTool: Background task tool
- SkillTool: Skill loading tool
- AskUserTool: User interaction tool
- WebSearchTool: Web search tool (DuckDuckGo)
- WebFetchTool: Web content scraping tool
"""

from .bash import BashTool
from .ask_user import AskUserTool
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .task_tool import TaskTool
from .skill_tool import SkillTool
from .background import BackgroundTool, BackgroundManager, BackgroundTaskRecord, get_background_manager
from .file_tools import ListFilesTool, ReadTool, WriteTool, EditTool, MultiEditTool, EditFileMultiTool
from .todowrite_tool import TodoWriteTool, TodoItem, TodoList
from .web_tool import WebSearchTool, WebFetchTool

__all__ = [
    "AskUserTool",
    "BackgroundTool",
    "BackgroundManager",
    "BackgroundTaskRecord",
    "get_background_manager",
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
    "WebSearchTool",
    "WebFetchTool",
]
