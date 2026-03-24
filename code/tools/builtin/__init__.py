"""Built-in Tools Module

A collection of built-in tools for the HelloAgents framework, including:
- ReadTool: File read tool (supports optimistic locking)
- WriteTool: File write tool (supports optimistic locking)
- DeleteTool: Safe file/directory deletion tool
- EditTool: File edit tool (supports optimistic locking)
- BashTool: Shell command execution tool
- GlobTool: File name pattern search tool
- GrepTool: Code content search tool
- TodoWriteTool: Unified task management tool (task graph + progress tracking)
- SkillTool: Skill loading tool
- AskUserTool: User interaction tool
- WebSearchTool: Web search tool (DuckDuckGo)
- WebFetchTool: Web content scraping tool
"""

from .bash import BashTool
from .ask_user import AskUserTool
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .skill_tool import SkillTool
from .file_tools import ListFilesTool, ReadTool, WriteTool, DeleteTool, EditTool
from .todowrite_tool import TodoWriteTool, TaskManager
from .web_tool import WebSearchTool, WebFetchTool

__all__ = [
    "AskUserTool",
    "BashTool",
    "ListFilesTool",
    "ReadTool",
    "WriteTool",
    "DeleteTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "TodoWriteTool",
    "TaskManager",
    "SkillTool",
    "WebSearchTool",
    "WebFetchTool",
]
