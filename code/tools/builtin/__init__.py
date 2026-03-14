"""内置工具模块

HelloAgents框架的内置工具集合，包括：
- ReadTool: 文件读取工具（支持乐观锁）
- WriteTool: 文件写入工具（支持乐观锁）
- EditTool: 文件编辑工具（支持乐观锁）
- MultiEditTool: 批量编辑工具（支持乐观锁）
- BashTool: Shell 命令执行工具
- GlobTool: 文件名模式搜索工具
- GrepTool: 代码内容搜索工具
- TodoWriteTool: 任务列表管理工具（进度管理）
- TaskTool: 持久化任务图工具
- BackgroundTool: 后台任务工具
- SkillTool: 技能加载工具
- AskUserTool: 用户交互工具
- WebSearchTool: 网页搜索工具（DuckDuckGo）
- WebFetchTool: 网页内容抓取工具
"""

from .ask_user import AskUserTool
from .background import BackgroundTool, BackgroundManager, BackgroundTaskRecord, get_background_manager
from .bash import BashTool
from .file_tools import ListFilesTool, ReadTool, WriteTool, EditTool, MultiEditTool, EditFileMultiTool
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .todowrite_tool import TodoWriteTool, TodoItem, TodoList
from .task_tool import TaskTool
from .skill_tool import SkillTool
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
