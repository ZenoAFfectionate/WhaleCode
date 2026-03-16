"""TodoWrite Progress Management Tool

Provides task list management capabilities, enforces single-threaded focus, and avoids task switching.

Features:
- Declarative Overwrite (submits the full list every time)
- Forced Single-threading (maximum 1 in_progress)
- Automatic Recap Generation
- Persistence to memory/todos/
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

from ..base import Tool, ToolParameter
from ..response import ToolResponse
from ..errors import ToolErrorCode
from ._code_utils import atomic_write


@dataclass
class TodoItem:
    """To-do item"""
    content: str  # Task content
    status: str   # "pending" | "in_progress" | "completed"
    created_at: str  # Creation time
    updated_at: str = ""  # Update time

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class TodoList:
    """To-do list"""
    summary: str  # Overall summary
    todos: List[TodoItem] = field(default_factory=list)

    def get_in_progress(self) -> Optional[TodoItem]:
        """Get the currently active task"""
        for todo in self.todos:
            if todo.status == "in_progress":
                return todo
        return None

    def get_pending(self, limit: int = 5) -> List[TodoItem]:
        """Get pending tasks"""
        return [
            todo for todo in self.todos
            if todo.status == "pending"
        ][:limit]

    def get_completed(self) -> List[TodoItem]:
        """Get completed tasks"""
        return [
            todo for todo in self.todos
            if todo.status == "completed"
        ]

    def get_stats(self) -> dict:
        """Get statistical information"""
        total = len(self.todos)
        completed = sum(1 for t in self.todos if t.status == "completed")
        in_progress = sum(1 for t in self.todos if t.status == "in_progress")
        pending = total - completed - in_progress

        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending
        }


class TodoWriteTool(Tool):
    """To-do list management tool
    
    Features:
    - Declarative overwrite (submits full list every time)
    - Single-thread enforcement (maximum 1 in_progress task)
    - Automatic Recap generation
    - Persistence to file
    """

    def __init__(
        self,
        project_root: str = ".",
        persistence_dir: str = "memory/todos"
    ):
        """Initialize TodoWriteTool
        
        Args:
            project_root: Project root directory
            persistence_dir: Persistence directory (relative to project_root)
        """
        super().__init__(
            name="TodoWrite",
            description="""Manage task lists and maintain single-thread focus.

Features:
- Submit full list every time (declarative)
- Maximum of 1 task marked as in_progress
- Automatically generate Recap to keep context concise
- Automatically save to memory/todos/

Usage Scenarios:
- Create a task list when starting complex tasks
- Track progress to avoid omissions
- Maintain state across multi-turn conversations

Parameters:
- summary: Overall task description (optional)
- todos: To-do list (JSON array)
- action: Action type (create/update/clear, default is create)""",
            expandable=False
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self.persistence_dir = self.project_root / persistence_dir
        
        # Ensure directory exists
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Current Todo list
        self.current_todos = TodoList(summary="")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="summary",
                type="string",
                description="Overall task description (brief, 1-2 sentences)",
                required=False,
                default=""
            ),
            ToolParameter(
                name="todos",
                type="array",
                description="""To-do list (JSON array)

Format: [
  {"content": "Task 1", "status": "pending"},
  {"content": "Task 2", "status": "in_progress"},
  {"content": "Task 3", "status": "completed"}
]

Rules:
- status can only be: pending, in_progress, completed
- A maximum of 1 task can be marked as in_progress
- Submit full list every time (declarative)""",
                required=False,
                default=[]
            ),
            ToolParameter(
                name="action",
                type="string",
                description="Action type: create|update|clear (default is create)",
                required=False,
                default="create"
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute tool

        Args:
            parameters: Tool parameters
                - summary: Overall description
                - todos: To-do list
                - action: Action type

        Returns:
            ToolResponse: Standardized response
        """
        action = parameters.get("action", "create")

        try:
            if action == "clear":
                # Clear task list
                self.current_todos = TodoList(summary="")
                recap = "✅ Task list cleared"

                return ToolResponse.success(
                    text=recap,
                    data={
                        "action": action,
                        "summary": "",
                        "stats": {"total": 0, "completed": 0, "in_progress": 0, "pending": 0}
                    }
                )

            # Get todos parameter
            todos_data = parameters.get("todos", [])

            # If it's a string, try parsing as JSON
            if isinstance(todos_data, str):
                try:
                    todos_data = json.loads(todos_data)
                except json.JSONDecodeError as e:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"todos JSON format error: {str(e)}"
                    )

            # Validate constraints
            validation = self._validate_todos(todos_data)
            if not validation["valid"]:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=validation["message"]
                )

            # Create TodoItem objects
            now = datetime.now().isoformat()
            todos = [
                TodoItem(
                    content=item["content"],
                    status=item["status"],
                    created_at=item.get("created_at", now),
                    updated_at=now
                )
                for item in todos_data
            ]

            # Create TodoList
            summary = parameters.get("summary", "")
            self.current_todos = TodoList(summary=summary, todos=todos)

            # Generate Recap
            recap = self._generate_recap()

            # Persist to disk
            self._persist_todos()

            return ToolResponse.success(
                text=recap,
                data={
                    "action": action,
                    "summary": self.current_todos.summary,
                    "stats": self.current_todos.get_stats()
                }
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to process task list: {str(e)}"
            )

    def _validate_todos(self, todos_data: list) -> dict:
        """Validate todos constraints

        Returns:
            {"valid": bool, "message": str}
        """
        if not isinstance(todos_data, list):
            return {
                "valid": False,
                "message": "todos must be an array"
            }

        in_progress_count = sum(1 for t in todos_data if t.get("status") == "in_progress")

        if in_progress_count > 1:
            return {
                "valid": False,
                "message": f"Only 1 in_progress task is allowed; currently there are {in_progress_count}"
            }

        for i, todo in enumerate(todos_data):
            if not isinstance(todo, dict):
                return {
                    "valid": False,
                    "message": f"Task {i+1} must be an object"
                }

            content = todo.get("content", "")
            status = todo.get("status", "")

            if not content.strip():
                return {
                    "valid": False,
                    "message": f"Content of task {i+1} cannot be empty"
                }

            if status not in ["pending", "in_progress", "completed"]:
                return {
                    "valid": False,
                    "message": f"Status of task {i+1} must be pending, in_progress, or completed"
                }

        return {"valid": True, "message": ""}

    def _generate_recap(self) -> str:
        """Generate Recap text

        Format: [2/5] In progress: xxx. Pending: yyy; zzz.
        """
        stats = self.current_todos.get_stats()

        if stats['total'] == 0:
            return "📋 [0/0] No active tasks"

        recap_parts = [f"📋 [{stats['completed']}/{stats['total']}]"]

        in_progress = self.current_todos.get_in_progress()
        if in_progress:
            recap_parts.append(f"In progress: {in_progress.content}")

        pending = self.current_todos.get_pending(limit=3)
        if pending:
            pending_texts = [t.content for t in pending]
            recap_parts.append(f"Pending: {'; '.join(pending_texts)}")

        if stats['pending'] > 3:
            recap_parts.append(f"There are {stats['pending'] - 3} more...")

        if stats['completed'] == stats['total'] and stats['total'] > 0:
            return f"✅ [{stats['completed']}/{stats['total']}] All tasks completed!"

        return ". ".join(recap_parts)

    def _persist_todos(self):
        """Persist to file (atomic write)"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"todoList-{timestamp}.json"
        filepath = self.persistence_dir / filename

        # Create serializable data
        data = {
            "summary": self.current_todos.summary,
            "todos": [
                {
                    "content": t.content,
                    "status": t.status,
                    "created_at": t.created_at,
                    "updated_at": t.updated_at
                }
                for t in self.current_todos.todos
            ],
            "created_at": datetime.now().isoformat(),
            "stats": self.current_todos.get_stats()
        }

        # Use shared atomic write
        atomic_write(filepath, json.dumps(data, indent=2, ensure_ascii=False))

    def load_todos(self, filepath: str):
        """Load task list from file

        Args:
            filepath: Task list file path
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        todos = [
            TodoItem(
                content=t["content"],
                status=t["status"],
                created_at=t["created_at"],
                updated_at=t.get("updated_at", t["created_at"])
            )
            for t in data["todos"]
        ]

        self.current_todos = TodoList(
            summary=data.get("summary", ""),
            todos=todos
        )
