"""Persistent task graph tools: task_create, task_update, task_list, task_get.

State is stored as JSON files in .tasks/ so it survives context compression.
Each task has a dependency graph (blockedBy/blocks) following the s07 design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..base import Tool, ToolResponse, tool_action
from ..errors import ToolErrorCode
from ._code_utils import atomic_write

if TYPE_CHECKING:
    from ...core.config import Config


VALID_STATUSES = {"pending", "in_progress", "completed"}


class TaskManager:
    """JSON-file-per-task graph manager with dependency resolution."""

    def __init__(self, tasks_dir: Path):
        self.dir = Path(tasks_dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    def create(self, subject: str, description: str = "") -> dict:
        task = {
            "id": self._next_id(),
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
        }
        self._save(task)
        return task

    def get(self, task_id: int) -> dict:
        return self._load(task_id)

    def update(
        self,
        task_id: int,
        *,
        status: Optional[str] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        add_blocked_by: Optional[List[int]] = None,
        add_blocks: Optional[List[int]] = None,
    ) -> dict:
        task = self._load(task_id)

        if subject is not None:
            task["subject"] = subject
        if description is not None:
            task["description"] = description
        if owner is not None:
            task["owner"] = owner

        if status is not None:
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid status '{status}'. Allowed: {', '.join(sorted(VALID_STATUSES))}")
            task["status"] = status

        # Dependency edges (bidirectional)
        for dep_id in add_blocked_by or []:
            self._load(dep_id)  # ensure exists
            if dep_id not in task["blockedBy"]:
                task["blockedBy"].append(dep_id)
            dep = self._load(dep_id)
            if task_id not in dep["blocks"]:
                dep["blocks"].append(task_id)
                self._save(dep)

        for blocked_id in add_blocks or []:
            self._load(blocked_id)  # ensure exists
            if blocked_id not in task["blocks"]:
                task["blocks"].append(blocked_id)
            blocked = self._load(blocked_id)
            if task_id not in blocked["blockedBy"]:
                blocked["blockedBy"].append(task_id)
                self._save(blocked)

        self._save(task)

        if status == "completed":
            self._clear_dependency(task_id)
            task = self._load(task_id)

        return task

    def list_all(self, status: Optional[str] = None) -> List[dict]:
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            task = json.loads(f.read_text(encoding="utf-8"))
            if status and status != "all" and task["status"] != status:
                continue
            tasks.append(task)
        return tasks

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, task: dict) -> None:
        atomic_write(
            self.dir / f"task_{task['id']}.json",
            json.dumps(task, ensure_ascii=False, indent=2) + "\n",
        )

    def _next_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return (max(ids) + 1) if ids else 1

    def _clear_dependency(self, completed_id: int) -> None:
        """Remove completed_id from all other tasks' blockedBy lists."""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text(encoding="utf-8"))
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)


def _format_task_list(tasks: List[dict]) -> str:
    if not tasks:
        return "No tasks."
    marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
    lines = []
    for t in tasks:
        blocked = f" blocked_by={t['blockedBy']}" if t.get("blockedBy") else ""
        blocks = f" blocks={t['blocks']}" if t.get("blocks") else ""
        owner = f" owner={t['owner']}" if t.get("owner") else ""
        lines.append(f"{marker.get(t['status'], '[?]')} #{t['id']} {t['subject']}{blocked}{blocks}{owner}")
    return "\n".join(lines)


def _format_task_detail(task: dict) -> str:
    return (
        f"Task #{task['id']}: {task['subject']}\n"
        f"Status: {task['status']}\n"
        f"Description: {task.get('description') or '[empty]'}\n"
        f"Blocked by: {task['blockedBy'] or 'none'}\n"
        f"Blocks: {task['blocks'] or 'none'}\n"
        f"Owner: {task.get('owner') or '[unassigned]'}"
    )


class TaskTool(Tool):
    """Persistent task graph: create, update, list, get."""

    def __init__(self, config: Optional["Config"] = None, persistence_dir: Optional[str] = None, **_kwargs):
        from ...core.config import Config

        super().__init__(
            name="TaskSystem",
            description=(
                "Persistent task graph for planning and tracking work that survives context compression. "
                "Tools: task_create, task_update, task_list, task_get."
            ),
            expandable=True,
        )
        self.config = config or Config()
        tasks_dir = persistence_dir or "memory/tasks"
        self.task_manager = TaskManager(tasks_dir)

    def get_parameters(self) -> List[Any]:
        return []

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message="TaskSystem is expandable. Use: task_create, task_update, task_list, task_get.",
        )

    @tool_action(name="task_create", description="Create a persistent task with optional dependencies.")
    def create_task(self, subject: str, description: str = "", blocked_by: list = None) -> ToolResponse:
        """Create a task.

        Args:
            subject: Short task title.
            description: Longer description.
            blocked_by: Task IDs that must complete before this task.
        """
        if not subject.strip():
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="subject cannot be empty.")
        try:
            task = self.task_manager.create(subject.strip(), description)
            if blocked_by:
                task = self.task_manager.update(task["id"], add_blocked_by=blocked_by)
        except ValueError as exc:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message=str(exc))
        return ToolResponse.success(
            text=f"Created task #{task['id']}: {task['subject']}",
            data={"task": task},
        )

    @tool_action(name="task_update", description="Update task status, fields, or dependencies.")
    def update_task(
        self,
        task_id: int,
        status: Optional[str] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        add_blocked_by: list = None,
        add_blocks: list = None,
    ) -> ToolResponse:
        """Update one task.

        Args:
            task_id: Task ID.
            status: New status: pending, in_progress, or completed.
            subject: New title.
            description: New description.
            owner: Owner label.
            add_blocked_by: Task IDs that block this task.
            add_blocks: Task IDs this task blocks.
        """
        try:
            task = self.task_manager.update(
                task_id, status=status, subject=subject, description=description,
                owner=owner, add_blocked_by=add_blocked_by, add_blocks=add_blocks,
            )
        except ValueError as exc:
            code = ToolErrorCode.NOT_FOUND if "not found" in str(exc).lower() else ToolErrorCode.INVALID_PARAM
            return ToolResponse.error(code=code, message=str(exc))
        return ToolResponse.success(
            text=f"Updated task #{task['id']}\n{_format_task_detail(task)}",
            data={"task": task},
        )

    @tool_action(name="task_list", description="List all tasks with status and dependency info.")
    def list_tasks(self, status: str = "all") -> ToolResponse:
        """List tasks.

        Args:
            status: Filter: all, pending, in_progress, or completed.
        """
        if status not in VALID_STATUSES | {"all"}:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="status must be: all, pending, in_progress, completed.")
        tasks = self.task_manager.list_all(status=status)
        return ToolResponse.success(
            text=_format_task_list(tasks),
            data={"tasks": tasks},
        )

    @tool_action(name="task_get", description="Get full details of one task by ID.")
    def get_task(self, task_id: int) -> ToolResponse:
        """Get one task.

        Args:
            task_id: Task ID.
        """
        try:
            task = self.task_manager.get(task_id)
        except ValueError as exc:
            return ToolResponse.error(code=ToolErrorCode.NOT_FOUND, message=str(exc))
        return ToolResponse.success(text=_format_task_detail(task), data={"task": task})
