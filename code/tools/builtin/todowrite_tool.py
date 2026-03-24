"""TodoWrite — Unified task management tool.

Single tool that handles create, update, list, get, and bulk_create via an
``action`` parameter.  Persists tasks as JSON files with dependency graph
(blockedBy / blocks) so state survives context compression and restarts.

Statuses: pending → in_progress → completed | cancelled
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter
from ..response import ToolResponse
from ..errors import ToolErrorCode
from ._code_utils import atomic_write

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
_LEGACY_PLACEHOLDER_SUBJECTS = {"Setup project", "Write code", "Write tests"}


class TaskManager:
    """Persistent task graph stored as one JSON file per task."""

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
                raise ValueError(
                    f"Invalid status '{status}'. Allowed: {', '.join(sorted(VALID_STATUSES))}"
                )
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

    def delete(self, task_id: int) -> dict:
        """Delete a task and remove it from all dependency lists."""
        task = self._load(task_id)
        # Remove from other tasks' blockedBy/blocks
        for f in self.dir.glob("task_*.json"):
            other = json.loads(f.read_text(encoding="utf-8"))
            changed = False
            if task_id in other.get("blockedBy", []):
                other["blockedBy"].remove(task_id)
                changed = True
            if task_id in other.get("blocks", []):
                other["blocks"].remove(task_id)
                changed = True
            if changed:
                self._save(other)
        # Remove task file
        path = self.dir / f"task_{task_id}.json"
        path.unlink(missing_ok=True)
        return task


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



_STATUS_MARKER = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[-]",
}


def _format_task_list(tasks: List[dict]) -> str:
    if not tasks:
        return "No tasks."
    lines = []
    for t in tasks:
        blocked = f"  blocked_by={t['blockedBy']}" if t.get("blockedBy") else ""
        blocks = f"  blocks={t['blocks']}" if t.get("blocks") else ""
        owner = f"  owner={t['owner']}" if t.get("owner") else ""
        lines.append(
            f"{_STATUS_MARKER.get(t['status'], '[?]')} #{t['id']} {t['subject']}"
            f"{blocked}{blocks}{owner}"
        )
    return "\n".join(lines)


def _format_task_detail(task: dict) -> str:
    return (
        f"Task #{task['id']}: {task['subject']}\n"
        f"  Status: {task['status']}\n"
        f"  Description: {task.get('description') or '[empty]'}\n"
        f"  Blocked by: {task['blockedBy'] or 'none'}\n"
        f"  Blocks: {task['blocks'] or 'none'}\n"
        f"  Owner: {task.get('owner') or '[unassigned]'}"
    )


def _format_recap(tasks: List[dict]) -> str:
    """One-line progress summary."""
    total = len(tasks)
    if total == 0:
        return "📋 [0/0] No tasks"
    completed = sum(1 for t in tasks if t["status"] == "completed")
    in_progress = [t for t in tasks if t["status"] == "in_progress"]
    pending = [t for t in tasks if t["status"] == "pending" and not t.get("blockedBy")]

    if completed == total:
        return f"✅ [{completed}/{total}] All tasks completed!"

    parts = [f"📋 [{completed}/{total}]"]
    if in_progress:
        parts.append(f"In progress: #{in_progress[0]['id']} {in_progress[0]['subject']}")
    if pending:
        names = [f"#{t['id']}" for t in pending[:3]]
        parts.append(f"Ready: {', '.join(names)}")
    return " | ".join(parts)




class TodoWriteTool(Tool):
    """Unified task management tool with dependency graph.

    Actions: create, update, list, get, bulk_create, delete
    Statuses: pending, in_progress, completed, cancelled
    """

    def __init__(
        self,
        project_root: str = ".",
        persistence_dir: str = "memory/tasks",
    ):
        super().__init__(
            name="TodoWrite",
            description=(
                "Manage tasks with dependencies. Single tool for all task operations.\n\n"
                "Actions:\n"
                "- create: Create a task. Params: subject (required), description, blocked_by\n"
                "- update: Update a task. Params: task_id (required), status, subject, description, owner, blocked_by, blocks\n"
                "- list: List tasks. Params: status (optional filter: all/pending/in_progress/completed/cancelled)\n"
                "- get: Get task details. Params: task_id (required)\n"
                "- bulk_create: Create multiple tasks. Params: tasks (array of {subject, description, blocked_by})\n"
                "- delete: Remove a task. Params: task_id (required)\n\n"
                "Statuses: pending → in_progress → completed | cancelled\n"
                "Dependencies: Use blocked_by=[1,2] to declare ordering. "
                "Completing a task auto-unblocks dependents."
            ),
            expandable=False,
        )
        root = Path(project_root).expanduser().resolve()
        self.task_manager = TaskManager(root / persistence_dir)
        self._purge_legacy_placeholder_tasks()

    def _purge_legacy_placeholder_tasks(self) -> None:
        """Remove legacy seeded placeholder tasks if they match exact old pattern.

        This is intentionally strict to avoid deleting user-authored tasks.
        It only removes tasks that look exactly like old boilerplate seeds:
        - subject in: Setup project / Write code / Write tests
        - status is pending
        - no deps (blockedBy/blocks)
        - no owner
        - empty description
        """
        try:
            tasks = self.task_manager.list_all()
            removable = []
            for task in tasks:
                subject = str(task.get("subject", "")).strip()
                if subject not in _LEGACY_PLACEHOLDER_SUBJECTS:
                    continue
                if task.get("status") != "pending":
                    continue
                if task.get("blockedBy"):
                    continue
                if task.get("blocks"):
                    continue
                if str(task.get("owner", "")).strip():
                    continue
                if str(task.get("description", "")).strip():
                    continue
                removable.append(task)

            for task in sorted(removable, key=lambda item: int(item.get("id", 0)), reverse=True):
                self.task_manager.delete(int(task["id"]))
        except Exception:
            # Non-critical compatibility cleanup; never block tool startup.
            pass

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Operation: create, update, list, get, bulk_create, delete",
                required=True,
            ),
            ToolParameter(
                name="task_id",
                type="integer",
                description="Task ID (for update, get, delete)",
                required=False,
            ),
            ToolParameter(
                name="subject",
                type="string",
                description="Task title (for create, update)",
                required=False,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="Task description (for create, update)",
                required=False,
            ),
            ToolParameter(
                name="status",
                type="string",
                description="Task status: pending, in_progress, completed, cancelled (for update); or filter for list",
                required=False,
            ),
            ToolParameter(
                name="blocked_by",
                type="array",
                description="Task IDs that must complete before this task (for create, update)",
                required=False,
            ),
            ToolParameter(
                name="blocks",
                type="array",
                description="Task IDs this task blocks (for update)",
                required=False,
            ),
            ToolParameter(
                name="owner",
                type="string",
                description="Owner label (for update)",
                required=False,
            ),
            ToolParameter(
                name="tasks",
                type="array",
                description='Array of task objects for bulk_create: [{"subject": "...", "description": "...", "blocked_by": [...]}, ...]',
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        action = (parameters.get("action") or "").strip().lower()

        dispatch = {
            "create": self._action_create,
            "update": self._action_update,
            "list": self._action_list,
            "get": self._action_get,
            "bulk_create": self._action_bulk_create,
            "delete": self._action_delete,
        }

        handler = dispatch.get(action)
        if not handler:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Unknown action '{action}'. Use: {', '.join(sorted(dispatch))}",
            )

        try:
            return handler(parameters)
        except ValueError as exc:
            code = ToolErrorCode.NOT_FOUND if "not found" in str(exc).lower() else ToolErrorCode.INVALID_PARAM
            return ToolResponse.error(code=code, message=str(exc))
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"TodoWrite failed: {exc}",
            )


    def _action_create(self, params: Dict[str, Any]) -> ToolResponse:
        subject = (params.get("subject") or "").strip()
        if not subject:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="subject is required for create")

        description = params.get("description", "")
        task = self.task_manager.create(subject, description)

        blocked_by = params.get("blocked_by")
        if blocked_by:
            task = self.task_manager.update(task["id"], add_blocked_by=self._to_int_list(blocked_by))

        recap = _format_recap(self.task_manager.list_all())
        return ToolResponse.success(
            text=f"Created #{task['id']}: {task['subject']}\n{recap}",
            data={"task": task},
        )

    def _action_update(self, params: Dict[str, Any]) -> ToolResponse:
        task_id = params.get("task_id")
        if task_id is None:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="task_id is required for update")

        task = self.task_manager.update(
            int(task_id),
            status=params.get("status"),
            subject=params.get("subject"),
            description=params.get("description"),
            owner=params.get("owner"),
            add_blocked_by=self._to_int_list(params.get("blocked_by")),
            add_blocks=self._to_int_list(params.get("blocks")),
        )

        recap = _format_recap(self.task_manager.list_all())
        return ToolResponse.success(
            text=f"Updated #{task['id']}\n{_format_task_detail(task)}\n{recap}",
            data={"task": task},
        )

    def _action_list(self, params: Dict[str, Any]) -> ToolResponse:
        status_filter = params.get("status", "all")
        if status_filter and status_filter != "all" and status_filter not in VALID_STATUSES:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid status filter '{status_filter}'. Use: all, {', '.join(sorted(VALID_STATUSES))}",
            )
        tasks = self.task_manager.list_all(status=status_filter)
        recap = _format_recap(self.task_manager.list_all())
        return ToolResponse.success(
            text=f"{_format_task_list(tasks)}\n\n{recap}",
            data={"tasks": tasks},
        )

    def _action_get(self, params: Dict[str, Any]) -> ToolResponse:
        task_id = params.get("task_id")
        if task_id is None:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="task_id is required for get")
        task = self.task_manager.get(int(task_id))
        return ToolResponse.success(
            text=_format_task_detail(task),
            data={"task": task},
        )

    def _action_bulk_create(self, params: Dict[str, Any]) -> ToolResponse:
        tasks_data = params.get("tasks", [])
        if isinstance(tasks_data, str):
            try:
                tasks_data = json.loads(tasks_data)
            except json.JSONDecodeError as e:
                return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message=f"Invalid tasks JSON: {e}")

        if not isinstance(tasks_data, list) or not tasks_data:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="tasks must be a non-empty array")

        created = []
        for item in tasks_data:
            subject = (item.get("subject") or "").strip()
            if not subject:
                continue
            task = self.task_manager.create(subject, item.get("description", ""))
            blocked_by = item.get("blocked_by")
            if blocked_by:
                task = self.task_manager.update(task["id"], add_blocked_by=self._to_int_list(blocked_by))
            created.append(task)

        recap = _format_recap(self.task_manager.list_all())
        names = [f"#{t['id']} {t['subject']}" for t in created]
        return ToolResponse.success(
            text=f"Created {len(created)} tasks:\n" + "\n".join(names) + f"\n\n{recap}",
            data={"tasks": created},
        )

    def _action_delete(self, params: Dict[str, Any]) -> ToolResponse:
        task_id = params.get("task_id")
        if task_id is None:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="task_id is required for delete")
        task = self.task_manager.delete(int(task_id))
        recap = _format_recap(self.task_manager.list_all())
        return ToolResponse.success(
            text=f"Deleted #{task['id']}: {task['subject']}\n{recap}",
            data={"task": task},
        )


    @staticmethod
    def _to_int_list(value) -> Optional[List[int]]:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if isinstance(value, list):
            return [int(v) for v in value]
        return None
