"""TodoWrite - session-scoped planning state with replace-all updates.

The tool exposes a single `todos` parameter. Each invocation replaces the
entire todo list for the current session and persists it atomically as one JSON
snapshot. This keeps the LLM interface simple while preserving state across
context compaction, auto-save, and explicit session restore.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import atomic_write

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
VALID_PRIORITIES = {"high", "medium", "low"}
TERMINAL_STATUSES = {"completed", "cancelled"}

_STATUS_MARKER = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[-]",
}

_LEGACY_TOP_LEVEL_KEYS = {
    "action",
    "task_id",
    "subject",
    "description",
    "status",
    "blocked_by",
    "blocks",
    "owner",
    "tasks",
    "summary",
}


class TodoValidationError(ValueError):
    """Raised when a todo snapshot is malformed or violates state rules."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compute_stats(todos: List[dict]) -> Dict[str, int]:
    stats = {status: 0 for status in sorted(VALID_STATUSES)}
    for todo in todos:
        stats[todo["status"]] += 1
    stats["total"] = len(todos)
    return stats


def _format_todos(todos: List[dict]) -> str:
    if not todos:
        return "No todos."

    lines: List[str] = []
    for todo in todos:
        priority = f" [{todo['priority']}]" if todo["priority"] != "medium" else ""
        lines.append(f"{_STATUS_MARKER[todo['status']]} {todo['content']}{priority}")
    return "\n".join(lines)


def _format_recap(todos: List[dict]) -> str:
    stats = _compute_stats(todos)
    if stats["total"] == 0:
        return "Todos [0/0 completed]"

    completed = stats["completed"]
    total = stats["total"]
    parts = [f"Todos [{completed}/{total} completed]"]

    in_progress = [todo["content"] for todo in todos if todo["status"] == "in_progress"]
    if in_progress:
        parts.append(f"In progress: {in_progress[0]}")

    ready = [todo["content"] for todo in todos if todo["status"] == "pending"]
    if ready:
        parts.append(f"Pending: {', '.join(ready[:3])}")

    return " | ".join(parts)


class TodoSessionStore:
    """Atomically persisted todo snapshots keyed by session id."""

    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self.dir / f"session-{session_id}.json"

    def load_state(self, session_id: str) -> Dict[str, Any]:
        path = self.path_for(session_id)
        if not path.exists():
            return self._empty_state(session_id)

        data = json.loads(path.read_text(encoding="utf-8"))
        todos = self._normalize_todos(data.get("todos", []))
        updated_at = data.get("updated_at")
        return {
            "session_id": session_id,
            "updated_at": updated_at,
            "todos": todos,
            "stats": _compute_stats(todos),
        }

    def replace_all(self, session_id: str, todos: List[dict]) -> Dict[str, Any]:
        previous = self.load_state(session_id)["todos"]
        normalized = self._normalize_todos(todos, previous_todos=previous)
        state = {
            "session_id": session_id,
            "updated_at": _utcnow_iso(),
            "todos": normalized,
            "stats": _compute_stats(normalized),
        }
        atomic_write(self.path_for(session_id), json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        return state

    def import_state(self, session_id: str, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        todos = [] if not state else state.get("todos", [])
        return self.replace_all(session_id, todos)

    @staticmethod
    def _empty_state(session_id: str) -> Dict[str, Any]:
        todos: List[dict] = []
        return {
            "session_id": session_id,
            "updated_at": None,
            "todos": todos,
            "stats": _compute_stats(todos),
        }

    @staticmethod
    def _normalize_todos(
        todos: Any,
        *,
        previous_todos: Optional[List[dict]] = None,
    ) -> List[dict]:
        if not isinstance(todos, list):
            raise TodoValidationError("todos must be an array of todo objects")

        normalized: List[dict] = []
        seen_contents: set[str] = set()
        in_progress_count = 0
        previous_status = {todo["content"]: todo["status"] for todo in previous_todos or []}

        for index, raw in enumerate(todos):
            if not isinstance(raw, dict):
                raise TodoValidationError(f"todos[{index}] must be an object")

            content = str(raw.get("content") or "").strip()
            if not content:
                raise TodoValidationError(f"todos[{index}].content is required")
            if content in seen_contents:
                raise TodoValidationError(f"Duplicate todo content is not allowed: {content!r}")

            status = str(raw.get("status") or "").strip().lower()
            if status not in VALID_STATUSES:
                allowed = ", ".join(sorted(VALID_STATUSES))
                raise TodoValidationError(f"Invalid status for {content!r}: {status!r}. Allowed: {allowed}")

            priority = str(raw.get("priority", "medium") or "medium").strip().lower()
            if priority not in VALID_PRIORITIES:
                allowed = ", ".join(sorted(VALID_PRIORITIES))
                raise TodoValidationError(f"Invalid priority for {content!r}: {priority!r}. Allowed: {allowed}")

            previous = previous_status.get(content)
            if previous in TERMINAL_STATUSES and status != previous:
                raise TodoValidationError(
                    f"Todo {content!r} is already {previous} and cannot transition back to {status}"
                )

            if status == "in_progress":
                in_progress_count += 1

            seen_contents.add(content)
            normalized.append(
                {
                    "content": content,
                    "status": status,
                    "priority": priority,
                }
            )

        if in_progress_count > 1:
            raise TodoValidationError("Only one todo may be in_progress at a time")

        return normalized


class TodoWriteTool(Tool):
    """Replace the entire todo list for the current session."""

    def __init__(
        self,
        project_root: str = ".",
        persistence_dir: str = "memory/todos",
        session_id: Optional[str] = None,
    ):
        super().__init__(
            name="TodoWrite",
            description=(
                "Replace the entire todo list for the current session.\n\n"
                "Parameters:\n"
                "- todos: Required array of todo objects. Each object must contain:\n"
                "  - content: brief task description\n"
                "  - status: pending, in_progress, completed, or cancelled\n"
                "  - priority: optional high, medium, or low (defaults to medium)\n\n"
                "Rules:\n"
                "- Always send the complete current todo list, not a partial patch.\n"
                "- Keep at most one todo in_progress.\n"
                "- Completed/cancelled todos are terminal and cannot be reopened with the same content.\n"
                "- Legacy action-based parameters are not supported."
            ),
            expandable=False,
        )
        root = Path(project_root).expanduser().resolve()
        self.store = TodoSessionStore(root / persistence_dir)
        self.session_id = session_id or f"local-{uuid4().hex[:8]}"

    def bind_session(self, session_id: str) -> None:
        self.session_id = session_id

    def export_state(self) -> Dict[str, Any]:
        return self.store.load_state(self.session_id)

    def import_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        imported = self.store.import_state(self.session_id, state)
        return imported

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todos",
                type="array",
                description=(
                    'The full replacement todo list. Example: [{"content": "Inspect failing tests", '
                    '"status": "in_progress", "priority": "high"}, {"content": "Apply fix", '
                    '"status": "pending"}]'
                ),
                required=True,
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        legacy_keys = sorted(_LEGACY_TOP_LEVEL_KEYS.intersection(parameters.keys()))
        if legacy_keys:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    "TodoWrite now only accepts the `todos` array. "
                    f"Legacy parameters are not supported: {', '.join(legacy_keys)}"
                ),
            )

        todos = parameters.get("todos")
        if todos is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="todos is required",
            )

        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError as exc:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"Invalid todos JSON: {exc}",
                )

        try:
            state = self.store.replace_all(self.session_id, todos)
        except TodoValidationError as exc:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message=str(exc))
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"TodoWrite failed: {exc}",
            )

        text = _format_todos(state["todos"])
        recap = _format_recap(state["todos"])
        return ToolResponse.success(
            text=f"{text}\n\n{recap}",
            data={
                "session_id": state["session_id"],
                "todos": state["todos"],
                "stats": state["stats"],
                "updated_at": state["updated_at"],
            },
        )
