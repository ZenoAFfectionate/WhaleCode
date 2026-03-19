"""AskUser tool - request user input during agent execution."""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse


class AskUserTool(Tool):
    """Ask the user a question and wait for an answer.

    Only the main agent is allowed to interact with the user.
    Sub-agents should be created with ``interactive=False`` so that
    calling this tool returns an error instead of blocking on stdin.
    """

    def __init__(self, interactive: bool = True):
        super().__init__(
            name="AskUser",
            description=(
                "Ask the user a clarifying question when critical information is "
                "missing and you cannot proceed without it. "
                "Do NOT use for greetings, casual conversation, or when you "
                "already have enough context to respond or act."
            ),
        )
        self._interactive = bool(interactive)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="questions",
                type="array",
                description=(
                    "List of questions. Each item is an object with fields: "
                    "id (string), text (string), type (string, optional), "
                    "options (array, optional), required (bool, optional)."
                ),
                required=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        if not self._interactive:
            return ToolResponse.error(
                code=ToolErrorCode.ASK_USER_UNAVAILABLE,
                message="AskUser is unavailable in sub-agent context. "
                        "Handle user interaction in the main agent.",
            )

        questions = parameters.get("questions")
        if not isinstance(questions, list) or not questions:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="'questions' must be a non-empty list.",
            )

        answers: list[dict[str, Any]] = []
        for item in questions:
            if not isinstance(item, dict):
                continue
            prompt_text = str(item.get("text", "")).strip()
            if not prompt_text:
                continue
            try:
                user_input = input(f"[Agent] {prompt_text}\n> ")
            except EOFError:
                user_input = ""
            answers.append({"id": item.get("id"), "answer": user_input})

        return ToolResponse.success(
            text=f"User answered {len(answers)} question(s).",
            data={"answers": answers},
        )
