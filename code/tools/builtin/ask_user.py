"""AskUser tool - request user input during agent execution."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse

_QUESTIONS_PARAM_ERROR = "'questions' must be a non-empty list, question object, or JSON string."

# answer_provider(questions) -> list[{"id": str, "answer": str}]
AnswerProvider = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


class AskUserTool(Tool):
    """Ask the user a question and wait for an answer.

    三种交互模式（按优先级）:
    1. ``answer_provider`` 提供时：调用它获取回答（Web 场景的事件通道，
       由宿主实现"提问 → 阻塞等待回答"）。
    2. ``interactive=True``：调用 ``input()`` 等待终端输入（CLI 场景）。
    3. 否则返回 ``ASK_USER_UNAVAILABLE`` 错误（子 agent / 无交互宿主）。
    """

    def __init__(
        self,
        interactive: bool = True,
        answer_provider: Optional[AnswerProvider] = None,
    ):
        super().__init__(
            name="AskUser",
            description=(
                "Ask the user a clarifying question when critical information is "
                "missing and you cannot proceed without it. "
                "Do NOT use for greetings, casual conversation, or when you "
                "already have enough context to respond or act."
            ),
            category="interactive",
        )
        self._interactive = bool(interactive)
        self._answer_provider = answer_provider

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="questions",
                type="array",
                description=(
                    "List of questions. Each item is an object with fields: "
                    "id (string), text (string), type (string, optional), "
                    "options (array, optional), required (bool, optional). "
                    "The runtime also accepts a single object or JSON-encoded "
                    "string for compatibility."
                ),
                required=True,
            ),
        ]

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        try:
            questions = self._parse_questions(parameters.get("questions"))
        except ValueError as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=str(exc),
            )

        # 模式 1：事件通道（Web）—— 由宿主实现提问与等待
        if self._answer_provider is not None:
            try:
                answers = self._answer_provider(questions)
            except Exception as exc:
                return ToolResponse.error(
                    code=ToolErrorCode.ASK_USER_UNAVAILABLE,
                    message=f"AskUser interaction failed: {exc}",
                )
            if not isinstance(answers, list):
                return ToolResponse.error(
                    code=ToolErrorCode.ASK_USER_UNAVAILABLE,
                    message="AskUser answer_provider returned invalid result.",
                )
            return self._success_with_answers(answers)

        # 模式 2：CLI 终端输入
        if self._interactive:
            answers: list[dict[str, Any]] = []
            for question in questions:
                try:
                    user_input = input(self._format_prompt(question))
                except EOFError:
                    user_input = ""
                answers.append(
                    {
                        "id": question["id"],
                        "answer": self._normalize_answer(user_input, question["options"]),
                    }
                )
            return self._success_with_answers(answers)

        # 模式 3：无交互宿主（子 agent 等）
        return ToolResponse.error(
            code=ToolErrorCode.ASK_USER_UNAVAILABLE,
            message="AskUser is unavailable in sub-agent context. "
                    "Handle user interaction in the main agent.",
        )

    @staticmethod
    def _success_with_answers(answers: list[dict[str, Any]]) -> ToolResponse:
        """构造成功响应 —— text 必须包含完整问答内容。

        模型消费工具结果时主要看 ``response.text``（data 不展示给模型，
        见 core/agent.py 的 _tool_observation_source_text），因此 text 里
        必须带上每个问题的 id 与用户的回答，否则模型"看不到回答"。
        """
        lines = [f"User answered {len(answers)} question(s):"]
        for item in answers:
            qid = str(item.get("id") or "?")
            answer = str(item.get("answer") or "")
            lines.append(f"- {qid}: {answer}")
        return ToolResponse.success(
            text="\n".join(lines),
            data={"answers": answers},
        )

    def _parse_questions(self, raw_questions: Any) -> list[dict[str, Any]]:
        payload = self._decode_questions_payload(raw_questions)
        if isinstance(payload, dict):
            items = [payload]
        elif isinstance(payload, list):
            items = payload
        elif isinstance(payload, str):
            text = payload.strip()
            items = [text] if text else []
        else:
            items = []

        questions = [
            question
            for index, item in enumerate(items, start=1)
            if (question := self._parse_question(item, index)) is not None
        ]
        if not questions:
            raise ValueError(_QUESTIONS_PARAM_ERROR)
        return questions

    @staticmethod
    def _decode_questions_payload(raw_questions: Any) -> Any:
        if not isinstance(raw_questions, str):
            return raw_questions

        stripped = raw_questions.strip()
        if not stripped:
            return None

        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON for 'questions': {exc.msg}.") from exc

        return stripped

    def _parse_question(self, item: Any, index: int) -> dict[str, Any] | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {"id": f"question_{index}", "text": text, "options": []}

        if not isinstance(item, dict):
            return None

        text = str(item.get("text") or "").strip()
        if not text:
            return None

        return {
            "id": str(item.get("id") or f"question_{index}"),
            "text": text,
            "options": self._parse_options(item.get("options")),
        }

    @staticmethod
    def _parse_options(raw_options: Any) -> list[dict[str, str]]:
        if isinstance(raw_options, str):
            items = [raw_options]
        elif isinstance(raw_options, list):
            items = raw_options
        else:
            items = []

        options: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("text") or item.get("value") or "").strip()
                value = str(item.get("value") or label).strip()
            else:
                label = str(item).strip()
                value = label
            if not label:
                continue
            options.append({"label": label, "value": value})
        return options

    @staticmethod
    def _format_prompt(question: dict[str, Any]) -> str:
        lines = [f"[Agent] {question['text']}"]
        options = question["options"]
        if options:
            lines.append("Options:")
            for index, option in enumerate(options, start=1):
                lines.append(f"{index}. {option['label']}")
        lines.append("> ")
        return "\n".join(lines)

    @staticmethod
    def _normalize_answer(user_input: str, options: list[dict[str, str]]) -> str:
        answer = (user_input or "").strip()
        if not options or not answer:
            return answer

        if answer.isdigit():
            selected_index = int(answer) - 1
            if 0 <= selected_index < len(options):
                return options[selected_index]["value"]

        lowered = answer.casefold()
        for option in options:
            if lowered in {option["label"].casefold(), option["value"].casefold()}:
                return option["value"]

        return answer
