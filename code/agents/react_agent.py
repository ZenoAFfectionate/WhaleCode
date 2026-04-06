"""ReAct agent built on top of tool-calling chat models."""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.lifecycle import EventType, LifecycleHook
from ..core.llm import HelloAgentsLLM
from ..core.message import Message
from ..core.reasoning import extract_reasoning_payload
from ..core.streaming import StreamEvent, StreamEventType
from ..tools.base import Tool, ToolParameter
from ..tools.registry import ToolRegistry
from ..tools.response import ToolResponse


THOUGHT_TOOL_NAME = "Thought"
FINISH_TOOL_NAME = "Finish"
STRUCTURED_OUTPUT_TOOL_NAME = "StructuredOutput"

DEFAULT_REACT_SYSTEM_PROMPT = """You are an AI assistant with reasoning and action capabilities.

Use tools whenever they are helpful for gathering information or performing work.
You may call tools multiple times.
Use the Thought tool to record concise reasoning when it helps.
If you already have enough information and no more tool work is needed, you may answer directly in plain text.
Use the Finish tool when you want to conclude explicitly through a tool call.
"""

THOUGHT_TOOL_DESCRIPTION = (
    "Record short reasoning or planning notes before or between actions."
)
FINISH_TOOL_DESCRIPTION = (
    "Return the final answer and conclude the current agent turn."
)
DEFAULT_STRUCTURED_OUTPUT_DESCRIPTION = (
    "Return the final answer in the required structured format. "
    "Call this tool exactly once after all other tool usage is complete."
)


class _ThoughtTool(Tool):
    """Registry-backed control tool for short reasoning notes."""

    def __init__(self, description: str):
        super().__init__(name=THOUGHT_TOOL_NAME, description=description)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="reasoning",
                type="string",
                description="Concise reasoning, plan, or decision note.",
                required=True,
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        reasoning = str(parameters.get("reasoning", "")).strip() or "[empty reasoning]"
        return ToolResponse.success(
            text=f"Reasoning: {reasoning}",
            data={
                "builtin_tool": True,
                "finished": False,
                "reasoning": reasoning,
            },
        )


class _FinishTool(Tool):
    """Registry-backed control tool for final answers."""

    def __init__(self, description: str):
        super().__init__(name=FINISH_TOOL_NAME, description=description)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="answer",
                type="string",
                description="The final answer for the user.",
                required=True,
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        answer = str(parameters.get("answer", "")).strip()
        return ToolResponse.success(
            text=f"Final answer: {answer}",
            data={
                "builtin_tool": True,
                "finished": True,
                "final_answer": answer,
            },
        )


@dataclass
class _ExecutionState:
    current_step: int
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    consecutive_no_diff_edits: int = 0
    last_test_output_hash: Optional[int] = None
    consecutive_same_tests: int = 0
    stagnation_detected: bool = False
    last_reasoning_content: Optional[str] = None
    last_reasoning_source: Optional[str] = None


@dataclass(frozen=True)
class _StructuredOutputSpec:
    name: str
    description: str
    schema: Dict[str, Any]


class ReActAgent(Agent):
    """ReAct-style agent that loops on tool calls until the model finishes."""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
    ):
        super().__init__(
            name,
            llm,
            system_prompt or DEFAULT_REACT_SYSTEM_PROMPT,
            config,
            tool_registry=tool_registry or ToolRegistry(config=config),
        )
        self._register_builtin_tools()
        self.max_steps = max_steps

    # ==================== Console / render hooks ====================

    def _console(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        """Default console sink used by render hooks."""
        print(message, end=end, flush=flush)

    def _render_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Structured render event hook."""
        if event_type == "agent_start":
            self._console(f"\n🤖 {self.name} processing: {payload.get('input_text', '')}")
        elif event_type == "step_start":
            self._console(f"\n--- Step {payload.get('step')} ---")
        elif event_type == "compaction_notice":
            self._console("[auto-compact triggered]")
        elif event_type == "llm_error":
            self._console(f"❌ LLM call failed: {payload.get('error')}")
        elif event_type == "direct_response":
            self._console(f"💬 Response: {payload.get('final_answer', '')}")
        elif event_type == "builtin_tool":
            tool_name = payload.get("tool_name", "")
            result_content = str(payload.get("result_content", ""))
            if tool_name == THOUGHT_TOOL_NAME:
                self._console(f"💭 {result_content.removeprefix('Reasoning: ')}")
            else:
                self._console(f"🔧 {tool_name}: {result_content}")
        elif event_type == "control_tool":
            self._console(f"🔧 {payload.get('tool_name')}: {payload.get('result_content', '')}")
        elif event_type == "tool_call":
            self._console(f"🎬 Tool call: {payload.get('tool_name')}({payload.get('arguments', {})})")
        elif event_type == "tool_result":
            result_content = str(payload.get("result_content", ""))
            if payload.get("status") == "error" or result_content.startswith("❌"):
                self._console(result_content)
            else:
                self._console(f"👀 Observation: {result_content}")
        elif event_type == "final_answer":
            self._console(f"🎉 Final answer: {payload.get('final_answer', '')}")
        elif event_type == "timeout":
            self._console("⏰ Maximum steps reached, stopping.")
        elif event_type == "stagnation_detected":
            self._console(f"🔄 Stagnation detected: {payload.get('reason')}. Stopping early.")
        elif event_type == "stream_chunk":
            self._console(payload.get("chunk", ""), end="", flush=True)
        elif event_type == "stream_newline":
            self._console("")
        elif event_type == "agent_error":
            self._console(f"❌ {payload.get('message', '')}")
        elif event_type == "background_update":
            step = payload.get("step")
            notification_text = payload.get("notification_text", "")
            self._console(f"📬 Background updates before step {step}:\n{notification_text}")
        elif event_type == "console":
            self._console(
                payload.get("message", ""),
                end=payload.get("end", "\n"),
                flush=payload.get("flush", False),
            )

    def _render_agent_start(self, input_text: str) -> None:
        self._render_event("agent_start", {"input_text": input_text, "agent_name": self.name})

    def _render_step_start(self, step: int) -> None:
        self._render_event("step_start", {"step": step})

    def _render_compaction_notice(self) -> None:
        self._render_event("compaction_notice", {})

    def _render_llm_error(self, error: Exception | str, *, step: Optional[int] = None) -> None:
        self._render_event("llm_error", {"error": str(error), "step": step})

    def _render_direct_response(
        self,
        final_answer: str,
        *,
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> None:
        payload = {"final_answer": final_answer}
        if reasoning_content is not None:
            payload["reasoning_content"] = reasoning_content
        if reasoning_source is not None:
            payload["reasoning_source"] = reasoning_source
        self._render_event("direct_response", payload)

    def _render_builtin_tool(
        self,
        tool_name: str,
        result_content: str,
        *,
        tool_call_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        self._render_event(
            "builtin_tool",
            {
                "tool_name": tool_name,
                "result_content": result_content,
                "tool_call_id": tool_call_id,
                "step": step,
            },
        )

    def _render_control_tool(
        self,
        tool_name: str,
        result_content: str,
        *,
        tool_call_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        self._render_event(
            "control_tool",
            {
                "tool_name": tool_name,
                "result_content": result_content,
                "tool_call_id": tool_call_id,
                "step": step,
            },
        )

    def _render_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        tool_call_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        self._render_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_call_id,
                "step": step,
            },
        )

    def _render_tool_result(
        self,
        result_content: str,
        *,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        status: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        if status is None:
            status = "error" if str(result_content).startswith("❌") else "success"
        self._render_event(
            "tool_result",
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "result_content": result_content,
                "status": status,
                "step": step,
            },
        )

    def _render_final_answer(
        self,
        final_answer: str,
        *,
        step: Optional[int] = None,
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> None:
        payload = {"final_answer": final_answer, "step": step}
        if reasoning_content is not None:
            payload["reasoning_content"] = reasoning_content
        if reasoning_source is not None:
            payload["reasoning_source"] = reasoning_source
        self._render_event("final_answer", payload)

    def _render_timeout(self) -> None:
        self._render_event("timeout", {})

    def _render_stream_chunk(self, chunk: str) -> None:
        self._render_event("stream_chunk", {"chunk": chunk})

    def _render_stream_newline(self) -> None:
        self._render_event("stream_newline", {})

    def _render_agent_error(self, message: str) -> None:
        self._render_event("agent_error", {"message": message})

    def add_tool(self, tool) -> None:
        self.tool_registry.register_tool(tool)

    @staticmethod
    def _builtin_tool_names() -> tuple[str, str]:
        return THOUGHT_TOOL_NAME, FINISH_TOOL_NAME

    def _register_builtin_tools(self) -> None:
        if not self.tool_registry:
            return
        builtin_specs = (
            (THOUGHT_TOOL_NAME, _ThoughtTool, THOUGHT_TOOL_DESCRIPTION),
            (FINISH_TOOL_NAME, _FinishTool, FINISH_TOOL_DESCRIPTION),
        )
        for tool_name, tool_cls, description in builtin_specs:
            existing_function = self.tool_registry.get_function(tool_name)
            if existing_function is not None:
                raise ValueError(
                    f"Builtin control tool name '{tool_name}' conflicts with an existing function tool."
                )

            existing_tool = self.tool_registry.get_tool(tool_name)
            if existing_tool is None:
                self.tool_registry.register_tool(tool_cls(description))
                continue

            if not isinstance(existing_tool, tool_cls):
                raise ValueError(
                    f"Builtin control tool name '{tool_name}' conflicts with an existing tool."
                )

            existing_tool.description = description

    @staticmethod
    def _invalid_tool_arguments_content(exc: json.JSONDecodeError) -> str:
        return f"Error: Invalid argument format - {exc}"

    def _extract_structured_output_spec(self, kwargs: Dict[str, Any]) -> Optional[_StructuredOutputSpec]:
        schema = kwargs.pop("structured_output_schema", kwargs.pop("output_schema", None))
        if schema is None:
            return None

        name = kwargs.pop("structured_output_name", STRUCTURED_OUTPUT_TOOL_NAME)
        description = kwargs.pop(
            "structured_output_description",
            DEFAULT_STRUCTURED_OUTPUT_DESCRIPTION,
        )

        if not isinstance(name, str) or not name.strip():
            raise ValueError("structured_output_name must be a non-empty string.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("structured_output_description must be a non-empty string.")

        normalized_schema = self._normalize_structured_output_schema(schema)
        self._ensure_structured_output_name_available(name.strip())

        return _StructuredOutputSpec(
            name=name.strip(),
            description=description.strip(),
            schema=normalized_schema,
        )

    @staticmethod
    def _normalize_structured_output_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            raise TypeError("structured_output_schema must be a JSON schema dictionary.")

        normalized = deepcopy(schema)
        if normalized.get("type") != "object":
            raise ValueError("structured_output_schema must declare type='object'.")

        properties = normalized.get("properties")
        if properties is None:
            normalized["properties"] = {}
        elif not isinstance(properties, dict):
            raise ValueError("structured_output_schema.properties must be a dictionary.")

        required = normalized.get("required")
        if required is not None and not isinstance(required, list):
            raise ValueError("structured_output_schema.required must be a list when provided.")

        return normalized

    def _ensure_structured_output_name_available(self, tool_name: str) -> None:
        if not self.tool_registry:
            return
        if self.tool_registry.get_tool(tool_name) or self.tool_registry.get_function(tool_name):
            raise ValueError(
                f"Structured output tool name '{tool_name}' conflicts with an existing tool."
            )

    @staticmethod
    def _structured_output_tool_name(
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> Optional[str]:
        return structured_output.name if structured_output else None

    def _is_builtin_tool_name(self, tool_name: str) -> bool:
        return tool_name in self._builtin_tool_names()

    def _is_structured_output_tool_name(
        self,
        tool_name: str,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> bool:
        structured_name = self._structured_output_tool_name(structured_output)
        return structured_name is not None and tool_name == structured_name

    def _is_finalizing_tool_name(
        self,
        tool_name: str,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> bool:
        return tool_name == FINISH_TOOL_NAME or self._is_structured_output_tool_name(
            tool_name,
            structured_output,
        )

    def _invalid_finalizing_tool_calls(
        self,
        tool_calls: List[Any],
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> Dict[str, str]:
        finalizing_calls = [
            tool_call
            for tool_call in tool_calls
            if self._is_finalizing_tool_name(tool_call.function.name, structured_output)
        ]
        if not finalizing_calls:
            return {}

        invalid_calls: Dict[str, str] = {}
        if len(finalizing_calls) > 1:
            structured_name = self._structured_output_tool_name(structured_output)
            for tool_call in finalizing_calls:
                tool_name = tool_call.function.name
                if structured_name and tool_name == structured_name:
                    invalid_calls[tool_call.id] = (
                        f"{structured_name} must be called at most once per response."
                    )
                else:
                    invalid_calls[tool_call.id] = (
                        f"{FINISH_TOOL_NAME} must be called at most once per response."
                    )
            return invalid_calls

        has_other_tool_work = any(
            tool_call.function.name != THOUGHT_TOOL_NAME
            and not self._is_finalizing_tool_name(tool_call.function.name, structured_output)
            for tool_call in tool_calls
        )
        if not has_other_tool_work:
            return invalid_calls

        finalizing_call = finalizing_calls[0]
        tool_name = finalizing_call.function.name
        if self._is_structured_output_tool_name(tool_name, structured_output):
            invalid_calls[finalizing_call.id] = (
                f"{tool_name} must be called alone after all other tool work is complete."
            )
        else:
            invalid_calls[finalizing_call.id] = (
                f"{FINISH_TOOL_NAME} must be called after all other tool work is complete."
            )
        return invalid_calls

    def _decode_tool_call(
        self,
        tool_call: Any,
    ) -> tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
        tool_name = tool_call.function.name
        tool_call_id = tool_call.id

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            return tool_name, tool_call_id, None, self._invalid_tool_arguments_content(exc)

        return tool_name, tool_call_id, arguments, None

    async def _emit_tool_call_event(
        self,
        on_tool_call: LifecycleHook,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
        current_step: int,
    ) -> None:
        await self._emit_event(
            EventType.TOOL_CALL,
            on_tool_call,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args=arguments,
            step=current_step,
        )

    def _trace_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
        *,
        step: int,
    ) -> None:
        if not self.trace_logger:
            return
        self.trace_logger.log_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "args": arguments,
            },
            step=step,
        )

    def _trace_tool_result(
        self,
        tool_name: str,
        tool_call_id: str,
        result_content: str,
        *,
        step: int,
        status: Optional[str] = None,
    ) -> None:
        if not self.trace_logger:
            return
        payload = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "result": result_content,
        }
        if status is not None:
            payload["status"] = status
        self.trace_logger.log_event("tool_result", payload, step=step)

    def _tool_error_result(
        self,
        tool_name: str,
        tool_call_id: str,
        error_content: str,
        *,
        step: int,
    ) -> tuple[str, str, Dict[str, str]]:
        self._render_agent_error(error_content)
        self._trace_tool_result(
            tool_name,
            tool_call_id,
            error_content,
            step=step,
            status="error",
        )
        return (tool_name, tool_call_id, {"content": error_content, "status": "error"})

    def _create_execution_state(self, start_step: int) -> _ExecutionState:
        return _ExecutionState(current_step=start_step)

    @staticmethod
    def _normalize_start_step(start_step: Any) -> int:
        normalized = int(start_step or 0)
        return normalized if normalized >= 0 else 0

    @staticmethod
    def _builtin_tool_instruction() -> str:
        return (
            "Builtin control tools are available.\n"
            f"- Use {THOUGHT_TOOL_NAME} to record concise reasoning or planning when helpful.\n"
            "- Use normal tools to gather information and perform work.\n"
            "- You may answer directly in plain text when no more tool work is needed.\n"
            f"- Use {FINISH_TOOL_NAME} when you want to conclude explicitly through a tool call."
        )

    def _apply_builtin_tool_prompt(self, messages: List[Dict[str, Any]]) -> None:
        instruction = self._builtin_tool_instruction()
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content", "")
            message["content"] = f"{content}\n\n{instruction}" if content else instruction
            return
        messages.insert(0, {"role": "system", "content": instruction})

    @staticmethod
    def _structured_output_instruction(spec: _StructuredOutputSpec) -> str:
        return (
            "Structured output mode is enabled.\n"
            f"- Use the {spec.name} tool for the final answer.\n"
            "- Do not return the final answer as plain text.\n"
            "- Keep using normal tools as needed before that final tool call.\n"
            f"- The {spec.name} arguments must match its JSON schema exactly."
        )

    def _apply_structured_output_prompt(
        self,
        messages: List[Dict[str, Any]],
        spec: _StructuredOutputSpec,
    ) -> None:
        instruction = self._structured_output_instruction(spec)
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content", "")
            message["content"] = f"{content}\n\n{instruction}" if content else instruction
            return
        messages.insert(0, {"role": "system", "content": instruction})

    @staticmethod
    def _tool_choice_for(structured_output: Optional[_StructuredOutputSpec]) -> str:
        return "required" if structured_output else "auto"

    def _build_structured_output_tool_schema(self, spec: _StructuredOutputSpec) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": deepcopy(spec.schema),
            },
        }

    def _build_tool_execution_result(self, tool_name: str, response) -> Dict[str, Any]:
        result = super()._build_tool_execution_result(tool_name, response)
        if tool_name not in self._builtin_tool_names():
            return result

        payload = response.data if isinstance(getattr(response, "data", None), dict) else {}
        result["builtin_tool"] = True

        finished = payload.get("finished")
        if isinstance(finished, bool):
            result["finished"] = finished

        final_answer = payload.get("final_answer")
        if isinstance(final_answer, str):
            result["final_answer"] = final_answer

        return result

    def _trace_user_message(self, input_text: str) -> None:
        if self.trace_logger:
            self.trace_logger.log_event(
                "message_written",
                {"role": "user", "content": input_text},
            )

    def _prepare_execution(
        self,
        input_text: str,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[Dict[str, Any]]:
        tool_schemas = self._build_tool_schemas(structured_output)
        self._trace_user_message(input_text)
        self._render_agent_start(input_text)
        self._append_history_message(Message(input_text, "user"), allow_compact=False)
        return tool_schemas

    def _maybe_compact_history(
        self,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        _ = tool_schemas
        result = self.history_manager.maybe_compact(
            llm=self.llm,
            system_prompt=self._get_context_system_prompt(),
            latest_prompt_tokens=getattr(self, "_last_prompt_tokens", 0),
        )
        self._sync_history_token_count()
        self._estimated_next_prompt_tokens = self.history_manager.estimate_tokens(
            system_prompt=self._get_context_system_prompt(),
        )
        if result is not None:
            self._render_compaction_notice()

    @staticmethod
    def _extract_response_usage(response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = getattr(usage, "input_tokens", 0)

            completion_tokens = getattr(usage, "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = getattr(usage, "output_tokens", 0)

            total_tokens = getattr(usage, "total_tokens", None)
            if total_tokens is None:
                total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
            return int(prompt_tokens or 0), int(completion_tokens or 0), int(total_tokens or 0)

        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            prompt_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
            total_tokens = getattr(usage_metadata, "total_token_count", None)
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
            return prompt_tokens, completion_tokens, int(total_tokens or 0)

        return 0, 0, 0

    def _record_model_response(self, response: Any, current_step: int, state: _ExecutionState) -> Any:
        response_message = response.choices[0].message
        message_reasoning = extract_reasoning_payload(response_message)
        choice_reasoning = (
            extract_reasoning_payload(response.choices[0])
            if message_reasoning.content is None
            else None
        )
        reasoning_content = None
        reasoning_source = None

        prompt_tokens, completion_tokens, total_tokens = self._extract_response_usage(response)
        if total_tokens or prompt_tokens or completion_tokens:
            state.total_tokens += total_tokens
            state.total_prompt_tokens += prompt_tokens
            state.total_completion_tokens += completion_tokens
            self._total_tokens = state.total_tokens
            self._turn_prompt_tokens = state.total_prompt_tokens
            self._turn_completion_tokens = state.total_completion_tokens
            self._last_prompt_tokens = prompt_tokens
            self.history_manager.record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            self._sync_history_token_count()

        if message_reasoning.content is not None:
            reasoning_content = message_reasoning.content
            reasoning_source = f"message.{message_reasoning.source}"
        elif choice_reasoning and choice_reasoning.content is not None:
            reasoning_content = choice_reasoning.content
            reasoning_source = f"choice.{choice_reasoning.source}"

        state.last_reasoning_content = reasoning_content
        state.last_reasoning_source = reasoning_source

        if reasoning_content is not None:
            self._render_event(
                "model_output",
                {
                    "content": response_message.content or "",
                    "tool_calls": len(response_message.tool_calls) if response_message.tool_calls else 0,
                    "reasoning_content": reasoning_content,
                    "reasoning_source": reasoning_source,
                    "step": current_step,
                },
            )

        if self.trace_logger:
            trace_payload = {
                "content": response_message.content or "",
                "tool_calls": len(response_message.tool_calls) if response_message.tool_calls else 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": 0.0,
                },
            }
            if reasoning_content is not None:
                trace_payload["reasoning_content"] = reasoning_content
                trace_payload["reasoning_source"] = reasoning_source
            self.trace_logger.log_event("model_output", trace_payload, step=current_step)

        return response_message

    def _append_assistant_tool_call_message(self, response_message: Any) -> None:
        tool_calls = response_message.tool_calls or []
        assistant_message = self.history_manager.build_assistant_tool_call_message(
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
            content=response_message.content,
        )
        self._append_history_message(assistant_message, allow_compact=False)

    def _append_tool_message(
        self,
        tool_name: str,
        tool_call_id: str,
        result_content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        }
        if metadata:
            payload.update(metadata)
        self._append_history_message(
            Message(result_content, "tool", metadata=payload),
            allow_compact=False,
        )

    @staticmethod
    def _assistant_reasoning_metadata(
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if reasoning_content is not None:
            metadata["reasoning_content"] = reasoning_content
        if reasoning_source is not None:
            metadata["reasoning_source"] = reasoning_source
        return metadata

    def _append_final_history(
        self,
        final_answer: str,
        *,
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> None:
        assistant_metadata = self._assistant_reasoning_metadata(
            reasoning_content=reasoning_content,
            reasoning_source=reasoning_source,
        )
        assistant_kwargs = {"metadata": assistant_metadata} if assistant_metadata else {}
        self.add_message(Message(final_answer, "assistant", **assistant_kwargs))

    @staticmethod
    def _state_reasoning_kwargs(state: _ExecutionState) -> Dict[str, Optional[str]]:
        return {
            "reasoning_content": state.last_reasoning_content,
            "reasoning_source": state.last_reasoning_source,
        }

    def _finalize_trace_session(
        self,
        session_start_time: datetime,
        current_step: int,
        final_answer: str,
        *,
        status: str,
    ) -> None:
        if not self.trace_logger:
            return
        duration = (datetime.now() - session_start_time).total_seconds()
        self.trace_logger.log_event(
            "session_end",
            {
                "duration": duration,
                "total_steps": current_step,
                "final_answer": final_answer,
                "status": status,
            },
        )
        self.trace_logger.finalize()

    def _complete_local_turn(
        self,
        final_answer: str,
        session_start_time: datetime,
        state: _ExecutionState,
        *,
        status: str,
        include_reasoning: bool = True,
    ) -> str:
        reasoning_kwargs = self._state_reasoning_kwargs(state) if include_reasoning else {}
        self._append_final_history(final_answer, **reasoning_kwargs)
        self._finalize_trace_session(
            session_start_time,
            state.current_step,
            final_answer,
            status=status,
        )
        return final_answer

    @staticmethod
    def _timeout_final_answer() -> str:
        return "Sorry, I could not complete this task within the step limit."

    @staticmethod
    def _coerce_optional_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        return None

    def _response_unfinished_flag(self, response_message: Any) -> bool:
        candidate_values = []
        for attr_name in ("unfinished", "unfinish"):
            candidate_values.append(getattr(response_message, attr_name, None))

        additional_kwargs = getattr(response_message, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            for key in ("unfinished", "unfinish"):
                candidate_values.append(additional_kwargs.get(key))

        metadata = getattr(response_message, "metadata", None)
        if isinstance(metadata, dict):
            for key in ("unfinished", "unfinish"):
                candidate_values.append(metadata.get(key))

        for value in candidate_values:
            normalized = self._coerce_optional_bool(value)
            if normalized is not None:
                return normalized
        return False

    @staticmethod
    def _direct_response_text(
        text_content: str,
        *,
        fallback_text: str = "",
    ) -> str:
        final_answer = text_content.strip() or fallback_text.strip()
        return final_answer or "Sorry, I could not answer this question."

    def _resolve_no_tool_call_response(
        self,
        response_message: Any,
        text_content: str,
        *,
        structured_output: Optional[_StructuredOutputSpec] = None,
        fallback_text: str = "",
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        if structured_output is None and not self._response_unfinished_flag(response_message):
            final_answer = self._direct_response_text(text_content, fallback_text=fallback_text)
            self._render_direct_response(
                final_answer,
                reasoning_content=reasoning_content,
                reasoning_source=reasoning_source,
            )
            return False, final_answer, "success"

        if structured_output is None and text_content.strip():
            assistant_metadata = self._assistant_reasoning_metadata(
                reasoning_content=reasoning_content,
                reasoning_source=reasoning_source,
            )
            assistant_metadata["unfinished_response"] = True
            self._append_history_message(
                Message(
                    text_content,
                    "assistant",
                    metadata=assistant_metadata,
                ),
                allow_compact=False,
            )

        return True, None, None

    @staticmethod
    def _tool_call_arguments_by_id(tool_calls: List[Any]) -> Dict[str, Dict[str, Any]]:
        arguments_by_id: Dict[str, Dict[str, Any]] = {}
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except Exception:
                continue
            if isinstance(arguments, dict):
                arguments_by_id[tool_call.id] = arguments
        return arguments_by_id

    def _update_stagnation_state(
        self,
        tool_name: str,
        tool_call_id: str,
        result_content: str,
        current_step: int,
        state: _ExecutionState,
        *,
        tool_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        if tool_name == "Edit":
            if "[no textual diff]" in result_content:
                state.consecutive_no_diff_edits += 1
            else:
                state.consecutive_no_diff_edits = 0
        else:
            state.consecutive_no_diff_edits = 0

        if tool_name == "Bash":
            cmd = ""
            if isinstance(tool_arguments, dict):
                cmd = str(tool_arguments.get("command", ""))
            if "tests.py" in cmd or "pytest" in cmd or "unittest" in cmd:
                test_hash = hash(result_content)
                if test_hash == state.last_test_output_hash:
                    state.consecutive_same_tests += 1
                else:
                    state.consecutive_same_tests = 0
                    state.last_test_output_hash = test_hash

        if state.consecutive_no_diff_edits >= 3 or state.consecutive_same_tests >= 3:
            reason = (
                "3 consecutive Edit calls with no textual diff"
                if state.consecutive_no_diff_edits >= 3
                else "3 consecutive identical test results"
            )
            self._render_event("stagnation_detected", {"reason": reason, "step": current_step})
            state.stagnation_detected = True

    def _process_tool_results(
        self,
        tool_calls: List[Any],
        tool_results: List[tuple],
        current_step: int,
        state: _ExecutionState,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> Optional[str]:
        tool_call_arguments = self._tool_call_arguments_by_id(tool_calls)

        for tool_name, tool_call_id, result in tool_results:
            if result.get("finished"):
                final_answer = result["final_answer"]
                self._render_final_answer(
                    final_answer,
                    step=current_step,
                    **self._state_reasoning_kwargs(state),
                )
                return final_answer

            result_content = result.get("content", str(result))
            self._append_tool_message(
                tool_name,
                tool_call_id,
                result_content,
                metadata=result.get("metadata"),
            )
            self._update_stagnation_state(
                tool_name,
                tool_call_id,
                result_content,
                current_step,
                state,
                tool_arguments=tool_call_arguments.get(tool_call_id),
            )
            if state.stagnation_detected:
                break

        return None

    @staticmethod
    def _format_structured_output(arguments: Dict[str, Any]) -> str:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    def _build_structured_output_result(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_answer = self._format_structured_output(arguments)
        return {
            "content": final_answer,
            "finished": True,
            "final_answer": final_answer,
            "structured_output": arguments,
            "status": "success",
        }

    def _record_tool_execution_result(
        self,
        tool_name: str,
        tool_call_id: str,
        result: Dict[str, Any],
        *,
        step: int,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> tuple[str, str, Dict[str, Any]]:
        result_content = result["content"]
        result_status = result.get("status")
        if self._is_builtin_tool_name(tool_name):
            self._render_builtin_tool(
                tool_name,
                result_content,
                tool_call_id=tool_call_id,
                step=step,
            )
        elif self._is_structured_output_tool_name(tool_name, structured_output):
            self._render_control_tool(
                tool_name,
                result_content,
                tool_call_id=tool_call_id,
                step=step,
            )
        else:
            self._render_tool_result(
                result_content,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                step=step,
                status=result_status,
            )
        self._trace_tool_result(
            tool_name,
            tool_call_id,
            result_content,
            step=step,
            status=result_status,
        )
        return tool_name, tool_call_id, result

    def _execute_one_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
        *,
        current_step: int,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> tuple[str, str, Dict[str, Any]]:
        self._trace_tool_call(tool_name, tool_call_id, arguments, step=current_step)
        if not self._is_builtin_tool_name(tool_name) and not self._is_structured_output_tool_name(
            tool_name,
            structured_output,
        ):
            self._render_tool_call(
                tool_name,
                arguments,
                tool_call_id=tool_call_id,
                step=current_step,
            )
        if self._is_structured_output_tool_name(tool_name, structured_output):
            result = self._build_structured_output_result(arguments)
        else:
            result = self._execute_tool_call_result(tool_name, arguments)
        return self._record_tool_execution_result(
            tool_name,
            tool_call_id,
            result,
            step=current_step,
            structured_output=structured_output,
        )

    async def _aexecute_one_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
        *,
        current_step: int,
        on_tool_call: LifecycleHook = None,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> tuple[str, str, Dict[str, Any]]:
        self._trace_tool_call(tool_name, tool_call_id, arguments, step=current_step)
        await self._emit_tool_call_event(
            on_tool_call,
            tool_name,
            tool_call_id,
            arguments,
            current_step,
        )
        if not self._is_builtin_tool_name(tool_name) and not self._is_structured_output_tool_name(
            tool_name,
            structured_output,
        ):
            self._render_tool_call(
                tool_name,
                arguments,
                tool_call_id=tool_call_id,
                step=current_step,
            )
        if self._is_structured_output_tool_name(tool_name, structured_output):
            result = self._build_structured_output_result(arguments)
        else:
            result = await self._aexecute_tool_call_result(tool_name, arguments)
        return self._record_tool_execution_result(
            tool_name,
            tool_call_id,
            result,
            step=current_step,
            structured_output=structured_output,
        )

    def _execute_tools(
        self,
        tool_calls: List[Any],
        current_step: int,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[tuple]:
        results: List[tuple] = []
        invalid_finalizing_calls = self._invalid_finalizing_tool_calls(
            tool_calls,
            structured_output=structured_output,
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            usage_error = invalid_finalizing_calls.get(tool_call_id)
            if usage_error is not None:
                results.append(
                    self._tool_error_result(
                        tool_name,
                        tool_call_id,
                        f"Error: {usage_error}",
                        step=current_step,
                    )
                )
                continue

            tool_name, tool_call_id, arguments, error_content = self._decode_tool_call(tool_call)
            if error_content is not None:
                results.append(
                    self._tool_error_result(tool_name, tool_call_id, error_content, step=current_step)
                )
                continue

            results.append(
                self._execute_one_tool_call(
                    tool_name,
                    tool_call_id,
                    arguments,
                    current_step=current_step,
                    structured_output=structured_output,
                )
            )

        return results

    def run(self, input_text: str, **kwargs) -> str:
        session_start_time = datetime.now()

        try:
            final_answer = self._run_impl(input_text, session_start_time, **kwargs)
            self._session_metadata["total_steps"] = getattr(self, "_current_step", 0)
            self._session_metadata["total_tokens"] = getattr(self, "_total_tokens", 0)
            return final_answer
        except KeyboardInterrupt:
            self._console("\n⚠️ User interrupted, auto-saving session...")
            if self.session_store:
                try:
                    filepath = self.save_session("session-interrupted")
                    self._console(f"✅ Session saved: {filepath}")
                except Exception as exc:
                    self._console(f"❌ Save failed: {exc}")
            raise
        except Exception as exc:
            self._console(f"\n❌ Error: {exc}")
            if self.session_store:
                try:
                    filepath = self.save_session("session-error")
                    self._console(f"✅ Session saved: {filepath}")
                except Exception as save_error:
                    self._console(f"❌ Save failed: {save_error}")
            raise

    def _run_impl(self, input_text: str, session_start_time, **kwargs) -> str:
        start_step = self._normalize_start_step(kwargs.pop("start_step", 0))

        structured_output = self._extract_structured_output_spec(kwargs)
        tool_choice = kwargs.pop("tool_choice", self._tool_choice_for(structured_output))

        tool_schemas = self._prepare_execution(
            input_text,
            structured_output=structured_output,
        )
        state = self._create_execution_state(start_step)

        while self.max_steps <= 0 or state.current_step < self.max_steps:
            state.current_step += 1
            self._render_step_start(state.current_step)

            self._current_step = state.current_step
            self._maybe_compact_history()

            messages = self._build_messages()
            self._apply_builtin_tool_prompt(messages)
            if structured_output:
                self._apply_structured_output_prompt(messages, structured_output)
            self._before_model_call(messages, state.current_step)

            try:
                response = self.llm.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            except Exception as exc:
                self._render_llm_error(exc, step=state.current_step)
                if self.trace_logger:
                    self.trace_logger.log_event(
                        "error",
                        {"error_type": "LLM_ERROR", "message": str(exc)},
                        step=state.current_step,
                    )
                break

            response_message = self._record_model_response(response, state.current_step, state)
            tool_calls = response_message.tool_calls
            if not tool_calls:
                text_content = (response_message.content or "").strip()
                should_continue, final_answer, status = self._resolve_no_tool_call_response(
                    response_message,
                    text_content,
                    structured_output=structured_output,
                    reasoning_content=state.last_reasoning_content,
                    reasoning_source=state.last_reasoning_source,
                )
                if should_continue:
                    continue

                return self._complete_local_turn(
                    final_answer,
                    session_start_time,
                    state,
                    status=status,
                )

            self._append_assistant_tool_call_message(response_message)

            tool_results = self._execute_tools(
                tool_calls,
                state.current_step,
                structured_output=structured_output,
            )
            final_answer = self._process_tool_results(
                tool_calls,
                tool_results,
                state.current_step,
                state,
                structured_output=structured_output,
            )
            if final_answer is not None:
                return self._complete_local_turn(
                    final_answer,
                    session_start_time,
                    state,
                    status="success",
                )

            if state.stagnation_detected:
                break

        if not state.stagnation_detected:
            self._render_timeout()
        final_answer = self._timeout_final_answer()
        return self._complete_local_turn(
            final_answer,
            session_start_time,
            state,
            status="timeout",
            include_reasoning=False,
        )

    def _build_messages(self, input_text: Optional[str] = None) -> List[Dict[str, str]]:
        return self.history_manager.build_llm_messages(
            system_prompt=self._get_context_system_prompt(),
            latest_user_input=input_text,
        )

    def _before_model_call(self, messages: List[Dict[str, Any]], current_step: int) -> None:
        return None

    async def _abefore_model_call(self, messages: List[Dict[str, Any]], current_step: int) -> None:
        self._before_model_call(messages, current_step)

    def _build_tool_schemas(
        self,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[Dict[str, Any]]:
        schemas = super()._build_tool_schemas()
        if structured_output is not None:
            schemas.append(self._build_structured_output_tool_schema(structured_output))
        return schemas

    async def arun(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_step: LifecycleHook = None,
        on_tool_call: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs,
    ) -> str:
        session_start_time = datetime.now()
        start_step = self._normalize_start_step(kwargs.pop("start_step", 0))

        structured_output = self._extract_structured_output_spec(kwargs)
        tool_choice = kwargs.pop("tool_choice", self._tool_choice_for(structured_output))

        await self._emit_event(
            EventType.AGENT_START,
            on_start,
            input_text=input_text,
        )

        try:
            tool_schemas = self._prepare_execution(
                input_text,
                structured_output=structured_output,
            )
            state = self._create_execution_state(start_step)

            while self.max_steps <= 0 or state.current_step < self.max_steps:
                state.current_step += 1
                self._render_step_start(state.current_step)

                self._current_step = state.current_step
                self._maybe_compact_history()

                await self._emit_event(
                    EventType.STEP_START,
                    on_step,
                    step=state.current_step,
                )

                messages = self._build_messages()
                self._apply_builtin_tool_prompt(messages)
                if structured_output:
                    self._apply_structured_output_prompt(messages, structured_output)
                await self._abefore_model_call(messages, state.current_step)

                try:
                    response = await self.llm.ainvoke_with_tools(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice=tool_choice,
                        **kwargs,
                    )
                except Exception as exc:
                    self._render_llm_error(exc, step=state.current_step)
                    await self._emit_event(
                        EventType.AGENT_ERROR,
                        on_error,
                        error=str(exc),
                        step=state.current_step,
                    )
                    break

                response_message = self._record_model_response(response, state.current_step, state)
                tool_calls = response_message.tool_calls
                if not tool_calls:
                    text_content = (response_message.content or "").strip()
                    should_continue, final_answer, status = self._resolve_no_tool_call_response(
                        response_message,
                        text_content,
                        structured_output=structured_output,
                        reasoning_content=state.last_reasoning_content,
                        reasoning_source=state.last_reasoning_source,
                    )
                    if should_continue:
                        continue

                    await self._emit_event(
                        EventType.AGENT_FINISH,
                        on_finish,
                        result=final_answer,
                        total_steps=state.current_step,
                        total_tokens=state.total_tokens,
                        status=status,
                    )
                    return self._complete_local_turn(
                        final_answer,
                        session_start_time,
                        state,
                        status=status,
                    )

                self._append_assistant_tool_call_message(response_message)

                tool_results = await self._execute_tools_async(
                    tool_calls,
                    state.current_step,
                    on_tool_call,
                    structured_output=structured_output,
                )
                final_answer = self._process_tool_results(
                    tool_calls,
                    tool_results,
                    state.current_step,
                    state,
                    structured_output=structured_output,
                )
                if final_answer is not None:
                    await self._emit_event(
                        EventType.AGENT_FINISH,
                        on_finish,
                        result=final_answer,
                        total_steps=state.current_step,
                        total_tokens=state.total_tokens,
                    )
                    return self._complete_local_turn(
                        final_answer,
                        session_start_time,
                        state,
                        status="success",
                    )

                if state.stagnation_detected:
                    break

                await self._emit_event(
                    EventType.STEP_FINISH,
                    on_step,
                    step=state.current_step,
                    tool_calls=len(tool_calls),
                )

            if not state.stagnation_detected:
                self._render_timeout()
            final_answer = self._timeout_final_answer()

            await self._emit_event(
                EventType.AGENT_FINISH,
                on_finish,
                result=final_answer,
                total_steps=state.current_step,
                total_tokens=state.total_tokens,
                status="timeout",
            )

            return self._complete_local_turn(
                final_answer,
                session_start_time,
                state,
                status="timeout",
                include_reasoning=False,
            )
        except Exception as exc:
            await self._emit_event(
                EventType.AGENT_ERROR,
                on_error,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

    async def _execute_tools_async(
        self,
        tool_calls: List[Any],
        current_step: int,
        on_tool_call: LifecycleHook = None,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[tuple]:
        results: List[tuple] = []
        invalid_finalizing_calls = self._invalid_finalizing_tool_calls(
            tool_calls,
            structured_output=structured_output,
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            usage_error = invalid_finalizing_calls.get(tool_call_id)
            if usage_error is not None:
                results.append(
                    self._tool_error_result(
                        tool_name,
                        tool_call_id,
                        f"Error: {usage_error}",
                        step=current_step,
                    )
                )
                continue

            tool_name, tool_call_id, arguments, error_content = self._decode_tool_call(tool_call)
            if error_content is not None:
                results.append(
                    self._tool_error_result(tool_name, tool_call_id, error_content, step=current_step)
                )
                continue

            results.append(
                await self._aexecute_one_tool_call(
                    tool_name,
                    tool_call_id,
                    arguments,
                    current_step=current_step,
                    on_tool_call=on_tool_call,
                    structured_output=structured_output,
                )
            )

        return results

    async def arun_stream(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_step: LifecycleHook = None,
        on_tool_call: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs,
    ) -> AsyncGenerator[StreamEvent, None]:
        session_start_time = datetime.now()
        start_step = self._normalize_start_step(kwargs.pop("start_step", 0))

        structured_output = self._extract_structured_output_spec(kwargs)
        tool_choice = kwargs.pop("tool_choice", self._tool_choice_for(structured_output))

        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text,
        )

        await self._emit_event(EventType.AGENT_START, on_start, input_text=input_text)

        try:
            tool_schemas = self._prepare_execution(
                input_text,
                structured_output=structured_output,
            )
            state = self._create_execution_state(start_step)
            final_answer = None

            while self.max_steps <= 0 or state.current_step < self.max_steps:
                state.current_step += 1
                self._current_step = state.current_step
                self._maybe_compact_history()

                messages = self._build_messages()
                self._apply_builtin_tool_prompt(messages)
                if structured_output:
                    self._apply_structured_output_prompt(messages, structured_output)

                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    step=state.current_step,
                    max_steps=self.max_steps,
                )
                await self._emit_event(EventType.STEP_START, on_step, step=state.current_step)

                self._render_step_start(state.current_step)
                await self._abefore_model_call(messages, state.current_step)

                full_response = ""
                try:
                    async for chunk in self.llm.astream_invoke(messages, **kwargs):
                        full_response += chunk
                        yield StreamEvent.create(
                            StreamEventType.LLM_CHUNK,
                            self.name,
                            chunk=chunk,
                            step=state.current_step,
                        )
                        self._render_stream_chunk(chunk)

                    self._render_stream_newline()
                except Exception as exc:
                    error_msg = f"LLM call failed: {exc}"
                    self._render_agent_error(error_msg)
                    yield StreamEvent.create(
                        StreamEventType.ERROR,
                        self.name,
                        error=error_msg,
                        step=state.current_step,
                    )
                    await self._emit_event(EventType.AGENT_ERROR, on_error, error=error_msg)
                    break

                try:
                    response = self.llm.invoke_with_tools(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice=tool_choice,
                        **kwargs,
                    )
                    response_message = self._record_model_response(
                        response,
                        state.current_step,
                        state,
                    )
                    tool_calls = response_message.tool_calls

                    if not tool_calls:
                        text_content = (response_message.content or "").strip()
                        should_continue, final_answer, status = self._resolve_no_tool_call_response(
                            response_message,
                            text_content,
                            structured_output=structured_output,
                            fallback_text=full_response.strip(),
                            reasoning_content=state.last_reasoning_content,
                            reasoning_source=state.last_reasoning_source,
                        )
                        if should_continue:
                            continue

                        yield StreamEvent.create(
                            StreamEventType.AGENT_FINISH,
                            self.name,
                            result=final_answer,
                            total_steps=state.current_step,
                            total_tokens=state.total_tokens,
                            status=status,
                        )
                        await self._emit_event(
                            EventType.AGENT_FINISH,
                            on_finish,
                            result=final_answer,
                            total_steps=state.current_step,
                            total_tokens=state.total_tokens,
                            status=status,
                        )
                        self._complete_local_turn(
                            final_answer,
                            session_start_time,
                            state,
                            status=status,
                        )
                        return

                    self._append_assistant_tool_call_message(response_message)
                    tool_results = await self._execute_tools_async_stream(
                        tool_calls,
                        state.current_step,
                        on_tool_call,
                        structured_output=structured_output,
                    )

                    tool_call_arguments = self._tool_call_arguments_by_id(tool_calls)
                    for tool_name, tool_call_id, result_dict in tool_results:
                        yield StreamEvent.create(
                            StreamEventType.TOOL_CALL_FINISH,
                            self.name,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            result=result_dict["content"],
                            step=state.current_step,
                        )

                        if result_dict.get("finished") and (
                            tool_name == FINISH_TOOL_NAME
                            or (structured_output and tool_name == structured_output.name)
                        ):
                            final_answer = result_dict.get("final_answer", result_dict["content"])
                            yield StreamEvent.create(
                                StreamEventType.AGENT_FINISH,
                                self.name,
                                result=final_answer,
                                total_steps=state.current_step,
                                total_tokens=state.total_tokens,
                            )
                            await self._emit_event(
                                EventType.AGENT_FINISH,
                                on_finish,
                                result=final_answer,
                                total_steps=state.current_step,
                                total_tokens=state.total_tokens,
                            )
                            self._complete_local_turn(
                                final_answer,
                                session_start_time,
                                state,
                                status="success",
                            )
                            return

                        self._append_tool_message(
                            tool_name,
                            tool_call_id,
                            result_dict["content"],
                            metadata=result_dict.get("metadata"),
                        )
                        self._update_stagnation_state(
                            tool_name,
                            tool_call_id,
                            result_dict["content"],
                            state.current_step,
                            state,
                            tool_arguments=tool_call_arguments.get(tool_call_id),
                        )
                        if state.stagnation_detected:
                            break

                    if state.stagnation_detected:
                        break

                    yield StreamEvent.create(
                        StreamEventType.STEP_FINISH,
                        self.name,
                        step=state.current_step,
                    )
                    await self._emit_event(
                        EventType.STEP_FINISH,
                        on_step,
                        step=state.current_step,
                        tool_calls=len(tool_calls),
                    )
                except Exception as exc:
                    error_msg = f"Tool execution failed: {exc}"
                    self._render_agent_error(error_msg)
                    yield StreamEvent.create(
                        StreamEventType.ERROR,
                        self.name,
                        error=error_msg,
                        step=state.current_step,
                    )
                    await self._emit_event(EventType.AGENT_ERROR, on_error, error=error_msg)
                    break

            if not final_answer:
                if not state.stagnation_detected:
                    self._render_timeout()
                final_answer = self._timeout_final_answer()
                yield StreamEvent.create(
                    StreamEventType.AGENT_FINISH,
                    self.name,
                    result=final_answer,
                    total_steps=state.current_step,
                    total_tokens=state.total_tokens,
                    max_steps_reached=not state.stagnation_detected,
                )
                await self._emit_event(
                    EventType.AGENT_FINISH,
                    on_finish,
                    result=final_answer,
                    total_steps=state.current_step,
                    total_tokens=state.total_tokens,
                    status="timeout",
                )
                self._complete_local_turn(
                    final_answer,
                    session_start_time,
                    state,
                    status="timeout",
                    include_reasoning=False,
                )
        except Exception as exc:
            error_msg = f"Agent execution failed: {exc}"
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=error_msg,
                error_type=type(exc).__name__,
            )
            await self._emit_event(EventType.AGENT_ERROR, on_error, error=error_msg)
            raise

    async def _execute_tools_async_stream(
        self,
        tool_calls: List[Any],
        current_step: int,
        on_tool_call: LifecycleHook = None,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[tuple]:
        return await self._execute_tools_async(
            tool_calls,
            current_step,
            on_tool_call,
            structured_output=structured_output,
        )
