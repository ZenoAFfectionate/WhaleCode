"""ReAct agent built on top of tool-calling chat models."""

import json
import asyncio
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.lifecycle import EventType, LifecycleHook
from ..core.reasoning import extract_reasoning_payload
from ..core.streaming import StreamEvent, StreamEventType
from ..tools.registry import ToolRegistry
from ..tools.response import ToolStatus
from ..context.compactor import ContextCompactor


DEFAULT_REACT_SYSTEM_PROMPT = """You are an AI assistant with reasoning and action capabilities.

Use tools whenever they are helpful for gathering information or performing work.
You may call tools multiple times. If you already have enough information and no
more tools are needed, respond directly with the final answer.
"""

STRUCTURED_OUTPUT_TOOL_NAME = "StructuredOutput"
DEFAULT_STRUCTURED_OUTPUT_DESCRIPTION = (
    "Return the final answer in the required structured format. "
    "Call this tool exactly once after all other tool usage is complete."
)


@dataclass
class _ExecutionState:
    current_step: int
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    no_tool_call_retries: int = 0
    max_no_tool_call_retries: int = 2
    is_retry: bool = False
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
        max_steps: int = 5
    ):
        """Initialize the agent with an optional tool registry."""
        super().__init__(
            name,
            llm,
            system_prompt or DEFAULT_REACT_SYSTEM_PROMPT,
            config,
            tool_registry=tool_registry or ToolRegistry()
        )

        # 上下文压缩引擎
        self._compactor = ContextCompactor(
            config=self.config, token_counter=self.token_counter
        )

        self.max_steps = max_steps

    # ==================== Console / render hooks ====================

    def _console(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        """Default console sink used by render hooks.

        Subclasses can override this to integrate richer UIs without touching
        the core ReAct control flow.
        """
        print(message, end=end, flush=flush)

    def _render_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Structured render event hook.

        Subclasses can override this single method to consume structured agent
        UI events without parsing raw strings from stdout.
        """
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
        elif event_type == "control_tool":
            self._console(f"🔧 {payload.get('tool_name')}: {payload.get('result_content', '')}")
        elif event_type == "tool_call":
            self._console(f"🎬 Tool call: {payload.get('tool_name')}({payload.get('arguments', {})})")
        elif event_type == "tool_result":
            result_content = str(payload.get('result_content', ''))
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
    
    def add_tool(self, tool):
        """添加工具到工具注册表"""
        self.tool_registry.register_tool(tool)

    def _parallel_user_tool_execution_enabled(self) -> bool:
        """Whether sync run() should execute user tools concurrently."""
        return False

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

    def _split_tool_calls(
        self,
        tool_calls: List[Any],
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> tuple[List[Any], List[Any]]:
        """Split tool calls into control and user-defined groups."""
        control_calls: List[Any] = []
        user_calls: List[Any] = []
        control_name = structured_output.name if structured_output else None

        for tool_call in tool_calls:
            target = (
                control_calls
                if control_name and tool_call.function.name == control_name
                else user_calls
            )
            target.append(tool_call)

        return control_calls, user_calls

    def _decode_tool_call(self, tool_call: Any) -> tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
        """Decode a single tool call payload into structured arguments."""
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
        return (tool_name, tool_call_id, {"content": error_content})

    def _create_execution_state(self, start_step: int) -> _ExecutionState:
        return _ExecutionState(current_step=start_step)

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

    def _build_structured_output_tool_schema(
        self,
        spec: _StructuredOutputSpec,
    ) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": deepcopy(spec.schema),
            },
        }

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
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        messages = self._build_messages(input_text)
        if structured_output:
            self._apply_structured_output_prompt(messages, structured_output)
        tool_schemas = self._build_tool_schemas(structured_output)
        self._trace_user_message(input_text)
        self._render_agent_start(input_text)
        return messages, tool_schemas

    def _maybe_compact_messages(self, messages: List[Dict[str, Any]]) -> None:
        if self._compactor and self.config.compact_enabled:
            self._compactor.micro_compact(messages)
            if self._compactor.should_compact(
                messages,
                latest_prompt_tokens=getattr(self, "_last_prompt_tokens", 0),
            ):
                self._render_compaction_notice()
                messages[:] = self._compactor.auto_compact(messages, self.llm)
                self._last_prompt_tokens = self._compactor.estimate_tokens(messages)

    def _record_model_response(self, response: Any, current_step: int, state: _ExecutionState) -> Any:
        response_message = response.choices[0].message
        message_reasoning = extract_reasoning_payload(response_message)
        choice_reasoning = extract_reasoning_payload(response.choices[0]) if message_reasoning.content is None else None
        reasoning_content = None
        reasoning_source = None

        if response.usage:
            state.total_tokens += response.usage.total_tokens
            state.total_prompt_tokens += getattr(response.usage, "prompt_tokens", 0)
            state.total_completion_tokens += getattr(response.usage, "completion_tokens", 0)
            self._total_tokens = state.total_tokens
            self._turn_prompt_tokens = state.total_prompt_tokens
            self._turn_completion_tokens = state.total_completion_tokens
            self._last_prompt_tokens = getattr(response.usage, "prompt_tokens", 0)

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
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                    "cost": 0.0,
                },
            }
            if reasoning_content is not None:
                trace_payload["reasoning_content"] = reasoning_content
                trace_payload["reasoning_source"] = reasoning_source

            self.trace_logger.log_event(
                "model_output",
                trace_payload,
                step=current_step,
            )

        return response_message

    def _append_assistant_tool_call_message(self, messages: List[Dict[str, Any]], response_message: Any) -> None:
        tool_calls = response_message.tool_calls or []
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
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
            }
        )

    @staticmethod
    def _append_tool_message(messages: List[Dict[str, Any]], tool_call_id: str, result_content: str) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content,
            }
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
        input_text: str,
        final_answer: str,
        *,
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> None:
        self.add_message(Message(input_text, "user"))
        assistant_metadata = self._assistant_reasoning_metadata(
            reasoning_content=reasoning_content,
            reasoning_source=reasoning_source,
        )
        assistant_kwargs = {"metadata": assistant_metadata} if assistant_metadata else {}
        self.add_message(Message(final_answer, "assistant", **assistant_kwargs))

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

    def _should_retry_without_tool_call(
        self,
        messages: List[Dict[str, Any]],
        text_content: str,
        state: _ExecutionState,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> bool:
        if structured_output is None and (
            text_content or state.no_tool_call_retries >= state.max_no_tool_call_retries
        ):
            return False

        if structured_output is not None and state.no_tool_call_retries >= state.max_no_tool_call_retries:
            return False

        state.no_tool_call_retries += 1
        state.is_retry = True
        if structured_output is None:
            reminder = (
                "You have access to tools - please use them to complete the task. "
                "Do not respond with text only. Call a tool now."
            )
        else:
            reminder = (
                f"Structured output is required. Do not answer with plain text. "
                f"When you are ready to finish, call the {structured_output.name} tool "
                "exactly once with arguments that match its schema."
            )
        messages.append(
            {
                "role": "user",
                "content": reminder,
            }
        )
        return True

    def _resolve_no_tool_call_response(
        self,
        messages: List[Dict[str, Any]],
        text_content: str,
        state: _ExecutionState,
        *,
        structured_output: Optional[_StructuredOutputSpec] = None,
        fallback_text: str = "",
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        if self._should_retry_without_tool_call(
            messages,
            text_content,
            state,
            structured_output=structured_output,
        ):
            return True, None, None

        if structured_output is not None:
            error_message = (
                f"Sorry, structured output was requested, but the model did not call "
                f"{structured_output.name} before finishing."
            )
            self._render_agent_error(error_message)
            return False, error_message, "error"

        final_answer = text_content or fallback_text or "Sorry, I cannot answer this question."
        self._render_direct_response(
            final_answer,
            reasoning_content=reasoning_content,
            reasoning_source=reasoning_source,
        )
        return False, final_answer, "success"

    @staticmethod
    def _extract_tool_command(tool_calls: List[Any], tool_call_id: str) -> str:
        for tool_call in tool_calls:
            if tool_call.id != tool_call_id:
                continue
            try:
                arguments = json.loads(tool_call.function.arguments)
            except Exception:
                return ""
            return arguments.get("command", "")
        return ""

    def _update_stagnation_state(
        self,
        tool_name: str,
        tool_call_id: str,
        result_content: str,
        tool_calls: List[Any],
        current_step: int,
        state: _ExecutionState,
    ) -> None:
        if tool_name == "Edit":
            if "[no textual diff]" in result_content:
                state.consecutive_no_diff_edits += 1
            else:
                state.consecutive_no_diff_edits = 0
        else:
            state.consecutive_no_diff_edits = 0

        if tool_name == "Bash":
            cmd = self._extract_tool_command(tool_calls, tool_call_id)
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
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
        tool_results: List[tuple],
        current_step: int,
        state: _ExecutionState,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> Optional[str]:
        for tool_name, tool_call_id, result in tool_results:
            if structured_output and tool_name == structured_output.name and result.get("finished"):
                final_answer = result["final_answer"]
                self._render_final_answer(
                    final_answer,
                    step=current_step,
                    reasoning_content=state.last_reasoning_content,
                    reasoning_source=state.last_reasoning_source,
                )
                return final_answer

            result_content = result.get("content", str(result))
            self._append_tool_message(messages, tool_call_id, result_content)
            self._update_stagnation_state(
                tool_name,
                tool_call_id,
                result_content,
                tool_calls,
                current_step,
                state,
            )
            if state.stagnation_detected:
                break

        return None

    def _control_tool_usage_error(
        self,
        tool_name: str,
        tool_call_id: str,
        message: str,
        *,
        step: int,
    ) -> tuple[str, str, Dict[str, str]]:
        return self._tool_error_result(tool_name, tool_call_id, f"Error: {message}", step=step)

    @staticmethod
    def _format_structured_output(arguments: Dict[str, Any]) -> str:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    def _handle_control_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        structured_output: _StructuredOutputSpec,
    ) -> Dict[str, Any]:
        final_answer = self._format_structured_output(arguments)
        return {
            "content": final_answer,
            "finished": True,
            "final_answer": final_answer,
            "structured_output": arguments,
        }

    def _execute_tools(
        self,
        tool_calls: List[Any],
        current_step: int,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[tuple]:
        """Execute control tools locally and user tools via the shared executor."""
        results = []
        control_calls, user_calls = self._split_tool_calls(tool_calls, structured_output)

        if control_calls:
            if len(control_calls) > 1:
                for tc in control_calls:
                    tool_name, tool_call_id, _, _ = self._decode_tool_call(tc)
                    results.append(
                        self._control_tool_usage_error(
                            tool_name,
                            tool_call_id,
                            f"{structured_output.name} must be called at most once per response.",
                            step=current_step,
                        )
                    )
                return results

            if user_calls:
                tc = control_calls[0]
                tool_name, tool_call_id, _, _ = self._decode_tool_call(tc)
                results.append(
                    self._control_tool_usage_error(
                        tool_name,
                        tool_call_id,
                        f"{structured_output.name} must be called alone after all other tool work is complete.",
                        step=current_step,
                    )
                )
            else:
                tc = control_calls[0]
                tool_name, tool_call_id, arguments, error_content = self._decode_tool_call(tc)
                if error_content is not None:
                    results.append(
                        self._tool_error_result(
                            tool_name,
                            tool_call_id,
                            error_content,
                            step=current_step,
                        )
                    )
                else:
                    self._trace_tool_call(tool_name, tool_call_id, arguments, step=current_step)
                    result = self._handle_control_tool(
                        tool_name,
                        arguments,
                        structured_output=structured_output,
                    )
                    self._render_control_tool(
                        tool_name,
                        result["content"],
                        tool_call_id=tool_call_id,
                        step=current_step,
                    )
                    self._trace_tool_result(
                        tool_name,
                        tool_call_id,
                        result["content"],
                        step=current_step,
                        status="success",
                    )
                    results.append((tool_name, tool_call_id, result))
                    return results

        prepared_user_calls = []
        for tc in user_calls:
            tool_name, tool_call_id, arguments, error_content = self._decode_tool_call(tc)
            if error_content is not None:
                results.append(
                    self._tool_error_result(
                        tool_name,
                        tool_call_id,
                        error_content,
                        step=current_step,
                    )
                )
                continue

            self._trace_tool_call(tool_name, tool_call_id, arguments, step=current_step)

            self._render_tool_call(
                tool_name,
                arguments,
                tool_call_id=tool_call_id,
                step=current_step,
            )
            prepared_user_calls.append((tool_name, tool_call_id, arguments))

        if not prepared_user_calls:
            return results

        max_concurrent = getattr(self.config, "max_concurrent_tools", 3)
        run_in_parallel = self._parallel_user_tool_execution_enabled() and len(prepared_user_calls) > 1

        if run_in_parallel:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_items = [
                    (
                        tool_name,
                        tool_call_id,
                        executor.submit(self._execute_tool_call, tool_name, arguments),
                    )
                    for tool_name, tool_call_id, arguments in prepared_user_calls
                ]

                for tool_name, tool_call_id, future in future_items:
                    try:
                        result_content = future.result()
                    except Exception as exc:
                        result_content = f"❌ Tool execution failed: {exc}"

                    self._trace_tool_result(
                        tool_name,
                        tool_call_id,
                        result_content,
                        step=current_step,
                    )

                    self._render_tool_result(
                        result_content,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        step=current_step,
                    )
                    results.append((tool_name, tool_call_id, {"content": result_content}))
        else:
            for tool_name, tool_call_id, arguments in prepared_user_calls:
                result_content = self._execute_tool_call(tool_name, arguments)

                self._trace_tool_result(
                    tool_name,
                    tool_call_id,
                    result_content,
                    step=current_step,
                )

                self._render_tool_result(
                    result_content,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    step=current_step,
                )
                results.append((tool_name, tool_call_id, {"content": result_content}))

        return results
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 ReAct Agent

        Args:
            input_text: 用户问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        session_start_time = datetime.now()

        try:
            # 执行主逻辑
            final_answer = self._run_impl(input_text, session_start_time, **kwargs)

            # 更新元数据
            self._session_metadata["total_steps"] = getattr(self, '_current_step', 0)
            self._session_metadata["total_tokens"] = getattr(self, '_total_tokens', 0)

            return final_answer

        except KeyboardInterrupt:
            # Ctrl+C 时自动保存
            self._console("\n⚠️ User interrupted, auto-saving session...")
            if self.session_store:
                try:
                    filepath = self.save_session("session-interrupted")
                    self._console(f"✅ Session saved: {filepath}")
                except Exception as e:
                    self._console(f"❌ Save failed: {e}")
            raise

        except Exception as e:
            # 错误时也尝试保存
            self._console(f"\n❌ Error: {e}")
            if self.session_store:
                try:
                    filepath = self.save_session("session-error")
                    self._console(f"✅ Session saved: {filepath}")
                except Exception as save_error:
                    self._console(f"❌ Save failed: {save_error}")
            raise

    def _run_impl(self, input_text: str, session_start_time, **kwargs) -> str:
        start_step = int(kwargs.pop("start_step", 0) or 0)
        if start_step < 0:
            start_step = 0

        structured_output = self._extract_structured_output_spec(kwargs)
        tool_choice = kwargs.pop("tool_choice", self._tool_choice_for(structured_output))

        messages, tool_schemas = self._prepare_execution(
            input_text,
            structured_output=structured_output,
        )
        state = self._create_execution_state(start_step)

        while self.max_steps <= 0 or state.current_step < self.max_steps:
            if not state.is_retry:
                state.current_step += 1
                self._render_step_start(state.current_step)
            state.is_retry = False

            self._current_step = state.current_step
            self._maybe_compact_messages(messages)

            self._before_model_call(messages, state.current_step)

            try:
                response = self.llm.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            except Exception as e:
                self._render_llm_error(e, step=state.current_step)
                if self.trace_logger:
                    self.trace_logger.log_event(
                        "error",
                        {"error_type": "LLM_ERROR", "message": str(e)},
                        step=state.current_step
                    )
                break

            response_message = self._record_model_response(response, state.current_step, state)
            tool_calls = response_message.tool_calls
            if not tool_calls:
                text_content = (response_message.content or "").strip()
                should_continue, final_answer, status = self._resolve_no_tool_call_response(
                    messages,
                    text_content,
                    state,
                    structured_output=structured_output,
                    reasoning_content=state.last_reasoning_content,
                    reasoning_source=state.last_reasoning_source,
                )
                if should_continue:
                    continue

                self._append_final_history(
                    input_text,
                    final_answer,
                    reasoning_content=state.last_reasoning_content,
                    reasoning_source=state.last_reasoning_source,
                )
                self._finalize_trace_session(
                    session_start_time,
                    state.current_step,
                    final_answer,
                    status=status,
                )
                return final_answer

            state.no_tool_call_retries = 0
            self._append_assistant_tool_call_message(messages, response_message)

            tool_results = self._execute_tools(
                tool_calls,
                state.current_step,
                structured_output=structured_output,
            )
            final_answer = self._process_tool_results(
                messages,
                tool_calls,
                tool_results,
                state.current_step,
                state,
                structured_output=structured_output,
            )
            if final_answer is not None:
                self._append_final_history(
                    input_text,
                    final_answer,
                    reasoning_content=state.last_reasoning_content,
                    reasoning_source=state.last_reasoning_source,
                )
                self._finalize_trace_session(
                    session_start_time,
                    state.current_step,
                    final_answer,
                    status="success",
                )
                return final_answer

            if state.stagnation_detected:
                break

        if not state.stagnation_detected:
            self._render_timeout()
        final_answer = "Sorry, I could not complete this task within the step limit."
        self._append_final_history(input_text, final_answer)
        self._finalize_trace_session(
            session_start_time,
            state.current_step,
            final_answer,
            status="timeout",
        )
        return final_answer

    def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        """Build the base message list for a single run."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": input_text})
        return messages

    def _before_model_call(self, messages: List[Dict[str, Any]], current_step: int) -> None:
        """Hook for subclasses to inject context before each model call."""
        return None

    async def _abefore_model_call(self, messages: List[Dict[str, Any]], current_step: int) -> None:
        """Async wrapper for the pre-model-call hook."""
        self._before_model_call(messages, current_step)

    def _build_tool_schemas(
        self,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[Dict[str, Any]]:
        """Build user tool schemas and an optional structured-output tool."""
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
        **kwargs
    ) -> str:
        """Run the ReAct agent asynchronously."""
        session_start_time = datetime.now()
        start_step = int(kwargs.pop("start_step", 0) or 0)
        if start_step < 0:
            start_step = 0
        structured_output = self._extract_structured_output_spec(kwargs)
        tool_choice = kwargs.pop("tool_choice", self._tool_choice_for(structured_output))

        await self._emit_event(
            EventType.AGENT_START,
            on_start,
            input_text=input_text,
        )

        try:
            messages, tool_schemas = self._prepare_execution(
                input_text,
                structured_output=structured_output,
            )
            state = self._create_execution_state(start_step)

            while self.max_steps <= 0 or state.current_step < self.max_steps:
                if not state.is_retry:
                    state.current_step += 1
                    self._render_step_start(state.current_step)
                state.is_retry = False

                self._current_step = state.current_step
                self._maybe_compact_messages(messages)

                await self._emit_event(
                    EventType.STEP_START,
                    on_step,
                    step=state.current_step,
                )

                await self._abefore_model_call(messages, state.current_step)

                try:
                    response = await self.llm.ainvoke_with_tools(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice=tool_choice,
                        **kwargs,
                    )
                except Exception as e:
                    self._render_llm_error(e, step=state.current_step)
                    await self._emit_event(
                        EventType.AGENT_ERROR,
                        on_error,
                        error=str(e),
                        step=state.current_step,
                    )
                    break

                response_message = self._record_model_response(response, state.current_step, state)
                tool_calls = response_message.tool_calls
                if not tool_calls:
                    text_content = (response_message.content or "").strip()
                    should_continue, final_answer, status = self._resolve_no_tool_call_response(
                        messages,
                        text_content,
                        state,
                        structured_output=structured_output,
                        reasoning_content=state.last_reasoning_content,
                        reasoning_source=state.last_reasoning_source,
                    )
                    if should_continue:
                        continue

                    self._append_final_history(
                        input_text,
                        final_answer,
                        reasoning_content=state.last_reasoning_content,
                        reasoning_source=state.last_reasoning_source,
                    )
                    await self._emit_event(
                        EventType.AGENT_FINISH,
                        on_finish,
                        result=final_answer,
                        total_steps=state.current_step,
                        total_tokens=state.total_tokens,
                        status=status,
                    )
                    self._finalize_trace_session(
                        session_start_time,
                        state.current_step,
                        final_answer,
                        status=status,
                    )
                    return final_answer

                state.no_tool_call_retries = 0
                self._append_assistant_tool_call_message(messages, response_message)

                tool_results = await self._execute_tools_async(
                    tool_calls,
                    state.current_step,
                    on_tool_call,
                    structured_output=structured_output,
                )

                final_answer = self._process_tool_results(
                    messages,
                    tool_calls,
                    tool_results,
                    state.current_step,
                    state,
                    structured_output=structured_output,
                )
                if final_answer is not None:
                    self._append_final_history(
                        input_text,
                        final_answer,
                        reasoning_content=state.last_reasoning_content,
                        reasoning_source=state.last_reasoning_source,
                    )
                    await self._emit_event(
                        EventType.AGENT_FINISH,
                        on_finish,
                        result=final_answer,
                        total_steps=state.current_step,
                        total_tokens=state.total_tokens,
                    )
                    self._finalize_trace_session(
                        session_start_time,
                        state.current_step,
                        final_answer,
                        status="success",
                    )
                    return final_answer

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
            final_answer = "Sorry, I could not complete this task within the step limit."
            self._append_final_history(input_text, final_answer)

            await self._emit_event(
                EventType.AGENT_FINISH,
                on_finish,
                result=final_answer,
                total_steps=state.current_step,
                total_tokens=state.total_tokens,
                status="timeout",
            )

            self._finalize_trace_session(
                session_start_time,
                state.current_step,
                final_answer,
                status="timeout",
            )
            return final_answer

        except Exception as e:
            await self._emit_event(
                EventType.AGENT_ERROR,
                on_error,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
    async def _execute_tools_async(
        self,
        tool_calls: List[Any],
        current_step: int,
        on_tool_call: LifecycleHook = None,
        structured_output: Optional[_StructuredOutputSpec] = None,
    ) -> List[tuple]:
        """Execute control tools locally and user tools concurrently."""
        results = []
        control_calls, user_calls = self._split_tool_calls(tool_calls, structured_output)

        if control_calls:
            if len(control_calls) > 1:
                for tc in control_calls:
                    tool_name, tool_call_id, _, _ = self._decode_tool_call(tc)
                    results.append(
                        self._control_tool_usage_error(
                            tool_name,
                            tool_call_id,
                            f"{structured_output.name} must be called at most once per response.",
                            step=current_step,
                        )
                    )
                return results

            if user_calls:
                tc = control_calls[0]
                tool_name, tool_call_id, _, _ = self._decode_tool_call(tc)
                results.append(
                    self._control_tool_usage_error(
                        tool_name,
                        tool_call_id,
                        f"{structured_output.name} must be called alone after all other tool work is complete.",
                        step=current_step,
                    )
                )
            else:
                tc = control_calls[0]
                tool_name, tool_call_id, arguments, error_content = self._decode_tool_call(tc)
                if error_content is not None:
                    results.append(
                        self._tool_error_result(
                            tool_name,
                            tool_call_id,
                            error_content,
                            step=current_step,
                        )
                    )
                else:
                    self._trace_tool_call(tool_name, tool_call_id, arguments, step=current_step)
                    await self._emit_tool_call_event(
                        on_tool_call,
                        tool_name,
                        tool_call_id,
                        arguments,
                        current_step,
                    )
                    result = self._handle_control_tool(
                        tool_name,
                        arguments,
                        structured_output=structured_output,
                    )
                    self._render_control_tool(
                        tool_name,
                        result["content"],
                        tool_call_id=tool_call_id,
                        step=current_step,
                    )
                    self._trace_tool_result(
                        tool_name,
                        tool_call_id,
                        result["content"],
                        step=current_step,
                        status="success",
                    )
                    results.append((tool_name, tool_call_id, result))
                    return results

        if user_calls:
            max_concurrent = getattr(self.config, "max_concurrent_tools", 3)
            semaphore = asyncio.Semaphore(max_concurrent)

            async def execute_one(tc):
                async with semaphore:
                    tool_name, tool_call_id, arguments, error_content = self._decode_tool_call(tc)
                    if error_content is not None:
                        return self._tool_error_result(
                            tool_name,
                            tool_call_id,
                            error_content,
                            step=current_step,
                        )

                    self._trace_tool_call(tool_name, tool_call_id, arguments, step=current_step)
                    await self._emit_tool_call_event(
                        on_tool_call,
                        tool_name,
                        tool_call_id,
                        arguments,
                        current_step,
                    )
                    self._render_tool_call(
                        tool_name,
                        arguments,
                        tool_call_id=tool_call_id,
                        step=current_step,
                    )

                    tool_response = await self._aexecute_tool_response(tool_name, arguments)
                    result_content = self._format_tool_response_text(tool_name, tool_response)
                    result_status = "error" if tool_response.status == ToolStatus.ERROR else "success"

                    self._trace_tool_result(
                        tool_name,
                        tool_call_id,
                        result_content,
                        step=current_step,
                        status=result_status,
                    )
                    self._render_tool_result(
                        result_content,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        status=result_status,
                        step=current_step,
                    )
                    return (tool_name, tool_call_id, {"content": result_content})

            user_results = await asyncio.gather(*[execute_one(tc) for tc in user_calls])
            results.extend(user_results)

        return results
    async def arun_stream(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_step: LifecycleHook = None,
        on_tool_call: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run the ReAct agent with streamed assistant text events."""
        session_start_time = datetime.now()
        start_step = int(kwargs.pop("start_step", 0) or 0)
        if start_step < 0:
            start_step = 0
        structured_output = self._extract_structured_output_spec(kwargs)
        tool_choice = kwargs.pop("tool_choice", self._tool_choice_for(structured_output))

        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text,
        )

        await self._emit_event(EventType.AGENT_START, on_start, input_text=input_text)

        try:
            messages, tool_schemas = self._prepare_execution(
                input_text,
                structured_output=structured_output,
            )
            state = self._create_execution_state(start_step)
            final_answer = None

            while self.max_steps <= 0 or state.current_step < self.max_steps:
                if not state.is_retry:
                    state.current_step += 1
                state.is_retry = False
                self._current_step = state.current_step
                self._maybe_compact_messages(messages)

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
                except Exception as e:
                    error_msg = f"LLM call failed: {str(e)}"
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
                            messages,
                            text_content,
                            state,
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
                        self._append_final_history(
                            input_text,
                            final_answer,
                            reasoning_content=state.last_reasoning_content,
                            reasoning_source=state.last_reasoning_source,
                        )
                        self._finalize_trace_session(
                            session_start_time,
                            state.current_step,
                            final_answer,
                            status=status,
                        )
                        return

                    state.no_tool_call_retries = 0
                    self._append_assistant_tool_call_message(messages, response_message)
                    tool_results = await self._execute_tools_async_stream(
                        tool_calls,
                        state.current_step,
                        on_tool_call,
                        structured_output=structured_output,
                    )

                    for tool_name, tool_call_id, result_dict in tool_results:
                        yield StreamEvent.create(
                            StreamEventType.TOOL_CALL_FINISH,
                            self.name,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            result=result_dict["content"],
                            step=state.current_step,
                        )

                        if structured_output and tool_name == structured_output.name and result_dict.get("finished"):
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
                            self._append_final_history(
                                input_text,
                                final_answer,
                                reasoning_content=state.last_reasoning_content,
                                reasoning_source=state.last_reasoning_source,
                            )
                            self._finalize_trace_session(
                                session_start_time,
                                state.current_step,
                                final_answer,
                                status="success",
                            )
                            return

                        self._append_tool_message(messages, tool_call_id, result_dict["content"])
                        self._update_stagnation_state(
                            tool_name,
                            tool_call_id,
                            result_dict["content"],
                            tool_calls,
                            state.current_step,
                            state,
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

                except Exception as e:
                    error_msg = f"Tool execution failed: {str(e)}"
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
                final_answer = "Sorry, I could not complete this task within the step limit."
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
                self._append_final_history(input_text, final_answer)
                self._finalize_trace_session(
                    session_start_time,
                    state.current_step,
                    final_answer,
                    status="timeout",
                )

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=error_msg,
                error_type=type(e).__name__,
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
        """Async streaming wrapper around the shared async tool executor."""
        return await self._execute_tools_async(
            tool_calls,
            current_step,
            on_tool_call,
            structured_output=structured_output,
        )
