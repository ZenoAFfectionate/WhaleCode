"""ReAct Agent - 基于 Function Calling 的实现"""

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.lifecycle import EventType, LifecycleHook
from ..core.streaming import StreamEvent, StreamEventType
from ..tools.registry import ToolRegistry
from ..tools.response import ToolStatus
from ..context.compactor import ContextCompactor


DEFAULT_REACT_SYSTEM_PROMPT = """You are an AI assistant with reasoning and action capabilities.

## Workflow
You complete tasks by calling tools:

1. **Thought tool**: Record your reasoning process and analysis
   - Call when you need to think
   - Parameter: reasoning (your reasoning content)

2. **Action tools**: Retrieve information or perform operations
   - Choose appropriate tools based on the task
   - You may call different tools multiple times

3. **Finish tool**: Return the final answer
   - Call when you have enough information to conclude
   - Parameter: answer (the final answer)

## Important
- Actively use the Thought tool to record your reasoning
- You may call tools multiple times to gather information
- Only call Finish when you are confident you have enough information
"""


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


class ReActAgent(Agent):
    """
    ReAct Agent - 基于 Function Calling 的推理与行动

    核心改进：
    - 使用 OpenAI Function Calling（结构化输出）
    - 支持 Thought 工具（显式推理）
    - 支持 Finish 工具（结束流程）
    - 无需正则解析，解析成功率 99%+
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5
    ):
        """
        初始化 ReActAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tool_registry: 工具注册表（可选）
            system_prompt: 系统提示词（可选）
            config: 配置对象
            max_steps: 最大执行步数
        """
        # 传递 tool_registry 到基类
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

        # 内置工具标记（用于特殊处理）
        self._builtin_tools = {"Thought", "Finish"}

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
        elif event_type == "builtin_tool":
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

    def _render_direct_response(self, final_answer: str) -> None:
        self._render_event("direct_response", {"final_answer": final_answer})

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

    def _render_final_answer(self, final_answer: str, *, step: Optional[int] = None) -> None:
        self._render_event("final_answer", {"final_answer": final_answer, "step": step})

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

    def _split_tool_calls(self, tool_calls: List[Any]) -> tuple[List[Any], List[Any]]:
        """Split tool calls into builtin and user-defined groups."""
        builtin_calls: List[Any] = []
        user_calls: List[Any] = []

        for tool_call in tool_calls:
            target = builtin_calls if tool_call.function.name in self._builtin_tools else user_calls
            target.append(tool_call)

        return builtin_calls, user_calls

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

    def _trace_user_message(self, input_text: str) -> None:
        if self.trace_logger:
            self.trace_logger.log_event(
                "message_written",
                {"role": "user", "content": input_text},
            )

    def _prepare_execution(self, input_text: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        messages = self._build_messages(input_text)
        tool_schemas = self._build_tool_schemas()
        self._trace_user_message(input_text)
        self._render_agent_start(input_text)
        return messages, tool_schemas

    def _maybe_compact_messages(self, messages: List[Dict[str, Any]]) -> None:
        if self._compactor and self.config.compact_enabled:
            self._compactor.micro_compact(messages)
            if self._compactor.estimate_tokens(messages) > self.config.compact_token_threshold:
                self._render_compaction_notice()
                messages[:] = self._compactor.auto_compact(messages, self.llm)

    def _record_model_response(self, response: Any, current_step: int, state: _ExecutionState) -> Any:
        response_message = response.choices[0].message

        if response.usage:
            state.total_tokens += response.usage.total_tokens
            state.total_prompt_tokens += getattr(response.usage, "prompt_tokens", 0)
            state.total_completion_tokens += getattr(response.usage, "completion_tokens", 0)
            self._total_tokens = state.total_tokens
            self._turn_prompt_tokens = state.total_prompt_tokens
            self._turn_completion_tokens = state.total_completion_tokens
            self._last_prompt_tokens = getattr(response.usage, "prompt_tokens", 0)

        if self.trace_logger:
            self.trace_logger.log_event(
                "model_output",
                {
                    "content": response_message.content or "",
                    "tool_calls": len(response_message.tool_calls) if response_message.tool_calls else 0,
                    "usage": {
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                        "cost": 0.0,
                    },
                },
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

    def _append_final_history(self, input_text: str, final_answer: str) -> None:
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

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
    ) -> bool:
        if text_content or state.no_tool_call_retries >= state.max_no_tool_call_retries:
            return False

        state.no_tool_call_retries += 1
        state.is_retry = True
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have access to tools — please use them to complete the task. "
                    "Do not respond with text only. Call a tool now."
                ),
            }
        )
        return True

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
    ) -> Optional[str]:
        for tool_name, tool_call_id, result in tool_results:
            if tool_name == "Finish" and result.get("finished"):
                final_answer = result["final_answer"]
                self._render_final_answer(final_answer, step=current_step)
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

    def _execute_tools(
        self,
        tool_calls: List[Any],
        current_step: int,
    ) -> List[tuple]:
        """Execute builtin tools serially and user tools with an optional shared executor."""
        results = []
        builtin_calls, user_calls = self._split_tool_calls(tool_calls)

        for tc in builtin_calls:
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

            result = self._handle_builtin_tool(tool_name, arguments)
            self._render_builtin_tool(
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
        """
        ReAct Agent 主逻辑实现

        Args:
            input_text: 用户问题
            session_start_time: 会话开始时间
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        start_step = int(kwargs.pop("start_step", 0) or 0)
        if start_step < 0:
            start_step = 0

        messages, tool_schemas = self._prepare_execution(input_text)
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
                    tool_choice="auto",
                    **kwargs
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

                if self._should_retry_without_tool_call(messages, text_content, state):
                    continue

                final_answer = text_content or "Sorry, I cannot answer this question."
                self._render_direct_response(final_answer)
                self._append_final_history(input_text, final_answer)
                self._finalize_trace_session(
                    session_start_time,
                    state.current_step,
                    final_answer,
                    status="success",
                )
                return final_answer

            state.no_tool_call_retries = 0
            self._append_assistant_tool_call_message(messages, response_message)

            tool_results = self._execute_tools(tool_calls, state.current_step)
            final_answer = self._process_tool_results(
                messages,
                tool_calls,
                tool_results,
                state.current_step,
                state,
            )
            if final_answer is not None:
                self._append_final_history(input_text, final_answer)
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
        """构建消息列表"""
        messages = []

        # 添加系统提示词
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # 添加用户问题
        messages.append({
            "role": "user",
            "content": input_text
        })

        return messages

    def _before_model_call(self, messages: List[Dict[str, Any]], current_step: int) -> None:
        """Hook for subclasses to inject context before each model call."""
        return None

    async def _abefore_model_call(self, messages: List[Dict[str, Any]], current_step: int) -> None:
        """Async wrapper for the pre-model-call hook."""
        self._before_model_call(messages, current_step)

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """构建工具 JSON Schema（包含内置工具和用户工具）

        复用基类的 _build_tool_schemas()，并追加 ReAct 内置工具
        """
        schemas = []

        # 1. 添加内置工具：Thought
        schemas.append({
            "type": "function",
            "function": {
                "name": "Thought",
                "description": "Analyze the problem, plan strategy, and record your reasoning process. Call this tool when you need to think.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Your reasoning process and analysis"
                        }
                    },
                    "required": ["reasoning"]
                }
            }
        })

        # 2. 添加内置工具：Finish
        schemas.append({
            "type": "function",
            "function": {
                "name": "Finish",
                "description": "When you have enough information to reach a conclusion, use this tool to return the final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                       "type": "string",
                            "description": "The final answer"
                        }
                    },
                    "required": ["answer"]
                }
            }
        })

        # 3. 添加用户工具（复用基类方法）
        if self.tool_registry:
            user_tool_schemas = super()._build_tool_schemas()
            schemas.extend(user_tool_schemas)

        return schemas

    def _handle_builtin_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理内置工具调用"""
        if tool_name == "Thought":
            reasoning = arguments.get("reasoning", "")
            return {
                "content": f"Reasoning: {reasoning}",
                "finished": False
            }
        elif tool_name == "Finish":
            answer = arguments.get("answer", "")
            return {
                "content": f"Final answer: {answer}",
                "finished": True,
                "final_answer": answer
            }
        else:
            return {
                "content": f"Unknown builtin tool: {tool_name}",
                "finished": False
            }


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
        """
        异步执行 ReAct Agent（完整版本）

        支持：
        - 工具并行执行（独立工具）
        - 生命周期钩子
        - 异步 LLM 调用

        Args:
            input_text: 用户问题
            on_start: Agent 开始执行时的钩子
            on_step: 每个推理步骤的钩子
            on_tool_call: 工具调用时的钩子
            on_finish: Agent 执行完成时的钩子
            on_error: 发生错误时的钩子
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        session_start_time = datetime.now()
        start_step = int(kwargs.pop("start_step", 0) or 0)
        if start_step < 0:
            start_step = 0

        # 触发开始事件
        await self._emit_event(
            EventType.AGENT_START,
            on_start,
            input_text=input_text
        )

        try:
            messages, tool_schemas = self._prepare_execution(input_text)
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
                    step=state.current_step
                )

                await self._abefore_model_call(messages, state.current_step)

                try:
                    response = await self.llm.ainvoke_with_tools(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="auto",
                        **kwargs
                    )
                except Exception as e:
                    self._render_llm_error(e, step=state.current_step)
                    await self._emit_event(
                        EventType.AGENT_ERROR,
                        on_error,
                        error=str(e),
                        step=state.current_step
                    )
                    break

                response_message = self._record_model_response(response, state.current_step, state)
                tool_calls = response_message.tool_calls
                if not tool_calls:
                    text_content = (response_message.content or "").strip()

                    if self._should_retry_without_tool_call(messages, text_content, state):
                        continue

                    final_answer = text_content or "Sorry, I cannot answer this question."
                    self._render_direct_response(final_answer)
                    self._append_final_history(input_text, final_answer)

                    await self._emit_event(
                        EventType.AGENT_FINISH,
                        on_finish,
                        result=final_answer,
                        total_steps=state.current_step,
                        total_tokens=state.total_tokens
                    )

                    self._finalize_trace_session(
                        session_start_time,
                        state.current_step,
                        final_answer,
                        status="success",
                    )
                    return final_answer

                state.no_tool_call_retries = 0
                self._append_assistant_tool_call_message(messages, response_message)

                tool_results = await self._execute_tools_async(
                    tool_calls,
                    state.current_step,
                    on_tool_call
                )

                final_answer = self._process_tool_results(
                    messages,
                    tool_calls,
                    tool_results,
                    state.current_step,
                    state,
                )
                if final_answer is not None:
                    self._append_final_history(input_text, final_answer)
                    await self._emit_event(
                        EventType.AGENT_FINISH,
                        on_finish,
                        result=final_answer,
                        total_steps=state.current_step,
                        total_tokens=state.total_tokens
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
                    tool_calls=len(tool_calls)
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
                status="timeout"
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
                error_type=type(e).__name__
            )
            raise

    async def _execute_tools_async(
        self,
        tool_calls: List[Any],
        current_step: int,
        on_tool_call: LifecycleHook = None
    ) -> List[tuple]:
        """
        异步并行执行工具

        策略：
        1. 内置工具（Thought/Finish）串行执行
        2. 用户工具并行执行（最多 max_concurrent_tools 个）

        Args:
            tool_calls: 工具调用列表
            current_step: 当前步骤
            on_tool_call: 工具调用钩子

        Returns:
            [(tool_name, tool_call_id, result), ...]
        """
        results = []
        builtin_calls, user_calls = self._split_tool_calls(tool_calls)

        # 1. 串行执行内置工具
        for tc in builtin_calls:
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

            await self._emit_tool_call_event(
                on_tool_call,
                tool_name,
                tool_call_id,
                arguments,
                current_step,
            )

            result = self._handle_builtin_tool(tool_name, arguments)
            self._render_builtin_tool(
                tool_name,
                result['content'],
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

        # 2. 并行执行用户工具
        if user_calls:
            max_concurrent = getattr(self.config, 'max_concurrent_tools', 3)

            # 使用 Semaphore 限制并发数
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

            # 并行执行
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
        """
        ReActAgent 真正的流式执行

        实时返回：
        - LLM 输出的每个文本块
        - 工具调用的开始和结束
        - 步骤的开始和结束

        Args:
            input_text: 用户问题
            on_start: 开始钩子
            on_step: 步骤钩子
            on_tool_call: 工具调用钩子
            on_finish: 完成钩子
            on_error: 错误钩子
            **kwargs: 其他参数

        Yields:
            StreamEvent: 流式事件
        """
        session_start_time = datetime.now()
        start_step = int(kwargs.pop("start_step", 0) or 0)
        if start_step < 0:
            start_step = 0

        # 发送开始事件
        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text
        )

        await self._emit_event(EventType.AGENT_START, on_start, input_text=input_text)

        try:
            messages, tool_schemas = self._prepare_execution(input_text)
            state = self._create_execution_state(start_step)
            final_answer = None

            while self.max_steps <= 0 or state.current_step < self.max_steps:
                state.current_step += 1
                self._current_step = state.current_step
                self._maybe_compact_messages(messages)

                # 发送步骤开始事件
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    step=state.current_step,
                    max_steps=self.max_steps
                )

                await self._emit_event(EventType.STEP_START, on_step, step=state.current_step)

                self._render_step_start(state.current_step)

                await self._abefore_model_call(messages, state.current_step)

                # LLM 流式调用
                full_response = ""

                try:
                    # 使用 LLM 的异步流式方法
                    async for chunk in self.llm.astream_invoke(messages, **kwargs):
                        full_response += chunk

                        # 发送 LLM 输出块
                        yield StreamEvent.create(
                            StreamEventType.LLM_CHUNK,
                            self.name,
                            chunk=chunk,
                            step=state.current_step
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
                        step=state.current_step
                    )

                    await self._emit_event(EventType.AGENT_ERROR, on_error, error=error_msg)
                    break

                # 解析工具调用（需要完整响应）
                # 注意：流式输出后需要重新调用 LLM 获取 tool_calls
                # 这里简化处理：使用非流式调用获取工具调用
                try:
                    response = self.llm.invoke_with_tools(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="auto",
                        **kwargs
                    )

                    response_message = self._record_model_response(
                        response,
                        state.current_step,
                        state,
                    )
                    tool_calls = response_message.tool_calls

                    if not tool_calls:
                        # 没有工具调用，直接返回
                        final_answer = response_message.content or full_response or "Sorry, I cannot answer this question."

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

                        self._append_final_history(input_text, final_answer)
                        self._finalize_trace_session(
                            session_start_time,
                            state.current_step,
                            final_answer,
                            status="success",
                        )

                        return

                    self._append_assistant_tool_call_message(messages, response_message)

                    # 执行工具调用
                    tool_results = await self._execute_tools_async_stream(
                        tool_calls,
                        state.current_step,
                        on_tool_call
                    )

                    # 发送工具结果事件并添加到消息
                    for tool_name, tool_call_id, result_dict in tool_results:
                        yield StreamEvent.create(
                            StreamEventType.TOOL_CALL_FINISH,
                            self.name,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            result=result_dict["content"],
                            step=state.current_step
                        )

                        self._append_tool_message(messages, tool_call_id, result_dict["content"])

                        self._update_stagnation_state(
                            tool_name,
                            tool_call_id,
                            result_dict["content"],
                            tool_calls,
                            state.current_step,
                            state,
                        )

                        # 检查是否是 Finish 工具
                        if tool_name == "Finish" and result_dict.get("finished"):
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

                            self._append_final_history(input_text, final_answer)
                            self._finalize_trace_session(
                                session_start_time,
                                state.current_step,
                                final_answer,
                                status="success",
                            )

                            return
                        if state.stagnation_detected:
                            break

                    if state.stagnation_detected:
                        break

                    # 发送步骤完成事件
                    yield StreamEvent.create(
                        StreamEventType.STEP_FINISH,
                        self.name,
                        step=state.current_step
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
                        step=state.current_step
                    )

                    await self._emit_event(EventType.AGENT_ERROR, on_error, error=error_msg)
                    break

            # 达到最大步数
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
                error_type=type(e).__name__
            )

            await self._emit_event(EventType.AGENT_ERROR, on_error, error=error_msg)
            raise

    async def _execute_tools_async_stream(
        self,
        tool_calls: List[Any],
        current_step: int,
        on_tool_call: LifecycleHook = None
    ) -> List[tuple]:
        """
        异步执行工具调用（流式版本，发送工具调用开始事件）

        Args:
            tool_calls: 工具调用列表
            current_step: 当前步骤
            on_tool_call: 工具调用钩子

        Returns:
            List[tuple]: (tool_name, tool_call_id, result_dict) 列表
        """
        results = []
        builtin_calls, user_calls = self._split_tool_calls(tool_calls)

        # 1. 串行执行内置工具
        for tc in builtin_calls:
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

            await self._emit_tool_call_event(
                on_tool_call,
                tool_name,
                tool_call_id,
                arguments,
                current_step,
            )

            result = self._handle_builtin_tool(tool_name, arguments)
            self._render_builtin_tool(
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

        # 2. 并行执行用户工具
        if user_calls:
            max_concurrent = getattr(self.config, 'max_concurrent_tools', 3)
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

                    self._render_tool_result(
                        result_content,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        status=result_status,
                        step=current_step,
                    )

                    return (tool_name, tool_call_id, {"content": result_content})

            # 并行执行
            user_results = await asyncio.gather(*[execute_one(tc) for tc in user_calls])
            results.extend(user_results)

        return results
