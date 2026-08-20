"""Task Tool — 主对话循环内的动态子代理派生 (改进项 A1).

让主 Agent 的 LLM 在 ReAct 循环中按需派生角色化子代理
(explorer / reviewer / tester), 在完全隔离的上下文中执行子任务并
返回蒸馏结果。

并行/串行语义:
    - 并行: 主循环 ``_execute_tools`` 天然支持同一轮多个工具调用并行
      执行 (ThreadPoolExecutor)。LLM 在同一轮响应中发起多个 Task 调用
      即可并行派生多个子代理。
    - 串行: 单次 Task 调用是阻塞的 (子代理跑完才返回)。依赖型子任务
      应等上游 Task 结果返回后再发起 —— ReAct "观察→思考→行动" 的
      分轮循环天然保证了这一点。

复用资产 (本工具是这些底层能力的"薄接线"):
    - ``Role.create_subagent()`` — 隔离工厂 (Config 深拷贝 / 独立
      ToolRegistry / 禁交互 / 防递归, 并发安全)
    - ``Agent._get_subagent_metadata`` / ``_generate_subagent_summary``
      — 执行元数据与蒸馏摘要
    - 共享 truncator — 子代理全量输出落盘 + 有界截断
    - ``Agent._render_event`` — subagent_start / subagent_finish 事件
      (CLI 端 ReActAgent._render_event 渲染; Web 端 WebCodeAgent 透传
      至 SSE, 前端渲染为系统行)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse

if TYPE_CHECKING:
    from ...agents.code_agent import CodeAgent


class TaskTool(Tool):
    """派生一个角色化隔离子代理执行子任务, 返回蒸馏结果与执行元数据."""

    def __init__(self, agent: "CodeAgent"):
        from ...agents.roles import list_roles

        self._available_roles = list_roles()
        role_names = ", ".join(self._available_roles)

        super().__init__(
            name="Task",
            description=(
                "Spawn an isolated role-specialized sub-agent to execute a subtask "
                "in a fresh context and return its distilled final report.\n\n"
                f"Available roles: {role_names}.\n\n"
                "Usage rules:\n"
                "- Independent subtasks: issue MULTIPLE Task calls in the SAME "
                "response — they run in parallel.\n"
                "- Dependent subtasks: wait for the upstream Task result to "
                "return before issuing the next one.\n"
                "- The sub-agent cannot see your conversation history: the task "
                "description must be self-contained (goal, relevant paths, "
                "constraints, expected output format).\n"
                "- Delegate bounded, delegatable work (codebase exploration, "
                "review, testing); keep the core reasoning and final synthesis "
                "to yourself."
            ),
            expandable=False,
            category="general",
        )
        self.agent = agent

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="role",
                type="string",
                description=(
                    f"Sub-agent role: one of {', '.join(self._available_roles)}"
                ),
                required=True,
            ),
            ToolParameter(
                name="task",
                type="string",
                description=(
                    "Self-contained subtask description: goal, relevant file "
                    "paths, constraints, and the expected output format"
                ),
                required=True,
            ),
        ]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """向宿主 Agent 的渲染/转发通道发事件 (失败不影响执行)."""
        try:
            render = getattr(self.agent, "_render_event", None)
            if callable(render):
                render(event_type, payload)
        except Exception:
            pass

    @staticmethod
    def _preview(text: str, max_chars: int = 160) -> str:
        """事件摘要用的单行预览."""
        single_line = " ".join((text or "").split())
        if len(single_line) > max_chars:
            single_line = single_line[: max_chars - 3] + "..."
        return single_line

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        role = str(parameters.get("role") or "").strip()
        task = str(parameters.get("task") or "").strip()

        if not role or not task:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Both 'role' and 'task' are required.",
                context={"params_input": parameters},
            )

        from ...agents.roles import get_role

        try:
            role_cls = get_role(role)
        except ValueError as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=str(exc),
                context={
                    "params_input": parameters,
                    "available_roles": list(self._available_roles),
                },
            )

        # 1. 创建完全隔离的角色化子代理 (深拷贝 Config / 独立 registry /
        #    interactive=False / Task 工具关闭 → 天然防递归)
        subagent = role_cls.create_subagent(
            llm=self.agent.llm,
            parent_config=self.agent.config,
            project_root=str(self.agent.project_root),
            working_dir=str(self.agent.working_dir),
        )

        self._emit_event("subagent_start", {"role": role, "task": task})

        # 2. 执行 (daemon 线程 + join(timeout) 的"放弃等待"式超时,
        #    与历史 orchestra 调度的已知限制语义一致: 超时后主循环继续,
        #    残留线程随进程退出)
        timeout = float(getattr(self.agent.config, "subagent_timeout_seconds", 300.0) or 0)
        start_time = time.time()
        outcome: Dict[str, Any] = {}
        timed_out = False

        def _worker() -> None:
            try:
                outcome["result"] = subagent.run(task)
            except BaseException as exc:  # noqa: BLE001 — 需完整捕获并回报
                outcome["error"] = exc

        worker = threading.Thread(
            target=_worker, name=f"whale-task-{role}", daemon=True
        )
        worker.start()
        worker.join(timeout if timeout > 0 else None)
        if worker.is_alive():
            timed_out = True
        elif "error" in outcome:
            pass  # 异常在下方统一转换为错误响应

        duration = time.time() - start_time

        # 3. 组装: 元数据 + 蒸馏结果 + 截断落盘
        if timed_out:
            error_message = (
                f"Sub-agent timed out after {timeout}s and was abandoned "
                "(its result, if any, is discarded)."
            )
            metadata = subagent._get_subagent_metadata(duration, error_message)
            summary = subagent._generate_subagent_summary(task, error_message, metadata)
            self._emit_event(
                "subagent_finish",
                {
                    "role": role,
                    "task": task,
                    "success": False,
                    "duration_seconds": round(duration, 2),
                    "summary": self._preview(summary),
                },
            )
            return ToolResponse.error(
                code=ToolErrorCode.TIMEOUT,
                message=error_message,
                context={"role": role, "task": task, "metadata": metadata},
            )

        if "error" in outcome:
            exc = outcome["error"]
            error_message = f"{type(exc).__name__}: {exc}"
            metadata = subagent._get_subagent_metadata(duration, error_message)
            summary = subagent._generate_subagent_summary(task, error_message, metadata)
            self._emit_event(
                "subagent_finish",
                {
                    "role": role,
                    "task": task,
                    "success": False,
                    "duration_seconds": round(duration, 2),
                    "summary": self._preview(summary),
                },
            )
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Sub-agent execution failed: {error_message}",
                context={"role": role, "task": task, "metadata": metadata},
            )

        result_text = str(outcome.get("result") or "")
        metadata = subagent._get_subagent_metadata(duration, None)
        truncated = self.agent.truncator.truncate_for_context(
            "Task", result_text, metadata
        )
        display = truncated.get("display_preview") or result_text

        steps = metadata.get("steps", 0)
        tokens = metadata.get("tokens", 0)
        text = (
            f"[subagent:{role}] {task}\n\n"
            f"{display}\n\n"
            f"(steps: {steps}, tokens: ~{tokens}, duration: {duration:.1f}s)"
        )

        self._emit_event(
            "subagent_finish",
            {
                "role": role,
                "task": task,
                "success": True,
                "duration_seconds": round(duration, 2),
                "summary": self._preview(result_text),
                "full_output_path": truncated.get("full_output_path"),
            },
        )
        return ToolResponse.success(
            text=text,
            data={
                "role": role,
                "task": task,
                "steps": steps,
                "tokens": tokens,
                "duration_seconds": round(duration, 2),
                "tools_used": metadata.get("tools_used", []),
                "truncated": bool(truncated.get("truncated")),
                "full_output_path": truncated.get("full_output_path"),
            },
        )
