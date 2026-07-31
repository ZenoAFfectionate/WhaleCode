"""Agent Orchestra — 主从式 (Orchestrator-Worker) 多智能体编排器.

主 Agent 将任务分解为 ExecutionPlan, 按模式调度完全隔离的角色化子 Agent
(Explorer / Reviewer / Tester), 最后汇总结果。

执行模式:
    - PIPELINE: 阶段串行, 阶段内并行, 上游结果注入下游 context_hint
    - PARALLEL: 全部子任务并行 (Semaphore 限流)
    - HYBRID:   同 PIPELINE (默认)

已知限制: asyncio.wait_for 超时只是「放弃等待」, Python 无法强杀线程池中的
线程; 超时子 Agent 实例被丢弃 (结果标记 error, 不注入下游), 线程随进程退出回收。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from ..core.config import Config
from .roles import get_role, list_roles

if TYPE_CHECKING:
    from .code_agent import CodeAgent


class ExecutionMode(Enum):
    PIPELINE = "pipeline"  # 阶段串行, 阶段内可并行
    PARALLEL = "parallel"  # 全部子任务并行执行
    HYBRID = "hybrid"      # Pipeline 阶段内包含 Parallel 子任务 (默认)

    @classmethod
    def from_value(cls, value: str, default: "ExecutionMode" = None) -> "ExecutionMode":
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return default or cls.HYBRID


@dataclass
class SubTask:
    """子任务定义."""

    id: str
    description: str
    role: str  # "explorer" | "reviewer" | "tester"
    dependencies: List[str] = field(default_factory=list)
    context_hint: str = ""  # 给子 Agent 的附加上下文 (含上游阶段注入)


@dataclass
class ExecutionPlan:
    """执行计划 — decompose() 的输出."""

    subtasks: List[SubTask]
    mode: ExecutionMode
    stages: List[List[str]] = field(default_factory=list)  # 仅 pipeline/hybrid
    original_task: str = ""


@dataclass
class SubtaskHooks:
    """子任务生命周期钩子 (全部可选; 钩子异常不影响执行)."""

    on_subtask_start: Optional[Callable[[SubTask], Awaitable[None]]] = None
    on_subtask_finish: Optional[Callable[[SubTask, "SubAgentResult"], Awaitable[None]]] = None
    on_subtask_error: Optional[Callable[[SubTask, Exception], Awaitable[None]]] = None


@dataclass
class SubAgentResult:
    """子 Agent 执行结果."""

    subtask_id: str
    success: bool
    summary: str
    full_result: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# 提示词集中维护于 code/prompts/orchestra/, 模板占位符为 {task}/{mode}/{results_text},
# 通过 str.replace 注入 (文件内 JSON 示例保持单大括号, 无需转义)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ORCHESTRA_PROMPTS_DIR = _PROJECT_ROOT / "code" / "prompts" / "orchestra"
_DECOMPOSE_PROMPT_TEMPLATE: str = (_ORCHESTRA_PROMPTS_DIR / "decompose.md").read_text(
    encoding="utf-8"
)
_AGGREGATE_PROMPT_TEMPLATE: str = (_ORCHESTRA_PROMPTS_DIR / "aggregate.md").read_text(
    encoding="utf-8"
)


class AgentOrchestra:
    """多智能体编排器 — 主从式 (Orchestrator-Worker).

    约束: main_agent 必须是 CodeAgent (需要 project_root/working_dir 与 LLM)。
    """

    def __init__(self, main_agent: "CodeAgent", config: Optional[Config] = None):
        self.main_agent = main_agent
        self.config = config or main_agent.config

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    async def decompose(
        self,
        task: str,
        mode: ExecutionMode = ExecutionMode.HYBRID,
    ) -> ExecutionPlan:
        """使用 LLM 将任务分解为 ExecutionPlan.

        无效 JSON → 重试一次 → 仍失败则降级为单个 Explorer 子任务。
        """
        prompt = self._build_decompose_prompt(task, mode)
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = await asyncio.to_thread(
                    self.main_agent.llm.invoke,
                    [{"role": "user", "content": prompt}],
                )
                raw = getattr(response, "content", None) or str(response)
                return self._parse_plan_from_llm_output(raw, mode, original_task=task)
            except Exception as exc:  # 解析/校验失败 → 重试一次
                last_error = exc
                prompt = (
                    f"{prompt}\n\n# 上次输出无法解析 ({exc})，请严格只输出合法 JSON。"
                )
        return self._fallback_plan(task, mode, reason=str(last_error))

    async def execute(
        self,
        plan: ExecutionPlan,
        *,
        hooks: Optional[SubtaskHooks] = None,
    ) -> List[SubAgentResult]:
        """按 plan.mode 调度执行所有子任务."""
        hooks = hooks or SubtaskHooks()
        if plan.mode is ExecutionMode.PARALLEL:
            return await self._execute_parallel(plan, hooks)
        return await self._execute_pipeline(plan, hooks)

    async def aggregate(
        self,
        plan: ExecutionPlan,
        results: List[SubAgentResult],
    ) -> str:
        """使用主 Agent 的 LLM 汇总子 Agent 结果 → 最终答案.

        ★ 上下文契约: 注入的是子 Agent 的蒸馏最终结果 (full_result, 有界截断),
        而非 500 字符的元数据摘要 —— 保证有效结果完整回传, 同时保持上下文有界。
        """
        results_text_parts: List[str] = []
        for r in results:
            payload = _result_payload(r)
            if r.success:
                results_text_parts.append(f"### 子任务 {r.subtask_id} (成功)\n{payload}")
            else:
                results_text_parts.append(
                    f"### 子任务 {r.subtask_id} (失败)\n错误: {r.error}\n{payload}"
                )
        prompt = self._build_aggregate_prompt(
            plan.original_task, plan, "\n\n".join(results_text_parts) or "(无子任务结果)"
        )
        response = await asyncio.to_thread(
            self.main_agent.llm.invoke,
            [{"role": "user", "content": prompt}],
        )
        return getattr(response, "content", None) or str(response)

    async def run(
        self,
        task: str,
        mode: ExecutionMode = ExecutionMode.HYBRID,
        *,
        confirm_hook: Optional[Callable[[ExecutionPlan], bool]] = None,
        hooks: Optional[SubtaskHooks] = None,
    ) -> str:
        """一站式入口: decompose → [confirm_hook 确认] → execute → aggregate."""
        plan = await self.decompose(task, mode)
        if confirm_hook is not None and not confirm_hook(plan):
            return "Orchestra plan was rejected by user; execution cancelled."
        results = await self.execute(plan, hooks=hooks)
        return await self.aggregate(plan, results)

    # ------------------------------------------------------------------
    # 子 Agent 创建与调度
    # ------------------------------------------------------------------

    def _create_subagent(self, role_name: str) -> "CodeAgent":
        """工厂方法: 创建完全隔离的角色化子 Agent."""
        role = get_role(role_name)
        return role.create_subagent(
            llm=self.main_agent.llm,
            parent_config=self.config,
            project_root=str(self.main_agent.project_root),
            working_dir=str(self.main_agent.working_dir),
        )

    async def _execute_pipeline(
        self, plan: ExecutionPlan, hooks: SubtaskHooks
    ) -> List[SubAgentResult]:
        """Pipeline/Hybrid 模式: 阶段串行, 阶段内并行, 阶段间注入上下文.

        stages 未覆盖的子任务不会被静默丢弃 —— 作为追加的最终阶段执行。
        """
        stages = [list(ids) for ids in (plan.stages or [[st.id for st in plan.subtasks]])]
        covered = {sid for ids in stages for sid in ids}
        uncovered = [st.id for st in plan.subtasks if st.id not in covered]
        if uncovered:
            stages.append(uncovered)

        all_results: List[SubAgentResult] = []
        for stage_idx, stage_ids in enumerate(stages):
            stage_subtasks = [st for st in plan.subtasks if st.id in stage_ids]
            stage_results = await asyncio.gather(
                *(self._execute_single_subtask(st, hooks) for st in stage_subtasks)
            )
            all_results.extend(stage_results)
            if stage_idx < len(stages) - 1:
                self._inject_stage_context(stage_results, plan, stages[stage_idx + 1])
        return all_results

    async def _execute_parallel(
        self, plan: ExecutionPlan, hooks: SubtaskHooks
    ) -> List[SubAgentResult]:
        """Parallel 模式: 全部并行, Semaphore 限流."""
        sem = asyncio.Semaphore(max(1, int(self.config.orchestra_max_parallel)))

        async def _bounded(subtask: SubTask) -> SubAgentResult:
            async with sem:
                return await self._execute_single_subtask(subtask, hooks)

        return await asyncio.gather(*(_bounded(st) for st in plan.subtasks))

    async def _execute_single_subtask(
        self, subtask: SubTask, hooks: SubtaskHooks
    ) -> SubAgentResult:
        """执行单个子任务 (线程池运行 + wait_for 超时 + 完备错误捕获)."""
        if hooks.on_subtask_start:
            try:
                await hooks.on_subtask_start(subtask)
            except Exception:
                pass

        try:
            subagent = self._create_subagent(subtask.role)
        except Exception as exc:
            # 子 Agent 创建失败 (如未知角色) 不得拖垮整个调度
            error = f"subagent creation failed: {type(exc).__name__}: {exc}"
            result = SubAgentResult(
                subtask_id=subtask.id,
                success=False,
                summary=f"子任务未能启动: {error}",
                full_result="",
                error=error,
            )
            if hooks.on_subtask_error:
                try:
                    await hooks.on_subtask_error(subtask, exc)
                except Exception:
                    pass
            return result

        prompt = self._build_subtask_prompt(subtask)
        start_time = time.time()
        success = False
        error: Optional[str] = None

        try:
            loop = asyncio.get_running_loop()
            result_text = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: subagent.run(prompt)),
                timeout=self.config.subagent_timeout_seconds,
            )
            success = True
        except asyncio.TimeoutError:
            error = f"timeout after {self.config.subagent_timeout_seconds}s"
            result_text = f"Subtask timed out after {self.config.subagent_timeout_seconds}s"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result_text = f"执行失败: {error}"

        duration = time.time() - start_time
        metadata = subagent._get_subagent_metadata(duration, error)
        result = SubAgentResult(
            subtask_id=subtask.id,
            success=success,
            summary=subagent._generate_subagent_summary(prompt, result_text, metadata),
            full_result=result_text,
            error=error,
            metadata=metadata,
        )

        if hooks.on_subtask_finish:
            try:
                await hooks.on_subtask_finish(subtask, result)
            except Exception:
                pass
        if not success and hooks.on_subtask_error:
            try:
                await hooks.on_subtask_error(subtask, Exception(error or "unknown"))
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # 上下文注入与 prompt 构建
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subtask_prompt(subtask: SubTask) -> str:
        parts = [subtask.description]
        if subtask.context_hint:
            parts.append(f"\n\n{subtask.context_hint}")
        return "\n".join(parts)

    def _inject_stage_context(
        self,
        stage_results: List[SubAgentResult],
        plan: ExecutionPlan,
        next_stage_ids: List[str],
    ) -> None:
        """将上游阶段结果注入下游子任务的 context_hint (流水线语义的关键).

        注入子 Agent 的蒸馏最终结果 (full_result, 有界截断), 保证下游拿到完整发现。
        """
        context_text = "\n\n".join(
            f"## 上游子任务 {r.subtask_id} 的结果:\n{_result_payload(r)}"
            for r in stage_results
            if r.success
        )
        if not context_text:
            return
        for subtask in plan.subtasks:
            if subtask.id in next_stage_ids:
                injection = f"## 上游阶段结果摘要\n{context_text}"
                subtask.context_hint = (
                    f"{subtask.context_hint}\n\n{injection}"
                    if subtask.context_hint
                    else injection
                )

    def _build_decompose_prompt(self, task: str, mode: ExecutionMode) -> str:
        return _DECOMPOSE_PROMPT_TEMPLATE.replace("{task}", task).replace(
            "{mode}", mode.value
        )

    def _build_aggregate_prompt(
        self, task: str, plan: ExecutionPlan, results_text: str
    ) -> str:
        return _AGGREGATE_PROMPT_TEMPLATE.replace("{task}", task).replace(
            "{results_text}", results_text
        )

    # ------------------------------------------------------------------
    # Plan 解析与校验
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan_from_llm_output(
        raw_json: str,
        mode: ExecutionMode,
        *,
        original_task: str = "",
    ) -> ExecutionPlan:
        """解析 LLM 输出的 JSON → 校验 → ExecutionPlan.

        校验: 角色已知 / stage ID 均在 subtasks 中 / 无循环依赖 / 无重复 ID。
        任何校验失败抛 ValueError, 由 decompose() 决定重试或降级。
        """
        data = _extract_json(raw_json)
        if data is None:
            raise ValueError("LLM output is not valid JSON")

        raw_subtasks = data.get("subtasks")
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise ValueError("plan has no subtasks")

        subtasks: List[SubTask] = []
        seen_ids: set = set()
        for i, item in enumerate(raw_subtasks):
            if not isinstance(item, dict):
                raise ValueError(f"subtask[{i}] is not an object")
            st_id = str(item.get("id", "")).strip() or f"task-{i + 1}"
            if st_id in seen_ids:
                raise ValueError(f"duplicate subtask id: {st_id}")
            seen_ids.add(st_id)
            role = str(item.get("role", "")).strip().lower()
            get_role(role)  # 未知角色 → ValueError
            deps = item.get("dependencies") or []
            if not isinstance(deps, list):
                raise ValueError(f"subtask {st_id}: dependencies must be a list")
            subtasks.append(
                SubTask(
                    id=st_id,
                    description=str(item.get("description", "")).strip(),
                    role=role,
                    dependencies=[str(d) for d in deps],
                    context_hint=str(item.get("context_hint", "") or ""),
                )
            )

        known = {st.id for st in subtasks}
        for st in subtasks:
            unknown = [d for d in st.dependencies if d not in known]
            if unknown:
                raise ValueError(f"subtask {st.id}: unknown dependencies {unknown}")
        _assert_no_cycle(subtasks)

        stages_raw = data.get("stages") or []
        stages: List[List[str]] = []
        for stage in stages_raw:
            if not isinstance(stage, list):
                raise ValueError("stages must be a list of id lists")
            ids = [str(s) for s in stage]
            unknown = [s for s in ids if s not in known]
            if unknown:
                raise ValueError(f"stages reference unknown subtask ids {unknown}")
            stages.append(ids)

        plan_mode = ExecutionMode.from_value(str(data.get("mode", mode.value)), default=mode)
        return ExecutionPlan(
            subtasks=subtasks, mode=plan_mode, stages=stages, original_task=original_task
        )

    def _fallback_plan(
        self, task: str, mode: ExecutionMode, *, reason: str = ""
    ) -> ExecutionPlan:
        """降级计划: 单个 Explorer 子任务."""
        hint = f"(decompose 降级: {reason})" if reason else ""
        return ExecutionPlan(
            subtasks=[
                SubTask(
                    id="exp-fallback",
                    description=f"探索并分析以下任务，给出结构化报告：\n{task}",
                    role="explorer",
                    context_hint=hint,
                )
            ],
            mode=mode,
            stages=[["exp-fallback"]],
            original_task=task,
        )


# ----------------------------------------------------------------------
# 模块级辅助
# ----------------------------------------------------------------------

# 子任务结果注入下游/聚合时的内容上限: 保证有效结果基本完整, 同时保持上下文有界
_RESULT_PAYLOAD_CAP = 4000


def _result_payload(result: "SubAgentResult") -> str:
    """取子任务的蒸馏最终结果 (full_result 优先, 回退 summary), 有界截断."""
    text = result.full_result or result.summary or "(无输出)"
    if len(text) > _RESULT_PAYLOAD_CAP:
        text = (
            text[:_RESULT_PAYLOAD_CAP]
            + f"\n... [结果共 {len(text)} 字符, 此处截断至 {_RESULT_PAYLOAD_CAP}]"
        )
    return text


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出提取 JSON 对象 (直接解析 → 去代码围栏 → 首个 {...} 块)."""
    import re

    text = raw.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _assert_no_cycle(subtasks: List[SubTask]) -> None:
    """DFS 检测依赖环; 发现环抛 ValueError."""
    deps = {st.id: st.dependencies for st in subtasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {st.id: WHITE for st in subtasks}

    def _visit(node: str) -> None:
        color[node] = GRAY
        for nxt in deps.get(node, []):
            if color.get(nxt) == GRAY:
                raise ValueError(f"circular dependency detected at subtask: {nxt}")
            if color.get(nxt) == WHITE:
                _visit(nxt)
        color[node] = BLACK

    for st_id in deps:
        if color[st_id] == WHITE:
            _visit(st_id)
