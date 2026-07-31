"""AgentOrchestra 编排器测试: decompose / execute / aggregate / run / 解析校验."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from hello_agents.agents.code_agent import CodeAgent
from hello_agents.agents.orchestra import (
    AgentOrchestra,
    ExecutionMode,
    ExecutionPlan,
    SubAgentResult,
    SubTask,
    SubtaskHooks,
)
from conftest import StubSubAgent


@pytest.fixture
def tmp_workspace(tmp_path):
    (tmp_path / "memory" / "tool-output").mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


@pytest.fixture
def main_agent(mock_llm, tmp_workspace):
    return CodeAgent(
        "main",
        mock_llm,
        project_root=tmp_workspace,
        register_default_tools=False,
    )


def _make_plan(mode=ExecutionMode.HYBRID):
    return ExecutionPlan(
        subtasks=[
            SubTask(id="exp-1", description="explore A", role="explorer"),
            SubTask(id="exp-2", description="explore B", role="explorer"),
            SubTask(id="rev-1", description="review", role="reviewer",
                    dependencies=["exp-1", "exp-2"]),
        ],
        mode=mode,
        stages=[["exp-1", "exp-2"], ["rev-1"]],
        original_task="analyze",
    )


class TestOrchestraDecompose:
    def test_simple_task_decomposition(self, main_agent, plan_llm):
        main_agent.llm = plan_llm
        orchestra = AgentOrchestra(main_agent)
        plan = asyncio.run(orchestra.decompose("analyze repo", ExecutionMode.HYBRID))
        assert plan.subtasks[0].role == "explorer"
        assert plan.original_task == "analyze repo"
        assert plan.stages == [["exp-1"]]

    def test_decompose_returns_valid_plan(self, main_agent, plan_llm):
        main_agent.llm = plan_llm
        orchestra = AgentOrchestra(main_agent)
        plan = asyncio.run(orchestra.decompose("t", ExecutionMode.HYBRID))
        assert isinstance(plan, ExecutionPlan)
        assert plan.mode is ExecutionMode.HYBRID

    def test_decompose_invalid_json_fallback(self, main_agent, mock_llm):
        response = MagicMock()
        response.content = "not a json at all"
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)
        plan = asyncio.run(orchestra.decompose("do something", ExecutionMode.HYBRID))
        # 降级为单个 explorer 子任务
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].role == "explorer"
        assert "do something" in plan.subtasks[0].description

    def test_decompose_empty_subtasks_fallback(self, main_agent, mock_llm):
        response = MagicMock()
        response.content = json.dumps({"subtasks": [], "mode": "hybrid", "stages": []})
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)
        plan = asyncio.run(orchestra.decompose("t", ExecutionMode.HYBRID))
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].role == "explorer"

    def test_decompose_unknown_role_validation(self, main_agent, mock_llm):
        response = MagicMock()
        response.content = json.dumps({
            "subtasks": [{"id": "x", "description": "d", "role": "wizard", "dependencies": []}],
        })
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)
        plan = asyncio.run(orchestra.decompose("t", ExecutionMode.HYBRID))
        # 未知角色 → 校验失败 → 重试仍失败 → fallback
        assert plan.subtasks[0].role == "explorer"


class TestOrchestraExecute:
    def test_parallel_execution(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        stub_subagent_factory(orchestra)
        plan = _make_plan(mode=ExecutionMode.PARALLEL)
        results = asyncio.run(orchestra.execute(plan))
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_parallel_respects_semaphore(self, main_agent, monkeypatch):
        """Semaphore(max_parallel) 限制并发数."""
        orchestra = AgentOrchestra(main_agent)
        orchestra.config.orchestra_max_parallel = 2
        concurrent = {"cur": 0, "max": 0}

        async def _fake_execute(subtask, hooks):
            concurrent["cur"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["cur"])
            await asyncio.sleep(0.01)
            concurrent["cur"] -= 1
            return SubAgentResult(subtask_id=subtask.id, success=True,
                                  summary="s", full_result="r")

        monkeypatch.setattr(orchestra, "_execute_single_subtask", _fake_execute)
        plan = ExecutionPlan(
            subtasks=[SubTask(id=f"t-{i}", description="d", role="explorer") for i in range(5)],
            mode=ExecutionMode.PARALLEL,
        )
        asyncio.run(orchestra.execute(plan))
        assert concurrent["max"] <= 2

    def test_pipeline_execution_order(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        created = stub_subagent_factory(orchestra)
        order: list[str] = []

        async def _on_finish(st, result):
            order.append(st.id)

        hooks = SubtaskHooks(on_subtask_finish=_on_finish)
        plan = _make_plan()
        results = asyncio.run(orchestra.execute(plan, hooks=hooks))
        assert len(results) == 3
        # rev-1 必须在 exp-1/exp-2 之后完成
        assert order.index("rev-1") > order.index("exp-1")
        assert order.index("rev-1") > order.index("exp-2")
        # 每个子任务各创建了一个独立子 Agent
        assert len(created) == 3

    def test_pipeline_context_injection(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        created = stub_subagent_factory(orchestra)
        plan = _make_plan()
        asyncio.run(orchestra.execute(plan))
        # 第三个创建的 stub 属于 rev-1, 其 prompt 应含上游结果注入
        rev_stub = created[2]
        assert len(rev_stub.run_prompts) == 1
        assert "上游阶段结果摘要" in rev_stub.run_prompts[0]
        # 第一阶段的子任务 prompt 不含注入
        assert "上游阶段结果摘要" not in created[0].run_prompts[0]

    def test_hybrid_execution(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        stub_subagent_factory(orchestra)
        plan = _make_plan(mode=ExecutionMode.HYBRID)
        results = asyncio.run(orchestra.execute(plan))
        assert len(results) == 3
        assert {r.subtask_id for r in results} == {"exp-1", "exp-2", "rev-1"}

    def test_subtask_timeout(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        orchestra.config.subagent_timeout_seconds = 0.05
        stub_subagent_factory(
            orchestra, factory=lambda role: StubSubAgent(delay=1.0)
        )
        plan = ExecutionPlan(
            subtasks=[SubTask(id="slow", description="d", role="explorer")],
            mode=ExecutionMode.PARALLEL,
        )
        results = asyncio.run(orchestra.execute(plan))
        assert len(results) == 1
        assert results[0].success is False
        assert "timeout" in (results[0].error or "")

    def test_subtask_error_propagation(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)

        def _factory(role):
            stub = StubSubAgent()
            def _boom(prompt, **kwargs):
                raise RuntimeError("boom")
            stub.run = _boom
            return stub

        stub_subagent_factory(orchestra, factory=_factory)
        plan = ExecutionPlan(
            subtasks=[SubTask(id="bad", description="d", role="explorer")],
            mode=ExecutionMode.PARALLEL,
        )
        results = asyncio.run(orchestra.execute(plan))
        assert results[0].success is False
        assert "RuntimeError" in (results[0].error or "")
        assert "boom" in (results[0].error or "")

    def test_failed_subtask_not_blocking(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        calls = {"n": 0}

        def _factory(role):
            calls["n"] += 1
            if calls["n"] == 1:
                stub = StubSubAgent()
                def _boom(prompt, **kwargs):
                    raise RuntimeError("first fails")
                stub.run = _boom
                return stub
            return StubSubAgent()

        stub_subagent_factory(orchestra, factory=_factory)
        plan = ExecutionPlan(
            subtasks=[
                SubTask(id="t-1", description="d", role="explorer"),
                SubTask(id="t-2", description="d", role="explorer"),
            ],
            mode=ExecutionMode.PARALLEL,
        )
        results = asyncio.run(orchestra.execute(plan))
        assert len(results) == 2
        statuses = {r.subtask_id: r.success for r in results}
        assert statuses["t-1"] is False
        assert statuses["t-2"] is True

    def test_hooks_exceptions_do_not_break_execution(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        stub_subagent_factory(orchestra)

        async def _bad_hook(st, *args):
            raise RuntimeError("hook bug")

        hooks = SubtaskHooks(on_subtask_start=_bad_hook, on_subtask_finish=_bad_hook)
        plan = ExecutionPlan(
            subtasks=[SubTask(id="t", description="d", role="explorer")],
            mode=ExecutionMode.PARALLEL,
        )
        results = asyncio.run(orchestra.execute(plan, hooks=hooks))
        assert results[0].success is True


class TestOrchestraAggregate:
    def _result(self, sid, success, summary="sum", error=None):
        return SubAgentResult(subtask_id=sid, success=success, summary=summary,
                              full_result=summary, error=error)

    def test_aggregate_with_all_success(self, main_agent, mock_llm):
        response = MagicMock()
        response.content = "final answer"
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)
        plan = _make_plan()
        out = asyncio.run(orchestra.aggregate(plan, [self._result("a", True)]))
        assert out == "final answer"
        prompt = mock_llm.invoke.call_args[0][0][0]["content"]
        assert "(成功)" in prompt

    def test_aggregate_with_partial_failure(self, main_agent, mock_llm):
        response = MagicMock()
        response.content = "partial"
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)
        plan = _make_plan()
        results = [self._result("a", True), self._result("b", False, error="boom")]
        asyncio.run(orchestra.aggregate(plan, results))
        prompt = mock_llm.invoke.call_args[0][0][0]["content"]
        assert "(失败)" in prompt and "boom" in prompt


class TestOrchestraRun:
    def test_run_end_to_end(self, main_agent, plan_llm, stub_subagent_factory):
        main_agent.llm = plan_llm  # decompose 返回单 explorer 计划
        orchestra = AgentOrchestra(main_agent)
        stub_subagent_factory(orchestra)
        out = asyncio.run(orchestra.run("task", ExecutionMode.HYBRID))
        # aggregate 也走 plan_llm.invoke → 返回 plan JSON 文本
        assert isinstance(out, str) and out

    def test_run_confirm_hook_reject(self, main_agent, plan_llm, stub_subagent_factory):
        main_agent.llm = plan_llm
        orchestra = AgentOrchestra(main_agent)
        created = stub_subagent_factory(orchestra)
        out = asyncio.run(
            orchestra.run("task", ExecutionMode.HYBRID, confirm_hook=lambda plan: False)
        )
        assert "cancelled" in out.lower() or "rejected" in out.lower()
        assert created == []  # 未执行任何子任务


class TestAgentOrchestraHelpers:
    def test_create_subagent_for_known_role(self, main_agent, tmp_workspace):
        orchestra = AgentOrchestra(main_agent)
        sub = orchestra._create_subagent("explorer")
        assert sub.tool_registry.get_tool("Write") is None
        assert sub.tool_registry.get_tool("Read") is not None

    def test_create_subagent_unknown_role_raises(self, main_agent):
        orchestra = AgentOrchestra(main_agent)
        with pytest.raises(ValueError):
            orchestra._create_subagent("unknown")

    def test_inject_stage_context(self, main_agent):
        orchestra = AgentOrchestra(main_agent)
        plan = _make_plan()
        results = [
            SubAgentResult(subtask_id="exp-1", success=True, summary="S1", full_result=""),
            SubAgentResult(subtask_id="exp-2", success=False, summary="S2", full_result="", error="e"),
        ]
        orchestra._inject_stage_context(results, plan, ["rev-1"])
        rev = next(st for st in plan.subtasks if st.id == "rev-1")
        assert "S1" in rev.context_hint
        # 失败结果不注入
        assert "S2" not in rev.context_hint
        # 同阶段子任务不受影响
        exp1 = next(st for st in plan.subtasks if st.id == "exp-1")
        assert exp1.context_hint == ""

    def test_parse_plan_from_valid_json(self):
        raw = json.dumps({
            "subtasks": [
                {"id": "a", "description": "d", "role": "explorer", "dependencies": []},
                {"id": "b", "description": "d", "role": "tester", "dependencies": ["a"]},
            ],
            "mode": "hybrid",
            "stages": [["a"], ["b"]],
        })
        plan = AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)
        assert len(plan.subtasks) == 2
        assert plan.subtasks[1].role == "tester"
        assert plan.stages == [["a"], ["b"]]

    def test_parse_plan_from_fenced_json(self):
        raw = "前言\n```json\n" + json.dumps({
            "subtasks": [{"id": "a", "description": "d", "role": "reviewer"}],
        }) + "\n```\n后叙"
        plan = AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.PARALLEL)
        assert plan.subtasks[0].id == "a"

    def test_parse_plan_from_invalid_json(self):
        with pytest.raises(ValueError):
            AgentOrchestra._parse_plan_from_llm_output("no json", ExecutionMode.HYBRID)

    def test_parse_plan_circular_dependency_rejected(self):
        raw = json.dumps({
            "subtasks": [
                {"id": "a", "description": "d", "role": "explorer", "dependencies": ["b"]},
                {"id": "b", "description": "d", "role": "explorer", "dependencies": ["a"]},
            ],
        })
        with pytest.raises(ValueError):
            AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)

    def test_parse_plan_unknown_stage_id_rejected(self):
        raw = json.dumps({
            "subtasks": [{"id": "a", "description": "d", "role": "explorer"}],
            "stages": [["a", "ghost"]],
        })
        with pytest.raises(ValueError):
            AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)

    def test_parse_plan_duplicate_id_rejected(self):
        raw = json.dumps({
            "subtasks": [
                {"id": "a", "description": "d", "role": "explorer"},
                {"id": "a", "description": "d", "role": "explorer"},
            ],
        })
        with pytest.raises(ValueError):
            AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)

    def test_parse_plan_unknown_dependency_names_correct_subtask(self):
        """回归: 错误消息必须指出真正的子任务 id (而非最后一个子任务)."""
        raw = json.dumps({
            "subtasks": [
                {"id": "first", "description": "d", "role": "explorer", "dependencies": ["ghost"]},
                {"id": "second", "description": "d", "role": "explorer"},
            ],
        })
        with pytest.raises(ValueError, match="first"):
            AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)

    def test_parse_plan_self_dependency_rejected(self):
        raw = json.dumps({
            "subtasks": [{"id": "a", "description": "d", "role": "explorer", "dependencies": ["a"]}],
        })
        with pytest.raises(ValueError):
            AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)

    def test_parse_plan_long_cycle_rejected(self):
        raw = json.dumps({
            "subtasks": [
                {"id": "a", "description": "d", "role": "explorer", "dependencies": ["c"]},
                {"id": "b", "description": "d", "role": "explorer", "dependencies": ["a"]},
                {"id": "c", "description": "d", "role": "explorer", "dependencies": ["b"]},
            ],
        })
        with pytest.raises(ValueError, match="circular"):
            AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)

    def test_parse_plan_non_dict_subtask_rejected(self):
        raw = json.dumps({"subtasks": ["not-a-dict"]})
        with pytest.raises(ValueError):
            AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)

    def test_parse_plan_preserves_context_hint(self):
        raw = json.dumps({
            "subtasks": [{"id": "a", "description": "d", "role": "explorer",
                          "context_hint": "hint-123"}],
        })
        plan = AgentOrchestra._parse_plan_from_llm_output(raw, ExecutionMode.HYBRID)
        assert plan.subtasks[0].context_hint == "hint-123"


class TestExecutionMode:
    def test_from_value_valid(self):
        assert ExecutionMode.from_value("pipeline") is ExecutionMode.PIPELINE
        assert ExecutionMode.from_value("PARALLEL") is ExecutionMode.PARALLEL
        assert ExecutionMode.from_value(" hybrid ") is ExecutionMode.HYBRID

    def test_from_value_invalid_uses_default(self):
        assert ExecutionMode.from_value("bogus") is ExecutionMode.HYBRID
        assert ExecutionMode.from_value("bogus", default=ExecutionMode.PARALLEL) is ExecutionMode.PARALLEL


class TestResultPayloadContract:
    """上下文契约验证: 完整性 (有效结果完整回传) 与有界性 (截断护栏)."""

    def test_aggregate_uses_full_result_not_truncated_summary(self, main_agent, mock_llm):
        """>500 字符的完整发现必须进入聚合 prompt (不被 500 字符摘要截断)."""
        response = MagicMock(); response.content = "ans"
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)
        long_finding = "关键发现X" * 200  # ~1000 字符, 超过 summary 的 500 截断
        results = [SubAgentResult(
            subtask_id="exp-1", success=True,
            summary="任务: ...\n结果: 短摘要...",
            full_result=f"报告开头 {long_finding} 报告结尾",
        )]
        plan = _make_plan()
        asyncio.run(orchestra.aggregate(plan, results))
        prompt = mock_llm.invoke.call_args[0][0][0]["content"]
        assert "报告开头" in prompt and "报告结尾" in prompt
        assert long_finding[:100] in prompt

    def test_inject_stage_context_uses_full_result(self, main_agent):
        orchestra = AgentOrchestra(main_agent)
        plan = _make_plan()
        long_finding = "深度发现Y" * 200
        results = [SubAgentResult(
            subtask_id="exp-1", success=True, summary="短摘要",
            full_result=f"完整报告 {long_finding} 完",
        )]
        orchestra._inject_stage_context(results, plan, ["rev-1"])
        rev = next(st for st in plan.subtasks if st.id == "rev-1")
        assert "完整报告" in rev.context_hint and "深度发现Y" in rev.context_hint

    def test_payload_falls_back_to_summary(self):
        from hello_agents.agents.orchestra import _result_payload

        r = SubAgentResult(subtask_id="a", success=True, summary="S", full_result="")
        assert _result_payload(r) == "S"

    def test_payload_cap_bounds_oversized_results(self):
        from hello_agents.agents.orchestra import _RESULT_PAYLOAD_CAP, _result_payload

        r = SubAgentResult(subtask_id="a", success=True, summary="",
                           full_result="z" * 10_000)
        payload = _result_payload(r)
        assert len(payload) < 10_000
        assert "截断" in payload
        assert payload.startswith("z" * 100)

    def test_payload_empty_result_placeholder(self):
        from hello_agents.agents.orchestra import _result_payload

        r = SubAgentResult(subtask_id="a", success=True, summary="", full_result="")
        assert _result_payload(r) == "(无输出)"


class TestOrchestraEdgeCases:
    def test_pipeline_uncovered_subtasks_still_executed(self, main_agent, stub_subagent_factory):
        """回归: stages 未覆盖的子任务不得被静默丢弃."""
        orchestra = AgentOrchestra(main_agent)
        stub_subagent_factory(orchestra)
        plan = ExecutionPlan(
            subtasks=[
                SubTask(id="a", description="d", role="explorer"),
                SubTask(id="b", description="d", role="explorer"),
            ],
            mode=ExecutionMode.HYBRID,
            stages=[["a"]],  # b 未被覆盖
            original_task="t",
        )
        results = asyncio.run(orchestra.execute(plan))
        assert {r.subtask_id for r in results} == {"a", "b"}

    def test_parallel_ignores_empty_stages(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        stub_subagent_factory(orchestra)
        plan = ExecutionPlan(
            subtasks=[SubTask(id="a", description="d", role="explorer")],
            mode=ExecutionMode.PARALLEL,
            stages=[],
            original_task="t",
        )
        results = asyncio.run(orchestra.execute(plan))
        assert len(results) == 1 and results[0].success

    def test_unknown_role_contained_as_error_result(self, main_agent):
        """手工构造的非法 role 不得拖垮 gather, 返回 error 结果."""
        orchestra = AgentOrchestra(main_agent)
        plan = ExecutionPlan(
            subtasks=[
                SubTask(id="bad", description="d", role="wizard"),
                SubTask(id="good", description="d", role="explorer"),
            ],
            mode=ExecutionMode.PARALLEL,
            original_task="t",
        )
        # good 子任务走真实创建 (explorer 有效) 但 run 会真实调用 LLM →
        # 用 monkeypatch 避免; 只验证 bad 被收容
        from conftest import StubSubAgent
        orchestra._create_subagent = lambda role: (
            StubSubAgent() if role == "explorer" else (_ for _ in ()).throw(ValueError("unknown role"))
        )
        results = asyncio.run(orchestra.execute(plan))
        statuses = {r.subtask_id: r for r in results}
        assert statuses["bad"].success is False
        assert "creation failed" in (statuses["bad"].error or "")
        assert statuses["good"].success is True

    def test_semaphore_clamped_when_max_parallel_zero(self, main_agent, monkeypatch):
        """orchestra_max_parallel=0 不应导致 Semaphore(0) 死锁."""
        orchestra = AgentOrchestra(main_agent)
        orchestra.config.orchestra_max_parallel = 0

        async def _fake_execute(subtask, hooks):
            return SubAgentResult(subtask_id=subtask.id, success=True, summary="s", full_result="r")

        monkeypatch.setattr(orchestra, "_execute_single_subtask", _fake_execute)
        plan = ExecutionPlan(
            subtasks=[SubTask(id="a", description="d", role="explorer")],
            mode=ExecutionMode.PARALLEL,
        )
        results = asyncio.run(asyncio.wait_for(orchestra.execute(plan), timeout=5))
        assert len(results) == 1

    def test_on_error_hook_fires_on_timeout(self, main_agent, stub_subagent_factory):
        orchestra = AgentOrchestra(main_agent)
        orchestra.config.subagent_timeout_seconds = 0.05
        stub_subagent_factory(orchestra, factory=lambda role: StubSubAgent(delay=1.0))
        errors: list[str] = []

        async def _on_error(st, exc):
            errors.append(st.id)

        hooks = SubtaskHooks(on_subtask_error=_on_error)
        plan = ExecutionPlan(
            subtasks=[SubTask(id="slow", description="d", role="explorer")],
            mode=ExecutionMode.PARALLEL,
        )
        asyncio.run(orchestra.execute(plan, hooks=hooks))
        assert errors == ["slow"]

    def test_decompose_retries_once_then_succeeds(self, main_agent, mock_llm):
        bad = MagicMock(); bad.content = "not json"
        good = MagicMock(); good.content = json.dumps({
            "subtasks": [{"id": "a", "description": "d", "role": "explorer"}],
        })
        mock_llm.invoke.side_effect = [bad, good]
        orchestra = AgentOrchestra(main_agent)
        plan = asyncio.run(orchestra.decompose("t", ExecutionMode.HYBRID))
        assert plan.subtasks[0].id == "a"
        assert mock_llm.invoke.call_count == 2
        # 重试 prompt 附带了解析失败提示
        retry_prompt = mock_llm.invoke.call_args_list[1][0][0][0]["content"]
        assert "无法解析" in retry_prompt

    def test_fallback_plan_structure(self, main_agent):
        orchestra = AgentOrchestra(main_agent)
        plan = orchestra._fallback_plan("my task", ExecutionMode.PIPELINE, reason="bad json")
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].role == "explorer"
        assert "my task" in plan.subtasks[0].description
        assert "bad json" in plan.subtasks[0].context_hint
        assert plan.stages == [["exp-fallback"]]
        assert plan.mode is ExecutionMode.PIPELINE

    def test_build_subtask_prompt_with_hint(self, main_agent):
        st = SubTask(id="a", description="do it", role="explorer", context_hint="ctx")
        prompt = AgentOrchestra._build_subtask_prompt(st)
        assert "do it" in prompt and "ctx" in prompt
        st2 = SubTask(id="b", description="do it", role="explorer")
        assert AgentOrchestra._build_subtask_prompt(st2) == "do it"

    def test_run_confirm_hook_accept_executes(self, main_agent, plan_llm, stub_subagent_factory):
        main_agent.llm = plan_llm
        orchestra = AgentOrchestra(main_agent)
        created = stub_subagent_factory(orchestra)
        seen_plans: list = []
        out = asyncio.run(
            orchestra.run("task", ExecutionMode.HYBRID,
                          confirm_hook=lambda p: seen_plans.append(p) or True)
        )
        assert isinstance(out, str)
        assert len(seen_plans) == 1
        assert len(created) == 1

    def test_aggregate_prompt_contains_original_task(self, main_agent, mock_llm):
        response = MagicMock(); response.content = "ans"
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)
        plan = _make_plan()
        asyncio.run(orchestra.aggregate(plan, []))
        prompt = mock_llm.invoke.call_args[0][0][0]["content"]
        assert "analyze" in prompt  # original_task

    def test_create_orchestra_factory(self, main_agent):
        from hello_agents.agents.factory import create_orchestra

        orchestra = create_orchestra(main_agent)
        assert isinstance(orchestra, AgentOrchestra)
        assert orchestra.main_agent is main_agent
        assert orchestra.config is main_agent.config
