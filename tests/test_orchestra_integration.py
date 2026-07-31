"""Orchestra 集成测试: 完整编排流程 / 阶段协作 / 取消 / 并发隔离."""

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
    SubTask,
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


def _three_stage_plan():
    return ExecutionPlan(
        subtasks=[
            SubTask(id="exp-1", description="探索模块结构", role="explorer"),
            SubTask(id="test-1", description="编写并运行测试", role="tester",
                    dependencies=["exp-1"]),
            SubTask(id="rev-1", description="复审测试质量", role="reviewer",
                    dependencies=["test-1"]),
        ],
        mode=ExecutionMode.HYBRID,
        stages=[["exp-1"], ["test-1"], ["rev-1"]],
        original_task="为模块补测试并检查质量",
    )


class TestOrchestraIntegration:
    def test_full_orchestra_flow(self, main_agent, plan_llm, stub_subagent_factory):
        """完整流程: decompose → execute → aggregate 均成功."""
        main_agent.llm = plan_llm
        orchestra = AgentOrchestra(main_agent)
        created = stub_subagent_factory(orchestra)

        answer = asyncio.run(orchestra.run("analyze the project", ExecutionMode.HYBRID))

        assert isinstance(answer, str) and answer
        # decompose 的计划为单个 explorer 子任务 → 恰好创建一个子 Agent
        assert len(created) == 1

    def test_reviewer_as_pipeline_stage(self, main_agent, stub_subagent_factory):
        """Explorer → Tester → Reviewer 三阶段上下文逐层传递."""
        orchestra = AgentOrchestra(main_agent)
        created = stub_subagent_factory(
            orchestra, factory=lambda role: StubSubAgent(result_text=f"{role}-output")
        )
        asyncio.run(orchestra.execute(_three_stage_plan()))

        exp_stub, test_stub, rev_stub = created
        # tester 收到 explorer 的结果
        assert "explorer-output" in test_stub.run_prompts[0]
        # reviewer 收到 tester 的结果
        assert "tester-output" in rev_stub.run_prompts[0]
        # explorer 无注入
        assert "上游阶段结果摘要" not in exp_stub.run_prompts[0]

    def test_concurrent_subagent_isolation(self, main_agent, stub_subagent_factory):
        """并行子 Agent 各自独立创建, 无共享实例."""
        orchestra = AgentOrchestra(main_agent)
        created = stub_subagent_factory(orchestra)
        plan = ExecutionPlan(
            subtasks=[
                SubTask(id=f"t-{i}", description="d", role="explorer") for i in range(4)
            ],
            mode=ExecutionMode.PARALLEL,
        )
        results = asyncio.run(orchestra.execute(plan))
        assert len(results) == 4
        assert len(created) == 4
        assert len({id(s) for s in created}) == 4  # 全部不同实例
        # 每个 stub 只收到自己的 prompt
        for stub in created:
            assert len(stub.run_prompts) == 1

    def test_cancellation_during_execution(self, main_agent, stub_subagent_factory):
        """execute 被取消 → CancelledError 传播, 已启动任务被 asyncio 取消."""
        orchestra = AgentOrchestra(main_agent)
        stub_subagent_factory(
            orchestra, factory=lambda role: StubSubAgent(delay=1.0)
        )
        plan = ExecutionPlan(
            subtasks=[SubTask(id="slow", description="d", role="explorer")],
            mode=ExecutionMode.PARALLEL,
        )

        async def _cancel_soon():
            task = asyncio.ensure_future(orchestra.execute(plan))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_cancel_soon())

    def test_end_to_end_three_roles_with_real_subagent_creation(
        self, main_agent, tmp_workspace
    ):
        """真实 Role.create_subagent 路径: 三个角色的子 Agent 都能被 Orchestra 创建."""
        orchestra = AgentOrchestra(main_agent)
        for role in ("explorer", "reviewer", "tester"):
            sub = orchestra._create_subagent(role)
            assert sub.config.trace_enabled is False
            assert sub.config.session_enabled is False
            assert str(sub.project_root) == str(main_agent.project_root)

    def test_aggregate_includes_stage_failures(
        self, main_agent, mock_llm, stub_subagent_factory
    ):
        """阶段失败后 aggregate 的 prompt 标注失败, 成功结果仍被保留."""
        response = MagicMock()
        response.content = "aggregated"
        mock_llm.invoke.return_value = response
        orchestra = AgentOrchestra(main_agent)

        def _factory(role):
            if role == "tester":
                stub = StubSubAgent()
                def _boom(prompt, **kwargs):
                    raise RuntimeError("test run failed")
                stub.run = _boom
                return stub
            return StubSubAgent()

        stub_subagent_factory(orchestra, factory=_factory)
        plan = _three_stage_plan()
        results = asyncio.run(orchestra.execute(plan))
        answer = asyncio.run(orchestra.aggregate(plan, results))
        assert answer == "aggregated"
        agg_prompt = mock_llm.invoke.call_args[0][0][0]["content"]
        assert "(失败)" in agg_prompt and "test-1" in agg_prompt
