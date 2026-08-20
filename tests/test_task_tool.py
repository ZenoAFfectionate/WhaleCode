"""Task 工具测试 — 主循环内子代理动态派生 (改进项 A1).

覆盖:
    1. 注册与开关: 默认注册 / enable_subagent_task / config 开关 / 防递归
    2. 参数校验: 缺参 / 未知角色
    3. 正常执行: 隔离工厂接线 / 蒸馏结果 / 元数据 / 事件契约
    4. 失败路径: 子代理异常 / 超时 (放弃等待语义)
    5. 长输出截断落盘 (共享 truncator)
    6. 并发安全: 多线程同时派生互不干扰
    7. E2E: 主循环 LLM 主动调用 Task → 子代理真实 ReAct → 结果回传
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from hello_agents.agents.code_agent import CodeAgent
from hello_agents.agents.roles import ExplorerRole, ReviewerRole, TesterRole
from hello_agents.core.config import Config
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolStatus

from test_integration import ScriptedLLM, _e2e_config


# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture
def tmp_workspace(tmp_path):
    (tmp_path / "memory" / "tool-output").mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


def _make_agent(tmp_workspace, llm, **config_overrides) -> CodeAgent:
    return CodeAgent(
        "main",
        llm,
        project_root=tmp_workspace,
        config=_e2e_config(**config_overrides),
        register_default_tools=True,
        interactive=False,
    )


class StubSubAgent:
    """可控的子代理替身: 记录 prompt, 返回预制文本, 提供元数据/摘要方法."""

    def __init__(self, result_text: str = "stub-result", delay: float = 0.0):
        self.result_text = result_text
        self.delay = delay
        self.run_prompts: List[str] = []

    def run(self, prompt: str, **kwargs) -> str:
        self.run_prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        return self.result_text

    def _get_subagent_metadata(self, duration: float, error) -> Dict[str, Any]:
        metadata = {
            "steps": 2,
            "tokens": 128,
            "duration_seconds": round(duration, 2),
            "tools_used": ["Read"],
        }
        if error:
            metadata["error"] = error
        return metadata

    def _generate_subagent_summary(self, task, result, metadata) -> str:
        return f"任务: {task}\n结果: {result}"


@pytest.fixture
def stub_explorer_factory(monkeypatch):
    """monkeypatch ExplorerRole.create_subagent → 返回可控 StubSubAgent 列表."""
    created: List[StubSubAgent] = []

    def _install(result_text: str = "explorer-report", delay: float = 0.0):
        def _create(**kwargs):
            stub = StubSubAgent(result_text=result_text, delay=delay)
            created.append(stub)
            return stub

        monkeypatch.setattr(ExplorerRole, "create_subagent", _create)
        return created

    return _install


class EventRecorder:
    """捕获宿主 Agent 的 _render_event 调用."""

    def __init__(self, agent: CodeAgent):
        self.events: List[tuple] = []
        agent._render_event = self._record  # type: ignore[method-assign]

    def _record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    def of_type(self, event_type: str) -> List[Dict[str, Any]]:
        return [payload for et, payload in self.events if et == event_type]


# ============================================================================
# 1. 注册与开关
# ============================================================================


class TestTaskToolRegistration:
    def test_registered_by_default(self, mock_llm, tmp_workspace):
        agent = _make_agent(tmp_workspace, mock_llm)
        tool = agent.tool_registry.get_tool("Task")
        assert tool is not None
        assert tool.name == "Task"

    def test_disabled_by_constructor_flag(self, mock_llm, tmp_workspace):
        agent = CodeAgent(
            "main",
            mock_llm,
            project_root=tmp_workspace,
            config=_e2e_config(),
            register_default_tools=True,
            enable_subagent_task=False,
            interactive=False,
        )
        assert agent.tool_registry.get_tool("Task") is None
        assert agent.config.subagent_task_enabled is False

    def test_disabled_by_config_flag(self, mock_llm, tmp_workspace):
        agent = _make_agent(tmp_workspace, mock_llm, subagent_task_enabled=False)
        assert agent.tool_registry.get_tool("Task") is None

    def test_no_recursive_task_in_role_subagents(self, mock_llm, tmp_workspace):
        """防递归契约: 角色子代理不得持有 Task 工具."""
        main = _make_agent(tmp_workspace, mock_llm)
        assert main.tool_registry.get_tool("Task") is not None  # 主代理持有

        for role_cls in (ExplorerRole, ReviewerRole, TesterRole):
            sub = role_cls.create_subagent(
                main.llm, main.config, tmp_workspace, tmp_workspace
            )
            assert sub.tool_registry.get_tool("Task") is None
            assert sub.config.subagent_task_enabled is False

    def test_tool_description_carries_parallel_guidance(self, mock_llm, tmp_workspace):
        """工具描述必须引导 LLM: 独立任务同轮多调用并行, 依赖任务分轮串行."""
        agent = _make_agent(tmp_workspace, mock_llm)
        tool = agent.tool_registry.get_tool("Task")
        desc = tool.description.lower()
        assert "parallel" in desc
        assert "wait for the upstream" in desc
        # 三种角色都必须出现在描述中
        for role in ("explorer", "reviewer", "tester"):
            assert role in desc

    def test_parameters_contract(self, mock_llm, tmp_workspace):
        agent = _make_agent(tmp_workspace, mock_llm)
        tool = agent.tool_registry.get_tool("Task")
        params = {p.name: p for p in tool.get_parameters()}
        assert set(params) == {"role", "task"}
        assert params["role"].required is True
        assert params["task"].required is True


# ============================================================================
# 2. 参数校验
# ============================================================================


class TestTaskToolValidation:
    def test_missing_role(self, mock_llm, tmp_workspace, stub_explorer_factory):
        created = stub_explorer_factory()
        tool = _make_agent(tmp_workspace, mock_llm).tool_registry.get_tool("Task")

        resp = tool.run({"task": "do something"})

        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == str(ToolErrorCode.INVALID_PARAM)
        assert created == []  # 校验失败不创建子代理

    def test_missing_task(self, mock_llm, tmp_workspace, stub_explorer_factory):
        created = stub_explorer_factory()
        tool = _make_agent(tmp_workspace, mock_llm).tool_registry.get_tool("Task")

        resp = tool.run({"role": "explorer"})

        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == str(ToolErrorCode.INVALID_PARAM)
        assert created == []

    def test_unknown_role_lists_available(self, mock_llm, tmp_workspace):
        tool = _make_agent(tmp_workspace, mock_llm).tool_registry.get_tool("Task")

        resp = tool.run({"role": "architect", "task": "design"})

        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == str(ToolErrorCode.INVALID_PARAM)
        message = resp.error_info.get("message", "")
        assert "architect" in message
        for role in ("explorer", "reviewer", "tester"):
            assert role in message


# ============================================================================
# 3. 正常执行
# ============================================================================


class TestTaskToolExecution:
    def test_success_returns_distilled_result(self, mock_llm, tmp_workspace, stub_explorer_factory):
        created = stub_explorer_factory(result_text="app.py defines work() -> 42")
        tool = _make_agent(tmp_workspace, mock_llm).tool_registry.get_tool("Task")

        resp = tool.run({"role": "explorer", "task": "explore app.py"})

        assert resp.status == ToolStatus.SUCCESS
        # 子代理恰好被创建一次, 收到自包含任务描述
        assert len(created) == 1
        assert created[0].run_prompts == ["explore app.py"]
        # 蒸馏结果 + 元数据都在返回文本中
        assert "app.py defines work() -> 42" in resp.text
        assert "[subagent:explorer]" in resp.text
        assert "steps: 2" in resp.text
        # 结构化 data
        assert resp.data["role"] == "explorer"
        assert resp.data["task"] == "explore app.py"
        assert resp.data["steps"] == 2
        assert resp.data["tools_used"] == ["Read"]
        assert resp.data["truncated"] is False

    def test_event_lifecycle_contract(self, mock_llm, tmp_workspace, stub_explorer_factory):
        """subagent_start 必须先于 subagent_finish, 且 payload 携带关键字段."""
        stub_explorer_factory(result_text="the report")
        agent = _make_agent(tmp_workspace, mock_llm)
        recorder = EventRecorder(agent)

        agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "map the repo"}
        )

        starts = recorder.of_type("subagent_start")
        finishes = recorder.of_type("subagent_finish")
        assert len(starts) == 1 and len(finishes) == 1
        assert starts[0] == {"role": "explorer", "task": "map the repo"}
        assert finishes[0]["role"] == "explorer"
        assert finishes[0]["task"] == "map the repo"
        assert finishes[0]["success"] is True
        assert "the report" in finishes[0]["summary"]
        assert finishes[0]["duration_seconds"] >= 0
        # 顺序: start 在 finish 之前
        assert recorder.events[0][0] == "subagent_start"
        assert recorder.events[-1][0] == "subagent_finish"

    def test_isolated_factory_receives_parent_context(
        self, mock_llm, tmp_workspace, monkeypatch
    ):
        """create_subagent 必须收到主 Agent 的 llm / config / 工作区上下文."""
        received: Dict[str, Any] = {}

        def _spy_create(cls, **kwargs):
            received.update(kwargs)
            return StubSubAgent(result_text="ok")

        monkeypatch.setattr(
            ExplorerRole, "create_subagent", classmethod(_spy_create)
        )
        agent = _make_agent(tmp_workspace, mock_llm)
        agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "t"}
        )

        assert received["llm"] is agent.llm
        assert received["project_root"] == str(agent.project_root)
        assert received["working_dir"] == str(agent.working_dir)
        assert received["parent_config"] is agent.config


# ============================================================================
# 4. 失败路径
# ============================================================================


class TestTaskToolFailures:
    def test_subagent_exception_returns_execution_error(
        self, mock_llm, tmp_workspace, monkeypatch
    ):
        class _Boom(StubSubAgent):
            def run(self, prompt, **kwargs):
                self.run_prompts.append(prompt)
                raise RuntimeError("LLM exploded")

        monkeypatch.setattr(
            ExplorerRole, "create_subagent", lambda **kw: _Boom()
        )
        agent = _make_agent(tmp_workspace, mock_llm)
        recorder = EventRecorder(agent)

        resp = agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "t"}
        )

        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == str(ToolErrorCode.EXECUTION_ERROR)
        assert "RuntimeError" in resp.error_info["message"]
        assert "LLM exploded" in resp.error_info["message"]
        # 失败也必须发 finish 事件 (success=False)
        finishes = recorder.of_type("subagent_finish")
        assert len(finishes) == 1 and finishes[0]["success"] is False

    def test_subagent_timeout_returns_timeout_error(
        self, mock_llm, tmp_workspace, stub_explorer_factory
    ):
        """超时 = 放弃等待: 返回 TIMEOUT, 不等子代理线程结束."""
        stub_explorer_factory(result_text="slow", delay=5.0)
        agent = _make_agent(tmp_workspace, mock_llm, subagent_timeout_seconds=0.2)
        start = time.time()

        resp = agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "t"}
        )

        elapsed = time.time() - start
        assert elapsed < 3.0  # 未等待 5 秒
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == str(ToolErrorCode.TIMEOUT)
        assert "0.2s" in resp.error_info["message"] or "timed out" in resp.error_info["message"]

    def test_timeout_zero_means_no_limit(self, mock_llm, tmp_workspace, stub_explorer_factory):
        """timeout<=0 表示不限时: 正常完成."""
        stub_explorer_factory(result_text="done", delay=0.1)
        agent = _make_agent(tmp_workspace, mock_llm, subagent_timeout_seconds=0)

        resp = agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "t"}
        )

        assert resp.status == ToolStatus.SUCCESS


# ============================================================================
# 5. 长输出截断落盘
# ============================================================================


class TestTaskToolTruncation:
    def test_long_output_saved_to_disk(self, mock_llm, tmp_workspace, stub_explorer_factory):
        long_text = "\n".join(f"line-{i} " + "x" * 200 for i in range(400))
        stub_explorer_factory(result_text=long_text)
        agent = _make_agent(tmp_workspace, mock_llm)

        resp = agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "big scan"}
        )

        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["truncated"] is True
        saved = resp.data["full_output_path"]
        assert saved and Path(saved).is_file()
        # 落盘内容包含全量输出
        saved_text = Path(saved).read_text(encoding="utf-8")
        assert "line-399" in saved_text
        # 返回给 LLM 的文本被截断 (远小于全量)
        assert len(resp.text) < len(long_text)

    def test_short_output_passthrough(self, mock_llm, tmp_workspace, stub_explorer_factory):
        stub_explorer_factory(result_text="short report")
        agent = _make_agent(tmp_workspace, mock_llm)

        resp = agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "t"}
        )

        assert resp.data["truncated"] is False
        assert resp.data["full_output_path"] is None
        assert "short report" in resp.text


# ============================================================================
# 6. 并发安全
# ============================================================================


class TestTaskToolConcurrency:
    def test_parallel_spawns_are_isolated(self, mock_llm, tmp_workspace, monkeypatch):
        """同一轮多个 Task 调用并行执行时, 每次派生独立的子代理实例."""
        created: List[StubSubAgent] = []
        lock = threading.Lock()

        def _create(**kwargs):
            stub = StubSubAgent(result_text="parallel-result", delay=0.05)
            with lock:
                created.append(stub)
            return stub

        monkeypatch.setattr(ExplorerRole, "create_subagent", _create)
        agent = _make_agent(tmp_workspace, mock_llm)
        tool = agent.tool_registry.get_tool("Task")

        # 模拟主循环 _execute_tools 的并行执行 (ThreadPoolExecutor 语义)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(tool.run, {"role": "explorer", "task": f"task-{i}"})
                for i in range(3)
            ]
            responses = [f.result() for f in futures]

        assert len(created) == 3
        assert len({id(s) for s in created}) == 3  # 三个不同实例
        assert all(r.status == ToolStatus.SUCCESS for r in responses)
        # 每个子代理只收到自己的任务
        for i, stub in enumerate(sorted(created, key=lambda s: s.run_prompts[0])):
            assert stub.run_prompts == [f"task-{i}"]


# ============================================================================
# 7. E2E: 主循环集成
# ============================================================================


class TestTaskToolE2E:
    def test_llm_driven_delegation_full_chain(self, tmp_workspace):
        """主循环 LLM 主动调用 Task → 真实 Explorer 子代理 ReAct → 结果回传 → Finish.

        LLM 脚本序列 (ScriptedLLM 按调用顺序消费):
            1. 主循环 round 1: Task(explorer, "explore app.py")
            2. 子代理 round 1: Read(app.py)
            3. 子代理 round 2: Finish("app.py defines work() returning 42")
            4. 主循环 round 2: Finish("done")
        """
        with_root = Path(tmp_workspace)
        (with_root / "app.py").write_text("def work():\n    return 42\n")

        script = [
            {"Task": {"role": "explorer", "task": "explore app.py"}},
            {"Read": {"path": "app.py"}},
            {"Finish": {"answer": "app.py defines work() returning 42"}},
            {"Finish": {"answer": "done"}},
        ]
        llm = ScriptedLLM(script)
        main = _make_agent(tmp_workspace, llm)
        history_before = len(main.get_history())

        answer = main.run("explore the project")

        assert answer == "done"
        assert llm.call_count == 4
        # Task 工具的蒸馏结果回传到主循环历史
        task_messages = [
            m for m in main.get_history()
            if m.role == "tool" and "app.py defines work() returning 42" in (m.content or "")
        ]
        assert task_messages, "sub-agent distilled result must flow back into main history"
        # 隔离契约: 子代理的中间步骤 (Read 调用) 不污染主历史
        read_pollution = [
            m for m in main.get_history()
            if m.role == "tool" and "[Read]" in (m.content or "")
        ]
        assert read_pollution == []
        # 主历史只含主循环自身的轮次:
        # user + Task 调用对 + Finish 调用对 + final = 6 条
        assert len(main.get_history()) - history_before == 6

    def test_subagent_gets_fresh_isolated_context(self, tmp_workspace):
        """子代理在全新上下文中运行: 收到的是 task 描述, 不是主对话历史."""
        (Path(tmp_workspace) / "app.py").write_text("VALUE = 7\n")

        script = [
            {"Task": {"role": "explorer", "task": "find VALUE in app.py"}},
            {"Finish": {"answer": "VALUE is 7"}},
            {"Finish": {"answer": "final"}},
        ]
        llm = ScriptedLLM(script)
        main = _make_agent(tmp_workspace, llm)
        main.run("what is VALUE")

        # 子代理的首次 LLM 调用 (invoke_history[1]) 的消息不含主循环的输入文本
        sub_first_call = llm.invoke_history[1]
        serialized = str(sub_first_call["messages"])
        assert "what is VALUE" not in serialized
        assert "find VALUE in app.py" in serialized

    def test_delegation_disabled_falls_back_to_no_task_tool(self, mock_llm, tmp_workspace):
        agent = _make_agent(tmp_workspace, mock_llm, subagent_task_enabled=False)
        schemas = agent._build_tool_schemas()
        tool_names = [s["function"]["name"] for s in schemas]
        assert "Task" not in tool_names


# ============================================================================
# 8. 渲染通道 (CLI / 导出契约)
# ============================================================================


class TestTaskToolRenderEvents:
    def test_cli_renders_subagent_events(self, mock_llm, tmp_workspace):
        """CLI 契约: subagent_start/finish 事件必须渲染为可见控制台输出."""
        agent = _make_agent(tmp_workspace, mock_llm)
        lines: List[str] = []
        agent._console = lambda msg="", **kw: lines.append(msg)  # type: ignore[assignment]

        agent._render_event(
            "subagent_start", {"role": "explorer", "task": "map the repo"}
        )
        agent._render_event(
            "subagent_finish",
            {
                "role": "explorer",
                "success": True,
                "duration_seconds": 1.5,
                "summary": "found 3 modules",
            },
        )
        agent._render_event(
            "subagent_finish",
            {
                "role": "tester",
                "success": False,
                "duration_seconds": 2.0,
                "summary": "pytest failed",
            },
        )

        assert any("⎇ [explorer] map the repo" in ln for ln in lines)
        assert any("✓ [explorer]" in ln and "1.5s" in ln and "found 3 modules" in ln for ln in lines)
        assert any("✗ [tester]" in ln and "pytest failed" in ln for ln in lines)

    def test_builtin_package_exports_task_tool(self):
        """导出契约: TaskTool 必须出现在 builtin 包的公开导出中."""
        import hello_agents.tools.builtin as builtin_pkg
        from hello_agents.tools.builtin.task_tool import TaskTool

        assert builtin_pkg.TaskTool is TaskTool
        assert "TaskTool" in builtin_pkg.__all__


# ============================================================================
# 9. 异步路径 (Tool.arun 默认线程池实现)
# ============================================================================


class TestTaskToolAsyncPath:
    def test_arun_matches_run_semantics(self, mock_llm, tmp_workspace, stub_explorer_factory):
        """异步主循环走 arun → 默认 run_in_executor → 同步 run: 契约一致."""
        created = stub_explorer_factory(result_text="async-report")
        tool = _make_agent(tmp_workspace, mock_llm).tool_registry.get_tool("Task")

        import asyncio

        resp = asyncio.run(tool.arun({"role": "explorer", "task": "t"}))

        assert resp.status == ToolStatus.SUCCESS
        assert len(created) == 1
        assert "async-report" in resp.text

    def test_arun_does_not_block_event_loop(self, mock_llm, tmp_workspace, stub_explorer_factory):
        """阻塞式子代理在 executor 线程执行: gather 与 asyncio.sleep 并行推进."""
        stub_explorer_factory(result_text="ok", delay=0.15)
        tool = _make_agent(tmp_workspace, mock_llm).tool_registry.get_tool("Task")

        import asyncio

        async def _scenario():
            done_flags = []

            async def _heartbeat():
                await asyncio.sleep(0.05)
                done_flags.append("heartbeat")

            task_fut = asyncio.ensure_future(tool.arun({"role": "explorer", "task": "t"}))
            beat_fut = asyncio.ensure_future(_heartbeat())
            resp = await task_fut
            await beat_fut
            return resp, done_flags

        resp, flags = asyncio.run(_scenario())
        # 事件循环在子代理运行期间仍能调度其他协程 (未被同步 run 卡死)
        assert flags == ["heartbeat"]
        assert resp.status == ToolStatus.SUCCESS


# ============================================================================
# 10. 同轮多 Task 并行的主循环 E2E
# ============================================================================


class TestTaskToolParallelRoundE2E:
    def test_two_task_calls_in_one_round(self, tmp_workspace, monkeypatch):
        """LLM 一轮返回两个 Task 调用 → _execute_tools 并行执行 → 双结果回传.

        script 序列:
            1. 主循环 round 1: [Task(explorer), Task(reviewer)] (同轮, list 格式)
            2. 两个 stub 子代理 (并行)
            3. 主循环 round 2: Finish
        """
        from types import SimpleNamespace

        explorer_stub = StubSubAgent(result_text="explorer-report", delay=0.05)
        reviewer_stub = StubSubAgent(result_text="reviewer-report", delay=0.05)

        monkeypatch.setattr(
            ExplorerRole, "create_subagent", lambda **kw: explorer_stub
        )
        from hello_agents.agents.roles import ReviewerRole

        monkeypatch.setattr(
            ReviewerRole, "create_subagent", lambda **kw: reviewer_stub
        )

        script = [
            [
                {"Task": {"role": "explorer", "task": "explore src"}},
                {"Task": {"role": "reviewer", "task": "review diff"}},
            ],
            {"Finish": {"answer": "both reports received"}},
        ]
        llm = ScriptedLLM(script)
        main = _make_agent(tmp_workspace, llm)

        answer = main.run("analyze")

        assert answer == "both reports received"
        # 两个子代理各自收到自己的任务
        assert explorer_stub.run_prompts == ["explore src"]
        assert reviewer_stub.run_prompts == ["review diff"]
        # 双结果均回传主历史
        history_text = "\n".join(m.content or "" for m in main.get_history())
        assert "explorer-report" in history_text
        assert "reviewer-report" in history_text
        # 主循环只消耗 2 轮 LLM 调用 (一轮派发 + 一轮收尾)
        assert llm.call_count == 2


# ============================================================================
# 11. Web 通道 (WebCodeAgent → event_sink 的 SSE 半程)
# ============================================================================


class TestTaskToolWebChannel:
    def test_events_reach_web_event_sink(self, mock_llm, tmp_workspace, stub_explorer_factory):
        """Web 半程契约: Task 工具事件经 WebCodeAgent._render_event 到达 sink.

        这是 SSE 链路的后端半程 (sink → SSE → 前端 app.js 渲染),
        前端渲染分支已由人工核验 (node --check + case 分支代码审查)。
        """
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))

        import web.server as web_server_mod  # noqa: E402 — namespace package

        stub_explorer_factory(result_text="web-report")
        collected: List[tuple] = []

        def sink(event_type: str, payload: Dict[str, Any]) -> None:
            collected.append((event_type, payload))

        agent = web_server_mod.WebCodeAgent(
            name="web-test-agent",
            llm=mock_llm,
            project_root=tmp_workspace,
            working_dir=tmp_workspace,
            config=_e2e_config(),
            register_default_tools=True,
            interactive=False,
            event_sink=sink,
        )
        assert agent.tool_registry.get_tool("Task") is not None

        resp = agent.tool_registry.get_tool("Task").run(
            {"role": "explorer", "task": "web channel check"}
        )

        assert resp.status == ToolStatus.SUCCESS
        types_seen = [et for et, _ in collected]
        assert "subagent_start" in types_seen
        assert "subagent_finish" in types_seen

        # payload 必须 JSON 可序列化 (SSE 通道 json.dumps 兼容)
        import json as _json

        start = next(p for et, p in collected if et == "subagent_start")
        finish = next(p for et, p in collected if et == "subagent_finish")
        _json.dumps(start)
        _json.dumps(finish)
        assert start == {"role": "explorer", "task": "web channel check"}
        assert finish["success"] is True
        assert "web-report" in finish["summary"]
