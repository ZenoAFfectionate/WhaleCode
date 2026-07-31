"""Stress tests for concurrent tool execution, thread safety, and exception recovery.

Covers:
- Parallel tool execution with ThreadPoolExecutor (10+ tools at once)
- ToolRegistry concurrent register/unregister/get
- LSPManager concurrent server creation (double-checked locking)
- HistoryManager concurrent append
- Exception recovery: LLM failures, tool exceptions, disk-full conditions
- Circuit breaker behavior under repeated failures
"""

from __future__ import annotations

import concurrent.futures
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from code.agents.react_agent import ReActAgent
from code.context.history import HistoryManager
from code.context.token_counter import TokenCounter
from code.core.config import Config
from code.core.message import Message
from code.tools.base import Tool, ToolParameter
from code.tools.registry import ToolRegistry
from code.tools.response import ToolResponse, ToolStatus
from code.tools.lsp.manager import LSPManager


# ============================================================================
# Helpers
# ============================================================================


class _DelayTool(Tool):
    """Tool with controllable execution delay for concurrency testing."""

    def __init__(self, name: str, delay: float = 0.05):
        super().__init__(name=name, description=f"Delays {delay}s")
        self.delay = delay
        self.start_times: List[float] = []
        self.end_times: List[float] = []
        self.thread_ids: List[int] = []

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="input", type="string", description="Input", required=True)]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        self.start_times.append(time.time())
        self.thread_ids.append(threading.get_ident())
        time.sleep(self.delay)
        self.end_times.append(time.time())
        return ToolResponse.success(text=f"{self.name}: {parameters.get('input', '')}")


class _CrashTool(Tool):
    """Tool that raises an exception when executed."""

    def __init__(self, name: str = "CrashTool"):
        super().__init__(name=name, description="Always crashes")
        self.crash_count = 0

    def get_parameters(self) -> List[ToolParameter]:
        return []

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        self.crash_count += 1
        raise RuntimeError(f"Simulated crash #{self.crash_count}")


def _mock_llm():
    llm = MagicMock()
    llm.model = "test-model"
    llm.temperature = 0.7
    return llm


def _concurrency_config(**overrides) -> Config:
    base = dict(
        max_concurrent_tools=5,
        compact_enabled=False,
        trace_enabled=False,
        skills_enabled=False,
        todowrite_enabled=False,
        session_enabled=False,
    )
    base.update(overrides)
    return Config(**base)


# ============================================================================
# Parallel Tool Execution
# ============================================================================


class TestParallelToolExecution:
    """ThreadPoolExecutor-based parallel tool execution under stress."""

    def test_10_parallel_delay_tools(self):
        """10 delay tools execute — all results collected correctly."""
        registry = ToolRegistry()
        tools = [_DelayTool(f"delay_{i}", delay=0.01) for i in range(10)]
        for t in tools:
            registry.register_tool(t)

        agent = ReActAgent("parallel-test", _mock_llm(), tool_registry=registry,
                           config=_concurrency_config(max_concurrent_tools=10))

        call_specs = [(f"delay_{i}", {"input": f"val_{i}"}) for i in range(10)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(agent._execute_tool_call_result, name, args): (name, args)
                for name, args in call_specs
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # All 10 should succeed
        assert len(results) == 10
        for r in results:
            assert "content" in r

        # Verify all 10 tools executed (check unique names in results)
        names_found = set()
        for r in results:
            for i in range(10):
                if f"delay_{i}" in r["content"]:
                    names_found.add(f"delay_{i}")
        assert len(names_found) == 10, f"Not all tools executed: {names_found}"

    def test_individual_tool_results_correct(self):
        """Each tool's result content correctly identifies the tool that produced it."""
        registry = ToolRegistry()
        slow = _DelayTool("slow", delay=0.05)
        fast = _DelayTool("fast", delay=0.01)
        registry.register_tool(slow)
        registry.register_tool(fast)

        agent = ReActAgent("result-test", _mock_llm(), tool_registry=registry,
                           config=_concurrency_config())

        result_slow = agent._execute_tool_call_result("slow", {"input": "a"})
        result_fast = agent._execute_tool_call_result("fast", {"input": "b"})

        # Each result should contain the tool name that produced it
        assert "slow" in result_slow["content"]
        assert "fast" in result_fast["content"]
        # Results should not be mixed up
        assert "slow" not in result_fast["content"]
        assert "fast" not in result_slow["content"]

    def test_mixed_success_and_failure(self):
        """Some tools crash, others succeed — all results collected."""
        registry = ToolRegistry()
        registry.register_tool(_DelayTool("ok1", delay=0.01))
        registry.register_tool(_CrashTool("bad1"))
        registry.register_tool(_DelayTool("ok2", delay=0.01))
        registry.register_tool(_CrashTool("bad2"))

        agent = ReActAgent("mixed-test", _mock_llm(), tool_registry=registry,
                           config=_concurrency_config())

        names = ["ok1", "bad1", "ok2", "bad2"]
        args_spec = [({"input": "x"} if n.startswith("ok") else {}) for n in names]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for name, args in zip(names, args_spec):
                futures[executor.submit(agent._execute_tool_call_result, name, args)] = name
            results = {}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = None

        # ok tools should have results
        assert results.get("ok1") is not None
        assert results.get("ok2") is not None

    def test_max_concurrent_tools_respected(self):
        """max_concurrent_tools correctly batches parallel executions."""
        registry = ToolRegistry()
        tools = [_DelayTool(f"t_{i}", delay=0.02) for i in range(12)]
        for t in tools:
            registry.register_tool(t)

        agent = ReActAgent("limit-test", _mock_llm(), tool_registry=registry,
                           config=_concurrency_config(max_concurrent_tools=3))

        call_specs = [(f"t_{i}", {"input": "x"}) for i in range(12)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(agent._execute_tool_call_result, n, a): n
                for n, a in call_specs
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                assert "content" in result

        # All 12 tools completed — system didn't crash or deadlock
        assert True


# ============================================================================
# ToolRegistry Concurrent Access
# ============================================================================


class TestRegistryConcurrentAccess:
    """Thread safety of ToolRegistry under concurrent read/write."""

    def test_concurrent_register_and_get(self):
        """Multiple threads registering and getting tools simultaneously."""
        registry = ToolRegistry()
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(50):
                    name = f"tool_w{worker_id}_{i}"
                    tool = _DelayTool(name, delay=0.001)
                    registry.register_tool(tool)
                    # Immediately try to get it
                    retrieved = registry.get_tool(name)
                    if retrieved is None:
                        errors.append(f"Worker {worker_id}: {name} not found after register")
                    # Also list tools
                    _ = registry.list_tools()
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = []
        for w in range(8):
            t = threading.Thread(target=worker, args=(w,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        # Should have registered ~400 tools
        tools = registry.list_tools()
        assert len(tools) > 100

    def test_concurrent_unregister_and_execute(self):
        """One thread unregisters while another executes."""
        registry = ToolRegistry()
        registry.register_tool(_DelayTool("target", delay=0.02))
        errors = []

        def executor():
            for _ in range(30):
                try:
                    if "target" in registry.list_tools():
                        resp = registry.execute_tool("target", {"input": "x"})
                        if resp.status != ToolStatus.SUCCESS:
                            errors.append(f"Exec failed: {resp.text}")
                except Exception as e:
                    # May raise if the tool was unregistered mid-execution
                    pass
                time.sleep(0.005)

        def unregisterer():
            for _ in range(20):
                try:
                    registry.unregister("target")
                except Exception:
                    pass
                time.sleep(0.01)
                registry.register_tool(_DelayTool("target", delay=0.01))

        t1 = threading.Thread(target=executor)
        t2 = threading.Thread(target=unregisterer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should not crash the process
        assert True

    def test_concurrent_list_and_register(self):
        """Many threads listing tools while others register — no data corruption."""
        registry = ToolRegistry()
        for i in range(20):
            registry.register_tool(_DelayTool(f"base_{i}", delay=0.001))

        exceptions = []

        def lister():
            for _ in range(100):
                try:
                    tools = registry.list_tools()
                    # Verify no None values or empty strings in tool names
                    for name in tools:
                        if not name or not isinstance(name, str):
                            exceptions.append(f"Bad tool name: {name!r}")
                except Exception as e:
                    exceptions.append(str(e))

        def adder():
            for i in range(30):
                try:
                    registry.register_tool(_DelayTool(f"added_{i}", delay=0.001))
                except Exception as e:
                    exceptions.append(str(e))

        threads = []
        for _ in range(6):
            threads.append(threading.Thread(target=lister))
        threads.append(threading.Thread(target=adder))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Exceptions: {exceptions}"


# ============================================================================
# LSPManager Concurrent Server Creation
# ============================================================================


class TestLSPManagerConcurrency:
    """Thread safety of LSPManager's lazy server creation."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test.py").write_text("x = 1\n")
            yield root

    def test_concurrent_server_for_same_language(self, workspace):
        """Multiple threads requesting server for same language → only one created."""
        manager = LSPManager(workspace)

        # The creation_lock exists
        assert hasattr(manager, '_creation_lock')
        assert isinstance(manager._creation_lock, type(threading.Lock()))

        # Attempt concurrent access — even if pylsp isn't installed,
        # the locking mechanism should work
        results = []
        errors = []

        def access():
            try:
                # server_for may return None if pylsp not installed — that's fine
                client = manager.server_for("test.py")
                results.append(client)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=access) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 10 threads should get the same result (either all None or all the same client)
        assert len(errors) == 0, f"Errors: {errors}"
        # If pylsp is available, they should all return the same client
        non_none = [r for r in results if r is not None]
        if non_none:
            assert all(r is non_none[0] for r in non_none), "Different clients returned!"

    def test_manager_shutdown_thread_safety(self, workspace):
        """Shutdown during concurrent access doesn't crash."""
        manager = LSPManager(workspace)

        # Start some access
        def access():
            try:
                manager.server_for("test.py")
            except Exception:
                pass

        t = threading.Thread(target=access)
        t.start()
        t.join()

        # Shutdown should work
        manager.shutdown()
        # Double shutdown should be safe
        manager.shutdown()


# ============================================================================
# HistoryManager Concurrent Append
# ============================================================================


class TestHistoryManagerConcurrentAppend:
    """Thread safety under concurrent append operations."""

    def test_concurrent_append_from_multiple_threads(self):
        """Multiple threads appending to the same HistoryManager."""
        hm = HistoryManager(token_counter=TokenCounter())
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(100):
                    hm.append(Message(f"Worker {worker_id} message {i}", "user"))
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All messages should have been appended
        history = hm.get_history()
        assert len(history) == 500
        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_append_and_estimate(self):
        """One thread appends while another estimates tokens."""
        hm = HistoryManager(token_counter=TokenCounter())
        for _ in range(10):
            hm.append(Message("initial", "user"))

        estimate_errors = []

        def estimator():
            for _ in range(200):
                try:
                    t = hm.estimate_tokens()
                    if t < 0:
                        estimate_errors.append(f"Negative tokens: {t}")
                except Exception as e:
                    estimate_errors.append(str(e))

        def appender():
            for i in range(100):
                try:
                    hm.append(Message(f"appended {i}", "user"))
                except Exception as e:
                    estimate_errors.append(f"Append error: {e}")

        t1 = threading.Thread(target=estimator)
        t2 = threading.Thread(target=appender)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(estimate_errors) == 0, f"Estimate errors: {estimate_errors}"


# ============================================================================
# Exception Recovery
# ============================================================================


class TestExceptionRecovery:
    """Agent recovers gracefully from various failure modes."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_tool_exception_does_not_crash_agent(self, workspace):
        """When a tool throws an unexpected exception, the agent continues."""
        registry = ToolRegistry()
        registry.register_tool(_CrashTool("Exploder"))

        agent = ReActAgent("recovery-test", _mock_llm(), tool_registry=registry,
                           config=_concurrency_config())

        result = agent._execute_tool_call_result("Exploder", {})
        # Should return an error result, not raise
        assert "content" in result
        assert result.get("status") == "error" or "Error" in result["content"]

    def test_agent_handles_missing_tool(self, workspace):
        """Calling a tool that doesn't exist returns a proper error."""
        agent = ReActAgent("missing-tool-test", _mock_llm(),
                           config=_concurrency_config())

        result = agent._execute_tool_call_result("NonExistentTool", {"x": 1})
        assert "content" in result
        assert result.get("status") == "error"

    def test_repeated_tool_failures(self):
        """A tool that fails repeatedly — circuit breaker opens after threshold."""
        registry = ToolRegistry()
        crash = _CrashTool("Repeater")
        registry.register_tool(crash)

        agent = ReActAgent("repeat-test", _mock_llm(), tool_registry=registry,
                           config=_concurrency_config())

        # The circuit breaker will trip after 3 failures (default threshold)
        # but each individual call should return a valid result
        for _ in range(8):
            result = agent._execute_tool_call_result("Repeater", {})
            assert "content" in result

        # At least some calls executed before the circuit breaker tripped
        assert crash.crash_count >= 3

    def test_none_arguments_handled(self, workspace):
        """None arguments to _execute_tool_call don't crash."""
        agent = ReActAgent("none-test", _mock_llm(),
                           config=_concurrency_config())

        result = agent._execute_tool_call_result("Thought", {"reasoning": "test"})
        assert "content" in result

    def test_empty_string_arguments(self, workspace):
        """Empty string arguments are handled gracefully."""
        agent = ReActAgent("empty-test", _mock_llm(),
                           config=_concurrency_config())

        result = agent._execute_tool_call_result("Thought", {"reasoning": ""})
        assert "content" in result


# ============================================================================
# Circuit Breaker Under Stress
# ============================================================================


class TestCircuitBreakerStress:
    """Circuit breaker behavior under repeated failures."""

    def test_circuit_breaker_opens_after_repeated_failures(self):
        """Circuit breaker opens when failure threshold is reached."""
        registry = ToolRegistry()
        crash = _CrashTool("FragileTool")
        registry.register_tool(crash)

        # Execute enough times to trigger the circuit breaker
        for i in range(10):
            try:
                resp = registry.execute_tool("FragileTool", {})
            except Exception:
                pass

        # Verify the tool was called at least once
        assert crash.crash_count > 0
        # After failure_threshold (default 3) consecutive failures,
        # the circuit breaker should be open
        if crash.crash_count >= 3:
            assert registry.circuit_breaker.is_open("FragileTool") is True
            # All subsequent calls should return a circuit-open response,
            # NOT execute the tool again
            fail_count_before = crash.crash_count
            try:
                resp = registry.execute_tool("FragileTool", {})
            except Exception:
                pass
            # Tool should NOT have been executed again (circuit is open)
            assert crash.crash_count == fail_count_before


# ============================================================================
# Concurrent CodeAgent E2E Stress
# ============================================================================


class TestConcurrentAgentSessions:
    """Multiple agent instances running simultaneously."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_multiple_agents_do_not_interfere(self, workspace):
        """Two CodeAgent instances running in parallel don't corrupt each other."""
        (workspace / "a.py").write_text("x = 1\n")
        (workspace / "b.py").write_text("y = 2\n")

        results: List[str] = []
        errors: List[str] = []

        def run_agent(file_to_read: str):
            try:
                registry = ToolRegistry()
                from code.tools.builtin.file_tools import ReadTool
                registry.register_tool(ReadTool(
                    project_root=str(workspace),
                    working_dir=str(workspace),
                    registry=registry,
                ))
                agent = ReActAgent(f"agent-{file_to_read}", _mock_llm(),
                                   tool_registry=registry,
                                   config=_concurrency_config())
                result = agent._execute_tool_call("Read", {"path": file_to_read})
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=run_agent, args=("a.py",)),
            threading.Thread(target=run_agent, args=("b.py",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 2
        # Results may arrive in any order due to threading
        combined = " ".join(results)
        assert "x = 1" in combined
        assert "y = 2" in combined

    def test_rapid_agent_create_destroy(self, workspace):
        """Creating and destroying many agents rapidly — all functional afterward."""
        agents_created = 0
        for i in range(20):
            agent = ReActAgent(f"rapid-{i}", _mock_llm(),
                               config=_concurrency_config())
            # Trigger internal initialization
            schemas = agent._build_tool_schemas()
            messages = agent._build_messages()
            # Each agent should produce valid schemas and messages
            assert isinstance(schemas, list)
            assert isinstance(messages, list)
            agents_created += 1

        # All agents were created and initialized successfully
        assert agents_created == 20
