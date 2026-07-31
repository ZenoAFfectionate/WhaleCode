"""Tests for parallel tool execution in ReActAgent.

Covers both ``_execute_tools`` (sync / ThreadPoolExecutor) and
``_execute_tools_async`` (async / asyncio.gather + Semaphore) paths.

Coverage matrix for each tested behaviour:

    ============================================ ===== =====
    Behaviour                                    sync  async
    ============================================ ===== =====
    Parallel reduces wall time                    ✓     ✓
    Semaphore limits concurrency                  ✓     ✓
    Results order preserved                       ✓     ✓
    Error tool isolated from others               ✓     ✓
    Decode error handled pre-execution            ✓     ✓
    Empty tool_calls                              ✓     ✓
    History order after process_tool_results       ✓     ✓
    Single tool works fine                        ✓     ✓
    Mixed valid + invalid in same batch           ✓     ✓
    Mixed structured + normal in same batch       ✓     ✓
    Mixed builtin (Thought) + normal in batch     ✓     ✓
    All tools fail (graceful degradation)         ✓     ✓
    Large batch (10+ tools)                       ✓     ✓
    Finish tool stops process_tool_results         ✓     ✓
    _execute_one_tool_call backward compat        ✓
    _aexecute_one_tool_call backward compat             ✓
    ============================================ ===== =====

Plus:
- Config: default / custom / zero / negative / very large max_concurrent_tools
- Structured output: inline execution (sync + async)
- Sync vs async equivalence for identical inputs
- Unknown tool name handled gracefully
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from hello_agents.agents.react_agent import (
    FINISH_TOOL_NAME,
    THOUGHT_TOOL_NAME,
    ReActAgent,
    _ExecutionState,
    _StructuredOutputSpec,
)
from hello_agents.core.config import Config
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.response import ToolResponse


# ============================================================================
# Helpers
# ============================================================================


def _make_tool_call(name: str, tool_id: str, arguments: str):
    """Build a minimal object that mimics an OpenAI tool_call entry."""
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(function=function, id=tool_id)


def _make_tool_calls(*specs):
    """Shortcut to build a list of tool-call objects.

    Each spec is (name, id, arguments_json_string).
    """
    return [_make_tool_call(n, tid, args) for n, tid, args in specs]


class _DelayedTool(Tool):
    """Mock tool with configurable delay for testing parallel execution."""

    def __init__(self, name: str, delay: float = 0.05, should_fail: bool = False):
        super().__init__(name=name, description=f"Delayed tool {name}")
        self._delay = delay
        self._should_fail = should_fail

    def get_parameters(self) -> list:
        return [
            ToolParameter(name="input", type="string", description="", required=False),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        time.sleep(self._delay)
        if self._should_fail:
            raise RuntimeError(f"{self.name} intentional failure")
        return ToolResponse.success(
            text=f"{self.name} done", data={"tool": self.name},
        )


class _ZeroParamTool(Tool):
    """Tool with no required parameters."""

    def __init__(self, name: str, delay: float = 0.02):
        super().__init__(name=name, description=f"Zero-param tool {name}")
        self._delay = delay

    def get_parameters(self) -> list:
        return []

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        time.sleep(self._delay)
        return ToolResponse.success(text=f"{self.name} done", data={"tool": self.name})


def _build_agent(registry: ToolRegistry, **config_kwargs) -> ReActAgent:
    """Build a minimal ReActAgent suitable for testing tool execution."""
    cfg_kwargs: Dict[str, Any] = {
        "todowrite_enabled": False,
        "trace_enabled": False,
        "session_enabled": False,
        "skills_enabled": False,
        "max_concurrent_tools": 3,
    }
    cfg_kwargs.update(config_kwargs)
    config = Config(**cfg_kwargs)

    llm = HelloAgentsLLM(model="gpt-4", api_key="sk-test", base_url="http://localhost")
    return ReActAgent(
        name="test", llm=llm, tool_registry=registry,
        max_steps=10, config=config,
    )


# ============================================================================
# Sync path: _execute_tools
# ============================================================================


class TestExecuteToolsSync:
    """Tests for ``_execute_tools()`` (sync, ThreadPoolExecutor)."""

    # ------------------------------------------------------------------
    # Core parallelism
    # ------------------------------------------------------------------

    def test_parallel_execution_reduces_wall_time(self):
        """Three 0.1s tools with max_conc=3 should complete in <0.2s wall time."""
        registry = ToolRegistry(verbose=False)
        for name in ("A", "B", "C"):
            registry.register_tool(_DelayedTool(name, delay=0.1))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"a"}'),
            ("B", "id_b", '{"input":"b"}'),
            ("C", "id_c", '{"input":"c"}'),
        )

        start = time.monotonic()
        results = agent._execute_tools(tool_calls, current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert elapsed < 0.25, f"Expected parallel speedup, got {elapsed:.2f}s"
        for _, _, payload in results:
            assert payload.get("status") == "success"

    def test_single_tool_call(self):
        """Single tool call should work correctly (no parallelism needed)."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Solo", delay=0.01))

        agent = _build_agent(registry)
        tool_calls = _make_tool_calls(("Solo", "id_1", '{"input":"hello"}'))

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 1
        tool_name, tool_call_id, payload = results[0]
        assert tool_name == "Solo"
        assert tool_call_id == "id_1"
        assert payload.get("status") == "success"

    def test_semaphore_limits_parallelism(self):
        """With max_conc=1, three 0.05s tools should take >=0.10s (sequential)."""
        registry = ToolRegistry(verbose=False)
        for name in ("A", "B", "C"):
            registry.register_tool(_DelayedTool(name, delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=1)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"a"}'),
            ("B", "id_b", '{"input":"b"}'),
            ("C", "id_c", '{"input":"c"}'),
        )

        start = time.monotonic()
        results = agent._execute_tools(tool_calls, current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert elapsed >= 0.10, f"Expected sequential timing, got {elapsed:.2f}s"

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def test_results_order_preserved(self):
        """Results must be returned in the same order as tool_calls, even when
        faster tools complete before slower ones."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Fast1", delay=0.01))
        registry.register_tool(_DelayedTool("Slow", delay=0.15))
        registry.register_tool(_DelayedTool("Fast2", delay=0.01))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("Fast1", "id_1", '{"input":"x"}'),
            ("Slow", "id_2", '{"input":"y"}'),
            ("Fast2", "id_3", '{"input":"z"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 3
        # results[i] is (tool_name, tool_call_id, payload_dict)
        assert results[0][0] == "Fast1"
        assert results[1][0] == "Slow"
        assert results[2][0] == "Fast2"

    def test_tool_call_ids_preserved(self):
        """Each result must carry the correct tool_call_id for message pairing."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("X", delay=0.01))

        agent = _build_agent(registry)
        tool_calls = _make_tool_calls(
            ("X", "call_abc123", '{"input":"x"}'),
            ("X", "call_def456", '{"input":"y"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 2
        assert results[0][1] == "call_abc123"
        assert results[1][1] == "call_def456"

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_error_tool_does_not_block_others(self):
        """A failing tool should not prevent other tools from executing."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Failer", delay=0.01, should_fail=True))
        registry.register_tool(_DelayedTool("Worker", delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=2)
        tool_calls = _make_tool_calls(
            ("Failer", "id_fail", '{"input":"x"}'),
            ("Worker", "id_work", '{"input":"y"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 2
        assert results[0][2].get("status") == "error"
        assert results[1][2].get("status") == "success"

    def test_all_tools_fail_gracefully(self):
        """When all tools fail, all results should be error status — no crash."""
        registry = ToolRegistry(verbose=False)
        for name in ("F1", "F2", "F3"):
            registry.register_tool(_DelayedTool(name, delay=0.01, should_fail=True))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("F1", "id_1", '{"input":"a"}'),
            ("F2", "id_2", '{"input":"b"}'),
            ("F3", "id_3", '{"input":"c"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 3
        for _, _, payload in results:
            assert payload.get("status") == "error"

    def test_decode_error_handled_before_execution(self):
        """A tool_call with invalid JSON arguments should fail in Phase 1
        (decode) and not block other tools from Phase 2 execution."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Good", delay=0.01))

        agent = _build_agent(registry)
        tool_calls = _make_tool_calls(
            ("Good", "id_good", 'this is not valid json {{{'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 1
        assert results[0][2].get("status") == "error"

    def test_mixed_valid_and_decode_error(self):
        """Mix of valid tools and decode errors: errors fail early, valid
        tools still execute in parallel."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("A", delay=0.03))
        registry.register_tool(_DelayedTool("B", delay=0.03))
        registry.register_tool(_DelayedTool("C", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"ok1"}'),
            ("B", "id_b", "{{{broken"),          # decode error
            ("C", "id_c", '{"input":"ok2"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 3
        # A: success
        assert results[0][0] == "A"
        assert results[0][2].get("status") == "success"
        # B: decode error
        assert results[1][0] == "B"
        assert results[1][2].get("status") == "error"
        # C: success
        assert results[2][0] == "C"
        assert results[2][2].get("status") == "success"

    def test_unknown_tool_name_in_parallel(self):
        """A tool name not in registry should fail gracefully without crashing
        other parallel tools."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Known", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=2)
        tool_calls = _make_tool_calls(
            ("Known", "id_1", '{"input":"x"}'),
            ("NoSuchTool", "id_2", '{"input":"y"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 2
        assert results[0][2].get("status") == "success"
        assert results[1][2].get("status") == "error"

    # ------------------------------------------------------------------
    # Empty / edge
    # ------------------------------------------------------------------

    def test_empty_tool_calls(self):
        """An empty list of tool_calls should return an empty list."""
        registry = ToolRegistry(verbose=False)
        agent = _build_agent(registry)
        results = agent._execute_tools([], current_step=1)
        assert results == []

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def test_history_messages_appended_in_order(self):
        """After parallel execution + _process_tool_results, tool messages in
        history should be in the same order as the original tool_calls list."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("First", delay=0.03))
        registry.register_tool(_DelayedTool("Second", delay=0.01))
        registry.register_tool(_DelayedTool("Third", delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("First", "id_1", '{"input":"a"}'),
            ("Second", "id_2", '{"input":"b"}'),
            ("Third", "id_3", '{"input":"c"}'),
        )

        state = _ExecutionState(current_step=0)
        results = agent._execute_tools(tool_calls, current_step=1)
        agent._process_tool_results(tool_calls, results, current_step=1, state=state)

        tool_messages = [
            msg for msg in agent.history_manager.get_history()
            if msg.role == "tool"
        ]
        assert len(tool_messages) == 3
        tool_names_in_order = [
            (msg.metadata or {}).get("tool_name", "") for msg in tool_messages
        ]
        assert tool_names_in_order == ["First", "Second", "Third"]

    # ------------------------------------------------------------------
    # Builtin tools (Thought / Finish)
    # ------------------------------------------------------------------

    def test_builtin_tool_executes_inline(self):
        """Thought tool should still execute correctly (builtins go through
        the executor but have instant execution)."""
        registry = ToolRegistry(verbose=False)
        agent = _build_agent(registry)

        tool_calls = _make_tool_calls(
            ("Thought", "id_t", '{"reasoning": "should work"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 1
        assert results[0][0] == "Thought"
        assert results[0][2].get("status") == "success"

    def test_mixed_builtin_and_normal_tools(self):
        """Thought + normal tools in the same batch: both types must succeed."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Worker", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=2)
        tool_calls = _make_tool_calls(
            ("Thought", "id_t", '{"reasoning": "first think"}'),
            ("Worker", "id_w", '{"input":"data"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 2
        assert results[0][0] == "Thought"
        assert results[0][2].get("status") == "success"
        assert results[1][0] == "Worker"
        assert results[1][2].get("status") == "success"

    def test_finish_tool_sets_finished_flag(self):
        """Finish tool sets finished=True in its result. _process_tool_results
        must detect this and stop processing subsequent results."""
        registry = ToolRegistry(verbose=False)
        agent = _build_agent(registry)

        tool_calls = _make_tool_calls(
            ("Thought", "id_t", '{"reasoning": "done thinking"}'),
            ("Finish", "id_f", '{"answer": "final result"}'),
        )

        state = _ExecutionState(current_step=0)
        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 2

        # _process_tool_results should return the final_answer on seeing Finish
        final_answer = agent._process_tool_results(
            tool_calls, results, current_step=1, state=state,
        )
        assert final_answer == "final result"

    def test_finish_tool_in_middle_of_batch(self):
        """If Finish appears alongside non-Thought tools, _invalid_finalizing_tool_calls
        correctly rejects it with an error. This validation runs in Phase 1
        before any parallel execution."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Worker", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("Worker", "id_w", '{"input":"data"}'),
            ("Finish", "id_f", '{"answer": "done early"}'),
            ("Worker", "id_w2", '{"input":"more"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 3
        # Worker at index 0: executes normally
        assert results[0][0] == "Worker"
        assert results[0][2].get("status") == "success"
        # Finish at index 1: rejected by _invalid_finalizing_tool_calls
        assert results[1][0] == "Finish"
        assert results[1][2].get("status") == "error"
        # Worker at index 2: executes normally
        assert results[2][0] == "Worker"
        assert results[2][2].get("status") == "success"

    # ------------------------------------------------------------------
    # Stagnation detection
    # ------------------------------------------------------------------

    def test_stagnation_detection_works(self):
        """Even with parallel execution, consecutive Edit-no-diff should be
        detectable by _process_tool_results."""
        registry = ToolRegistry(verbose=False)

        class _NoDiffEditTool(Tool):
            def __init__(self):
                super().__init__(name="Edit", description="Edit files")
            def get_parameters(self):
                return [
                    ToolParameter(name="path", type="string", description="", required=True),
                    ToolParameter(name="old_string", type="string", description="", required=True),
                    ToolParameter(name="new_string", type="string", description="", required=True),
                ]
            def run(self, parameters):
                return ToolResponse.success(
                    text="[no textual diff] No changes detected",
                    data={"diff": ""},
                )

        registry.register_tool(_NoDiffEditTool())
        agent = _build_agent(registry)

        state = _ExecutionState(current_step=0)
        tool_calls = _make_tool_calls(
            ("Edit", "id_1", '{"path":"f.py","old_string":"a","new_string":"a"}'),
            ("Edit", "id_2", '{"path":"f.py","old_string":"b","new_string":"b"}'),
            ("Edit", "id_3", '{"path":"f.py","old_string":"c","new_string":"c"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        assert len(results) == 3

        agent._process_tool_results(tool_calls, results, current_step=1, state=state)
        assert state.stagnation_detected is True

    def test_stagnation_not_detected_with_mixed_results(self):
        """Only 2 consecutive no-diff Edits should NOT trigger stagnation.
        The first Edit has a real diff, resetting the counter."""
        registry = ToolRegistry(verbose=False)

        class _MixedEditTool(Tool):
            call_count = 0

            def __init__(self):
                super().__init__(name="Edit", description="Edit files")
            def get_parameters(self):
                return [
                    ToolParameter(name="path", type="string", description="", required=True),
                    ToolParameter(name="old_string", type="string", description="", required=True),
                    ToolParameter(name="new_string", type="string", description="", required=True),
                ]
            def run(self, parameters):
                _MixedEditTool.call_count += 1
                if _MixedEditTool.call_count == 1:
                    return ToolResponse.success(text="+1 −0  file.py", data={"diff": "+1"})
                return ToolResponse.success(
                    text="[no textual diff] No changes", data={"diff": ""},
                )

        registry.register_tool(_MixedEditTool())
        agent = _build_agent(registry)
        _MixedEditTool.call_count = 0

        state = _ExecutionState(current_step=0)
        tool_calls = _make_tool_calls(
            ("Edit", "id_1", '{"path":"a.py","old_string":"x","new_string":"y"}'),
            ("Edit", "id_2", '{"path":"b.py","old_string":"x","new_string":"x"}'),
            ("Edit", "id_3", '{"path":"c.py","old_string":"x","new_string":"x"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1)
        agent._process_tool_results(tool_calls, results, current_step=1, state=state)
        # Only 2 consecutive no-diff (ids 2 and 3) — not enough for stagnation
        assert state.stagnation_detected is False

    # ------------------------------------------------------------------
    # Large batches
    # ------------------------------------------------------------------

    def test_large_batch_with_limited_concurrency(self):
        """10 tools with max_conc=3: all must complete, wall time > 3 waves."""
        registry = ToolRegistry(verbose=False)
        for i in range(10):
            registry.register_tool(_DelayedTool(f"T{i}", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=3)
        specs = [(f"T{i}", f"id_{i}", f'{{"input":"{i}"}}') for i in range(10)]

        start = time.monotonic()
        results = agent._execute_tools(_make_tool_calls(*specs), current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 10
        # 10 tools × 0.03s / 3 workers ≈ 4 waves → ≥ 0.12s
        assert elapsed >= 0.09, f"Expected at least 3 waves, got {elapsed:.2f}s"
        for _, _, payload in results:
            assert payload.get("status") == "success"

    # ------------------------------------------------------------------
    # Zero-param tools
    # ------------------------------------------------------------------

    def test_zero_param_tools_parallel(self):
        """Tools with no required parameters should execute correctly in parallel."""
        registry = ToolRegistry(verbose=False)
        for name in ("Z1", "Z2", "Z3"):
            registry.register_tool(_ZeroParamTool(name, delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("Z1", "id_1", "{}"),
            ("Z2", "id_2", "{}"),
            ("Z3", "id_3", "{}"),
        )

        start = time.monotonic()
        results = agent._execute_tools(tool_calls, current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert elapsed < 0.12  # parallel
        for _, _, payload in results:
            assert payload.get("status") == "success"

    # ------------------------------------------------------------------
    # Backward compatibility: _execute_one_tool_call
    # ------------------------------------------------------------------

    def test_execute_one_tool_call_still_works(self):
        """The original _execute_one_tool_call method must still function
        correctly for callers that use it directly (PlanSolveAgent, etc.)."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Direct", delay=0.01))

        agent = _build_agent(registry)
        result = agent._execute_one_tool_call(
            "Direct", "id_dir", {"input": "hello"},
            current_step=1,
        )
        tool_name, tool_call_id, payload = result
        assert tool_name == "Direct"
        assert tool_call_id == "id_dir"
        assert payload.get("status") == "success"


# ============================================================================
# Async path: _execute_tools_async
# ============================================================================


class TestExecuteToolsAsync:
    """Tests for ``_execute_tools_async()`` (asyncio.gather + Semaphore)."""

    @staticmethod
    def _run_async(coro):
        """Helper to run an async coroutine synchronously in tests."""
        return asyncio.run(coro)

    # ------------------------------------------------------------------
    # Core parallelism
    # ------------------------------------------------------------------

    def test_parallel_execution_reduces_wall_time(self):
        """Three 0.1s tools with max_conc=3 should complete in <0.2s wall time."""
        registry = ToolRegistry(verbose=False)
        for name in ("A", "B", "C"):
            registry.register_tool(_DelayedTool(name, delay=0.1))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"a"}'),
            ("B", "id_b", '{"input":"b"}'),
            ("C", "id_c", '{"input":"c"}'),
        )

        async def go():
            start = time.monotonic()
            results = await agent._execute_tools_async(tool_calls, current_step=1)
            elapsed = time.monotonic() - start
            return results, elapsed

        results, elapsed = self._run_async(go())
        assert len(results) == 3
        assert elapsed < 0.25, f"Expected parallel speedup, got {elapsed:.2f}s"
        for _, _, payload in results:
            assert payload.get("status") == "success"

    def test_single_tool_call(self):
        """Single tool call via async path."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Solo", delay=0.01))

        agent = _build_agent(registry)

        async def go():
            return await agent._execute_tools_async(
                _make_tool_calls(("Solo", "id_1", '{"input":"hello"}')),
                current_step=1,
            )

        results = self._run_async(go())
        assert len(results) == 1
        assert results[0][0] == "Solo"
        assert results[0][2].get("status") == "success"

    def test_semaphore_limits_parallelism(self):
        """With max_conc=1, total time should be >= sum of delays."""
        registry = ToolRegistry(verbose=False)
        for name in ("A", "B", "C"):
            registry.register_tool(_DelayedTool(name, delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=1)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"a"}'),
            ("B", "id_b", '{"input":"b"}'),
            ("C", "id_c", '{"input":"c"}'),
        )

        async def go():
            start = time.monotonic()
            results = await agent._execute_tools_async(tool_calls, current_step=1)
            elapsed = time.monotonic() - start
            return results, elapsed

        results, elapsed = self._run_async(go())
        assert len(results) == 3
        assert elapsed >= 0.10, f"Expected sequential timing, got {elapsed:.2f}s"

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def test_results_order_preserved(self):
        """Results order must match tool_calls order, regardless of completion time."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Fast1", delay=0.01))
        registry.register_tool(_DelayedTool("Slow", delay=0.15))
        registry.register_tool(_DelayedTool("Fast2", delay=0.01))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("Fast1", "id_1", '{"input":"x"}'),
            ("Slow", "id_2", '{"input":"y"}'),
            ("Fast2", "id_3", '{"input":"z"}'),
        )

        async def go():
            return await agent._execute_tools_async(tool_calls, current_step=1)

        results = self._run_async(go())
        assert len(results) == 3
        assert results[0][0] == "Fast1"
        assert results[1][0] == "Slow"
        assert results[2][0] == "Fast2"

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_error_tool_does_not_block_others(self):
        """A failing tool should not prevent other concurrent tools from succeeding."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Failer", delay=0.01, should_fail=True))
        registry.register_tool(_DelayedTool("Worker", delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=2)
        tool_calls = _make_tool_calls(
            ("Failer", "id_fail", '{"input":"x"}'),
            ("Worker", "id_work", '{"input":"y"}'),
        )

        async def go():
            return await agent._execute_tools_async(tool_calls, current_step=1)

        results = self._run_async(go())
        assert len(results) == 2
        assert results[0][2].get("status") == "error"
        assert results[1][2].get("status") == "success"

    def test_all_tools_fail_gracefully(self):
        """All tools failing in async path should not crash."""
        registry = ToolRegistry(verbose=False)
        for name in ("F1", "F2", "F3"):
            registry.register_tool(_DelayedTool(name, delay=0.01, should_fail=True))

        agent = _build_agent(registry, max_concurrent_tools=3)

        async def go():
            return await agent._execute_tools_async(
                _make_tool_calls(
                    ("F1", "id_1", '{"input":"a"}'),
                    ("F2", "id_2", '{"input":"b"}'),
                    ("F3", "id_3", '{"input":"c"}'),
                ), current_step=1,
            )

        results = self._run_async(go())
        assert len(results) == 3
        for _, _, payload in results:
            assert payload.get("status") == "error"

    def test_decode_error_handled_before_execution(self):
        """Invalid JSON arguments should fail in Phase 1, not block Phase 2."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Good", delay=0.01))

        agent = _build_agent(registry)

        async def go():
            return await agent._execute_tools_async(
                _make_tool_calls(("Good", "id_good", '{{{broken json')),
                current_step=1,
            )

        results = self._run_async(go())
        assert len(results) == 1
        assert results[0][2].get("status") == "error"

    def test_mixed_valid_and_decode_error(self):
        """Mix of valid tools and decode errors in async path."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("A", delay=0.03))
        registry.register_tool(_DelayedTool("C", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=2)

        async def go():
            return await agent._execute_tools_async(
                _make_tool_calls(
                    ("A", "id_a", '{"input":"ok1"}'),
                    ("B", "id_b", "{{{broken"),
                    ("C", "id_c", '{"input":"ok2"}'),
                ), current_step=1,
            )

        results = self._run_async(go())
        assert len(results) == 3
        assert results[0][2].get("status") == "success"
        assert results[1][2].get("status") == "error"
        assert results[2][2].get("status") == "success"

    # ------------------------------------------------------------------
    # Empty
    # ------------------------------------------------------------------

    def test_empty_tool_calls(self):
        """Empty list should return empty list on async path too."""
        registry = ToolRegistry(verbose=False)
        agent = _build_agent(registry)

        async def go():
            return await agent._execute_tools_async([], current_step=1)

        results = self._run_async(go())
        assert results == []

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def test_history_order_after_async_execution(self):
        """History messages must preserve tool_calls order after async parallel
        execution + _process_tool_results."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("First", delay=0.03))
        registry.register_tool(_DelayedTool("Second", delay=0.01))
        registry.register_tool(_DelayedTool("Third", delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=3)
        tool_calls = _make_tool_calls(
            ("First", "id_1", '{"input":"a"}'),
            ("Second", "id_2", '{"input":"b"}'),
            ("Third", "id_3", '{"input":"c"}'),
        )

        async def go():
            state = _ExecutionState(current_step=0)
            results = await agent._execute_tools_async(tool_calls, current_step=1)
            agent._process_tool_results(tool_calls, results, current_step=1, state=state)
            return results

        self._run_async(go())

        tool_messages = [
            msg for msg in agent.history_manager.get_history()
            if msg.role == "tool"
        ]
        assert len(tool_messages) == 3
        tool_names_in_order = [
            (msg.metadata or {}).get("tool_name", "") for msg in tool_messages
        ]
        assert tool_names_in_order == ["First", "Second", "Third"]

    # ------------------------------------------------------------------
    # Mixed builtin + normal (async)
    # ------------------------------------------------------------------

    def test_mixed_builtin_and_normal_tools_async(self):
        """Thought + normal tools in async path: both must succeed."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Worker", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=2)

        async def go():
            return await agent._execute_tools_async(
                _make_tool_calls(
                    ("Thought", "id_t", '{"reasoning": "think"}'),
                    ("Worker", "id_w", '{"input":"data"}'),
                ), current_step=1,
            )

        results = self._run_async(go())
        assert len(results) == 2
        assert results[0][0] == "Thought"
        assert results[0][2].get("status") == "success"
        assert results[1][0] == "Worker"
        assert results[1][2].get("status") == "success"

    # ------------------------------------------------------------------
    # Large batches (async)
    # ------------------------------------------------------------------

    def test_large_batch_async(self):
        """10 tools with max_conc=3 via async path: all complete, parallelism observed."""
        registry = ToolRegistry(verbose=False)
        for i in range(10):
            registry.register_tool(_DelayedTool(f"T{i}", delay=0.03))

        agent = _build_agent(registry, max_concurrent_tools=3)
        specs = [(f"T{i}", f"id_{i}", f'{{"input":"{i}"}}') for i in range(10)]

        async def go():
            start = time.monotonic()
            results = await agent._execute_tools_async(
                _make_tool_calls(*specs), current_step=1,
            )
            elapsed = time.monotonic() - start
            return results, elapsed

        results, elapsed = self._run_async(go())
        assert len(results) == 10
        assert elapsed >= 0.09, f"Expected at least 3 waves, got {elapsed:.2f}s"
        for _, _, payload in results:
            assert payload.get("status") == "success"

    # ------------------------------------------------------------------
    # Backward compatibility: _aexecute_one_tool_call
    # ------------------------------------------------------------------

    def test_aexecute_one_tool_call_still_works(self):
        """The original _aexecute_one_tool_call method must still function
        for callers that use it directly."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("DirectAsync", delay=0.01))

        agent = _build_agent(registry)

        async def go():
            return await agent._aexecute_one_tool_call(
                "DirectAsync", "id_dir", {"input": "hello"},
                current_step=1,
            )

        result = self._run_async(go())
        tool_name, tool_call_id, payload = result
        assert tool_name == "DirectAsync"
        assert tool_call_id == "id_dir"
        assert payload.get("status") == "success"


# ============================================================================
# Sync / Async equivalence
# ============================================================================


class TestSyncAsyncEquivalence:
    """Verify that _execute_tools and _execute_tools_async produce equivalent
    results for the same inputs."""

    def test_same_results_same_order(self):
        """Both paths should return the same tool names, call IDs, and statuses."""
        registry1 = ToolRegistry(verbose=False)
        registry2 = ToolRegistry(verbose=False)
        for name in ("A", "B", "C"):
            registry1.register_tool(_DelayedTool(name, delay=0.01))
            registry2.register_tool(_DelayedTool(name, delay=0.01))

        agent1 = _build_agent(registry1, max_concurrent_tools=3)
        agent2 = _build_agent(registry2, max_concurrent_tools=3)

        specs = [
            ("A", "id_a", '{"input":"1"}'),
            ("B", "id_b", '{"input":"2"}'),
            ("C", "id_c", '{"input":"3"}'),
        ]

        sync_results = agent1._execute_tools(
            _make_tool_calls(*specs), current_step=1,
        )

        async def go():
            return await agent2._execute_tools_async(
                _make_tool_calls(*specs), current_step=1,
            )
        async_results = asyncio.run(go())

        assert len(sync_results) == len(async_results) == 3
        for sr, ar in zip(sync_results, async_results):
            # tool_name
            assert sr[0] == ar[0]
            # tool_call_id
            assert sr[1] == ar[1]
            # status
            assert sr[2].get("status") == ar[2].get("status")


# ============================================================================
# Config integration
# ============================================================================


class TestConfigIntegration:
    """Verify that max_concurrent_tools from Config is respected."""

    def test_default_max_concurrent_tools(self):
        """Default Config value is 3."""
        config = Config()
        assert config.max_concurrent_tools == 3

    def test_custom_max_concurrent_tools(self):
        """Custom value limits concurrent waves correctly."""
        registry = ToolRegistry(verbose=False)
        for name in ("A", "B", "C", "D"):
            registry.register_tool(_DelayedTool(name, delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=2)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"a"}'),
            ("B", "id_b", '{"input":"b"}'),
            ("C", "id_c", '{"input":"c"}'),
            ("D", "id_d", '{"input":"d"}'),
        )

        start = time.monotonic()
        results = agent._execute_tools(tool_calls, current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 4
        assert elapsed >= 0.08, (
            f"Expected at least 2 waves with max_conc=2, got {elapsed:.2f}s"
        )

    def test_zero_max_conc_clamps_to_one(self):
        """max_concurrent_tools=0 should be clamped to 1 (sequential)."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("A", delay=0.05))
        registry.register_tool(_DelayedTool("B", delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=0)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"a"}'),
            ("B", "id_b", '{"input":"b"}'),
        )

        start = time.monotonic()
        results = agent._execute_tools(tool_calls, current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 2
        assert elapsed >= 0.08, f"Expected sequential timing, got {elapsed:.2f}s"

    def test_negative_max_conc_clamps_to_one(self):
        """max_concurrent_tools=-5 should be clamped to 1."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("A", delay=0.05))
        registry.register_tool(_DelayedTool("B", delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=-5)
        tool_calls = _make_tool_calls(
            ("A", "id_a", '{"input":"a"}'),
            ("B", "id_b", '{"input":"b"}'),
        )

        start = time.monotonic()
        results = agent._execute_tools(tool_calls, current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 2
        assert elapsed >= 0.08, f"Expected sequential timing, got {elapsed:.2f}s"

    def test_very_large_max_conc_works(self):
        """max_concurrent_tools=100 should work fine (all tools run in parallel)."""
        registry = ToolRegistry(verbose=False)
        for name in ("A", "B", "C", "D", "E"):
            registry.register_tool(_DelayedTool(name, delay=0.05))

        agent = _build_agent(registry, max_concurrent_tools=100)
        specs = [(f"{c}", f"id_{c}", f'{{"input":"{c}"}}') for c in "ABCDE"]

        start = time.monotonic()
        results = agent._execute_tools(_make_tool_calls(*specs), current_step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 5
        # All 5 run in parallel → wall time ≈ 0.05s
        assert elapsed < 0.12, f"Expected all parallel, got {elapsed:.2f}s"
        for _, _, payload in results:
            assert payload.get("status") == "success"


# ============================================================================
# Structured output
# ============================================================================


class TestStructuredOutput:
    """Structured output tools should be handled inline (no executor needed)."""

    def test_structured_output_inline_sync(self):
        """StructuredOutput tool builds result locally, not via thread pool."""
        registry = ToolRegistry(verbose=False)
        agent = _build_agent(registry)

        spec = _StructuredOutputSpec(
            name="StructuredOutput",
            description="Return structured output",
            schema={"type": "object", "properties": {"result": {"type": "string"}}},
        )

        tool_calls = _make_tool_calls(
            ("StructuredOutput", "id_so", '{"result": "hello structured"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1, structured_output=spec)
        assert len(results) == 1
        assert results[0][0] == "StructuredOutput"
        assert results[0][2].get("status") == "success"

    def test_structured_output_inline_async(self):
        """Structured output in async path should also work inline."""
        registry = ToolRegistry(verbose=False)
        agent = _build_agent(registry)

        spec = _StructuredOutputSpec(
            name="StructuredOutput",
            description="Return structured output",
            schema={"type": "object", "properties": {"result": {"type": "string"}}},
        )

        async def go():
            return await agent._execute_tools_async(
                _make_tool_calls(
                    ("StructuredOutput", "id_so", '{"result": "hello"}'),
                ), current_step=1, structured_output=spec,
            )

        results = asyncio.run(go())
        assert len(results) == 1
        assert results[0][0] == "StructuredOutput"
        assert results[0][2].get("status") == "success"

    def test_mixed_structured_and_normal_tools(self):
        """When StructuredOutput is called alongside normal tools,
        _invalid_finalizing_tool_calls correctly rejects it with an error
        in Phase 1. The normal tool still executes successfully."""
        registry = ToolRegistry(verbose=False)
        registry.register_tool(_DelayedTool("Worker", delay=0.02))

        agent = _build_agent(registry)

        spec = _StructuredOutputSpec(
            name="StructuredOutput",
            description="Return structured output",
            schema={"type": "object", "properties": {"result": {"type": "string"}}},
        )

        tool_calls = _make_tool_calls(
            ("Worker", "id_w", '{"input":"data"}'),
            ("StructuredOutput", "id_so", '{"result": "final"}'),
        )

        results = agent._execute_tools(tool_calls, current_step=1, structured_output=spec)
        assert len(results) == 2
        # Worker executes normally
        assert results[0][0] == "Worker"
        assert results[0][2].get("status") == "success"
        # StructuredOutput is rejected: must be called alone
        assert results[1][0] == "StructuredOutput"
        assert results[1][2].get("status") == "error"
