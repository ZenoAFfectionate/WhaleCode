"""Tests for benchmark progress display — step tracking, status updates, and
the full event → progress → display chain.

Covers:
- BenchmarkProgressManager: begin_task reset, update(max), status_line format
- describe_progress_update: step extraction from all event types
- build_progress_update: structured update creation
- Step consistency across rounds (multi-round benchmarks reset step)
- LCB6 per-round step counter behavior
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import pytest

from hello_agents.benchmark._utils import (
    BenchmarkProgressManager,
    build_progress_update,
    describe_progress_update,
    _clip_display,
    _human_elapsed,
)


# ═══════════════════════════════════════════════════════════════════
# 1. BenchmarkProgressManager — step lifecycle
# ═══════════════════════════════════════════════════════════════════

class TestProgressManagerStepLifecycle:
    """Verify that begin_task resets step and update respects max()."""

    @pytest.fixture
    def mgr(self):
        return BenchmarkProgressManager(benchmark_name="test", total=10)

    def test_initial_step_is_zero(self, mgr):
        assert mgr.current_step == 0

    def test_begin_task_resets_step(self, mgr):
        mgr.current_step = 42  # simulate previous task's step
        mgr.begin_task(index=3, task_id="task-3")
        assert mgr.current_step == 0
        assert mgr.current_task_id == "task-3"
        assert mgr.current_status == "Running"
        assert mgr.current_detail == "starting"

    def test_update_step_increases(self, mgr):
        mgr.update(step=5, status="Running", detail="Thinking")
        assert mgr.current_step == 5

    def test_update_step_uses_max_not_replace(self, mgr):
        """The display uses max() so step never goes backward within a task."""
        mgr.update(step=10, status="Running", detail="Thinking")
        assert mgr.current_step == 10
        mgr.update(step=3, status="Running", detail="Thinking")
        assert mgr.current_step == 10  # max(10, 3) = 10

    def test_update_step_none_preserves_current(self, mgr):
        mgr.update(step=7)
        assert mgr.current_step == 7
        mgr.update(status="Running")  # step=None
        assert mgr.current_step == 7  # unchanged

    def test_begin_task_then_update_resets_then_climbs(self, mgr):
        # Simulate task A: reaches step 25
        mgr.begin_task(index=1, task_id="A")
        mgr.update(step=1, detail="Thinking")
        mgr.update(step=10, detail="Bash: pytest")
        mgr.update(step=25, detail="Finish")
        assert mgr.current_step == 25

        # Simulate task B: only reaches step 5
        mgr.begin_task(index=2, task_id="B")
        assert mgr.current_step == 0  # reset
        mgr.update(step=3, detail="Thinking")
        mgr.update(step=5, detail="Finish")
        assert mgr.current_step == 5  # correctly shows max of task B

    def test_status_line_includes_step(self, mgr):
        mgr.begin_task(index=1, task_id="test-task")
        mgr.current_step = 7
        mgr.current_status = "Running"
        mgr.current_detail = "Thinking"
        line = mgr._status_line()
        assert "Step 7" in line
        assert "test-task" in line
        assert "Running" not in line  # status itself is not in the line, detail is
        # "Thinking" is the detail — but the status line may truncate it.
        # Ensure "Step 7" is there regardless.

    def test_status_line_init_when_step_zero(self, mgr):
        mgr.begin_task(index=1, task_id="init-task")
        mgr.current_step = 0
        line = mgr._status_line()
        assert "Init" in line


# ═══════════════════════════════════════════════════════════════════
# 2. describe_progress_update — step extraction from all event types
# ═══════════════════════════════════════════════════════════════════

class TestDescribeProgressUpdate:
    """Every event type must correctly extract step from payload."""

    def test_agent_start(self):
        update = build_progress_update("t1", "agent_start", {"step": 0})
        step, status, detail = describe_progress_update(update)
        assert step == 0
        assert status == "Running"
        assert detail == "Agent init"

    def test_step_start(self):
        update = build_progress_update("t1", "step_start", {"step": 3})
        step, status, detail = describe_progress_update(update)
        assert step == 3
        assert status == "Running"
        assert detail == "Thinking"

    def test_tool_call_with_bash(self):
        update = build_progress_update(
            "t1", "tool_call",
            {"tool_name": "Bash", "arguments": {"command": "pytest tests/"}, "step": 7},
        )
        step, status, detail = describe_progress_update(update)
        assert step == 7
        assert status == "Running"
        assert detail.startswith("Bash:")

    def test_tool_call_with_read(self):
        update = build_progress_update(
            "t1", "tool_call",
            {"tool_name": "Read", "arguments": {"path": "src/main.py"}, "step": 2},
        )
        step, status, detail = describe_progress_update(update)
        assert step == 2
        assert status == "Running"
        assert detail == "Read"

    def test_tool_result_success(self):
        update = build_progress_update(
            "t1", "tool_result",
            {"tool_name": "Read", "status": "success", "step": 2},
        )
        step, status, detail = describe_progress_update(update)
        assert step == 2
        assert status == "Running"
        assert detail == "Read"

    def test_tool_result_error(self):
        update = build_progress_update(
            "t1", "tool_result",
            {"tool_name": "Bash", "status": "error", "step": 5},
        )
        step, status, detail = describe_progress_update(update)
        assert step == 5
        assert status == "Error"
        assert detail == "Bash: error"

    def test_final_answer(self):
        update = build_progress_update("t1", "final_answer", {"step": 12})
        step, status, detail = describe_progress_update(update)
        assert step == 12
        assert status == "Completing"
        assert detail == "Final answer"

    def test_timeout(self):
        update = build_progress_update("t1", "timeout", {"step": 48})
        step, status, detail = describe_progress_update(update)
        assert step == 48
        assert status == "Timeout"

    def test_stagnation_detected(self):
        reason = "Verification loop: 6 consecutive .py writes without Finish"
        update = build_progress_update("t1", "stagnation_detected", {"step": 30, "reason": reason})
        step, status, detail = describe_progress_update(update)
        assert step == 30
        assert status == "Stalled"
        assert "consecutive .py writes" in detail

    def test_llm_error(self):
        update = build_progress_update("t1", "llm_error", {"step": 8, "error": "Connection refused"})
        step, status, detail = describe_progress_update(update)
        assert step == 8
        assert status == "Error"
        assert "Connection refused" in detail

    def test_agent_error(self):
        update = build_progress_update("t1", "agent_error", {"step": 15, "message": "Crash"})
        step, status, detail = describe_progress_update(update)
        assert step == 15
        assert status == "Error"
        assert detail == "Crash"

    def test_compaction_notice(self):
        update = build_progress_update("t1", "compaction_notice", {"step": 20})
        step, status, detail = describe_progress_update(update)
        assert step == 20
        assert status == "Running"
        assert detail == "compacting context"

    def test_unknown_event_returns_step_only(self):
        update = build_progress_update("t1", "unknown_thing", {"step": 99})
        step, status, detail = describe_progress_update(update)
        assert step == 99
        assert status is None
        assert detail is None

    def test_control_tool_falls_through_to_tool_call(self):
        update = build_progress_update(
            "t1", "control_tool",
            {"tool_name": "Finish", "arguments": {}, "step": 42},
        )
        step, status, detail = describe_progress_update(update)
        assert step == 42
        assert status == "Running"
        assert detail == "Finish"

    def test_no_step_in_payload_returns_none(self):
        update = build_progress_update("t1", "step_start", {"other": "data"})
        step, status, detail = describe_progress_update(update)
        assert step is None
        assert status == "Running"  # status is from event_type, not payload


# ═══════════════════════════════════════════════════════════════════
# 3. build_progress_update — structured update creation
# ═══════════════════════════════════════════════════════════════════

class TestBuildProgressUpdate:
    def test_includes_task_id_and_event_type(self):
        update = build_progress_update("my-task", "step_start", {"step": 1})
        assert update["task_id"] == "my-task"
        assert update["event_type"] == "step_start"

    def test_includes_payload_and_full_payload(self):
        payload = {"step": 3, "detail": "some long string" * 50}
        update = build_progress_update("t", "tool_call", payload)
        assert "payload" in update
        assert "payload_full" in update
        # payload should be truncated-safe
        assert isinstance(update["payload"], dict)
        # payload_full preserved intact (deep copy for trajectory)
        assert update["payload_full"]["step"] == 3

    def test_none_task_id_still_string(self):
        update = build_progress_update(None, "step_start", {"step": 1})
        assert isinstance(update["task_id"], str)


# ═══════════════════════════════════════════════════════════════════
# 4. Multi-round step consistency
# ═══════════════════════════════════════════════════════════════════

class TestMultiRoundStepConsistency:
    """Simulate LCB6/ClassEval multi-round behavior.

    In multi-round benchmarks, each agent.run() starts from step 0
    (or start_step=1 effectively).  The progress display uses max(),
    so step never decreases within a task, but it IS a ceiling that
    doesn't reflect the step *within the current round* directly.
    """

    def test_round_one_steps_visible(self):
        """Round 1: agent runs, steps are emitted and displayed."""
        mgr = BenchmarkProgressManager(benchmark_name="test", total=1)
        mgr.begin_task(index=1, task_id="LCB6/test")

        # Simulate agent events in Round 1
        for event_type, payload in [
            ("agent_start", {"step": 0}),
            ("step_start", {"step": 1}),
            ("tool_call", {"tool_name": "Read", "step": 1}),
            ("tool_result", {"tool_name": "Read", "status": "success", "step": 1}),
            ("step_start", {"step": 2}),
            ("tool_call", {"tool_name": "Write", "step": 2}),
            ("tool_result", {"tool_name": "Write", "status": "success", "step": 2}),
            ("final_answer", {"step": 2}),
        ]:
            step, status, detail = describe_progress_update(
                {"task_id": "LCB6/test", "event_type": event_type, "payload": payload, "payload_full": payload}
            )
            mgr.update(step=step, status=status, detail=detail)

        assert mgr.current_step == 2  # Round 1's max step

    def test_round_two_starts_from_zero_but_display_uses_max(self):
        """Round 2: agent resets to step 1 internally, but display
        shows max(R1, R2) because of `max(self.current_step, step)`."""
        mgr = BenchmarkProgressManager(benchmark_name="test", total=1)
        mgr.begin_task(index=1, task_id="LCB6/test-hard")

        # Round 1: agent takes 45 steps
        for s in range(1, 46):
            mgr.update(step=s, status="Running", detail="Thinking")
        assert mgr.current_step == 45

        # Round 2: agent restarts from step 1 (internally)
        # The display receives step=1, step=2, ...
        # Because of max(), the display stays at 45
        for s in range(1, 10):
            mgr.update(step=s, status="Running", detail="Thinking")
        assert mgr.current_step == 45  # max(45, 1..9) = 45

        # Round 2 eventually surpasses round 1
        for s in range(10, 51):
            mgr.update(step=s, status="Running", detail="Thinking")
        assert mgr.current_step == 50  # max(45, 50) = 50

    def test_fresh_task_resets_step(self):
        """A new task always starts at step 0."""
        mgr = BenchmarkProgressManager(benchmark_name="test", total=2)

        # Task 1: reaches step 30
        mgr.begin_task(index=1, task_id="task-1")
        mgr.update(step=30, status="Running")
        assert mgr.current_step == 30
        mgr.finish_task({"task_id": "task-1", "passed": True})

        # Task 2: fresh start
        mgr.begin_task(index=2, task_id="task-2")
        assert mgr.current_step == 0  # reset!
        mgr.update(step=3, status="Running", detail="Thinking")
        assert mgr.current_step == 3


# ═══════════════════════════════════════════════════════════════════
# 5. Real agent step tracking
# ═══════════════════════════════════════════════════════════════════

class TestAgentStepTracking:
    """Verify that the agent's execution state starts fresh each run."""

    def test_create_execution_state_starts_from_start_step(self):
        """_create_execution_state(start_step) sets current_step=start_step."""
        import sys, types
        CODE_DIR = Path(__file__).resolve().parents[1] / "code"
        pkg = types.ModuleType("hello_agents")
        pkg.__path__ = [str(CODE_DIR)]
        pkg.__file__ = str(CODE_DIR / "__init__.py")
        sys.modules["hello_agents"] = pkg

        from hello_agents.agents.react_agent import ReActAgent, _ExecutionState

        # _create_execution_state is an instance method — test via direct
        # _ExecutionState construction which is what it delegates to
        state = _ExecutionState(current_step=0)
        assert state.current_step == 0

        state2 = _ExecutionState(current_step=5)
        assert state2.current_step == 5

    def test_normalize_start_step_handles_all_inputs(self):
        """_normalize_start_step accepts int, str, None, negative."""
        import sys, types
        CODE_DIR = Path(__file__).resolve().parents[1] / "code"
        pkg = types.ModuleType("hello_agents")
        pkg.__path__ = [str(CODE_DIR)]
        pkg.__file__ = str(CODE_DIR / "__init__.py")
        sys.modules["hello_agents"] = pkg

        from hello_agents.agents.react_agent import ReActAgent

        assert ReActAgent._normalize_start_step(0) == 0
        assert ReActAgent._normalize_start_step("0") == 0
        assert ReActAgent._normalize_start_step(None) == 0
        assert ReActAgent._normalize_start_step(-5) == 0
        assert ReActAgent._normalize_start_step(42) == 42

    def test_execution_state_increments_correctly(self):
        """Verify that the increment loop works as expected."""
        import sys, types
        CODE_DIR = Path(__file__).resolve().parents[1] / "code"
        pkg = types.ModuleType("hello_agents")
        pkg.__path__ = [str(CODE_DIR)]
        pkg.__file__ = str(CODE_DIR / "__init__.py")
        sys.modules["hello_agents"] = pkg

        from hello_agents.agents.react_agent import _ExecutionState

        state = _ExecutionState(current_step=0)
        for expected in range(1, 11):
            state.current_step += 1
            assert state.current_step == expected


# ═══════════════════════════════════════════════════════════════════
# 6. Utility functions used in display
# ═══════════════════════════════════════════════════════════════════

class TestClipDisplay:
    def test_short_text_unchanged(self):
        assert _clip_display("hello", 20) == "hello"

    def test_long_text_truncated_with_ellipsis(self):
        result = _clip_display("a" * 50, 20)
        assert len(result) <= 20

    def test_exact_fit(self):
        result = _clip_display("12345", 5)
        assert result == "12345"


class TestHumanElapsed:
    def test_seconds(self):
        result = _human_elapsed(5.3)
        assert "05" in result  # "0:05"

    def test_minutes(self):
        result = _human_elapsed(125.0)
        assert ":" in result

    def test_hours(self):
        result = _human_elapsed(4000.0)
        assert ":" in result

    def test_zero(self):
        result = _human_elapsed(0.0)
        assert result is not None and len(result) > 0


# ═══════════════════════════════════════════════════════════════════
# 7. LCB6 step behavior — integration-style
# ═══════════════════════════════════════════════════════════════════

class TestLCB6StepBehavior:
    """LCB6 runs multiple agent.run() calls per task (one per round).
    Each round starts from step 0 internally. Test that events flow
    correctly regardless.
    """

    def test_lcb6_per_round_step_emission(self):
        """Simulate LCB6-style multi-round agent execution:
        Round 1: 20 steps, Round 2: 8 steps."""
        mgr = BenchmarkProgressManager(benchmark_name="test", total=1)
        mgr.begin_task(index=1, task_id="LCB6/hard-problem")

        # Round 1
        for s in range(1, 21):
            payload = {"step": s}
            _, status, detail = describe_progress_update(
                build_progress_update("t", "step_start", payload)
            )
            mgr.update(step=s, status=status, detail=detail)
        # Agent emitted final_answer at step 20
        _, status, detail = describe_progress_update(
            build_progress_update("t", "final_answer", {"step": 20})
        )
        mgr.update(step=20, status=status, detail=detail)
        assert mgr.current_step == 20

        # Round 2: agent.run() called again, step counter resets internally
        for s in range(1, 9):
            payload = {"step": s}
            _, status, detail = describe_progress_update(
                build_progress_update("t", "step_start", payload)
            )
            mgr.update(step=s, status=status, detail=detail)
        # Display still at 20 (max of round 1)
        assert mgr.current_step == 20

        _, status, detail = describe_progress_update(
            build_progress_update("t", "final_answer", {"step": 8})
        )
        mgr.update(step=8, status=status, detail=detail)
        assert mgr.current_step == 20  # still max = 20

        # Round 3: takes 35 steps this time
        for s in range(1, 36):
            mgr.update(step=s)
        assert mgr.current_step == 35  # surpasses round 1

    def test_display_includes_correct_step_in_final_status(self):
        """finish_task preserves the last step value in current_step."""
        mgr = BenchmarkProgressManager(benchmark_name="test", total=1)
        mgr.begin_task(index=1, task_id="A")
        mgr.update(step=12)
        mgr.finish_task({"task_id": "A", "passed": True, "error": None})

        line = mgr._status_line()
        assert "Step 12" in line
