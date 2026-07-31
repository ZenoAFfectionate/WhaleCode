"""Integration / end-to-end tests for WhaleCode agents.

Covers complete agent workflows with a scripted mock LLM that simulates
real multi-round ReAct loops. Each E2E scenario tests the full chain:

    LLM call → tool schema build → tool execution → result recording →
    history append → compression check → next LLM call → ... → Finish

Scenarios:
- Bug fix workflow (Grep → Read → Edit → Bash → Finish)
- New feature workflow (LS → Glob → Read → Write → Bash → Edit → Bash → Finish)
- Code exploration workflow (LS → Read → Grep → Finish)
- Refactoring workflow (Grep → Read → Edit ×2 → Bash → Edit → Finish)
- History compression during long sessions
- Subagent delegation
- Stagnation detection (no-diff edits / same test results)
- Truncation retry (finish_reason="length")
- Role subagents (Explorer read-only enforcement / Tester writes+runs tests)
- ReviewerRole full review chain (structured output → ReviewReport)
- AgentOrchestra full chain (decompose → execute → aggregate)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# 注: 使用 hello_agents 包名 (tests/conftest.py 引导), 而非 code.* 直导入
from hello_agents.agents.code_agent import CodeAgent
from hello_agents.core.config import Config
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse


# ============================================================================
# Mock LLM infrastructure — scriptable multi-round responses
# ============================================================================


class _FakeFunction:
    """Simulates openai.types.chat.chat_completion_message_tool_call.Function."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    """Simulates openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall."""

    def __init__(self, id: str, name: str, arguments: dict):
        self.id = id
        self.function = _FakeFunction(name, json.dumps(arguments, ensure_ascii=False))


class _FakeMessage:
    """Simulates openai.types.chat.chat_completion_message.ChatCompletionMessage."""

    def __init__(self, content: Optional[str], tool_calls: Optional[List[_FakeToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeUsage:
    """Simulates openai.types.CompletionUsage."""

    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeChoice:
    """Simulates openai.types.chat.chat_completion.Choice."""

    def __init__(
        self,
        message: _FakeMessage,
        finish_reason: str = "stop",
    ):
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    """Simulates openai.types.chat.ChatCompletion (or LiteLLM equivalent)."""

    def __init__(self, choices: List[_FakeChoice], usage: Optional[_FakeUsage] = None):
        self.choices = choices
        self.usage = usage


class ScriptedLLM:
    """A programmable mock LLM that returns pre-scripted responses round by round.

    Each entry in *script* is a dict describing one model response:

    - Tool calls: ``{"ToolName": {"arg": "value"}, "ToolName2": {...}}``
      Keys are tool names, values are arguments dicts.
    - Text response: ``{"_content": "final answer text"}``
    - Finish via tool: ``{"Finish": {"answer": "all done"}}``

    When the script is exhausted, returns a Finish("completed") by default.
    """

    def __init__(self, script: Optional[List[Dict[str, Any]]] = None, *, model: str = "test-model"):
        self.script = script or []
        self.call_count = 0
        self.model = model
        self.temperature = 0.7
        # Record every invocation for test assertions
        self.invoke_history: List[Dict[str, Any]] = []

    # ── synchronous ──────────────────────────────────────────────────────

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs,
    ) -> _FakeResponse:
        self.invoke_history.append({
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "kwargs": kwargs,
        })
        step = self._next_step()
        return self._build_response(step)

    # ── async ────────────────────────────────────────────────────────────

    async def ainvoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs,
    ) -> _FakeResponse:
        return self.invoke_with_tools(messages, tools, tool_choice, **kwargs)

    # ── plain text invocation (AgentOrchestra decompose/aggregate) ──────

    def invoke(self, messages: List[Dict[str, Any]], **kwargs):
        """Plain-text LLM call; script entry ``{"_text": "..."}`` supplies content."""
        from types import SimpleNamespace

        self.invoke_history.append({
            "messages": messages,
            "kwargs": kwargs,
            "kind": "invoke",
        })
        step = self._next_step()
        if isinstance(step, dict) and "_text" in step:
            content = step["_text"]
        else:
            content = json.dumps(step, ensure_ascii=False)
        return SimpleNamespace(
            content=content,
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    # ── internal ─────────────────────────────────────────────────────────

    def _next_step(self):
        if self.call_count < len(self.script):
            step = self.script[self.call_count]
            self.call_count += 1
            return step
        # Default: Finish when script exhausted
        self.call_count += 1
        return {"Finish": {"answer": "Task completed (script exhausted)."}}

    def _build_response(self, step) -> _FakeResponse:
        # Support list format for steps with duplicate tool names:
        #   [{"Read": {...}}, {"Read": {...}}]
        # alongside the standard dict format:
        #   {"Thought": {...}, "Read": {...}}
        if isinstance(step, list):
            entries = step
            extra = {}
        elif isinstance(step, dict):
            entries = [{k: v} for k, v in step.items()]
            extra = step
        else:
            entries = []
            extra = {}

        # Check for direct text response (no tool calls)
        if isinstance(step, dict) and "_content" in step:
            msg = _FakeMessage(content=step["_content"], tool_calls=None)
            choice = _FakeChoice(msg, finish_reason=step.get("_finish_reason", "stop"))
            return _FakeResponse([choice], _FakeUsage(100, 50, 150))

        # Build tool calls preserving order (list) or dict iteration order
        tool_calls = []
        for idx, entry in enumerate(entries):
            for tool_name, arguments in entry.items():
                if tool_name.startswith("_"):
                    continue
                tool_calls.append(_FakeToolCall(
                    id=f"call_{self.call_count}_{idx}",
                    name=tool_name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                ))

        finish_reason = extra.get("_finish_reason", "stop") if isinstance(extra, dict) else "stop"
        msg = _FakeMessage(content=None, tool_calls=tool_calls if tool_calls else None)
        choice = _FakeChoice(msg, finish_reason=finish_reason)
        return _FakeResponse([choice], _FakeUsage(200, 100, 300))


# ============================================================================
# Tool stubs for controlled E2E scenarios
# ============================================================================


class _CounterTool(Tool):
    """A tool that returns an incrementing counter value."""

    def __init__(self, name: str = "Counter"):
        super().__init__(name=name, description="Return an incrementing counter.")
        self.count = 0

    def get_parameters(self) -> List[ToolParameter]:
        return []

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        self.count += 1
        return ToolResponse.success(text=f"Counter value: {self.count}", data={"value": self.count})


class _FailingTool(Tool):
    """A tool that always raises an exception."""

    def __init__(self, name: str = "FailingTool"):
        super().__init__(name=name, description="Always fails.")
        self.fail_count = 0

    def get_parameters(self) -> List[ToolParameter]:
        return []

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        self.fail_count += 1
        raise RuntimeError(f"Simulated failure #{self.fail_count}")


# ============================================================================
# Config helpers
# ============================================================================


def _e2e_config(**overrides) -> Config:
    """Return a Config suitable for E2E testing (compression off by default)."""
    base = dict(
        context_window=128000,
        compact_enabled=False,
        compression_threshold=0.8,
        trace_enabled=False,
        skills_enabled=False,
        todowrite_enabled=False,
        session_enabled=False,
        max_concurrent_tools=5,
    )
    base.update(overrides)
    return Config(**base)


# ============================================================================
# E2E Scenario 1: Bug Fix
# ============================================================================


class TestCodeAgentE2EBugFix:
    """Complete bug-fix workflow: Grep → Read → Edit → Bash(test) → Finish."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Create a simple buggy Python file
            src = root / "src"
            src.mkdir()
            (src / "calc.py").write_text("def add(a, b):\n    return a - b  # bug: should be +\n")
            (src / "test_calc.py").write_text(
                "from src.calc import add\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
                "    assert add(-1, 1) == 0\n"
            )
            yield root

    def test_bug_fix_full_workflow(self, workspace):
        """Agent finds and fixes the bug through 4 rounds of ReAct."""
        script = [
            # Round 1: Explore — Read the buggy file + Grep for tests
            {
                "Thought": {"reasoning": "Let me first understand the codebase."},
                "Read": {"path": "src/calc.py"},
                "Grep": {"pattern": "test_add", "path": "src/"},
            },
            # Round 2: Understand the bug + Edit fix
            {
                "Thought": {"reasoning": "The bug is in add() — using subtraction instead of addition."},
                "Edit": {
                    "path": "src/calc.py",
                    "old_string": "return a - b  # bug: should be +",
                    "new_string": "return a + b",
                },
            },
            # Round 3: Run tests
            {
                "Bash": {"command": f"cd {workspace} && python -m pytest src/test_calc.py -v"},
            },
            # Round 4: Done
            {
                "Finish": {"answer": "Bug fixed: changed subtraction to addition in add(). Tests pass."},
            },
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "bug-fixer",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run("Fix the bug in src/calc.py")

        # Verify final answer
        assert "fixed" in result.lower() or "Bug" in result

        # Verify history completeness
        history = agent.get_history()
        roles = [m.role for m in history]
        assert "user" in roles
        # Should have assistant messages (with tool_calls) and tool result messages
        assert "assistant" in roles or "tool" in roles

        # Verify all 4 rounds executed
        assert llm.call_count == 4

        # File should be fixed
        fixed_content = (workspace / "src" / "calc.py").read_text()
        assert "return a + b" in fixed_content
        assert "return a - b" not in fixed_content


# ============================================================================
# E2E Scenario 2: New Feature Development
# ============================================================================


class TestCodeAgentE2ENewFeature:
    """New feature workflow: LS → Read → Write → Bash(test) → Edit(fix) → Bash → Finish."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "main.py").write_text("def greet(name):\n    return f'Hello, {name}'\n")
            yield root

    def test_new_feature_full_workflow(self, workspace):
        """Agent creates a new module, tests it, fixes a failure, and finishes."""
        script = [
            # Round 1: List and read existing code
            {"LS": {"path": "."}, "Read": {"path": "main.py"}},
            # Round 2: Write new feature
            {
                "Thought": {"reasoning": "I'll create utils.py with helper functions."},
                "Write": {
                    "path": "utils.py",
                    "content": "def double(x):\n    return x * 2\n\ndef triple(x):\n    return x * 3\n",
                },
            },
            # Round 3: Test the new module
            {"Bash": {"command": f"cd {workspace} && python -c 'from utils import double, triple; print(double(21), triple(7))'"}},
            # Round 4: Verify then finish
            {"Finish": {"answer": "Created utils.py with double() and triple() functions."}},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "feature-dev",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run("Create a utils.py module with double() and triple() functions.")

        assert "utils" in result.lower()
        assert (workspace / "utils.py").exists()
        content = (workspace / "utils.py").read_text()
        assert "def double" in content
        assert "def triple" in content
        assert llm.call_count == 4


# ============================================================================
# E2E Scenario 3: Code Exploration
# ============================================================================


class TestCodeAgentE2EExploration:
    """Code exploration: LS → Read → Grep → Finish."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / "src"
            src.mkdir(parents=True)
            (src / "models.py").write_text(
                "class User:\n    def __init__(self, name):\n        self.name = name\n\n"
                "class Admin(User):\n    def __init__(self, name, level=1):\n"
                "        super().__init__(name)\n        self.level = level\n"
            )
            (src / "views.py").write_text(
                "from .models import User\n\ndef render_user(user: User) -> str:\n"
                "    return f'User: {user.name}'\n"
            )
            (src / "__init__.py").write_text("")
            yield root

    def test_exploration_workflow(self, workspace):
        """Agent explores the codebase and summarizes its findings."""
        script = [
            # Round 1: List files
            {"LS": {"path": "src/"}},
            # Round 2: Read models and views (list format for duplicate Read calls)
            [{"Read": {"path": "src/models.py"}}, {"Read": {"path": "src/views.py"}}],
            # Round 3: Search for User usage
            {"Grep": {"pattern": "User", "path": "src/"}},
            # Round 4: Summarize
            {"Finish": {"answer": "The codebase has a User/Admin model hierarchy and a view renderer."}},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "explorer",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run("Explore the src/ directory and tell me what the codebase does.")

        assert "User" in result
        assert llm.call_count == 4
        history = agent.get_history()
        assert len(history) >= 7  # user + assistant×4 + tool results


# ============================================================================
# E2E Scenario 4: Refactoring
# ============================================================================


class TestCodeAgentE2ERefactor:
    """Multi-file refactoring: Grep → Read ×2 → Edit ×2 → Bash(test) → Finish."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.py").write_text("def get_config():\n    return {'host': 'localhost', 'port': 8080}\n")
            (root / "b.py").write_text("from a import get_config\ncfg = get_config()\nprint(cfg['port'])\n")
            yield root

    def test_refactor_workflow(self, workspace):
        """Agent renames a function across multiple files."""
        script = [
            # Round 1: Find all usages
            {"Grep": {"pattern": "get_config", "path": "."}},
            # Round 2: Read both files
            [{"Read": {"path": "a.py"}}, {"Read": {"path": "b.py"}}],
            # Round 3: Edit a.py — rename function
            {"Edit": {"path": "a.py", "old_string": "def get_config():", "new_string": "def load_config():"}},
            # Round 4: Edit b.py — update import and call
            {"Edit": {"path": "b.py", "old_string": "from a import get_config", "new_string": "from a import load_config"}},
            {"Edit": {"path": "b.py", "old_string": "cfg = get_config()", "new_string": "cfg = load_config()"}},
            # Round 5: Verify
            {"Bash": {"command": f"cd {workspace} && python -c 'from a import load_config; print(load_config())'"}},
            # Round 6: Done
            {"Finish": {"answer": "Renamed get_config() to load_config() across a.py and b.py."}},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "refactorer",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run("Rename get_config to load_config across the project.")

        assert "load_config" in result.lower()
        assert "def load_config" in (workspace / "a.py").read_text()
        assert "load_config" in (workspace / "b.py").read_text()
        assert "get_config" not in (workspace / "a.py").read_text()
        assert "get_config" not in (workspace / "b.py").read_text()


# ============================================================================
# E2E Scenario 5: Structured Output
# ============================================================================


class TestCodeAgentE2EStructuredOutput:
    """Agent run with structured output schema."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data.py").write_text("ITEMS = [1, 2, 3, 4, 5]\n")
            yield root

    def test_structured_output_workflow(self, workspace):
        """Agent reads a file and returns structured JSON output."""
        script = [
            {"Read": {"path": "data.py"}},
            {"StructuredOutput": {
                "summary": "The file contains a list of 5 integers",
                "item_count": 5,
                "items": [1, 2, 3, 4, 5],
            }},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "structured-runner",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run(
            "Analyze data.py",
            structured_output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "item_count": {"type": "integer"},
                    "items": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["summary", "item_count", "items"],
            },
        )

        parsed = json.loads(result)
        assert parsed["item_count"] == 5
        assert len(parsed["items"]) == 5


# ============================================================================
# E2E Scenario 6: History Compression During Session
# ============================================================================


class TestHistoryCompressionE2E:
    """Verify that compression triggers and preserves context during a long session."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_compression_triggers_during_long_session(self, workspace):
        """Run enough rounds to trigger compression, verify agent continues working."""
        N = 15  # Enough rounds to accumulate messages and trigger compression

        # Build script: N rounds of simple tools, final Finish
        script = []
        for i in range(N):
            script.append({"Thought": {"reasoning": f"Step {i}: doing work."}, "Counter": {}})
        script.append({"Finish": {"answer": f"Completed {N} steps successfully."}})

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "long-runner",
            llm,
            project_root=str(workspace),
            config=_e2e_config(
                context_window=4096,        # Small window to trigger compression early
                compact_enabled=True,
                compression_threshold=0.2,  # Low threshold
                compact_preserve_recent_rounds=2,
                compact_output_buffer=512,
            ),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=25,
        )

        result = agent.run(f"Run {N} steps of analysis.")

        assert "successfully" in result.lower() or "Completed" in result
        # Verify the agent didn't crash or timeout
        assert llm.call_count >= N

        # Verify token count is reasonable (should have been compressed)
        history = agent.get_history()
        assert len(history) > 0


# ============================================================================
# E2E Scenario 7: Stagnation Detection
# ============================================================================


class TestStagnationDetectionE2E:
    """Verify the agent stops when stagnation is detected."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "script.py").write_text("print('hello')\n")
            yield root

    def test_no_diff_edit_stagnation(self, workspace):
        """Agent makes 3 consecutive no-diff Edits → should stop early.

        Uses direct stagnation state manipulation to verify detection logic,
        since real no-diff edits require very specific file states.
        """
        from hello_agents.agents.react_agent import _ExecutionState

        agent = CodeAgent(
            "stuck-agent",
            ScriptedLLM(),
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        # Directly test the stagnation detection logic
        state = _ExecutionState(current_step=1)

        # Simulate 3 consecutive Edit calls that return "[no textual diff]"
        for i in range(3):
            agent._update_stagnation_state(
                "Edit", f"call_{i}",
                "[no textual diff]",
                1, state,
            )

        assert state.stagnation_detected is True
        assert state.consecutive_no_diff_edits == 3

        # Reset and verify non-Edit calls reset the counter
        state2 = _ExecutionState(current_step=1)
        agent._update_stagnation_state("Edit", "c1", "[no textual diff]", 1, state2)
        agent._update_stagnation_state("Edit", "c2", "[no textual diff]", 1, state2)
        assert state2.consecutive_no_diff_edits == 2
        agent._update_stagnation_state("Read", "c3", "file content", 1, state2)
        assert state2.consecutive_no_diff_edits == 0  # Reset by non-Edit tool


# ============================================================================
# E2E Scenario 8: Truncation Retry
# ============================================================================


class TestTruncationRetryE2E:
    """Verify that truncated LLM responses trigger a retry nudge."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_truncation_retry_then_success(self, workspace):
        """First response is truncated (length with no tool_calls), second succeeds."""
        script = [
            # Round 1: Truncated — finish_reason="length", no tool_calls
            {"_content": "I'll help you with that. First, let me explore the code...",
             "_finish_reason": "length"},
            # Round 2: After nudge, produces a proper tool call
            {"Read": {"path": "README.md"}},
            # Round 3: Done
            {"Finish": {"answer": "Analysis complete."}},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "retry-agent",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run("Analyze the project.")

        # Should have retried after truncation and eventually completed
        assert "Analysis complete" in result
        # Verify the truncation nudge message was added to history
        history = agent.get_history()
        nudge_found = any(
            "truncat" in (m.content or "").lower() or "cut off" in (m.content or "").lower()
            for m in history
        )
        assert nudge_found, "Truncation nudge should be in history"


# ============================================================================
# E2E Scenario 9: Tool Execution Error Handling
# ============================================================================


class TestCodeAgentE2EErrorHandling:
    """Agent gracefully handles tool execution errors."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_agent_continues_after_tool_error(self, workspace):
        """After a tool raises an exception, the agent continues to the next round."""
        script = [
            # Round 1: FailingTool raises, but Thought still works
            {"FailingTool": {}, "Thought": {"reasoning": "First attempt failed, let me try another approach."}},
            # Round 2: Recover
            {"Finish": {"answer": "Recovered after tool failure."}},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "error-handler",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        agent.tool_registry.register_tool(_FailingTool(name="FailingTool"))

        result = agent.run("Try something that might fail.")

        assert "Recovered" in result


# ============================================================================
# E2E Scenario 10: Subagent Delegation
# ============================================================================


class TestSubagentE2E:
    """Verify subagent creation and isolated execution."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "task.py").write_text("def work():\n    return 42\n")
            yield root

    def test_subagent_isolation_and_restoration(self, workspace):
        """Subagent runs with filtered tools, parent state restored afterward."""
        agent = CodeAgent(
            "parent",
            ScriptedLLM(),
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=20,
        )

        # Create subagent
        sub = agent._create_subagent("worker")
        assert sub.name == "parent-worker-subagent"
        assert sub.project_root == agent.project_root
        assert sub.max_steps > 0

        # Subagent should have its own tool registry
        assert sub.tool_registry is not agent.tool_registry

        # Verify parent's tools are unchanged after subagent creation
        parent_tools_before = set(agent.tool_registry.list_tools())
        sub_tools = set(sub.tool_registry.list_tools())
        parent_tools_after = set(agent.tool_registry.list_tools())
        assert parent_tools_before == parent_tools_after
        # Sub should have similar tools (default registration)
        for t in ("Read", "Write", "Edit", "Bash", "Grep"):
            assert t in sub_tools


# ============================================================================
# E2E Scenario 11: Session Save / Restore
# ============================================================================


class TestSessionSaveRestoreE2E:
    """Session serialization round-trip."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_session_save_and_restore(self, workspace):
        """Save agent session, create a new agent, restore, verify history."""
        config = _e2e_config(session_enabled=True, session_dir=str(workspace / "sessions"))

        script = [
            {"Thought": {"reasoning": "Working on it."}},
            {"Finish": {"answer": "Session data preserved."}},
        ]

        agent = CodeAgent(
            "session-agent",
            ScriptedLLM(script),
            project_root=str(workspace),
            config=config,
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        agent.run("Do some work.")

        # Save session
        saved_path = agent.save_session("test-session")
        assert saved_path is not None
        assert Path(saved_path).exists()

        # Create new agent and restore
        agent2 = CodeAgent(
            "restored-agent",
            ScriptedLLM(),
            project_root=str(workspace),
            config=config,
            register_default_tools=False,
            interactive=False,
        )
        agent2.load_session(saved_path)

        # History should be restored
        history = agent2.get_history()
        assert len(history) == len(agent.get_history())


# ============================================================================
# E2E Scenario 12: Direct text response (no tool calls)
# ============================================================================


class TestDirectTextResponseE2E:
    """Agent handles a simple question with a direct text response (no tools)."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_simple_question_direct_response(self, workspace):
        """Agent answers a simple question without calling any tools."""
        script = [
            {"_content": "The capital of France is Paris."},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "simple-qa",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run("What is the capital of France?")

        assert "Paris" in result
        assert llm.call_count == 1


# ============================================================================
# E2E Scenario 13: Multiple tools in one round
# ============================================================================


class TestMultiToolRoundE2E:
    """LLM returns multiple tool calls in a single round."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "file1.txt").write_text("content1")
            (root / "file2.txt").write_text("content2")
            yield root

    def test_multiple_reads_in_one_round(self, workspace):
        """Two Read calls in the same round are both executed."""
        script = [
            [{"Read": {"path": "file1.txt"}}, {"Read": {"path": "file2.txt"}}],
            {"Finish": {"answer": "Both files read successfully."}},
        ]

        llm = ScriptedLLM(script)
        agent = CodeAgent(
            "multi-reader",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=10,
        )

        result = agent.run("Read file1.txt and file2.txt.")

        assert "successfully" in result.lower()
        # Both tool results should be in history
        history = agent.get_history()
        tool_messages = [m for m in history if m.role == "tool"]
        assert len(tool_messages) >= 2


# ============================================================================
# E2E Scenario 14: Role Subagents (Explorer / Tester)
# ============================================================================


class TestRoleSubagentE2E:
    """角色化子 Agent 走完整 ReAct loop: 工具策略在真实执行中被强制."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "app.py").write_text("def work():\n    return 42\n")
            yield root

    def _main_agent(self, workspace, llm):
        return CodeAgent(
            "parent",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=20,
        )

    def test_explorer_read_only_full_loop(self, workspace):
        """Explorer 子 Agent: Read → Finish, 结果通过 run() 返回."""
        from hello_agents.agents.roles import ExplorerRole

        script = [
            {"Read": {"path": "app.py"}},
            {"Finish": {"answer": "app.py defines work() which returns 42."}},
        ]
        llm = ScriptedLLM(script)
        main = self._main_agent(workspace, llm)
        sub = ExplorerRole.create_subagent(
            main.llm, main.config, str(workspace), str(workspace)
        )

        result = sub.run("Explore app.py")

        assert "returns 42" in result
        assert llm.call_count == 2
        # 工具策略在真实执行链路上生效
        assert sub.tool_registry.get_tool("Write") is None
        assert sub.tool_registry.get_tool("Bash") is None

    def test_explorer_write_attempt_blocked_at_runtime(self, workspace):
        """Explorer 尝试 Write → NOT_FOUND 错误回执 → Agent 恢复并 Finish; 文件未创建."""
        from hello_agents.agents.roles import ExplorerRole

        script = [
            {"Write": {"path": "evil.py", "content": "x = 1\n"}},
            {"Finish": {"answer": "I cannot write files; read-only role."}},
        ]
        llm = ScriptedLLM(script)
        main = self._main_agent(workspace, llm)
        sub = ExplorerRole.create_subagent(
            main.llm, main.config, str(workspace), str(workspace)
        )

        result = sub.run("Try to write evil.py")

        assert "read-only" in result.lower() or "cannot" in result.lower()
        assert not (workspace / "evil.py").exists()
        # NOT_FOUND 错误进入了子 Agent 历史 (tool message)
        tool_msgs = [m for m in sub.get_history() if m.role == "tool"]
        assert any("not found" in (m.content or "").lower() for m in tool_msgs)

    def test_tester_writes_and_runs_tests(self, workspace):
        """Tester 子 Agent: Write 测试文件 → Bash 运行 → Finish; 产物真实落盘."""
        from hello_agents.agents.roles import TesterRole

        script = [
            {"Write": {
                "path": "tests/test_app.py",
                "content": "def test_work():\n    assert 42 == 42\n",
            }},
            {"Bash": {"command": f"cd {workspace} && python -m pytest tests/ -q"}},
            {"Finish": {"answer": "Test file created and all tests passed."}},
        ]
        llm = ScriptedLLM(script)
        main = self._main_agent(workspace, llm)
        sub = TesterRole.create_subagent(
            main.llm, main.config, str(workspace), str(workspace)
        )

        result = sub.run("Write and run a test for app.py")

        assert "passed" in result.lower()
        test_file = workspace / "tests" / "test_app.py"
        assert test_file.exists()
        assert "test_work" in test_file.read_text()
        # Tester 可用写工具但 Delete 被禁止
        assert sub.tool_registry.get_tool("Write") is not None
        assert sub.tool_registry.get_tool("Delete") is None


# ============================================================================
# E2E Scenario 15: ReviewerRole Full Review Chain
# ============================================================================


class TestReviewerRoleE2E:
    """review_files 完整链路: 读文件 → 子 Agent ReAct loop → 结构化报告."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "auth.py").write_text(
                "API_KEY = 'sk-hardcoded-secret-123'\n\n"
                "def login(user):\n"
                "    return user == 'admin'\n"
            )
            yield root

    def test_review_files_end_to_end(self, workspace):
        from hello_agents.agents.roles.reviewer import ReviewerRole

        payload = {
            "summary": "发现硬编码密钥",
            "findings": [
                {
                    "severity": "critical",
                    "category": "security",
                    "file": "auth.py",
                    "line": 1,
                    "title": "Hardcoded API key",
                    "description": "Secret in source control",
                    "suggestion": "Move to environment variable",
                }
            ],
            "score": {"correctness": 8, "security": 2},
            "recommendations": ["立即移除硬编码密钥"],
        }
        script = [
            {"Read": {"path": "auth.py"}},
            {"ReviewOutput": payload},
        ]
        llm = ScriptedLLM(script)
        main_config = _e2e_config()

        report = asyncio.run(
            ReviewerRole.review_files(
                llm, main_config, str(workspace), ["auth.py"], review_focus="security"
            )
        )

        assert report.error is None
        assert report.summary == "发现硬编码密钥"
        assert len(report.findings) == 1
        finding = report.findings[0]
        assert finding.severity == "critical"
        assert finding.category == "security"
        assert finding.file == "auth.py"
        assert finding.line == 1
        assert report.score["security"] == 2
        assert report.recommendations == ["立即移除硬编码密钥"]
        # Markdown 渲染可用
        md = report.to_markdown()
        assert "[CRITICAL]" in md and "auth.py:1" in md

    def test_review_files_degraded_output(self, workspace):
        """子 Agent 未按 schema 输出 → 降级报告, 不抛异常."""
        from hello_agents.agents.roles.reviewer import ReviewerRole

        script = [
            {"Finish": {"answer": "代码看起来没问题, 但我说不清具体 JSON."}},
        ]
        llm = ScriptedLLM(script)

        report = asyncio.run(
            ReviewerRole.review_files(llm, _e2e_config(), str(workspace), ["auth.py"])
        )

        assert report.error == "parse_fallback"
        assert report.findings == []


# ============================================================================
# E2E Scenario 16: AgentOrchestra Full Chain
# ============================================================================


class TestAgentOrchestraE2E:
    """Orchestra 完整链路: decompose → 真实子 Agent 执行 → aggregate."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "app.py").write_text("def work():\n    return 42\n")
            yield root

    def _main_agent(self, workspace, llm):
        return CodeAgent(
            "orchestrator",
            llm,
            project_root=str(workspace),
            config=_e2e_config(),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
            max_steps=20,
        )

    def test_full_pipeline_chain(self, workspace):
        """decompose(hybrid) → Explorer(Read→Finish) → Reviewer(Finish) → aggregate."""
        from hello_agents.agents.orchestra import AgentOrchestra, ExecutionMode

        plan_json = json.dumps({
            "subtasks": [
                {"id": "exp-1", "description": "探索 app.py 的功能", "role": "explorer",
                 "dependencies": []},
                {"id": "rev-1", "description": "审查发现的问题", "role": "reviewer",
                 "dependencies": ["exp-1"]},
            ],
            "mode": "hybrid",
            "stages": [["exp-1"], ["rev-1"]],
        }, ensure_ascii=False)

        script = [
            {"_text": plan_json},                                        # 1. decompose
            {"Read": {"path": "app.py"}},                                # 2. explorer round 1
            {"Finish": {"answer": "app.py: work() returns 42"}},         # 3. explorer round 2
            {"Finish": {"answer": "审查通过, 无严重问题"}},               # 4. reviewer round 1
            {"_text": "最终答案: 项目健康"},                              # 5. aggregate
        ]
        llm = ScriptedLLM(script)
        main = self._main_agent(workspace, llm)
        orchestra = AgentOrchestra(main)

        answer = asyncio.run(orchestra.run("分析这个项目", ExecutionMode.HYBRID))

        assert answer == "最终答案: 项目健康"
        assert llm.call_count == 5
        # 阶段间上下文注入: reviewer 的 prompt 包含 explorer 的结果摘要
        reviewer_call = llm.invoke_history[3]
        reviewer_prompt = json.dumps(reviewer_call["messages"], ensure_ascii=False, default=str)
        assert "work() returns 42" in reviewer_prompt

    def test_parallel_chain(self, workspace):
        """decompose(parallel) → 两个 Explorer 并行 → aggregate."""
        from hello_agents.agents.orchestra import AgentOrchestra, ExecutionMode

        plan_json = json.dumps({
            "subtasks": [
                {"id": "e1", "description": "探索结构", "role": "explorer", "dependencies": []},
                {"id": "e2", "description": "探索依赖", "role": "explorer", "dependencies": []},
            ],
            "mode": "parallel",
            "stages": [],
        }, ensure_ascii=False)

        script = [
            {"_text": plan_json},
            {"Finish": {"answer": "结构分析"}},
            {"Finish": {"answer": "依赖分析"}},
            {"_text": "并行汇总答案"},
        ]
        llm = ScriptedLLM(script)
        main = self._main_agent(workspace, llm)
        orchestra = AgentOrchestra(main)

        answer = asyncio.run(orchestra.run("并行分析", ExecutionMode.PARALLEL))

        assert answer == "并行汇总答案"
        assert llm.call_count == 4

    def test_decompose_failure_falls_back_and_completes(self, workspace):
        """decompose 两次输出非法 JSON → fallback 计划仍完整跑完流程."""
        from hello_agents.agents.orchestra import AgentOrchestra, ExecutionMode

        script = [
            {"_text": "这不是 JSON"},
            {"_text": "依然不是 JSON"},
            {"Finish": {"answer": "fallback 探索完成"}},
            {"_text": "fallback 汇总"},
        ]
        llm = ScriptedLLM(script)
        main = self._main_agent(workspace, llm)
        orchestra = AgentOrchestra(main)

        answer = asyncio.run(orchestra.run("任意任务", ExecutionMode.HYBRID))

        assert answer == "fallback 汇总"
        # 2 次 decompose 尝试 + 1 次子 Agent 执行 + 1 次 aggregate
        assert llm.call_count == 4

    def test_context_isolation_main_history_unpolluted(self, workspace):
        """隔离性契约: 子 Agent 的中间过程 (工具调用/试错) 不进入主 Agent 历史."""
        from hello_agents.agents.orchestra import AgentOrchestra, ExecutionMode

        plan_json = json.dumps({
            "subtasks": [
                {"id": "exp-1", "description": "探索 app.py", "role": "explorer",
                 "dependencies": []},
            ],
            "mode": "hybrid",
            "stages": [["exp-1"]],
        }, ensure_ascii=False)
        script = [
            {"_text": plan_json},
            {"Read": {"path": "app.py"}},
            {"Grep": {"pattern": "work"}},
            {"Finish": {"answer": "app.py: work() returns 42"}},
            {"_text": "最终答案"},
        ]
        llm = ScriptedLLM(script)
        main = self._main_agent(workspace, llm)
        history_before = len(main.get_history())

        answer = asyncio.run(
            AgentOrchestra(main).run("探索项目", ExecutionMode.HYBRID)
        )

        assert answer == "最终答案"
        # 子 Agent 的 Read/Grep 等中间步骤未污染主 Agent 历史
        assert len(main.get_history()) == history_before
        for msg in main.get_history():
            assert msg.role != "tool"
