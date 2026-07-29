"""CLI rendering smoke tests — verify CLI-1 through CLI-10 improvements."""
import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

# Bootstrap hello_agents
CODE = Path(__file__).resolve().parents[1] / "code"
if "hello_agents" not in sys.modules:
    pkg = types.ModuleType("hello_agents")
    pkg.__path__ = [str(CODE)]
    pkg.__file__ = str(CODE / "__init__.py")
    sys.modules["hello_agents"] = pkg
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── CLI-1: model_output / reasoning display ──────────────────────────

def test_model_output_shows_reasoning():
    """CLI-1: model_output event must render reasoning_content (not silently drop)."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    captured = io.StringIO()

    class _FakeUI:
        use_rich = False
        def print(self, msg="", end="\n", flush=False):
            captured.write(str(msg) + end)
        def info(self, msg): captured.write(f"[info] {msg}\n")
        def render_log_block(self, kind, text): captured.write(f"[{kind}] {text}\n")

    ui = _FakeUI()
    mixin = cli.CLICodeAgentMixin()
    mixin.ui = ui
    mixin._streaming_line_buffer = ""

    # Simulate model_output with reasoning_content
    mixin._render_event("model_output", {
        "content": "some reply",
        "tool_calls": 1,
        "reasoning_content": "Let me think about this carefully...",
        "step": 1,
    })
    output = captured.getvalue()
    assert "Let me think" in output, f"CLI-1 FAIL: reasoning missing from: {output}"
    assert "thinking" in output.lower() or "[thinking]" in output


# ── CLI-2: step visibility ──────────────────────────────────────────

def test_step_start_visible():
    """CLI-2: step_start must NOT be ignored — user must see step number."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    captured = io.StringIO()

    class _FakeUI:
        use_rich = False
        def print(self, msg="", end="\n", flush=False):
            captured.write(str(msg) + end)
        def info(self, msg): captured.write(f"[info] {msg}\n")
        render_log_block = Mock()

    ui = _FakeUI()
    mixin = cli.CLICodeAgentMixin()
    mixin.ui = ui
    mixin._streaming_line_buffer = ""

    # step_start must NOT be in IGNORED_EVENTS
    assert "step_start" not in mixin._IGNORED_EVENTS, "CLI-2 FAIL: step_start still ignored"

    mixin._render_event("step_start", {"step": 5})
    output = captured.getvalue()
    assert "Step 5" in output, f"CLI-2 FAIL: step not shown in: {output}"


# ── CLI-3: compact tool call args ────────────────────────────────────

def test_compact_args_truncation():
    """CLI-3: tool-call args must be compact, not full dict dump."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    m = cli.CLICodeAgentMixin()

    # Named keys preferred
    assert m._compact_args({"path": "src/main.py"}) == "src/main.py"
    assert m._compact_args({"command": "pytest tests/ -v"}) == "pytest tests/ -v"
    assert m._compact_args({"pattern": "*.py"}) == "*.py"
    assert m._compact_args({"query": "search term"}) == "search term"

    # Long values truncated
    long = "x" * 200
    compact = m._compact_args({"path": long})
    assert len(compact) <= 123, f"CLI-3 FAIL: compacted len={len(compact)}"
    assert compact.endswith("…")

    # Empty dict
    assert m._compact_args({}) == ""


# ── CLI-4: tool result truncation ────────────────────────────────────

def test_truncate_observation():
    """CLI-4: long tool output must be head+tail truncated."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    m = cli.CLICodeAgentMixin()

    short = "line1\nline2\nline3"
    assert m._truncate_observation(short) == short  # no change

    long = "\n".join(f"line {i}" for i in range(50))
    truncated = m._truncate_observation(long)
    assert "omitted" in truncated, f"CLI-4 FAIL: long output not truncated: {truncated[:100]}"
    assert len(truncated.splitlines()) <= 13  # 10 lines + omitted marker + 2 tail


# ── CLI-6: final answer structural separator ────────────────────────

def test_render_assistant_uses_rule_separator():
    """CLI-6: render_assistant must use Rule not Panel."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    captured = io.StringIO()

    class _FakeConsole:
        def __init__(self):
            self.width = 120
        def print(self, *args, **kw):
            pass
        def rule(self, title="", **kw):
            pass
        def markdown(self, text):
            captured.write(text)

    # Must not raise; verify the method signature.
    import CodeingAgent.WhaleCode.scripts.cli as cli_module
    # Smoke: call with a plain UI (no rich).
    ui = cli.CLIUI(use_rich=False)
    ui.render_assistant("Hello world")
    # Just verify no crash and output contains the separator text.
    captured2 = io.StringIO()
    sys.stdout = captured2
    ui.render_assistant("Hello world")
    sys.stdout = sys.__stdout__
    out = captured2.getvalue()
    assert "── Assistant ──" in out or "Assistant" in out, f"CLI-6: {out}"
    assert "Hello world" in out


# ── CLI-9: enhanced /info ───────────────────────────────────────────

def test_show_runtime_info_includes_new_fields():
    """CLI-9: /info must show max_steps, tokens, context usage, bash params."""
    # Check source that the info function references these.
    run_cli_src = Path("run_cli.py").read_text()
    assert "max_steps" in run_cli_src, "CLI-9 FAIL: max_steps missing from /info"
    assert "Tokens used" in run_cli_src or "total_tokens" in run_cli_src
    assert "Context:" in run_cli_src  # context usage line
    assert "Bash CPU" in run_cli_src or "bash_cpu" in run_cli_src


# ── CLI-10: visual hierarchy ────────────────────────────────────────

def test_log_block_tiered_rendering():
    """CLI-10: tool calls / thinking / observation use compact dim lines, not Panel."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    ui = cli.CLIUI(use_rich=False)

    # All secondary-tier calls should not throw.
    for kind in ("action", "thinking", "observation", "info"):
        ui.render_log_block(kind, "test content")

    # Error/warning should still work.
    ui.render_log_block("error", "critical failure")
    ui.render_log_block("warning", "heads up")


# ── mixin override sanity ────────────────────────────────────────────

def test_mixin_has_all_new_methods():
    """Smoke: all CLI improvement helper methods exist on the mixin."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    m = cli.CLICodeAgentMixin()
    for method in (
        "_compact_args",
        "_truncate_observation",
        "_truncate_observation_info",
        "_context_snapshot",
        "get_cli_events",
        "_reasoning_display_text",
    ):
        assert hasattr(m, method), f"CLI method {method} missing"


# ── CLI-N3/N4/N6/N8/N9/N10: end-to-end transcript behavior ──────────

def test_run_agent_turn_transcript_pairs_tool_and_preserves_full_outputs(tmp_path, capsys):
    """CLI transcript should show compact status while saving full reasoning/output."""
    import CodeingAgent.WhaleCode.scripts.cli as cli

    class _Config:
        context_window = 100000

    class _FakeAgent(cli.CLICodeAgentMixin):
        def __init__(self):
            self.ui = cli.CLIUI(use_rich=False)
            self.working_dir = str(tmp_path)
            self.project_root = str(tmp_path)
            self.config = _Config()
            self._streaming_line_buffer = ""
            self._last_prompt_tokens = 2048
            self.reasoning_mode = "summary"

        def run(self, prompt):
            long_reasoning = "first diagnostic line\n" + ("hidden detail " * 80)
            long_output = "\n".join(f"line {i}" for i in range(40))
            self._render_event("step_start", {"step": 1})
            self._render_event(
                "model_output",
                {"step": 1, "reasoning_content": long_reasoning},
            )
            self._render_event(
                "tool_call",
                {
                    "tool_call_id": "call-1",
                    "tool_name": "Bash",
                    "arguments": {"command": "pytest tests/test_cli_render.py -q"},
                },
            )
            self._render_event(
                "tool_result",
                {
                    "tool_call_id": "call-1",
                    "tool_name": "Bash",
                    "status": "success",
                    "result_content": long_output,
                },
            )
            return "done"

    agent = _FakeAgent()
    ui = agent.ui
    result = cli.run_agent_turn(agent, "check cli", ui)
    out = capsys.readouterr().out

    assert result == "done"
    assert "✦ Step 1" in out
    assert "Thinking..." in out
    assert "first diagnostic line" in out
    assert "hidden detail hidden detail" not in out
    assert "full reasoning:" in out
    assert "Running Bash..." in out
    assert "▸ Bash: pytest tests/test_cli_render.py -q" in out
    assert "✓ Bash" in out
    assert "truncated; full output:" in out
    assert "/trace shows event metadata" in out
    assert "── Assistant ──" in out
    assert "done" in out

    artifact_root = tmp_path / "memory" / "cli_artifacts"
    reasoning_files = list((artifact_root / "reasoning").glob("*.txt"))
    output_files = list((artifact_root / "tool_outputs").glob("*.txt"))
    assert reasoning_files, "full reasoning artifact was not saved"
    assert output_files, "full tool output artifact was not saved"
    assert "hidden detail" in reasoning_files[0].read_text(encoding="utf-8")
    assert "line 20" in output_files[0].read_text(encoding="utf-8")

    events = agent.get_cli_events()
    assert [event["event"] for event in events] == [
        "step",
        "model_output",
        "tool_call",
        "tool_result",
    ]
    tool_result = events[-1]
    assert tool_result["tool_call_id"] == "call-1"
    assert tool_result["truncated"] is True
    assert tool_result["line_count"] == 40


def test_reasoning_off_hides_terminal_preview_but_keeps_trace_and_file(tmp_path):
    """CLI-N3: off mode hides terminal reasoning while preserving audit data."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    captured = io.StringIO()

    class _FakeUI:
        use_rich = False
        def print(self, msg="", end="\n", flush=False):
            captured.write(str(msg) + end)
        def info(self, msg): captured.write(f"[info] {msg}\n")
        def status(self, msg): captured.write(f"[status] {msg}\n")
        def render_log_block(self, kind, text): captured.write(f"[{kind}] {text}\n")

    mixin = cli.CLICodeAgentMixin()
    mixin.ui = _FakeUI()
    mixin.working_dir = str(tmp_path)
    mixin.project_root = str(tmp_path)
    mixin.reasoning_mode = "off"
    full = "private reasoning that should not render"

    mixin._render_event("model_output", {"step": 1, "reasoning_content": full})

    out = captured.getvalue()
    assert full not in out
    events = mixin.get_cli_events()
    assert events[-1]["reasoning"] == full
    assert Path(events[-1]["reasoning_path"]).exists()
    assert Path(events[-1]["reasoning_path"]).read_text(encoding="utf-8") == full
    assert str(tmp_path / "memory" / "cli_artifacts" / "reasoning") in events[-1]["reasoning_path"]


def test_cli_artifacts_can_use_custom_dir_or_be_disabled(tmp_path):
    """CLI-A1: artifact storage defaults to memory and can be configured."""
    import CodeingAgent.WhaleCode.scripts.cli as cli

    class _FakeUI:
        use_rich = False
        def print(self, msg="", end="\n", flush=False): pass
        def info(self, msg): pass
        def status(self, msg): pass
        def render_log_block(self, kind, text): pass

    mixin = cli.CLICodeAgentMixin()
    mixin.ui = _FakeUI()
    mixin.working_dir = str(tmp_path)
    mixin.project_root = str(tmp_path)
    mixin.reasoning_mode = "off"
    mixin.cli_artifact_dir = "memory/custom_cli_artifacts"
    mixin._render_event("model_output", {"step": 1, "reasoning_content": "audit me"})
    events = mixin.get_cli_events()
    assert "memory/custom_cli_artifacts/reasoning" in events[-1]["reasoning_path"]

    disabled = cli.CLICodeAgentMixin()
    disabled.ui = _FakeUI()
    disabled.working_dir = str(tmp_path)
    disabled.project_root = str(tmp_path)
    disabled.reasoning_mode = "off"
    disabled.save_cli_artifacts = False
    disabled._render_event("model_output", {"step": 1, "reasoning_content": "trace only"})
    disabled_events = disabled.get_cli_events()
    assert disabled_events[-1]["reasoning_path"] is None
    assert not (tmp_path / "memory" / "cli_artifacts" / "reasoning" / "trace_only.txt").exists()


def test_event_history_renderer_shows_structured_timeline(capsys):
    """CLI-N8: /trace data can be rendered as event timeline."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    ui = cli.CLIUI(use_rich=False)
    ui.render_event_history(
        [
            {"event": "step", "summary": "Step 1"},
            {"event": "tool_call", "summary": "▸ Bash: pytest"},
            {"event": "tool_result", "summary": "✓ Bash 0.1s 2 lines, 10 chars"},
        ],
        limit=2,
    )
    out = capsys.readouterr().out
    assert "1. [step]" not in out
    assert "2. [tool_call] ▸ Bash: pytest" in out
    assert "3. [tool_result] ✓ Bash" in out


def test_parser_accepts_reasoning_modes_and_trace_command_is_registered():
    """CLI-N3/N8: public CLI knobs are wired into parser and command lists."""
    import CodeingAgent.WhaleCode.scripts.cli as cli
    parser = cli.build_parser()
    args = parser.parse_args(["--reasoning", "full", "--plain", "hello"])
    assert args.reasoning == "full"
    artifact_args = parser.parse_args(["--artifact-dir", "memory/custom_cli_artifacts", "--no-artifacts", "hello"])
    assert artifact_args.artifact_dir == "memory/custom_cli_artifacts"
    assert artifact_args.no_artifacts is True
    assert "/trace" in cli.INTERACTIVE_EXACT_COMMANDS
