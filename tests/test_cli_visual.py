"""VR-1..VR-7 CLI visual-redesign tests.

These assert the *structure* of the new rich-mode rendering (boxes, titles,
spacing, status markers) and the per-tool argument summarizers, while the
plain-mode string contract stays covered by ``test_cli_render.py``.
"""
import contextlib
import io
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("rich")

# Bootstrap hello_agents + repo root on sys.path (mirror test_cli_render.py).
CODE = Path(__file__).resolve().parents[1] / "code"
if "hello_agents" not in sys.modules:
    pkg = types.ModuleType("hello_agents")
    pkg.__path__ = [str(CODE)]
    pkg.__file__ = str(CODE / "__init__.py")
    sys.modules["hello_agents"] = pkg
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import CodeingAgent.WhaleCode.scripts.cli as cli  # noqa: E402


def _rich_ui(width: int = 80):
    """A CLIUI whose console records output at a deterministic width."""
    from rich.console import Console

    ui = cli.CLIUI(use_rich=True)
    ui.console = Console(
        record=True, width=width, force_terminal=True, theme=cli.WHALE_THEME
    )
    return ui


def _plain_capture(fn, *args, **kwargs) -> str:
    ui = cli.CLIUI(use_rich=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(ui, *args, **kwargs)
    return buf.getvalue()


# ── VR-6: centralized theme/palette ──────────────────────────────────

def test_theme_palette_defined():
    """VR-6: a named theme must exist with the semantic style names."""
    assert hasattr(cli, "WHALE_THEME")
    for name in ("thinking", "accent", "success", "error", "warning", "muted"):
        assert name in cli.WHALE_THEME.styles, f"missing theme style: {name}"


# ── VR-2: thinking box ───────────────────────────────────────────────

def test_thinking_rendered_as_panel_in_rich():
    ui = _rich_ui()
    ui.render_log_block("thinking", "let me reason about the failing import")
    out = ui.console.export_text()
    assert "let me reason about the failing import" in out
    assert "╭" in out and "╮" in out, "thinking should be boxed in rich mode"
    assert "thinking" in out.lower(), "thinking box should be titled"


def test_thinking_plain_mode_line_preserved():
    out = _plain_capture(cli.CLIUI.render_log_block, "thinking", "reason text")
    assert "reason text" in out
    assert "╭" not in out, "plain mode must stay boxless"


# ── VR-3: assistant output box ───────────────────────────────────────

def test_assistant_rendered_as_panel_in_rich():
    ui = _rich_ui()
    ui.render_assistant("Here is the final answer.")
    out = ui.console.export_text()
    assert "Here is the final answer." in out
    assert "Assistant" in out
    assert "╭" in out, "assistant answer should be boxed in rich mode"


def test_assistant_plain_keeps_rule_text():
    out = _plain_capture(cli.CLIUI.render_assistant, "hi there")
    assert "── Assistant ──" in out
    assert "hi there" in out


# ── VR-4: tool-call card ─────────────────────────────────────────────

def test_tool_card_rendered_as_panel_in_rich():
    ui = _rich_ui()
    ui.render_tool_card(
        tool_name="Bash",
        arg_summary="pytest -q",
        is_error=False,
        elapsed=0.4,
        meta="3 lines, 20 chars",
        body="ok\ndone",
    )
    out = ui.console.export_text()
    assert "Bash" in out and "pytest -q" in out
    assert "╭" in out
    assert "✓" in out, "success card should carry the success marker"


def test_tool_card_error_uses_error_marker():
    ui = _rich_ui()
    ui.render_tool_card(
        tool_name="Bash",
        arg_summary="do bad thing",
        is_error=True,
        elapsed=0.1,
        meta="1 lines, 4 chars",
        body="boom",
    )
    out = ui.console.export_text()
    assert "✗" in out, "error card should carry the error marker"


# ── VR-4: per-tool argument summarizers ──────────────────────────────

def _mixin():
    return cli.CLICodeAgentMixin()


def test_summarize_bash_first_line():
    m = _mixin()
    assert (
        m._summarize_tool_args("Bash", {"command": "pytest tests/test_cli_render.py -q"})
        == "pytest tests/test_cli_render.py -q"
    )
    multi = m._summarize_tool_args("Bash", {"command": "cd proj\nmake\n./run"})
    assert multi.startswith("cd proj") and "…" in multi


def test_summarize_read_shows_path_and_offset():
    m = _mixin()
    assert m._summarize_tool_args("Read", {"path": "src/main.py"}) == "src/main.py"
    assert (
        m._summarize_tool_args("Read", {"path": "src/main.py", "offset": 120})
        == "src/main.py:120"
    )


def test_summarize_edit_flags_replace_all():
    m = _mixin()
    s = m._summarize_tool_args(
        "Edit", {"path": "a.py", "old_string": "x", "new_string": "y", "replace_all": True}
    )
    assert "a.py" in s and "all" in s.lower()


def test_summarize_write_shows_size():
    m = _mixin()
    s = m._summarize_tool_args("Write", {"path": "out.txt", "content": "abcde"})
    assert "out.txt" in s and "5" in s


def test_summarize_grep_pattern_and_path():
    m = _mixin()
    s = m._summarize_tool_args("Grep", {"pattern": "TODO", "path": "src", "include": "*.py"})
    assert "TODO" in s and "src" in s


def test_summarize_todowrite_counts():
    m = _mixin()
    todos = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ]
    s = m._summarize_tool_args("TodoWrite", {"todos": todos})
    assert "3" in s  # total task count surfaced


def test_summarize_unknown_tool_falls_back():
    m = _mixin()
    # Unknown tool → generic _compact_args behavior (first meaningful field).
    assert m._summarize_tool_args("Mystery", {"path": "p.txt"}) == "p.txt"


# ── VR-5: spacing + step header ──────────────────────────────────────

def test_spacer_emits_blank_line():
    ui = _rich_ui()
    ui.spacer(1)
    assert "\n" in ui.console.export_text()


def test_step_header_plain_keeps_marker():
    out = _plain_capture(cli.CLIUI.render_step_header, 3, "[ctx 10/100 10%]")
    assert "✦ Step 3" in out


def test_step_header_rich_is_boxless_rule():
    ui = _rich_ui()
    ui.render_step_header(2, "")
    out = ui.console.export_text()
    assert "Step 2" in out


def test_step_header_rich_preserves_bracketed_ctx():
    # The context snapshot contains square brackets; it must not be swallowed by
    # Rich markup parsing when rendered into the step rule.
    ui = _rich_ui(width=100)
    ui.render_step_header(1, "[ctx 3,200 / 100,000  3%]")
    out = ui.console.export_text()
    assert "Step 1" in out
    assert "ctx 3,200" in out


# ── VR-4: rich tool_call defers the header to the card ───────────────

def test_tool_call_defers_header_in_rich(tmp_path):
    ui = _rich_ui()

    class _A(cli.CLICodeAgentMixin):
        pass

    a = _A()
    a.ui = ui
    a.working_dir = str(tmp_path)
    a.project_root = str(tmp_path)
    a.reasoning_mode = "off"
    a._render_event(
        "tool_call",
        {"tool_call_id": "c1", "tool_name": "Bash", "arguments": {"command": "ls -la"}},
    )
    out = ui.console.export_text()
    # In rich mode the call header is deferred; no box is drawn until the result.
    assert "╭" not in out


# ── VR-4b: /tools grouped view ───────────────────────────────────────

class _FakeTool:
    def __init__(self, name, description="does things"):
        self.name = name
        self.description = description


class _FakeToolAgent:
    def __init__(self, names):
        tools = [_FakeTool(n) for n in names]

        class _Reg:
            def get_all_tools(self_inner):
                return list(tools)

        self.tool_registry = _Reg()
        self._tools = tools


def test_tool_category_classifier():
    assert cli.CLIUI._tool_category("Read") == "File"
    assert cli.CLIUI._tool_category("Bash") == "Shell"
    assert cli.CLIUI._tool_category("WebSearch") == "Web"
    assert cli.CLIUI._tool_category("TodoWrite") == "Planning"
    assert cli.CLIUI._tool_category("Whatever") == "Other"


def test_render_tools_grouped_shows_categories():
    ui = _rich_ui(width=100)
    ui.render_tools(_FakeToolAgent(["Read", "Bash", "WebFetch", "TodoWrite", "Mystery"]))
    out = ui.console.export_text()
    for cat in ("File", "Shell", "Web", "Planning", "Other"):
        assert cat in out, f"missing category {cat}"
    assert "Read" in out and "Bash" in out


def test_render_tools_full_shows_descriptions():
    a = _FakeToolAgent(["Read"])
    a._tools[0].description = "reads a file from disk"
    ui = _rich_ui(width=100)
    ui.render_tools(a, full=True)
    assert "reads a file from disk" in ui.console.export_text()


# ── VR-1: input zone has a dark background ───────────────────────────

def test_input_style_has_background_chip_and_toolbar():
    tokens = cli._input_style_tokens()
    # The prompt glyph chip and the status toolbar must both be backed by a
    # dark surface (the user's core ask: the input should look like a field).
    assert tokens["chip"].startswith("bg:")
    assert tokens["toolbar"].startswith("bg:")
    assert "placeholder" in tokens


