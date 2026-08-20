"""Tests for the CLI interaction enhancements (phase 4.1).

Covers:
    1. Syntax highlighting — diff lexer detection + Markdown code theme
    2. Turn spinner — status routing + context-manager lifecycle
    3. Auto-completion — slash commands / paths / session names

All tests run without a TTY; Rich and prompt_toolkit paths skip gracefully
when the optional dependencies are missing.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Bootstrap hello_agents + repo-root import path (same pattern as test_cli_render.py)
CODE = Path(__file__).resolve().parents[1] / "code"
if "hello_agents" not in sys.modules:
    pkg = types.ModuleType("hello_agents")
    pkg.__path__ = [str(CODE)]
    pkg.__file__ = str(CODE / "__init__.py")
    sys.modules["hello_agents"] = pkg
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import cli  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rich_ui():
    """CLIUI backed by an in-memory console (force_terminal for full paths)."""
    ui = cli.CLIUI(use_rich=True)
    if not ui.use_rich:
        pytest.skip("rich is not available")
    buf = io.StringIO()
    ui.console = cli.Console(file=buf, force_terminal=True, width=100, theme=cli.WHALE_THEME)
    return ui, buf


def _plain_ui():
    return cli.CLIUI(use_rich=False)


# ===========================================================================
# 1. Syntax highlighting
# ===========================================================================


class TestGuessOutputLexer:
    DIFF = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n-old\n+new\n"
    )

    def test_diff_tool_with_git_header(self):
        assert cli._guess_output_lexer("GitDiff", self.DIFF) == "diff"

    def test_edit_tool_with_hunks(self):
        body = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        assert cli._guess_output_lexer("Edit", body) == "diff"

    def test_plain_output_no_lexer(self):
        assert cli._guess_output_lexer("Read", "12:def foo():\n13:    pass\n") is None
        assert cli._guess_output_lexer("Bash", "total 12\ndrwxr-xr-x 4 user\n") is None

    def test_empty_input(self):
        assert cli._guess_output_lexer("Edit", "") is None

    def test_non_diff_tool_with_plain_hunkless_text(self):
        assert cli._guess_output_lexer("Grep", "a.py:10:--- not a diff header") is None


class TestSyntaxHighlightingRendering:
    def test_tool_card_uses_syntax_for_diff(self, monkeypatch):
        ui, buf = _rich_ui()
        captured = {}
        real_syntax = cli.Syntax

        def _spy_syntax(code, lexer, **kwargs):
            captured["lexer"] = lexer
            captured["theme"] = kwargs.get("theme")
            return real_syntax(code, lexer, **kwargs)

        monkeypatch.setattr(cli, "Syntax", _spy_syntax)
        diff_body = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        ui.render_tool_card("GitDiff", "HEAD~1", is_error=False, elapsed=0.2,
                            meta="1 lines, 80 chars", body=diff_body)
        assert captured.get("lexer") == "diff"
        assert captured.get("theme") == cli.Palette.CODE_THEME

    def test_tool_card_plain_body_uses_text(self, monkeypatch):
        ui, buf = _rich_ui()
        spy = Mock(side_effect=AssertionError("Syntax must not be used for plain output"))
        monkeypatch.setattr(cli, "Syntax", spy)
        ui.render_tool_card("Bash", "ls", is_error=False, elapsed=0.1,
                            meta="2 lines, 10 chars", body="file1\nfile2")
        spy.assert_not_called()
        assert "file1" in buf.getvalue()

    def test_render_assistant_passes_code_theme(self, monkeypatch):
        ui, buf = _rich_ui()
        captured = {}
        real_markdown = cli.Markdown

        def _spy_markdown(text, **kwargs):
            captured.update(kwargs)
            return real_markdown(text, **kwargs)

        monkeypatch.setattr(cli, "Markdown", _spy_markdown)
        ui.render_assistant("Here is code:\n```python\nprint(1)\n```")
        assert captured.get("code_theme") == cli.Palette.CODE_THEME


# ===========================================================================
# 2. Turn spinner
# ===========================================================================


class TestTurnSpinner:
    def test_status_routes_into_live_spinner(self):
        ui, buf = _rich_ui()
        live = Mock()
        ui._live_status = live
        ui.status("Running Bash...")
        live.update.assert_called_once()
        assert "Running Bash..." in live.update.call_args[0][0]

    def test_status_prints_when_no_spinner(self):
        ui, buf = _rich_ui()
        ui.status("idle hint")
        assert "idle hint" in buf.getvalue()

    def test_spinner_noop_for_plain_mode(self):
        ui = _plain_ui()
        with ui.turn_spinner("working..."):
            assert ui._live_status is None
        assert ui._live_status is None

    def test_spinner_noop_for_non_terminal_console(self):
        ui = cli.CLIUI(use_rich=True)
        if not ui.use_rich:
            pytest.skip("rich is not available")
        # record=True console without a TTY → is_terminal False
        ui.console = cli.Console(file=io.StringIO(), force_terminal=False)
        with ui.turn_spinner("working..."):
            assert ui._live_status is None

    def test_spinner_sets_and_clears_live_status_on_terminal(self):
        from unittest.mock import MagicMock

        ui, buf = _rich_ui()
        status_obj = MagicMock()
        ui.console.status = Mock(return_value=status_obj)
        with ui.turn_spinner("Thinking..."):
            assert ui._live_status is status_obj
            ui.status("Running Read...")
            status_obj.update.assert_called_once()
        assert ui._live_status is None
        status_obj.__enter__.assert_called_once()
        status_obj.__exit__.assert_called_once()

    def test_spinner_clears_state_on_exception(self):
        from unittest.mock import MagicMock

        ui, buf = _rich_ui()
        status_obj = MagicMock()
        ui.console.status = Mock(return_value=status_obj)
        with pytest.raises(RuntimeError):
            with ui.turn_spinner("x"):
                raise RuntimeError("boom")
        assert ui._live_status is None


# ===========================================================================
# 3. Auto-completion
# ===========================================================================


needs_ptk = pytest.mark.skipif(
    not getattr(cli, "PROMPT_TOOLKIT_AVAILABLE", False),
    reason="prompt_toolkit is not available",
)


def _doc(text: str):
    from prompt_toolkit.document import Document

    return Document(text, cursor_position=len(text))


def _completions(completer, text: str):
    return list(completer.get_completions(_doc(text), None))


@needs_ptk
class TestWhaleCompleter:
    def test_all_commands_on_bare_slash(self):
        comp = cli.WhaleCompleter()
        results = _completions(comp, "/")
        names = {c.text for c in results}
        expected = {name for name, _ in cli.SLASH_COMMANDS}
        assert expected <= names

    def test_prefix_filtering_with_meta(self):
        comp = cli.WhaleCompleter()
        results = _completions(comp, "/he")
        assert [c.text for c in results] == ["/help"]
        assert results[0].display_meta_text

    def test_no_completion_for_natural_language(self):
        comp = cli.WhaleCompleter()
        assert _completions(comp, "please refactor") == []
        assert _completions(comp, "") == []

    def test_cd_completes_directories_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pkg").mkdir()
        (tmp_path / "readme.txt").write_text("x")
        comp = cli.WhaleCompleter()
        results = _completions(comp, "/cd p")
        # PathCompleter replaces the current word ("p"), so completion text is
        # the *suffix*; the full name shows up in the display text.
        displays = {getattr(c, "display_text", c.text) for c in results}
        assert any("pkg" in d for d in displays)
        assert not any("readme.txt" in d for d in displays)

    def test_review_completes_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "alpha.py").write_text("x")
        comp = cli.WhaleCompleter()
        results = _completions(comp, "/review al")
        displays = {getattr(c, "display_text", c.text) for c in results}
        assert any("alpha.py" in d for d in displays)

    def test_resume_completes_session_names(self):
        comp = cli.WhaleCompleter(get_sessions=lambda: ["session-latest", "demo-run", "other"])
        results = _completions(comp, "/resume session")
        texts = {c.text for c in results}
        assert texts == {"session-latest"}

    def test_resume_graceful_when_provider_fails(self):
        def _boom():
            raise RuntimeError("fs error")

        comp = cli.WhaleCompleter(get_sessions=_boom)
        assert _completions(comp, "/resume x") == []

    def test_help_lists_every_command(self, capsys):
        ui = _plain_ui()
        cli.print_help(ui)
        out = capsys.readouterr().out
        for name, _ in cli.SLASH_COMMANDS:
            assert name in out
        assert "exit" in out
