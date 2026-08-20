"""CLI /review 命令的参数解析与渲染测试."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保项目根在 sys.path (scripts.cli 可导入)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hello_agents.agents.roles.reviewer import ReviewFinding, ReviewReport  # noqa: E402
from scripts.cli import (  # noqa: E402
    CLIUI,
    _parse_review_args,
    _render_review_report,
)


class TestParseReviewArgs:
    def test_no_args(self):
        opts = _parse_review_args("/review")
        assert opts == {"focus": None, "staged": False, "targets": []}

    def test_staged(self):
        opts = _parse_review_args("/review --staged")
        assert opts["staged"] is True
        assert opts["targets"] == []

    def test_focus(self):
        opts = _parse_review_args("/review --focus security")
        assert opts["focus"] == "security"

    def test_full_resets_focus(self):
        opts = _parse_review_args("/review --focus security --full")
        assert opts["focus"] is None

    def test_single_file(self):
        opts = _parse_review_args("/review src/main.py")
        assert opts["targets"] == ["src/main.py"]

    def test_multiple_files_collected(self):
        """多个文件目标全部保留 (修复覆盖 bug)."""
        opts = _parse_review_args("/review a.py b.py c.py")
        assert opts["targets"] == ["a.py", "b.py", "c.py"]

    def test_pr_url(self):
        opts = _parse_review_args("/review https://github.com/o/r/pull/12")
        assert opts["targets"] == ["https://github.com/o/r/pull/12"]

    def test_pr_number(self):
        opts = _parse_review_args("/review #12")
        assert opts["targets"] == ["#12"]

    def test_focus_with_files(self):
        opts = _parse_review_args("/review --focus performance a.py b.py")
        assert opts["focus"] == "performance"
        assert opts["targets"] == ["a.py", "b.py"]

    def test_bare_focus_at_end_ignored(self):
        """末尾裸 --focus 不得落入 targets."""
        opts = _parse_review_args("/review a.py --focus")
        assert opts["focus"] is None
        assert opts["targets"] == ["a.py"]

    def test_staged_and_files_combined(self):
        opts = _parse_review_args("/review --staged a.py")
        assert opts["staged"] is True
        assert opts["targets"] == ["a.py"]


class TestRenderReviewReport:
    def _report(self):
        return ReviewReport(
            summary="发现 1 个问题",
            findings=[
                ReviewFinding(
                    severity="critical", category="security", file="a.py", line=1,
                    title="hardcoded key", description="d", suggestion="s",
                )
            ],
            score={"security": 5},
            recommendations=["fix it"],
        )

    def test_plain_mode_falls_back_to_markdown(self, capsys):
        ui = CLIUI(use_rich=False)
        _render_review_report(ui, self._report())
        out = capsys.readouterr().out
        assert "Review Report" in out
        assert "[CRITICAL]" in out
        assert "a.py:1" in out

    def test_rich_mode_renders_without_error(self):
        ui = CLIUI(use_rich=True)
        _render_review_report(ui, self._report())
        text = ui.console.export_text()
        assert "CRITICAL" in text
        assert "hardcoded key" in text
        assert "security" in text

    def test_empty_findings_success_message(self, capsys):
        ui = CLIUI(use_rich=False)
        _render_review_report(ui, ReviewReport(summary="clean"))
        assert "clean" in capsys.readouterr().out

    def test_error_report_shows_warning(self, capsys):
        ui = CLIUI(use_rich=False)
        report = ReviewReport(summary="gh unavailable", error="gh_cli_unavailable")
        _render_review_report(ui, report)
        assert "gh unavailable" in capsys.readouterr().out
