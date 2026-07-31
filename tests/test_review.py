"""Review 功能测试: ReviewReport / review_diff / review_pr / review_files."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from hello_agents.agents.roles import reviewer as reviewer_mod
from hello_agents.agents.roles.reviewer import (
    REVIEW_OUTPUT_SCHEMA,
    ReviewerRole,
    ReviewFinding,
    ReviewReport,
    _extract_json_payload,
    _parse_review_output,
)
from hello_agents.core.config import Config


def _valid_report_dict():
    return {
        "summary": "整体质量良好",
        "findings": [
            {
                "severity": "critical",
                "category": "security",
                "file": "src/auth.py",
                "line": 42,
                "title": "Hardcoded API key",
                "description": "API key written in source",
                "suggestion": "Move to env var",
            },
            {
                "severity": "medium",
                "category": "performance",
                "file": "src/parser.py",
                "line": 128,
                "title": "O(n^2) loop",
                "description": "hot path",
                "suggestion": "use dict",
            },
        ],
        "score": {"correctness": 8, "security": 6},
        "recommendations": ["fix critical first"],
    }


class _StubReviewSubAgent:
    """run() 返回预制 JSON 字符串 (结构化输出路径)."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def run(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self.payload


def _install_stub(monkeypatch, payload):
    stub = _StubReviewSubAgent(payload)
    monkeypatch.setattr(
        ReviewerRole,
        "create_subagent",
        classmethod(lambda cls, *a, **k: stub),
    )
    return stub


class TestReviewReport:
    def test_to_markdown_formatting(self):
        report = ReviewReport(
            summary="ok",
            findings=[ReviewFinding(severity="high", category="security",
                                    file="a.py", line=3, title="t",
                                    description="d", suggestion="s")],
            score={"security": 6},
            recommendations=["do x"],
        )
        md = report.to_markdown()
        assert "## Review Report" in md
        assert "[HIGH]" in md
        assert "a.py:3" in md
        assert "security: 6/10" in md
        assert "1. do x" in md

    def test_to_dict_roundtrip(self):
        report = ReviewReport(
            summary="s",
            findings=[ReviewFinding(severity="low", category="maintainability",
                                    file="b.py", title="t", description="d",
                                    suggestion="g")],
            score={"maintainability": 7},
            recommendations=["r1"],
        )
        data = report.to_dict()
        assert data["summary"] == "s"
        assert data["findings"][0]["file"] == "b.py"
        assert data["score"]["maintainability"] == 7
        assert data["error"] is None

    def test_empty_findings(self):
        report = ReviewReport(summary="clean")
        md = report.to_markdown()
        assert "Findings" not in md
        assert "clean" in md


class TestParseReviewOutput:
    def test_parse_valid_json(self):
        report = _parse_review_output(json.dumps(_valid_report_dict()))
        assert report.summary == "整体质量良好"
        assert len(report.findings) == 2
        assert report.findings[0].severity == "critical"
        assert report.findings[0].line == 42
        assert report.score == {"correctness": 8, "security": 6}
        assert report.recommendations == ["fix critical first"]
        assert report.error is None

    def test_parse_fenced_json(self):
        raw = "一些说明\n```json\n" + json.dumps(_valid_report_dict()) + "\n```"
        report = _parse_review_output(raw)
        assert len(report.findings) == 2

    def test_parse_invalid_json_fallback(self):
        report = _parse_review_output("完全是自然语言, 没有 JSON")
        assert report.error == "parse_fallback"
        assert report.findings == []

    def test_invalid_severity_category_normalized(self):
        data = _valid_report_dict()
        data["findings"][0]["severity"] = "fatal"
        data["findings"][0]["category"] = "unknown-cat"
        report = _parse_review_output(json.dumps(data))
        assert report.findings[0].severity == "info"
        assert report.findings[0].category == "maintainability"

    def test_max_findings_truncation(self):
        data = _valid_report_dict()
        data["findings"] = data["findings"] * 10
        report = _parse_review_output(json.dumps(data), max_findings=5)
        assert len(report.findings) == 5

    def test_score_clamped_to_range(self):
        data = _valid_report_dict()
        data["score"] = {"correctness": 99, "security": 0}
        report = _parse_review_output(json.dumps(data))
        assert report.score["correctness"] == 10
        assert report.score["security"] == 1

    def test_extract_json_payload_variants(self):
        assert _extract_json_payload('{"a": 1}') == {"a": 1}
        assert _extract_json_payload('```json\n{"a": 2}\n```') == {"a": 2}
        assert _extract_json_payload("no json") is None


class TestReviewDiff:
    def test_basic_diff_review(self, mock_llm, monkeypatch, tmp_path):
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        config = Config()
        report = asyncio.run(
            ReviewerRole.review_diff(
                mock_llm, config, str(tmp_path),
                "diff --git a/f.py b/f.py\n+secret = 'x'\n",
            )
        )
        assert len(report.findings) == 2
        # 结构化输出 schema 确实传给 run()
        call = stub.calls[0]
        assert call["kwargs"].get("structured_output_schema") is REVIEW_OUTPUT_SCHEMA
        assert call["kwargs"].get("structured_output_name") == "ReviewOutput"

    def test_empty_diff(self, mock_llm, tmp_path):
        report = asyncio.run(
            ReviewerRole.review_diff(mock_llm, Config(), str(tmp_path), "   ")
        )
        assert "No changes" in report.summary
        assert report.findings == []

    def test_large_diff_truncation_warning(self, mock_llm, monkeypatch, tmp_path):
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        config = Config(review_max_files=1)
        diff = "diff --git a/a.py b/a.py\n+x\ndiff --git a/b.py b/b.py\n+y\n"
        asyncio.run(
            ReviewerRole.review_diff(mock_llm, config, str(tmp_path), diff)
        )
        assert "NOTE" in stub.calls[0]["prompt"]

    def test_security_focus(self, mock_llm, monkeypatch, tmp_path):
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        asyncio.run(
            ReviewerRole.review_diff(
                mock_llm, Config(), str(tmp_path),
                "diff --git a/f.py b/f.py\n+x\n",
                review_focus="security",
            )
        )
        assert "Security" in stub.calls[0]["prompt"]

    def test_review_diff_llm_returns_invalid_json(self, mock_llm, monkeypatch, tmp_path):
        _install_stub(monkeypatch, "not json output")
        report = asyncio.run(
            ReviewerRole.review_diff(
                mock_llm, Config(), str(tmp_path), "diff --git a/f.py b/f.py\n+x\n"
            )
        )
        assert report.error == "parse_fallback"


class TestReviewPR:
    def test_review_pr_with_number(self, mock_llm, monkeypatch, tmp_path):
        _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        mock_fetch = MagicMock(return_value="diff --git a/f.py b/f.py\n+x\n")
        monkeypatch.setattr(reviewer_mod, "_fetch_pr_diff", mock_fetch)
        report = asyncio.run(
            ReviewerRole.review_pr(mock_llm, Config(), str(tmp_path), "#12")
        )
        mock_fetch.assert_called_once()
        assert mock_fetch.call_args[0][1] == "#12"
        assert len(report.findings) == 2

    def test_review_pr_gh_cli_unavailable(self, mock_llm, monkeypatch, tmp_path):
        monkeypatch.setattr(reviewer_mod, "_fetch_pr_diff", lambda *a: None)
        report = asyncio.run(
            ReviewerRole.review_pr(mock_llm, Config(), str(tmp_path), "#1")
        )
        assert report.error == "gh_cli_unavailable"

    def test_review_pr_disabled_by_config(self, mock_llm, tmp_path):
        report = asyncio.run(
            ReviewerRole.review_pr(
                mock_llm, Config(review_gh_cli_enabled=False), str(tmp_path), "#1"
            )
        )
        assert report.error == "gh_cli_disabled"


class TestReviewFiles:
    def test_review_single_file(self, mock_llm, monkeypatch, tmp_path):
        target = tmp_path / "main.py"
        target.write_text("def f():\n    return 1\n", encoding="utf-8")
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        report = asyncio.run(
            ReviewerRole.review_files(mock_llm, Config(), str(tmp_path), ["main.py"])
        )
        assert len(report.findings) == 2
        assert "### File: main.py" in stub.calls[0]["prompt"]

    def test_review_nonexistent_file(self, mock_llm, tmp_path):
        report = asyncio.run(
            ReviewerRole.review_files(mock_llm, Config(), str(tmp_path), ["ghost.py"])
        )
        assert report.error == "files_not_found"

    def test_review_multiple_files(self, mock_llm, monkeypatch, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("b = 2\n", encoding="utf-8")
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        asyncio.run(
            ReviewerRole.review_files(mock_llm, Config(), str(tmp_path), ["a.py", "b.py"])
        )
        prompt = stub.calls[0]["prompt"]
        assert "### File: a.py" in prompt and "### File: b.py" in prompt

    def test_review_outside_workspace_rejected(self, mock_llm, tmp_path):
        report = asyncio.run(
            ReviewerRole.review_files(mock_llm, Config(), str(tmp_path), ["../outside.py"])
        )
        assert report.error == "files_not_found"


class TestReviewAgentHelpers:
    def test_get_git_diff_returns_string(self, tmp_path):
        from hello_agents.agents.review_agent import get_git_diff

        # 非 git 仓库 → 返回空串 (不抛异常)
        assert get_git_diff(str(tmp_path)) == ""

    def test_review_working_diff_empty(self, mock_llm, tmp_path):
        from hello_agents.agents.review_agent import review_working_diff

        report = asyncio.run(review_working_diff(str(tmp_path), mock_llm, Config()))
        assert "No changes" in report.summary

    def test_review_staged_diff_empty(self, mock_llm, tmp_path):
        from hello_agents.agents.review_agent import review_staged_diff

        report = asyncio.run(review_staged_diff(str(tmp_path), mock_llm, Config()))
        assert "No changes" in report.summary

    def test_get_git_diff_staged_flag(self, monkeypatch, tmp_path):
        from hello_agents.agents import review_agent

        calls = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            m = MagicMock()
            m.returncode = 0
            m.stdout = "diff"
            return m

        monkeypatch.setattr(review_agent.subprocess, "run", _fake_run)
        review_agent.get_git_diff(str(tmp_path), staged=True)
        assert "--cached" in calls[0]
        review_agent.get_git_diff(str(tmp_path), staged=False)
        assert "--cached" not in calls[1]

    def test_get_git_diff_timeout_returns_empty(self, monkeypatch, tmp_path):
        import subprocess as sp

        from hello_agents.agents import review_agent

        def _boom(cmd, **kwargs):
            raise sp.TimeoutExpired(cmd, 10)

        monkeypatch.setattr(review_agent.subprocess, "run", _boom)
        assert review_agent.get_git_diff(str(tmp_path)) == ""

    def test_fetch_pr_diff_strips_hash(self, monkeypatch, tmp_path):
        calls = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            m = MagicMock()
            m.returncode = 0
            m.stdout = "pr-diff"
            return m

        monkeypatch.setattr(reviewer_mod.subprocess, "run", _fake_run)
        out = reviewer_mod._fetch_pr_diff(str(tmp_path), "#42")
        assert out == "pr-diff"
        assert "42" in calls[0] and "#42" not in calls[0]

    def test_fetch_pr_diff_nonzero_exit_returns_none(self, monkeypatch, tmp_path):
        m = MagicMock()
        m.returncode = 1
        m.stdout = ""
        monkeypatch.setattr(reviewer_mod.subprocess, "run", MagicMock(return_value=m))
        assert reviewer_mod._fetch_pr_diff(str(tmp_path), "7") is None

    def test_fetch_pr_diff_gh_missing_returns_none(self, monkeypatch, tmp_path):
        def _boom(cmd, **kwargs):
            raise FileNotFoundError("gh not found")

        monkeypatch.setattr(reviewer_mod.subprocess, "run", _boom)
        assert reviewer_mod._fetch_pr_diff(str(tmp_path), "7") is None


class TestReviewEdgeCases:
    def test_to_markdown_severity_ordering(self):
        """findings 按严重度排序输出, 与插入顺序无关."""
        report = ReviewReport(
            summary="s",
            findings=[
                ReviewFinding(severity="info", category="maintainability", file="a", title="i"),
                ReviewFinding(severity="critical", category="security", file="b", title="c"),
            ],
        )
        md = report.to_markdown()
        assert md.index("[CRITICAL]") < md.index("[INFO]")

    def test_review_files_mixed_missing_and_present(self, mock_llm, monkeypatch, tmp_path):
        (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        report = asyncio.run(
            ReviewerRole.review_files(
                mock_llm, Config(), str(tmp_path), ["ok.py", "ghost.py"]
            )
        )
        assert report.error is None  # 有可读文件 → 正常审查
        prompt = stub.calls[0]["prompt"]
        assert "### File: ok.py" in prompt
        assert "ghost.py" in prompt  # 在 Skipped files 说明中

    def test_review_files_large_file_truncated(self, mock_llm, monkeypatch, tmp_path):
        big = tmp_path / "big.py"
        big.write_text("x" * 60_000, encoding="utf-8")
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        asyncio.run(
            ReviewerRole.review_files(mock_llm, Config(), str(tmp_path), ["big.py"])
        )
        prompt = stub.calls[0]["prompt"]
        assert "truncated at 50000 chars" in prompt
        assert len(prompt) < 60_000

    def test_review_diff_large_diff_truncated(self, mock_llm, monkeypatch, tmp_path):
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        huge_diff = "diff --git a/f.py b/f.py\n" + ("+line\n" * 50_000)  # >200k chars
        asyncio.run(
            ReviewerRole.review_diff(mock_llm, Config(), str(tmp_path), huge_diff)
        )
        prompt = stub.calls[0]["prompt"]
        assert "truncated" in prompt.lower()
        assert len(prompt) < len(huge_diff)

    def test_unknown_focus_produces_no_instruction(self, mock_llm, monkeypatch, tmp_path):
        stub = _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        asyncio.run(
            ReviewerRole.review_diff(
                mock_llm, Config(), str(tmp_path),
                "diff --git a/f.py b/f.py\n+x\n",
                review_focus="nonexistent-dimension",
            )
        )
        assert "Focus PRIMARILY" not in stub.calls[0]["prompt"]

    def test_parse_findings_non_list_tolerated(self):
        data = {"summary": "s", "findings": "not-a-list", "score": {}, "recommendations": []}
        report = _parse_review_output(json.dumps(data))
        assert report.findings == []
        assert report.error is None

    def test_parse_finding_non_int_line_dropped(self):
        data = _valid_report_dict()
        data["findings"][0]["line"] = "42-not-int"
        report = _parse_review_output(json.dumps(data))
        assert report.findings[0].line is None

    def test_review_pr_url_passthrough(self, mock_llm, monkeypatch, tmp_path):
        _install_stub(monkeypatch, json.dumps(_valid_report_dict()))
        mock_fetch = MagicMock(return_value="diff --git a/f.py b/f.py\n+x\n")
        monkeypatch.setattr(reviewer_mod, "_fetch_pr_diff", mock_fetch)
        url = "https://github.com/o/r/pull/12"
        asyncio.run(ReviewerRole.review_pr(mock_llm, Config(), str(tmp_path), url))
        assert mock_fetch.call_args[0][1] == url

    def test_review_output_schema_wellformed(self):
        assert REVIEW_OUTPUT_SCHEMA["type"] == "object"
        required = set(REVIEW_OUTPUT_SCHEMA["required"])
        assert {"summary", "findings", "score", "recommendations"} <= required
        sev = REVIEW_OUTPUT_SCHEMA["properties"]["findings"]["items"]["properties"]["severity"]
        assert "critical" in sev["enum"]
