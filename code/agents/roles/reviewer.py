"""Reviewer 角色 — 代码审查专家 (双用途).

用途 1: Orchestra pipeline 中的审查阶段 (作为子 Agent 被调度)
用途 2: 独立审查 API (review_diff / review_pr / review_files), 供 CLI /review 使用
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.config import Config
from ...core.llm import HelloAgentsLLM
from .base import Role, RoleConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
REVIEWER_SYSTEM_PROMPT: str = (
    _PROJECT_ROOT / "code" / "prompts" / "roles" / "reviewer.md"
).read_text(encoding="utf-8")

REVIEW_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Overall assessment"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "info"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "correctness",
                            "security",
                            "performance",
                            "maintainability",
                            "test-coverage",
                        ],
                    },
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["severity", "category", "file", "title", "description", "suggestion"],
            },
        },
        "score": {
            "type": "object",
            "properties": {
                "correctness": {"type": "integer", "minimum": 1, "maximum": 10},
                "security": {"type": "integer", "minimum": 1, "maximum": 10},
                "performance": {"type": "integer", "minimum": 1, "maximum": 10},
                "maintainability": {"type": "integer", "minimum": 1, "maximum": 10},
                "test_coverage": {"type": "integer", "minimum": 1, "maximum": 10},
            },
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "findings", "score", "recommendations"],
}

_SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")
_VALID_SEVERITIES = frozenset(_SEVERITY_ORDER)
_VALID_CATEGORIES = frozenset(
    {"correctness", "security", "performance", "maintainability", "test-coverage"}
)

# 输入规模护栏: 防止超大 diff / 文件内容撑爆子 Agent 上下文
_MAX_DIFF_CHARS = 200_000
_MAX_FILE_CHARS = 50_000

_FOCUS_HINTS = {
    "security": "Focus PRIMARILY on the Security dimension; still report critical issues of other dimensions.",
    "performance": "Focus PRIMARILY on the Performance dimension; still report critical issues of other dimensions.",
    "perf": "Focus PRIMARILY on the Performance dimension; still report critical issues of other dimensions.",
    "correctness": "Focus PRIMARILY on the Correctness dimension; still report critical issues of other dimensions.",
    "maintainability": "Focus PRIMARILY on the Maintainability dimension; still report critical issues of other dimensions.",
    "test-coverage": "Focus PRIMARILY on Test Coverage; still report critical issues of other dimensions.",
}


@dataclass
class ReviewFinding:
    """单个审查发现."""

    severity: str       # critical | high | medium | low | info
    category: str       # correctness | security | performance | maintainability | test-coverage
    file: str           # 文件路径 (相对 project_root)
    line: Optional[int] = None
    title: str = ""
    description: str = ""
    suggestion: str = ""


@dataclass
class ReviewReport:
    """结构化审查报告."""

    summary: str
    findings: List[ReviewFinding] = field(default_factory=list)
    score: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    error: Optional[str] = None  # 流程级错误 (如 gh CLI 不可用), 与内容降级区分

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "file": f.file,
                    "line": f.line,
                    "title": f.title,
                    "description": f.description,
                    "suggestion": f.suggestion,
                }
                for f in self.findings
            ],
            "score": dict(self.score),
            "recommendations": list(self.recommendations),
            "error": self.error,
        }

    def to_markdown(self) -> str:
        lines = [f"## Review Report", "", self.summary or "(no summary)", ""]
        if self.error:
            lines += [f"**Error**: {self.error}", ""]
        if self.score:
            lines.append("### Scores")
            for cat, val in self.score.items():
                lines.append(f"- {cat}: {val}/10")
            lines.append("")
        if self.findings:
            lines.append("### Findings")
            for sev in _SEVERITY_ORDER:
                for f in [x for x in self.findings if x.severity == sev]:
                    loc = f.file + (f":{f.line}" if f.line else "")
                    lines.append(f"- **[{sev.upper()}] [{f.category}] {f.title}** ({loc})")
                    if f.description:
                        lines.append(f"  - {f.description}")
                    if f.suggestion:
                        lines.append(f"  - Suggestion: {f.suggestion}")
            lines.append("")
        if self.recommendations:
            lines.append("### Recommendations")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
        return "\n".join(lines).rstrip() + "\n"


class ReviewerRole(Role):
    """代码审查专家 — 只读 + Bash (仅 git 检查), 禁写入."""

    @staticmethod
    def get_config() -> RoleConfig:
        return RoleConfig(
            name="reviewer",
            description="Code review and PR review specialist",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            allowed_tools=["Bash"],  # 显式放行 Bash (git diff/log/show)
            denied_tools=["Write", "Edit", "Delete"],
            allowed_categories={"readonly"},
            # 不得含 "dangerous": BashTool.category="dangerous", 黑名单优先会误删 Bash
            denied_categories={"write"},
            max_steps=30,
        )

    # ------------------------------------------------------------------
    # 独立 Review API (用途 2)
    # ------------------------------------------------------------------

    @classmethod
    async def review_diff(
        cls,
        llm: HelloAgentsLLM,
        config: Config,
        project_root: str,
        diff_content: str,
        *,
        review_focus: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> ReviewReport:
        """审查 git diff 内容, 返回结构化 ReviewReport."""
        if not diff_content.strip():
            return ReviewReport(summary="No changes to review (empty diff).")

        limit = max_files if max_files is not None else config.review_max_files
        file_count = len(re.findall(r"^diff --git ", diff_content, flags=re.MULTILINE))
        size_note = ""
        if file_count > limit:
            size_note = (
                f"\n\nNOTE: This diff touches {file_count} files (limit {limit}). "
                "Prioritize the highest-risk changes."
            )
        if len(diff_content) > _MAX_DIFF_CHARS:
            diff_content = diff_content[:_MAX_DIFF_CHARS]
            size_note += (
                f"\n\nNOTE: Diff truncated to {_MAX_DIFF_CHARS} characters; "
                "review only what is visible."
            )

        prompt = (
            "Review the following git diff. Produce a structured review report.\n"
            f"{_focus_instruction(review_focus)}{size_note}\n\n"
            f"```diff\n{diff_content}\n```"
        )
        return await cls._review_with_subagent(llm, config, project_root, prompt)

    @classmethod
    async def review_pr(
        cls,
        llm: HelloAgentsLLM,
        config: Config,
        project_root: str,
        pr_ref: str,
        *,
        review_focus: Optional[str] = None,
    ) -> ReviewReport:
        """审查 GitHub PR (通过 gh CLI 获取 diff 后委托 review_diff)."""
        if not config.review_gh_cli_enabled:
            return ReviewReport(
                summary="PR review is disabled (review_gh_cli_enabled=False).",
                error="gh_cli_disabled",
            )
        diff = await asyncio.to_thread(_fetch_pr_diff, project_root, pr_ref)
        if diff is None:
            return ReviewReport(
                summary=(
                    f"Failed to fetch PR '{pr_ref}' via gh CLI. "
                    "Ensure `gh` is installed and authenticated (`gh auth status`)."
                ),
                error="gh_cli_unavailable",
            )
        return await cls.review_diff(
            llm, config, project_root, diff, review_focus=review_focus
        )

    @classmethod
    async def review_files(
        cls,
        llm: HelloAgentsLLM,
        config: Config,
        project_root: str,
        file_paths: List[str],
        *,
        review_focus: Optional[str] = None,
    ) -> ReviewReport:
        """审查指定文件列表 (完整文件内容, 非 diff)."""
        from pathlib import Path

        # root 也必须 resolve: macOS 上 /var → /private/var 符号链接会导致
        # relative_to 误判为越界
        root = Path(project_root).resolve()
        sections: List[str] = []
        missing: List[str] = []
        for rel in file_paths:
            path = (root / rel).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                missing.append(f"{rel} (outside workspace)")
                continue
            if not path.is_file():
                missing.append(rel)
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                missing.append(rel)
                continue
            if len(content) > _MAX_FILE_CHARS:
                content = (
                    content[:_MAX_FILE_CHARS]
                    + f"\n... [truncated at {_MAX_FILE_CHARS} chars]"
                )
            sections.append(f"### File: {rel}\n```\n{content}\n```")

        if not sections:
            return ReviewReport(
                summary=f"No readable files to review. Missing: {', '.join(missing)}",
                error="files_not_found",
            )

        note = f"\n\nSkipped files: {', '.join(missing)}" if missing else ""
        prompt = (
            "Review the following source files in full. Produce a structured review report.\n"
            f"{_focus_instruction(review_focus)}{note}\n\n" + "\n\n".join(sections)
        )
        return await cls._review_with_subagent(llm, config, project_root, prompt)

    @classmethod
    async def _review_with_subagent(
        cls,
        llm: HelloAgentsLLM,
        config: Config,
        project_root: str,
        review_prompt: str,
    ) -> ReviewReport:
        """核心审查逻辑: 创建隔离 Reviewer 子 Agent, 以结构化输出执行审查.

        子 Agent 的 run() 是同步阻塞调用, 放到线程中执行以避免阻塞事件循环。
        """
        subagent = cls.create_subagent(llm, config, project_root, project_root)
        raw = await asyncio.to_thread(
            subagent.run,
            review_prompt,
            structured_output_schema=REVIEW_OUTPUT_SCHEMA,
            structured_output_name="ReviewOutput",
        )
        return _parse_review_output(raw, max_findings=config.review_max_findings)


# ----------------------------------------------------------------------
# 模块级辅助函数
# ----------------------------------------------------------------------


def _focus_instruction(review_focus: Optional[str]) -> str:
    if not review_focus:
        return ""
    hint = _FOCUS_HINTS.get(review_focus.strip().lower())
    return f"\n{hint}" if hint else ""


def _fetch_pr_diff(project_root: str, pr_ref: str) -> Optional[str]:
    """通过 gh CLI 获取 PR diff; 失败返回 None."""
    ref = pr_ref.strip()
    if ref.startswith("#"):
        ref = ref[1:]
    try:
        result = subprocess.run(
            ["gh", "pr", "diff", ref],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _extract_json_payload(raw: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中提取 JSON 对象 (直接解析 → 去围栏 → 首个 {...} 块)."""
    text = raw.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _parse_review_output(raw: str, *, max_findings: int = 30) -> ReviewReport:
    """解析子 Agent 输出 → ReviewReport; 失败时降级为含原文的报告."""
    data = _extract_json_payload(raw)
    if data is None:
        return ReviewReport(
            summary="Review completed but the output was not valid JSON; raw output follows.",
            findings=[],
            score={},
            recommendations=[],
            error="parse_fallback",
        )

    findings: List[ReviewFinding] = []
    for item in data.get("findings") or []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "info")).lower()
        category = str(item.get("category", "maintainability")).lower()
        line = item.get("line")
        findings.append(
            ReviewFinding(
                severity=severity if severity in _VALID_SEVERITIES else "info",
                category=category if category in _VALID_CATEGORIES else "maintainability",
                file=str(item.get("file", "")),
                line=line if isinstance(line, int) else None,
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                suggestion=str(item.get("suggestion", "")),
            )
        )
    findings = findings[: max(0, max_findings)]

    score: Dict[str, int] = {}
    for key, val in (data.get("score") or {}).items():
        if isinstance(val, (int, float)):
            score[str(key)] = max(1, min(10, int(val)))

    recommendations = [str(r) for r in (data.get("recommendations") or [])]

    return ReviewReport(
        summary=str(data.get("summary", "")),
        findings=findings,
        score=score,
        recommendations=recommendations,
    )
