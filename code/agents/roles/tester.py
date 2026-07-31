"""Tester 角色 — 代码测试专家 (只读 + Write/Edit + Bash, 禁 Delete).

系统提示词集中维护于 ``code/prompts/roles/tester.md``。
"""

from __future__ import annotations

from pathlib import Path

from .base import Role, RoleConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
TESTER_SYSTEM_PROMPT: str = (
    _PROJECT_ROOT / "code" / "prompts" / "roles" / "tester.md"
).read_text(encoding="utf-8")


class TesterRole(Role):
    """代码测试专家 — 编写并运行测试, 分析报告.

    注意: ``allowed_categories`` 含 ``"write"`` 会同时匹配 Write/Edit/Delete
    (三者 category 均为 "write"), 因此必须用 ``denied_tools=["Delete"]``
    点名排除 (黑名单优先)。``denied_categories`` 不得含 ``"dangerous"``,
    否则显式放行的 Bash 会被黑名单误删。
    """

    @staticmethod
    def get_config() -> RoleConfig:
        return RoleConfig(
            name="tester",
            description="Code testing specialist: writes, runs and analyzes tests",
            system_prompt=TESTER_SYSTEM_PROMPT,
            allowed_tools=["Bash"],
            denied_tools=["Delete"],
            allowed_categories={"readonly", "write"},
            denied_categories=set(),
            max_steps=25,
        )
