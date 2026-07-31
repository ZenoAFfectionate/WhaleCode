"""Explorer 角色 — 代码探索专家 (只读工具).

系统提示词集中维护于 ``code/prompts/roles/explorer.md``。
"""

from __future__ import annotations

from pathlib import Path

from .base import Role, RoleConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPLORER_SYSTEM_PROMPT: str = (
    _PROJECT_ROOT / "code" / "prompts" / "roles" / "explorer.md"
).read_text(encoding="utf-8")


class ExplorerRole(Role):
    """代码探索专家 — 只读工具 (Read/Glob/Grep/LS/LSP), 禁止一切写入与 Bash."""

    @staticmethod
    def get_config() -> RoleConfig:
        return RoleConfig(
            name="explorer",
            description="Code exploration and architecture analysis specialist",
            system_prompt=EXPLORER_SYSTEM_PROMPT,
            allowed_categories={"readonly"},
            denied_tools=["Write", "Edit", "Delete", "Bash"],
            denied_categories={"write", "dangerous"},
            max_steps=20,
        )
