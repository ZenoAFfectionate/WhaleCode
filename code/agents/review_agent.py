"""面向 CLI 的 Review 便捷封装 — 处理 git 操作, 委托给 ReviewerRole."""

from __future__ import annotations

import subprocess
from typing import Optional

from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from .roles.reviewer import ReviewerRole, ReviewReport


async def review_working_diff(
    project_root: str,
    llm: HelloAgentsLLM,
    config: Config,
    focus: Optional[str] = None,
) -> ReviewReport:
    """审查 working directory 的未提交变更 (git diff)."""
    diff = get_git_diff(project_root, staged=False)
    return await ReviewerRole.review_diff(
        llm, config, project_root, diff, review_focus=focus
    )


async def review_staged_diff(
    project_root: str,
    llm: HelloAgentsLLM,
    config: Config,
    focus: Optional[str] = None,
) -> ReviewReport:
    """审查已暂存的变更 (git diff --cached)."""
    diff = get_git_diff(project_root, staged=True)
    return await ReviewerRole.review_diff(
        llm, config, project_root, diff, review_focus=focus
    )


def get_git_diff(project_root: str, staged: bool = False) -> str:
    """获取 git diff 内容 (同步, 10s 超时, 异常返回空串)."""
    cmd = ["git", "-C", project_root, "diff"]
    if staged:
        cmd.append("--cached")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout
