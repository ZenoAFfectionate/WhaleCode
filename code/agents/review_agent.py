"""面向 CLI 的 Review 便捷封装 — 处理 git 操作, 委托给 ReviewerRole."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..tools.builtin.git_tools import _GitFailure, _run_git
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
    """获取 git diff 内容 (异常返回空串).

    Q2-9: 复用 git_tools._run_git 的沙箱化封装（脱敏 env、
    GIT_TERMINAL_PROMPT=0 防凭证阻塞、统一超时），替代原先的裸
    subprocess.run——后者绕过了 BashTool 的环境一致性控制。
    """
    root = Path(project_root)
    args = ["diff"]
    if staged:
        args.append("--cached")
    try:
        returncode, stdout, _stderr = _run_git(args, cwd=root, project_root=root)
    except (_GitFailure, OSError):
        return ""
    if returncode != 0:
        return ""
    return stdout
