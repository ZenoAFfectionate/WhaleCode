"""角色注册表 — 预定义子 Agent 角色 (Explorer / Reviewer / Tester)."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import Role, RoleConfig
from .explorer import ExplorerRole
from .reviewer import ReviewerRole, ReviewFinding, ReviewReport
from .tester import TesterRole

_ROLE_REGISTRY: Dict[str, Type[Role]] = {
    "explorer": ExplorerRole,
    "reviewer": ReviewerRole,
    "tester": TesterRole,
}


def get_role(name: str) -> Type[Role]:
    """按名称获取角色类 (大小写不敏感); 无效角色抛 ValueError."""
    key = (name or "").strip().lower()
    role = _ROLE_REGISTRY.get(key)
    if role is None:
        raise ValueError(
            f"Unknown role: {name!r}. Available roles: {', '.join(sorted(_ROLE_REGISTRY))}"
        )
    return role


def list_roles() -> List[str]:
    """返回所有已注册角色名."""
    return sorted(_ROLE_REGISTRY)


__all__ = [
    "Role",
    "RoleConfig",
    "ExplorerRole",
    "ReviewerRole",
    "TesterRole",
    "ReviewFinding",
    "ReviewReport",
    "get_role",
    "list_roles",
]
