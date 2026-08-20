"""Role 抽象基类 — 角色化子 Agent 的配置与工厂.

角色 (Role) 是对子 Agent 的静态定义: 专用 system prompt + 工具访问策略 +
步数预算。通过 ``Role.create_subagent()`` 创建的子 Agent 完全隔离:
独立 Config 深拷贝 / 独立 ToolRegistry / 独立 HistoryManager,
可安全并发执行 (Task 工具即走此路径)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Set

from ...core.config import Config
from ...core.llm import HelloAgentsLLM
from ...tools.registry import ToolRegistry
from ..react_agent import FINISH_TOOL_NAME, THOUGHT_TOOL_NAME

# ReAct 控制工具: 任何角色都必须保留, 否则子 Agent 无法结束 ReAct loop
_CONTROL_TOOL_NAMES = frozenset({THOUGHT_TOOL_NAME, FINISH_TOOL_NAME})

if TYPE_CHECKING:
    from ..code_agent import CodeAgent


@dataclass
class RoleConfig:
    """角色配置 — 每个 Role 子类的静态定义.

    工具过滤规则 (按序求值):
        1. 黑名单优先: ``denied_tools`` / ``denied_categories`` 命中 → 移除
        2. 显式放行: ``allowed_tools`` 命中 → 保留 (可突破 category 限制,
           例如 Reviewer/Tester 显式放行 category="dangerous" 的 Bash)
        3. 类别白名单: ``allowed_categories`` 命中 → 保留
        4. 其余: 只要定义了任一白名单 → 移除
    """

    name: str
    description: str
    system_prompt: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    allowed_categories: Set[str] = field(default_factory=set)
    denied_categories: Set[str] = field(default_factory=set)
    max_steps: int = 15


class Role(ABC):
    """角色抽象基类.

    子类只需实现 :meth:`get_config`。:meth:`create_subagent` 工厂方法
    封装了隔离逻辑与工具策略执行。
    """

    @staticmethod
    @abstractmethod
    def get_config() -> RoleConfig:
        """返回角色的静态配置."""
        raise NotImplementedError

    @classmethod
    def create_subagent(
        cls,
        llm: HelloAgentsLLM,
        parent_config: Config,
        project_root: str,
        working_dir: str,
    ) -> "CodeAgent":
        """创建完全隔离的角色化子 Agent.

        隔离保证:
            1. Config 深拷贝 (model_copy(deep=True)), 关闭 trace/session/skills
            2. 全新 ToolRegistry 实例 (非共享, 并发安全)
            3. register_default_tools=True 注册全部默认工具后, 按策略移除
            4. 角色专用 system_prompt (CodeAgent 自动追加 workspace 信息)
        """
        from ..code_agent import CodeAgent

        role_cfg = cls.get_config()
        config = cls._build_isolated_config(parent_config)
        registry = ToolRegistry(config=config, verbose=False)

        agent = CodeAgent(
            name=f"subagent-{role_cfg.name}",
            llm=llm,
            tool_registry=registry,
            project_root=project_root,
            working_dir=working_dir,
            system_prompt=role_cfg.system_prompt,
            config=config,
            max_steps=role_cfg.max_steps,
            register_default_tools=True,
            enable_task_tool=False,
            enable_subagent_task=False,
            interactive=False,
        )

        cls._enforce_tool_policy(agent.tool_registry)
        return agent

    @classmethod
    def _enforce_tool_policy(cls, registry: ToolRegistry) -> None:
        """根据 RoleConfig 从 registry 中移除不允许的工具.

        使用 ``unregister()`` 而非交换 ``_tools`` dict —— 每个子 Agent 持有
        独立 registry, 移除操作不会波及其他 Agent, 并发安全。
        """
        cfg = cls.get_config()
        has_whitelist = bool(cfg.allowed_tools or cfg.allowed_categories)
        for tool_name in list(registry.list_tools()):
            if tool_name in _CONTROL_TOOL_NAMES:
                continue  # Thought/Finish 是 ReAct 控制面, 不参与工具策略
            tool = registry.get_tool(tool_name)
            if tool is None:
                # function 工具无 category 属性; 默认工具集均为 Tool 对象, 跳过即可
                continue
            # 1. 黑名单优先
            if tool_name in cfg.denied_tools or tool.category in cfg.denied_categories:
                registry.unregister(tool_name)
                continue
            # 2. 显式放行 (可突破 category 限制, 如 category="dangerous" 的 Bash)
            if tool_name in cfg.allowed_tools:
                continue
            # 3. 类别白名单
            if tool.category in cfg.allowed_categories:
                continue
            # 4. 定义了白名单则移除其余
            if has_whitelist:
                registry.unregister(tool_name)

    @classmethod
    def _build_isolated_config(cls, parent_config: Config) -> Config:
        """构建隔离的子 Agent Config (深拷贝 + 关闭无关子系统)."""
        from ..code_agent import _copy_config

        config = _copy_config(parent_config)
        config.trace_enabled = False
        config.session_enabled = False
        config.skills_enabled = False
        config.todowrite_enabled = False
        config.subagent_task_enabled = False  # 防递归双保险: 子代理不得再派生子代理
        return config
