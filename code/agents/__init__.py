from .simple_agent import SimpleAgent
from .react_agent import ReActAgent
from .reflection_agent import ReflectionAgent
from .plan_solve_agent import PlanSolveAgent
from .code_agent import CodeAgent

from .factory import create_agent, create_orchestra, default_subagent_factory

from .orchestra import (
    AgentOrchestra,
    ExecutionMode,
    ExecutionPlan,
    SubAgentResult,
    SubTask,
    SubtaskHooks,
)
from .review_agent import review_staged_diff, review_working_diff
from .roles import (
    ExplorerRole,
    ReviewerRole,
    ReviewFinding,
    ReviewReport,
    Role,
    RoleConfig,
    TesterRole,
    get_role,
    list_roles,
)


__all__ = [
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanSolveAgent",

    "CodeAgent",  # a specialized agent for Coding

    "create_agent",
    "create_orchestra",
    "default_subagent_factory",

    # Multi-agent orchestra
    "AgentOrchestra",
    "ExecutionMode",
    "ExecutionPlan",
    "SubAgentResult",
    "SubTask",
    "SubtaskHooks",

    # Roles
    "Role",
    "RoleConfig",
    "ExplorerRole",
    "ReviewerRole",
    "TesterRole",
    "get_role",
    "list_roles",

    # Code review
    "ReviewFinding",
    "ReviewReport",
    "review_staged_diff",
    "review_working_diff",
]
