"""Agent factory functions

Used to create different types of Agent instances, supporting sub-agent mechanism.
"""

from typing import Optional, TYPE_CHECKING
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


def create_agent(
    agent_type: str,
    name: str,
    llm: HelloAgentsLLM,
    tool_registry: Optional['ToolRegistry'] = None,
    config: Optional[Config] = None,
    system_prompt: Optional[str] = None
) -> Agent:
    """Create an Agent instance

    Args:
        agent_type: Agent type, supports:
            - "react": ReActAgent (Reasoning-Action loop)
            - "code": CodeAgent (repository-aware coding agent)
        name: Agent name
        llm: LLM instance
        tool_registry: Tool registry (optional)
        config: Configuration object (optional)
        system_prompt: System prompt (optional)

    Returns:
        Agent instance

    Raises:
        ValueError: Unsupported agent_type
    """
    agent_type = agent_type.lower()

    if agent_type == "react":
        from .react_agent import ReActAgent
        return ReActAgent(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            config=config,
            system_prompt=system_prompt
        )

    elif agent_type == "code":
        # 建议-8: expose the flagship CodeAgent through the factory too.
        from .code_agent import CodeAgent
        return CodeAgent(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            config=config,
            system_prompt=system_prompt,
        )

    else:
        raise ValueError(
            f"Unsupported agent_type: {agent_type}. "
            f"Supported types: react, code"
        )
