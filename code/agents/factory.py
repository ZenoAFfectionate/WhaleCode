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
            - "reflection": ReflectionAgent (Reflective type)
            - "plan": PlanAndSolveAgent (Planning-Execution)
            - "simple": SimpleAgent (Simple conversation)
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
    
    elif agent_type == "reflection":
        from .reflection_agent import ReflectionAgent
        return ReflectionAgent(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            config=config,
            system_prompt=system_prompt
        )
    
    elif agent_type == "plan":
        from .plan_solve_agent import PlanSolveAgent
        return PlanSolveAgent(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            config=config,
            system_prompt=system_prompt
        )
    
    elif agent_type == "simple":
        from .simple_agent import SimpleAgent
        return SimpleAgent(
            name=name,
            llm=llm,
            config=config,
            system_prompt=system_prompt
        )
    
    else:
        raise ValueError(
            f"Unsupported agent_type: {agent_type}. "
            f"Supported types: react, reflection, plan, simple"
        )


def default_subagent_factory(
    agent_type: str,
    llm: HelloAgentsLLM,
    tool_registry: Optional['ToolRegistry'] = None,
    config: Optional[Config] = None
) -> Agent:
    """Default sub-agent factory function
    
    Default implementation provided by the framework, which users can customize and replace.
    
    Args:
        agent_type: Agent type
        llm: LLM instance
        tool_registry: Tool registry (optional)
        config: Configuration object (optional)
        
    Returns:
        Configured sub-agent instance
    """
    config = config or Config()
    
    # Sub-agent name
    name = f"subagent-{agent_type}"
    
    # Select system prompt based on type
    system_prompt = _get_system_prompt_for_type(agent_type)
    
    # Create sub-agent
    subagent = create_agent(
        agent_type=agent_type,
        name=name,
        llm=llm,
        tool_registry=tool_registry,
        config=config,
        system_prompt=system_prompt
    )
    
    # Configure sub-agent specific parameters
    if hasattr(subagent, 'max_steps'):
        subagent.max_steps = config.subagent_max_steps
    
    return subagent


def _get_system_prompt_for_type(agent_type: str) -> str:
    """Get type-specific system prompt
    
    Args:
        agent_type: Agent type
        
    Returns:
        System prompt
    """
    prompts = {
        "react": """You are an efficient task execution expert.

Goal: Quickly complete the specified subtask.

Rules:
- Use available tools to complete tasks efficiently
- Keep output concise and clear
- Complete within the given step limit
""",
        "reflection": """You are a reflective expert.

Goal: Deeply analyze problems and provide high-quality solutions.

Rules:
- First give an initial solution
- Reflect and improve the solution
- Output the final optimized result
""",
        "plan": """You are a task planning expert.

Goal: Break down complex tasks into executable steps.

Rules:
- Analyze task requirements
- Develop a detailed execution plan
- Mark step dependencies
""",
        "simple": """You are a concise and efficient assistant.

Goal: Directly answer questions or complete tasks.

Rules:
- Keep answers concise
- Provide results directly
- Avoid redundant information
"""
    }
    
    return prompts.get(agent_type.lower(), prompts["simple"])