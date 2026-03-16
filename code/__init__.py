"""
HelloAgents - A flexible, scalable multi-agent framework

Built on the native OpenAI API, providing a simple and efficient agent development experience.
"""

# Configure the log level of third-party libraries to reduce noise
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

# Core components
from .core.llm import HelloAgentsLLM
from .core.config import Config
from .core.message import Message
from .core.exceptions import HelloAgentsException

# Agent implementations
from .agents.simple_agent import SimpleAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.plan_solve_agent import PlanSolveAgent

# Tool system
from .tools.registry import ToolRegistry, global_registry

__all__ = [
    # Core components
    "HelloAgentsLLM",
    "Config",
    "Message",
    "HelloAgentsException",

    # Agent paradigms
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanSolveAgent",

    # Tool system
    "ToolRegistry",
    "global_registry",
]