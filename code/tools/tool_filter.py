"""Tool Filter

Used for the sub-agent mechanism to control which tools different types of Agents can access.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional


class ToolFilter(ABC):
    """Tool Filter Base Class
    
    Used to restrict the set of available tools when a sub-agent is running.
    """
    
    @abstractmethod
    def filter(self, all_tools: List[str]) -> List[str]:
        """Filter the list of tools
        
        Args:
            all_tools: List of all available tool names
            
        Returns:
            Filtered list of tool names
        """
        pass
    
    @abstractmethod
    def is_allowed(self, tool_name: str) -> bool:
        """Check if a single tool is allowed
        
        Args:
            tool_name: Tool name
            
        Returns:
            Whether the tool is allowed to be used
        """
        pass


class ReadOnlyFilter(ToolFilter):
    """Read-Only Tool Filter
    
    Only allows the use of read-only tools, suitable for:
    - explore (exploring the codebase)
    - plan (planning tasks)
    - summary (summarizing information)
    """
    
    # Read-only tool whitelist
    READONLY_TOOLS: Set[str] = {
        "Read", "ReadTool",
        "LS", "LSTool",
        "Glob", "GlobTool",
        "Grep", "GrepTool",
        "Skill", "SkillTool",
    }
    
    def __init__(self, additional_allowed: Optional[List[str]] = None):
        """Initialize the read-only filter
        
        Args:
            additional_allowed: List of additionally allowed tool names
        """
        self.allowed_tools = self.READONLY_TOOLS.copy()
        if additional_allowed:
            self.allowed_tools.update(additional_allowed)
    
    def filter(self, all_tools: List[str]) -> List[str]:
        """Keep only read-only tools"""
        return [tool for tool in all_tools if self.is_allowed(tool)]
    
    def is_allowed(self, tool_name: str) -> bool:
        """Check if it is a read-only tool"""
        return tool_name in self.allowed_tools


class FullAccessFilter(ToolFilter):
    """Full Access Filter
    
    Allows the use of all tools (except explicitly denied dangerous tools), suitable for:
    - code (code implementation)
    """
    
    # Dangerous tool blacklist
    DENIED_TOOLS: Set[str] = {
        "Bash", "BashTool",
        "Terminal", "TerminalTool",
        "Execute", "ExecuteTool",
    }
    
    def __init__(self, additional_denied: Optional[List[str]] = None):
        """Initialize the full access filter
        
        Args:
            additional_denied: List of additionally denied tool names
        """
        self.denied_tools = self.DENIED_TOOLS.copy()
        if additional_denied:
            self.denied_tools.update(additional_denied)
    
    def filter(self, all_tools: List[str]) -> List[str]:
        """Exclude dangerous tools"""
        return [tool for tool in all_tools if self.is_allowed(tool)]
    
    def is_allowed(self, tool_name: str) -> bool:
        """Check if allowed (not in the blacklist)"""
        return tool_name not in self.denied_tools


class CustomFilter(ToolFilter):
    """Custom Tool Filter
    
    Users can explicitly specify a list of allowed or denied tools.
    """
    
    def __init__(
        self,
        allowed: Optional[List[str]] = None,
        denied: Optional[List[str]] = None,
        mode: str = "whitelist"
    ):
        """Initialize the custom filter
        
        Args:
            allowed: List of allowed tool names (whitelist mode)
            denied: List of denied tool names (blacklist mode)
            mode: Filter mode, "whitelist" or "blacklist"
        """
        self.allowed = set(allowed) if allowed else set()
        self.denied = set(denied) if denied else set()
        self.mode = mode
        
        if mode not in ("whitelist", "blacklist"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'whitelist' or 'blacklist'")
    
    def filter(self, all_tools: List[str]) -> List[str]:
        """Filter tools based on the mode"""
        return [tool for tool in all_tools if self.is_allowed(tool)]
    
    def is_allowed(self, tool_name: str) -> bool:
        """Check if allowed"""
        if self.mode == "whitelist":
            return tool_name in self.allowed
        else:  # blacklist
            return tool_name not in self.denied