"""Tool Filter

Used for the sub-agent mechanism to control which tools different types of Agents can access.

Supports two filtering modes (evaluated in order):
1. Category-based (primary): tools declare a ``category`` attribute; filters match on
   ``READONLY_CATEGORIES`` / ``DENIED_CATEGORIES``.
2. Name-based (fallback): explicit tool-name allow/deny sets for backward compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set


class ToolFilter(ABC):
    """Tool Filter Base Class

    Used to restrict the set of available tools when a sub-agent is running.
    """

    def __init__(self, tool_categories: Optional[Dict[str, str]] = None):
        """Initialize the filter.

        Args:
            tool_categories: Optional name→category mapping from ToolRegistry.
                             When provided, filters use category-based matching as
                             the primary mechanism.
        """
        self._tool_categories = tool_categories or {}

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

    def _category(self, tool_name: str) -> str:
        """Return the category of a tool, or 'general' if unknown."""
        return self._tool_categories.get(tool_name, "general")


class ReadOnlyFilter(ToolFilter):
    """Read-Only Tool Filter

    Allows only tools whose category is ``"readonly"`` (plus any additional
    explicitly allowed names).  Suitable for explore / plan / summary sub-agents.
    """

    # Primary: categories allowed in read-only mode.
    READONLY_CATEGORIES: Set[str] = {"readonly"}

    # Fallback: explicit tool names allowed (backward-compatible).
    READONLY_TOOLS: Set[str] = {
        "Read", "ReadTool",
        "LS", "LSTool",
        "Glob", "GlobTool",
        "Grep", "GrepTool",
        "Skill", "SkillTool",
    }

    def __init__(
        self,
        additional_allowed: Optional[List[str]] = None,
        tool_categories: Optional[Dict[str, str]] = None,
    ):
        """Initialize the read-only filter

        Args:
            additional_allowed: List of additionally allowed tool names.
            tool_categories: Optional name→category mapping.
        """
        super().__init__(tool_categories=tool_categories)
        self.allowed_tools = self.READONLY_TOOLS.copy()
        if additional_allowed:
            self.allowed_tools.update(additional_allowed)

    def filter(self, all_tools: List[str]) -> List[str]:
        """Keep only readable tools"""
        return [tool for tool in all_tools if self.is_allowed(tool)]

    def is_allowed(self, tool_name: str) -> bool:
        """Check if it is a read-only tool (category first, then name fallback)."""
        if tool_name in self.allowed_tools:
            return True
        cat = self._category(tool_name)
        return cat in self.READONLY_CATEGORIES


class FullAccessFilter(ToolFilter):
    """Full Access Filter

    Allows all tools *except* those whose category is ``"dangerous"`` (plus any
    additional explicitly denied names).  Suitable for the code sub-agent.
    """

    # Primary: categories denied (even with full access).
    DENIED_CATEGORIES: Set[str] = {"dangerous"}

    # Fallback: explicit tool names denied (backward-compatible).
    DENIED_TOOLS: Set[str] = {
        "Bash", "BashTool",
        "Terminal", "TerminalTool",
        "Execute", "ExecuteTool",
    }

    def __init__(
        self,
        additional_denied: Optional[List[str]] = None,
        tool_categories: Optional[Dict[str, str]] = None,
    ):
        """Initialize the full access filter

        Args:
            additional_denied: List of additionally denied tool names.
            tool_categories: Optional name→category mapping.
        """
        super().__init__(tool_categories=tool_categories)
        self.denied_tools = self.DENIED_TOOLS.copy()
        if additional_denied:
            self.denied_tools.update(additional_denied)

    def filter(self, all_tools: List[str]) -> List[str]:
        """Exclude dangerous tools"""
        return [tool for tool in all_tools if self.is_allowed(tool)]

    def is_allowed(self, tool_name: str) -> bool:
        """Check if allowed (category first, then name fallback)."""
        if tool_name in self.denied_tools:
            return False
        cat = self._category(tool_name)
        if cat in self.DENIED_CATEGORIES:
            return False
        return True


class CustomFilter(ToolFilter):
    """Custom Tool Filter

    Users can explicitly specify a list of allowed or denied tools,
    with optional category-based filtering as well.
    """

    def __init__(
        self,
        allowed: Optional[List[str]] = None,
        denied: Optional[List[str]] = None,
        mode: str = "whitelist",
        tool_categories: Optional[Dict[str, str]] = None,
        allowed_categories: Optional[Set[str]] = None,
        denied_categories: Optional[Set[str]] = None,
    ):
        """Initialize the custom filter

        Args:
            allowed: List of allowed tool names (whitelist mode).
            denied: List of denied tool names (blacklist mode).
            mode: Filter mode, "whitelist" or "blacklist".
            tool_categories: Optional name→category mapping.
            allowed_categories: Optional set of categories to allow (whitelist mode).
            denied_categories: Optional set of categories to deny (blacklist mode).
        """
        super().__init__(tool_categories=tool_categories)
        self.allowed = set(allowed) if allowed else set()
        self.denied = set(denied) if denied else set()
        self.mode = mode
        self.allowed_categories = allowed_categories or set()
        self.denied_categories = denied_categories or set()

        if mode not in ("whitelist", "blacklist"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'whitelist' or 'blacklist'")

    def filter(self, all_tools: List[str]) -> List[str]:
        """Filter tools based on the mode"""
        return [tool for tool in all_tools if self.is_allowed(tool)]

    def is_allowed(self, tool_name: str) -> bool:
        """Check if allowed (category-aware)."""
        cat = self._category(tool_name)
        if self.mode == "whitelist":
            if tool_name in self.allowed:
                return True
            if cat in self.allowed_categories:
                return True
            if tool_name in self.denied:
                return False
            if cat in self.denied_categories:
                return False
            return not self.allowed and not self.allowed_categories
        else:  # blacklist
            if tool_name in self.denied:
                return False
            if cat in self.denied_categories:
                return False
            return True
