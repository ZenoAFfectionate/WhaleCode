"""Tool Registry - HelloAgents Native Tool System"""

from typing import Optional, Any, Callable, Dict
import time
from .base import Tool
from .response import ToolResponse, ToolStatus
from .errors import ToolErrorCode
from .circuit_breaker import CircuitBreaker


class ToolRegistry:
    """
    HelloAgents Tool Registry

    Provides tool registration, management, and execution functionalities.
    Supports two tool registration methods:
    1. Tool object registration (Recommended)
    2. Direct function registration (Simple)
    """

    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None, verbose: bool = True):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}
        self.verbose = verbose

        # File metadata cache (used for optimistic locking mechanism)
        self.read_metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Circuit breaker (enabled by default)
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    def _emit(self, message: str) -> None:
        if self.verbose:
            print(message)

    def register_tool(self, tool: Tool, auto_expand: bool = True):
        """
        Register a Tool object

        Args:
            tool: Tool instance
            auto_expand: Whether to automatically expand expandable tools (default is True)
        """
        # Check if the tool is expandable
        if auto_expand and hasattr(tool, 'expandable') and tool.expandable:
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                # Register all expanded sub-tools
                for sub_tool in expanded_tools:
                    if sub_tool.name in self._tools:
                        self._emit(f"⚠️ Warning: Tool '{sub_tool.name}' already exists and will be overwritten.")
                    self._tools[sub_tool.name] = sub_tool
                self._emit(f"✅ Tool '{tool.name}' has been expanded into {len(expanded_tools)} independent tools.")
                return

        # Normal tool or non-expanded tool
        if tool.name in self._tools:
            self._emit(f"⚠️ Warning: Tool '{tool.name}' already exists and will be overwritten.")

        self._tools[tool.name] = tool
        self._emit(f"✅ Tool '{tool.name}' has been registered.")

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Directly register a function as a tool (simple method)

        Supports two calling methods:
        1. Traditional method: register_function(name, description, func)
        2. New method: register_function(func, name=None, description=None)
           - Automatically extracts info from function name and docstring

        Args:
            func: Tool function
            name: Tool name (optional, defaults to function name)
            description: Tool description (optional, defaults to function docstring)

        Usage Example:
            >>> def my_tool(input: str) -> str:
            ...     '''This is my tool'''
            ...     return f"Processing: {input}"
            >>> registry.register_function(my_tool)
            >>> # Or specify name and description
            >>> registry.register_function(my_tool, name="custom_name", description="Custom description")
        """
        # Compatible with old calling method: register_function(name, description, func)
        if isinstance(func, str) and callable(description):
            # Old method: first parameter is name, second is description, third is func
            name, description, func = func, name, description

        # Automatically extract name
        if name is None:
            name = func.__name__

        # Automatically extract description
        if description is None:
            import inspect
            doc = inspect.getdoc(func)
            if doc:
                # Extract the first line as description
                description = doc.split('\n')[0].strip()
            else:
                description = f"Execute {name}"

        if name in self._functions:
            self._emit(f"⚠️ Warning: Tool '{name}' already exists and will be overwritten.")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        self._emit(f"✅ Function tool '{name}' has been registered.")

    def unregister(self, name: str):
        """Unregister tool"""
        if name in self._tools:
            del self._tools[name]
            self._emit(f"🗑️ Tool '{name}' has been unregistered.")
        elif name in self._functions:
            del self._functions[name]
            self._emit(f"🗑️ Tool '{name}' has been unregistered.")
        else:
            self._emit(f"⚠️ Tool '{name}' does not exist.")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get Tool object"""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        """Get tool function"""
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> ToolResponse:
        """
        Execute tool, returning a ToolResponse object (with circuit breaker protection)

        Args:
            name: Tool name
            input_text: Input parameters

        Returns:
            ToolResponse: Standardized tool response object
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open(name):
            status = self.circuit_breaker.get_status(name)
            return ToolResponse.error(
                code=ToolErrorCode.CIRCUIT_OPEN,
                message=f"Tool '{name}' is currently disabled due to consecutive failures. Available in {status['recover_in_seconds']} seconds.",
                context={
                    "tool_name": name,
                    "circuit_status": status
                }
            )

        # Execute tool
        response = None

        # Prioritize finding Tool object (new protocol)
        if name in self._tools:
            tool = self._tools[name]
            try:
                # Parse parameters (supports JSON string or dictionary)
                import json
                if isinstance(input_text, str):
                    try:
                        parameters = json.loads(input_text)
                    except json.JSONDecodeError:
                        # If not JSON, treat as a normal string
                        parameters = {"input": input_text}
                elif isinstance(input_text, dict):
                    parameters = input_text
                else:
                    parameters = {"input": str(input_text)}

                # Use run_with_timing to automatically add time statistics
                response = tool.run_with_timing(parameters)
            except Exception as e:
                response = ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"Exception occurred while executing tool '{name}': {str(e)}",
                    context={"tool_name": name, "input": input_text}
                )

        # Find function tool (automatically wrap to new protocol)
        elif name in self._functions:
            func = self._functions[name]["func"]
            start_time = time.time()

            try:
                result = func(input_text)
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Wrap as ToolResponse
                response = ToolResponse.success(
                    text=str(result),
                    data={"output": result},
                    stats={"time_ms": elapsed_ms},
                    context={"tool_name": name, "input": input_text}
                )
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                response = ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"Function execution failed: {str(e)}",
                    stats={"time_ms": elapsed_ms},
                    context={"tool_name": name, "input": input_text}
                )

        # Tool does not exist
        else:
            response = ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"Tool named '{name}' not found",
                context={"tool_name": name}
            )

        # Record circuit breaker result
        self.circuit_breaker.record_result(name, response)

        return response

    def get_tools_description(self) -> str:
        """
        Get formatted description string of all available tools

        Returns:
            Tool description string, used for building prompts
        """
        descriptions = []

        # Tool object descriptions
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        # Function tool descriptions
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "No available tools at the moment"

    def list_tools(self) -> list[str]:
        """List all tool names"""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """Get all Tool objects"""
        return list(self._tools.values())

    def clear(self):
        """Clear all tools"""
        self._tools.clear()
        self._functions.clear()
        print("🧹 All tools have been cleared.")

    # ==================== Optimistic Locking Mechanism Support ====================

    def cache_read_metadata(self, file_path: str, metadata: Dict[str, Any]):
        """Cache file metadata obtained by the Read tool

        Args:
            file_path: File path (relative to project_root)
            metadata: File metadata dictionary, containing:
                - file_mtime_ms: File modification time (millisecond timestamp)
                - file_size_bytes: File size (bytes)
        """
        self.read_metadata_cache[file_path] = metadata

    def get_read_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached file metadata

        Args:
            file_path: File path

        Returns:
            File metadata dictionary, or None if it does not exist
        """
        return self.read_metadata_cache.get(file_path)

    def clear_read_cache(self, file_path: Optional[str] = None):
        """Clear file metadata cache

        Args:
            file_path: Specific file path, if None then clear all cache
        """
        if file_path:
            self.read_metadata_cache.pop(file_path, None)
        else:
            self.read_metadata_cache.clear()

# Global tool registry
global_registry = ToolRegistry()