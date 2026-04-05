"""Tool Registry - HelloAgents Native Tool System"""

import asyncio
import json
import time
from typing import Optional, Any, Callable, Dict
from .base import Tool
from .response import ToolResponse
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

    @staticmethod
    def _default_circuit_breaker(config: Optional[Any] = None) -> CircuitBreaker:
        if config is None:
            return CircuitBreaker()
        return CircuitBreaker(
            failure_threshold=int(getattr(config, "circuit_failure_threshold", 3) or 3),
            recovery_timeout=int(getattr(config, "circuit_recovery_timeout", 300) or 300),
            enabled=bool(getattr(config, "circuit_enabled", True)),
        )

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        *,
        config: Optional[Any] = None,
        verbose: bool = True,
    ):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}
        self.verbose = verbose

        # File metadata cache (used for optimistic locking mechanism)
        self.read_metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Circuit breaker (enabled by default)
        self.circuit_breaker = circuit_breaker or self._default_circuit_breaker(config)

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

    def _circuit_open_response(self, name: str) -> ToolResponse:
        status = self.circuit_breaker.get_status(name)
        return ToolResponse.error(
            code=ToolErrorCode.CIRCUIT_OPEN,
            message=(
                f"Tool '{name}' is currently disabled due to consecutive failures. "
                f"Available in {status['recover_in_seconds']} seconds."
            ),
            context={"tool_name": name, "circuit_status": status},
        )

    @staticmethod
    def _normalize_tool_parameters(input_payload: Any) -> Dict[str, Any]:
        if isinstance(input_payload, dict):
            return input_payload
        if isinstance(input_payload, str):
            try:
                parsed = json.loads(input_payload)
            except json.JSONDecodeError:
                return {"input": input_payload}
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        return {"input": str(input_payload)}

    @staticmethod
    def _tool_exception_response(name: str, input_payload: Any, exc: Exception) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=f"Exception occurred while executing tool '{name}': {str(exc)}",
            context={"tool_name": name, "input": input_payload},
        )

    @staticmethod
    def _function_success_response(name: str, input_payload: Any, result: Any, elapsed_ms: int) -> ToolResponse:
        return ToolResponse.success(
            text=str(result),
            data={"output": result},
            stats={"time_ms": elapsed_ms},
            context={"tool_name": name, "input": input_payload},
        )

    @staticmethod
    def _function_error_response(name: str, input_payload: Any, exc: Exception, elapsed_ms: int) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=f"Function execution failed: {str(exc)}",
            stats={"time_ms": elapsed_ms},
            context={"tool_name": name, "input": input_payload},
        )

    def _execute_function_sync(self, name: str, input_payload: Any) -> ToolResponse:
        func = self._functions[name]["func"]
        start_time = time.time()
        try:
            result = func(input_payload)
            elapsed_ms = int((time.time() - start_time) * 1000)
            return self._function_success_response(name, input_payload, result, elapsed_ms)
        except Exception as exc:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return self._function_error_response(name, input_payload, exc, elapsed_ms)

    async def _execute_function_async(self, name: str, input_payload: Any) -> ToolResponse:
        func = self._functions[name]["func"]
        start_time = time.time()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: func(input_payload))
            elapsed_ms = int((time.time() - start_time) * 1000)
            return self._function_success_response(name, input_payload, result, elapsed_ms)
        except Exception as exc:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return self._function_error_response(name, input_payload, exc, elapsed_ms)

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
            return self._circuit_open_response(name)

        # Execute tool
        # Prioritize finding Tool object (new protocol)
        if name in self._tools:
            tool = self._tools[name]
            try:
                parameters = self._normalize_tool_parameters(input_text)
                response = tool.run_with_timing(parameters)
            except Exception as exc:
                response = self._tool_exception_response(name, input_text, exc)

        # Find function tool (automatically wrap to new protocol)
        elif name in self._functions:
            response = self._execute_function_sync(name, input_text)

        # Tool does not exist
        else:
            response = ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"Tool named '{name}' not found",
                context={"tool_name": name},
            )

        # Record circuit breaker result
        self.circuit_breaker.record_result(name, response)

        return response

    async def aexecute_tool(self, name: str, input_text: Any) -> ToolResponse:
        """Async version of execute_tool with the same dispatch and breaker behavior."""
        if self.circuit_breaker.is_open(name):
            return self._circuit_open_response(name)

        if name in self._tools:
            tool = self._tools[name]
            try:
                parameters = self._normalize_tool_parameters(input_text)
                response = await tool.arun_with_timing(parameters)
            except Exception as exc:
                response = self._tool_exception_response(name, input_text, exc)
        elif name in self._functions:
            response = await self._execute_function_async(name, input_text)
        else:
            response = ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"Tool named '{name}' not found",
                context={"tool_name": name},
            )

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
