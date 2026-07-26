import re
import time
import inspect
import asyncio
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, get_type_hints

from .response import ToolResponse
from .errors import ToolErrorCode


def tool_action(name: str = None, description: str = None):
    """Decorator: Mark a method as an expandable tool action

    Usage:
        @tool_action("memory_add", "Add new memory")
        def _add_memory(self, content: str, importance: float = 0.5) -> str:
            '''Add memory

            Args:
                content: Memory content
                importance: Importance score
            '''
            ...

    Args:
        name: Tool name (if not provided, automatically generated from method name)
        description: Tool description (if not provided, extracted from docstring)
    """
    def decorator(func: Callable):
        func._is_tool_action = True
        func._tool_name = name
        func._tool_description = description
        return func
    return decorator


class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """Tool Base Class - New Protocol Version

    Supports two usage modes:
    1. Normal Mode: The tool is used as a single entity
    2. Expandable Mode: The tool can be expanded into multiple independent sub-tools (each corresponding to a function)

    Expandable mode supports two implementation methods:
    - Manually defining sub-tool classes (traditional method)
    - Automatically generating using the @tool_action decorator (recommended)

    New protocol features:
    - The run() method returns a ToolResponse object (instead of a string)
    - Provides run_with_timing() to automatically add time statistics
    - Supports structured status, data, and error information
    """

    def __init__(self, name: str, description: str, expandable: bool = False, category: str = "general"):
        """Initialize the tool

        Args:
            name: Tool name
            description: Tool description
            expandable: Whether it can be expanded into multiple sub-tools
            category: Tool category for filtering (readonly/write/dangerous/interactive/network/general)
        """
        self.name = name
        self.description = description
        self.expandable = expandable
        self.category = category

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute the tool, returning a ToolResponse object

        Use convenient methods to create responses:
        - ToolResponse.success(text="...", data={...})
        - ToolResponse.partial(text="...", data={...})
        - ToolResponse.error(code="NOT_FOUND", message="...")

        Args:
            parameters: Tool parameter dictionary

        Returns:
            ToolResponse: Standardized tool response object
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        pass

    def _augment_timing_response(self, start_time: float, parameters: Dict[str, Any], response: ToolResponse) -> ToolResponse:
        """Add elapsed time, parameter and tool-name metadata to a response."""
        elapsed_ms = int((time.time() - start_time) * 1000)
        if response.stats is None:
            response.stats = {}
        response.stats["time_ms"] = elapsed_ms
        if response.context is None:
            response.context = {}
        response.context["params_input"] = parameters
        response.context["tool_name"] = self.name
        return response

    def _build_timing_error_response(self, start_time: float, parameters: Dict[str, Any], exc: Exception) -> ToolResponse:
        """Build an error response for an unhandled exception during timed execution."""
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ToolResponse.error(
            code=ToolErrorCode.INTERNAL_ERROR,
            message=f"Unhandled exception occurred during tool execution: {str(exc)}",
            stats={"time_ms": elapsed_ms},
            context={"params_input": parameters, "tool_name": self.name},
        )

    def run_with_timing(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute the tool and automatically add time statistics and context information."""
        start_time = time.time()
        try:
            response = self.run(parameters)
        except Exception as e:
            return self._build_timing_error_response(start_time, parameters, e)
        return self._augment_timing_response(start_time, parameters, response)

    async def arun(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Asynchronously execute the tool

        Default implementation: Run the synchronous run() method in a thread pool
        Subclasses can override this method to implement true asynchronous execution

        Args:
            parameters: Tool parameter dictionary

        Returns:
            ToolResponse: Standardized tool response object
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run(parameters)
        )

    async def arun_with_timing(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Asynchronously execute the tool and automatically add time statistics."""
        start_time = time.time()
        try:
            response = await self.arun(parameters)
        except Exception as e:
            return self._build_timing_error_response(start_time, parameters, e)
        return self._augment_timing_response(start_time, parameters, response)

    def get_expanded_tools(self) -> Optional[List['Tool']]:
        """Get the list of expanded sub-tools

        Default implementation: Automatically generate sub-tools from methods marked with @tool_action
        Subclasses can override this method to provide custom expansion logic

        Returns:
            If the tool supports expansion, return the list of sub-tools; otherwise return None
        """
        if not self.expandable:
            return None

        # Automatically generate tools from methods marked by the decorator
        tools = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_is_tool_action'):
                tool = AutoGeneratedTool(
                    parent=self,
                    method=method,
                    name=method._tool_name,
                    description=method._tool_description
                )
                tools.append(tool)

        return tools if tools else None

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters — checks required fields exist"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def _validate_against_schema(self, parameters: Dict[str, Any]) -> Optional[ToolResponse]:
        """Validate parameters against the declared schema from get_parameters().

        Checks required fields and basic type compatibility.  This is a first-pass
        check; tools may still perform deeper validation inside ``run()``.

        Returns None on success, or a ToolResponse error on failure.
        """
        schema = {p.name: p for p in self.get_parameters()}

        # Check required parameters
        for p in self.get_parameters():
            if p.required and p.name not in parameters:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"Missing required parameter: {p.name}",
                )

        # Check types of provided parameters
        for name, value in parameters.items():
            if name not in schema:
                continue  # Allow extra parameters
            expected = schema[name]
            expected_type = expected.type

            if expected_type == "string":
                if not isinstance(value, str):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Invalid parameter `{name}`: expected string, got {type(value).__name__}.",
                    )
            elif expected_type == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Invalid parameter `{name}`: expected integer, got {type(value).__name__}.",
                    )
            elif expected_type == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Invalid parameter `{name}`: expected number, got {type(value).__name__}.",
                    )
            elif expected_type == "boolean":
                if not isinstance(value, bool):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Invalid parameter `{name}`: expected boolean, got {type(value).__name__}.",
                    )
            elif expected_type == "array":
                if not isinstance(value, (list, str)):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Invalid parameter `{name}`: expected array, got {type(value).__name__}.",
                    )
            elif expected_type == "object":
                if not isinstance(value, (dict, str)):
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Invalid parameter `{name}`: expected object, got {type(value).__name__}.",
                    )

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                param.model_dump() if hasattr(param, "model_dump") else param.dict()
                for param in self.get_parameters()
            ]
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()


class AutoGeneratedTool(Tool):
    """Automatically generated tool - automatically extracts parameters from method signature and docstring"""

    def __init__(self, parent: Tool, method: Callable, name: str = None, description: str = None):
        """Initialize automatically generated tool

        Args:
            parent: Parent tool instance
            method: Decorated method
            name: Tool name (if None, generate from method name)
            description: Tool description (if None, extract from docstring)
        """
        self.parent = parent
        self.method = method

        # Generate tool name
        if name is None:
            # Generate from method name: _add_memory -> parent_name_add_memory
            method_name = method.__name__.lstrip('_')
            name = f"{parent.name}_{method_name}"

        # Extract description
        if description is None:
            description = self._extract_description_from_docstring()

        super().__init__(name=name, description=description)

        # Automatically parse parameters
        self._parameters = self._parse_parameters()

    def _extract_description_from_docstring(self) -> str:
        """Extract description from docstring"""
        doc = inspect.getdoc(self.method)
        if not doc:
            return f"Execute {self.method.__name__}"

        # Extract the first line as description
        lines = doc.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Args:') and not line.startswith('Returns:'):
                return line

        return f"Execute {self.method.__name__}"

    def _parse_parameters(self) -> List[ToolParameter]:
        """Automatically extract parameters from method signature and docstring"""
        sig = inspect.signature(self.method)
        type_hints = get_type_hints(self.method)
        docstring = inspect.getdoc(self.method) or ""

        # Parse parameter descriptions from docstring
        param_descriptions = self._parse_param_descriptions(docstring)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Get type
            param_type_hint = type_hints.get(param_name, str)
            param_type = self._python_type_to_tool_type(param_type_hint)

            # Determine if required
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            # Get description
            description = param_descriptions.get(param_name, f"Parameter {param_name}")

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=description,
                required=required,
                default=default
            ))

        return parameters

    def _parse_param_descriptions(self, docstring: str) -> Dict[str, str]:
        """Parse parameter descriptions from docstring

        Supported format:
            Args:
                param_name: Parameter description
                another_param: Another parameter description
        """
        descriptions = {}

        # Find Args: section
        args_match = re.search(r'Args:\s*\n(.*?)(?:\n\s*\n|Returns:|$)', docstring, re.DOTALL)
        if not args_match:
            return descriptions

        args_section = args_match.group(1)

        # Parse each parameter
        # Match format: param_name: description OR param_name (type): description
        param_pattern = r'^\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=^\s*\w+\s*(?:\([^)]+\))?\s*:|$)'
        matches = re.finditer(param_pattern, args_section, re.MULTILINE | re.DOTALL)

        for match in matches:
            param_name = match.group(1).strip()
            param_desc = match.group(2).strip()
            # Clean up extra whitespace in description
            param_desc = re.sub(r'\s+', ' ', param_desc)
            descriptions[param_name] = param_desc

        return descriptions

    def _python_type_to_tool_type(self, py_type) -> str:
        """Convert Python type to tool type string"""
        # Handle generic types
        origin = getattr(py_type, '__origin__', None)
        if origin is not None:
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"

        # Handle basic types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        return type_map.get(py_type, "string")

    def get_parameters(self) -> List[ToolParameter]:
        """Get parameter list"""
        return self._parameters

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute method, return ToolResponse object

        If the decorated method returns a string, automatically wrap it in a success response
        If it returns a ToolResponse, return directly
        """
        try:
            result = self.method(**parameters)

            # If the method already returns a ToolResponse, return directly
            if isinstance(result, ToolResponse):
                return result

            # Otherwise wrap it in a success response
            return ToolResponse.success(
                text=str(result),
                data={"output": result}
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Method execution failed: {str(e)}"
            )
