"""Tool Response Protocol

Standardized tool response format, providing structured status, data, and error information.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import json


class ToolStatus(Enum):
    """Tool execution status enumeration"""
    SUCCESS = "success"  # Task executed completely as expected
    PARTIAL = "partial"  # Result available but compromised (truncated, fallback, partial failure)
    ERROR = "error"      # No valid result (fatal error)


@dataclass
class ToolResponse:
    """Tool response data class

    Standardized tool response format, containing:
    - status: Execution status (success/partial/error)
    - text: Formatted text for LLM to read
    - data: Structured data payload
    - error_info: Error information (only when status=error)
    - stats: Runtime statistics (time, tokens, etc.)
    - context: Context information (parameters, environment, etc.)

    Example:
        >>> # Success response
        >>> resp = ToolResponse.success(
        ...     text="Calculation result: 42",
        ...     data={"result": 42, "expression": "6*7"}
        ... )

        >>> # Error response
        >>> resp = ToolResponse.error(
        ...     code="INVALID_PARAM",
        ...     message="Expression cannot be empty"
        ... )
    """

    status: ToolStatus
    text: str
    data: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, str]] = None
    stats: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        result = {
            "status": self.status.value,
            "text": self.text,
            "data": self.data,
        }
        if self.error_info:
            result["error"] = self.error_info
        if self.stats:
            result["stats"] = self.stats
        if self.context:
            result["context"] = self.context
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResponse':
        """Create ToolResponse from dictionary"""
        status_str = data.get("status", "success")
        status = ToolStatus(status_str)

        return cls(
            status=status,
            text=data.get("text", ""),
            data=data.get("data", {}),
            error_info=data.get("error"),
            stats=data.get("stats"),
            context=data.get("context")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ToolResponse':
        """Create ToolResponse from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def success(
        cls,
        text: str,
        data: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> 'ToolResponse':
        """Quickly create a success response
        
        Args:
            text: Text for LLM to read
            data: Structured data
            stats: Runtime statistics
            context: Context information
        """
        return cls(
            status=ToolStatus.SUCCESS,
            text=text,
            data=data or {},
            stats=stats,
            context=context
        )
    
    @classmethod
    def partial(
        cls,
        text: str,
        data: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> 'ToolResponse':
        """Quickly create a partial success response
        
        Args:
            text: Text for LLM to read (should explain the reason for partial success)
            data: Structured data
            stats: Runtime statistics
            context: Context information
        """
        return cls(
            status=ToolStatus.PARTIAL,
            text=text,
            data=data or {},
            stats=stats,
            context=context
        )
    
    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        stats: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> 'ToolResponse':
        """Quickly create an error response

        Args:
            code: Error code (from ToolErrorCode)
            message: Error message
            stats: Runtime statistics
            context: Context information
        """
        return cls(
            status=ToolStatus.ERROR,
            text=message,
            data={},
            error_info={"code": code, "message": message},
            stats=stats,
            context=context
        )