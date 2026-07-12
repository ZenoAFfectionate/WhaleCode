"""Tool Error Code Definitions

Standardized tool error codes for unified error handling and tracking.

建议-1: this is now a real ``str``-based ``Enum`` (rather than a bag of string
constants). Members remain drop-in string-compatible — ``ToolErrorCode.INVALID_PARAM
== "INVALID_PARAM"``, ``str(...)`` / f-strings render the plain code, and JSON
serialization emits the string value — so existing comparisons and the circuit
breaker's ``code in FAULT_ERROR_CODES`` check keep working unchanged.
"""

from enum import Enum


class ToolErrorCode(str, Enum):
    """Tool Error Code Enumeration

    Facilitates:
    - Unified error handling at the Agent layer
    - Circuit breaker mechanism identifying failure types
    - Observability system tracking errors
    - User-friendly error messages
    """

    # Resource-related errors
    NOT_FOUND = "NOT_FOUND"                    # Resource does not exist (file, tool, etc.)
    ACCESS_DENIED = "ACCESS_DENIED"            # Access denied
    PERMISSION_DENIED = "PERMISSION_DENIED"    # Insufficient permissions
    IS_DIRECTORY = "IS_DIRECTORY"              # Expected file but got directory
    BINARY_FILE = "BINARY_FILE"                # Binary file cannot be processed

    # Parameter-related errors
    INVALID_PARAM = "INVALID_PARAM"            # Invalid or missing parameters
    INVALID_FORMAT = "INVALID_FORMAT"          # Format error

    # Execution-related errors
    EXECUTION_ERROR = "EXECUTION_ERROR"        # Error occurred during execution
    TIMEOUT = "TIMEOUT"                        # Execution timeout
    INTERNAL_ERROR = "INTERNAL_ERROR"          # Internal error

    # Status-related errors
    CONFLICT = "CONFLICT"                      # Conflict (e.g., optimistic locking conflict)
    CIRCUIT_OPEN = "CIRCUIT_OPEN"              # Circuit breaker is open, execution rejected

    # Interaction-related errors
    ASK_USER_UNAVAILABLE = "ASK_USER_UNAVAILABLE"  # User interaction unavailable (e.g., in a sub-agent)

    # Network-related errors
    NETWORK_ERROR = "NETWORK_ERROR"            # Network request failed
    API_ERROR = "API_ERROR"                    # API call failed
    RATE_LIMIT = "RATE_LIMIT"                  # Rate limit

    def __str__(self) -> str:
        # Keep ``str(code)`` / f"{code}" as the bare code (not "ToolErrorCode.X")
        # so existing string formatting and equality keep working.
        return str(self.value)

    @classmethod
    def get_all_codes(cls) -> list[str]:
        """Get all error codes as plain strings."""
        return [member.value for member in cls]

    @classmethod
    def is_valid_code(cls, code) -> bool:
        """Check if it is a valid error code (accepts str or enum member)."""
        try:
            cls(code)
            return True
        except ValueError:
            return False
