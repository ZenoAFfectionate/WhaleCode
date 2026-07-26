"""Tool Error Code Definitions

Standardized tool error codes for unified error handling and tracking.

Each member carries an ``is_fault`` flag.  Fault codes (e.g. INTERNAL_ERROR,
TIMEOUT) represent genuine tool *malfunctions* and are used by the circuit
breaker to decide when to trip.  Non-fault codes (e.g. INVALID_PARAM,
NOT_FOUND) represent the tool correctly rejecting bad or inapplicable input
and do NOT count toward circuit-breaker failures.

重要-7: ``is_fault`` is now a first-class attribute of each member — the
circuit breaker queries it directly instead of maintaining a separate
``FAULT_ERROR_CODES`` frozenset that can drift out of sync.

Usage::

    >>> ToolErrorCode.TIMEOUT.is_fault
    True
    >>> ToolErrorCode.INVALID_PARAM.is_fault
    False
    >>> ToolErrorCode.TIMEOUT == "TIMEOUT"   # still drop-in string-compatible
    True
    >>> str(ToolErrorCode.TIMEOUT)            # f-strings / JSON serialise cleanly
    'TIMEOUT'
"""

from enum import Enum


class ToolErrorCode(str, Enum):
    """Tool Error Code Enumeration

    Facilitates:
    - Unified error handling at the Agent layer
    - Circuit breaker mechanism identifying failure types (via ``is_fault``)
    - Observability system tracking errors
    - User-friendly error messages
    """

    # fmt: off
    # Resource-related errors
    NOT_FOUND           = ("NOT_FOUND",            False)  # Resource does not exist (file, tool, etc.)
    ACCESS_DENIED       = ("ACCESS_DENIED",        False)  # Access denied
    PERMISSION_DENIED   = ("PERMISSION_DENIED",    False)  # Insufficient permissions
    IS_DIRECTORY        = ("IS_DIRECTORY",         False)  # Expected file but got directory
    BINARY_FILE         = ("BINARY_FILE",          False)  # Binary file cannot be processed

    # Parameter-related errors
    INVALID_PARAM       = ("INVALID_PARAM",        False)  # Invalid or missing parameters
    INVALID_FORMAT      = ("INVALID_FORMAT",       False)  # Format error

    # Execution-related errors (— these are genuine *faults*)
    EXECUTION_ERROR     = ("EXECUTION_ERROR",      True)   # Error occurred during execution
    TIMEOUT             = ("TIMEOUT",              True)   # Execution timeout
    INTERNAL_ERROR      = ("INTERNAL_ERROR",       True)   # Internal error

    # Status-related errors
    CONFLICT            = ("CONFLICT",             False)  # Conflict (e.g., optimistic locking conflict)
    CIRCUIT_OPEN        = ("CIRCUIT_OPEN",         False)  # Circuit breaker is open, execution rejected

    # Interaction-related errors
    ASK_USER_UNAVAILABLE = ("ASK_USER_UNAVAILABLE", False)  # User interaction unavailable (e.g., in a sub-agent)

    # Network-related errors (— these are genuine *faults*)
    NETWORK_ERROR       = ("NETWORK_ERROR",        True)   # Network request failed
    API_ERROR           = ("API_ERROR",            True)   # API call failed
    RATE_LIMIT          = ("RATE_LIMIT",           False)  # Rate limit
    # fmt: on

    def __new__(cls, value, is_fault=False):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.is_fault = is_fault
        return obj

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

    @classmethod
    def fault_codes(cls) -> list[str]:
        """Return the plain-string values of every fault code."""
        return [m.value for m in cls if m.is_fault]
