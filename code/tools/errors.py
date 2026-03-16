"""Tool Error Code Definitions

Standardized tool error codes for unified error handling and tracking.
"""


class ToolErrorCode:
    """Tool Error Code Enumeration
    
    Defines standard error codes that tools may return, facilitating:
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
    
    @classmethod
    def get_all_codes(cls) -> list[str]:
        """Get all error codes"""
        return [
            value for name, value in vars(cls).items()
            if not name.startswith('_') and isinstance(value, str)
        ]
    
    @classmethod
    def is_valid_code(cls, code: str) -> bool:
        """Check if it is a valid error code"""
        return code in cls.get_all_codes()