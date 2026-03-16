"""Circuit Breaker Mechanism - Prevents infinite loops caused by continuous tool failures"""

import time
from typing import Any, Dict
from collections import defaultdict
from .response import ToolResponse, ToolStatus


class CircuitBreaker:
    """
    Tool Circuit Breaker

    Features:
    - Automatically disables tools upon continuous failures
    - Automatically recovers after timeout
    - Judges errors based on the ToolResponse protocol

    State Machine:
    Closed (Normal) → Open (Tripped) → Closed (Recovered)
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 300,
        enabled: bool = True
    ):
        """
        Initialize the circuit breaker

        Args:
            failure_threshold: Number of consecutive failures before tripping (default 3)
            recovery_timeout: Recovery time in seconds after tripping (default 300)
            enabled: Whether to enable the circuit breaker (default True)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.enabled = enabled

        # Failure counts (per tool)
        self.failure_counts: Dict[str, int] = defaultdict(int)

        # Timestamps when the circuit was opened (tripped)
        self.open_timestamps: Dict[str, float] = {}

    def is_open(self, tool_name: str) -> bool:
        """
        Check if the tool's circuit breaker is open (tripped)

        Args:
            tool_name: Tool name

        Returns:
            True: Tool is disabled
            False: Tool is available
        """
        if not self.enabled:
            return False

        # Check if it is in the tripped list
        if tool_name not in self.open_timestamps:
            return False

        # Check if it can be recovered
        open_time = self.open_timestamps[tool_name]
        if time.time() - open_time > self.recovery_timeout:
            # Automatic recovery
            self.close(tool_name)
            return False

        return True

    def record_result(self, tool_name: str, response: ToolResponse):
        """
        Record the tool execution result

        Args:
            tool_name: Tool name
            response: Tool response object
        """
        if not self.enabled:
            return

        # Check if it is an error
        is_error = response.status == ToolStatus.ERROR

        if is_error:
            self._on_failure(tool_name)
        else:
            self._on_success(tool_name)

    def _on_failure(self, tool_name: str):
        """Handle failure"""
        # Increment failure count
        self.failure_counts[tool_name] += 1

        # Check if the threshold is reached
        if self.failure_counts[tool_name] >= self.failure_threshold:
            self.open_timestamps[tool_name] = time.time()
            print(f"🔴 Circuit Breaker: Tool '{tool_name}' tripped ({self.failure_counts[tool_name]} consecutive failures)")

    def _on_success(self, tool_name: str):
        """Handle success"""
        # Reset failure count
        self.failure_counts[tool_name] = 0

    def open(self, tool_name: str):
        """Manually trip the circuit breaker"""
        if not self.enabled:
            return

        self.open_timestamps[tool_name] = time.time()
        print(f"🔴 Circuit Breaker: Tool '{tool_name}' manually tripped")

    def close(self, tool_name: str):
        """Close the circuit breaker, recovering the tool"""
        self.failure_counts[tool_name] = 0
        self.open_timestamps.pop(tool_name, None)
        print(f"🟢 Circuit Breaker: Tool '{tool_name}' recovered")

    def get_status(self, tool_name: str) -> Dict[str, Any]:
        """
        Get the circuit breaker status of a tool

        Args:
            tool_name: Tool name

        Returns:
            Status dictionary containing:
            - state: "open" | "closed"
            - failure_count: Number of failures
            - open_since: Time when tripped (only for open state)
            - recover_in_seconds: Countdown to recovery (only for open state)
        """
        is_open = tool_name in self.open_timestamps

        if is_open:
            open_time = self.open_timestamps[tool_name]
            time_since_open = time.time() - open_time
            time_to_recover = max(0, self.recovery_timeout - time_since_open)

            return {
                "state": "open",
                "failure_count": self.failure_counts[tool_name],
                "open_since": open_time,
                "recover_in_seconds": int(time_to_recover)
            }
        else:
            return {
                "state": "closed",
                "failure_count": self.failure_counts[tool_name]
            }

    def get_all_status(self) -> Dict[str, Dict]:
        """
        Get the circuit breaker status of all tools

        Returns:
            Tool name -> Status dictionary
        """
        # Collect all known tool names
        all_tools = set(self.failure_counts.keys()) | set(self.open_timestamps.keys())

        return {
            tool_name: self.get_status(tool_name)
            for tool_name in all_tools
        }