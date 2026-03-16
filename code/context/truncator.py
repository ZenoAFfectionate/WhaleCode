"""ObservationTruncator - Tool Output Truncator

Responsibilities:
- Unify tool output truncation (avoiding each tool implementing it individually)
- Support multiple truncation directions (head/tail/head_tail)
- Return ToolResponse.partial() status
- Save full output to file
"""

import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class ObservationTruncator:
    """Tool Output Truncator
    
    Features:
    - Multi-directional truncation (head/tail/head_tail)
    - Automatically save full output
    - Return standard ToolResponse.partial() response
    
    Usage Example:
    ```python
    truncator = ObservationTruncator(
        max_lines=2000,
        max_bytes=51200,
        truncate_direction="head"
    )
    
    # Truncate tool output
    result = truncator.truncate(
        tool_name="search",
        output=long_output,
        metadata={"query": "test"}
    )
    
    # result is a dictionary containing:
    # - truncated: bool
    # - preview: str (truncated preview)
    # - full_output_path: str (full output path)
    # - stats: dict (statistics)
    ```
    """
    
    def __init__(
        self,
        max_lines: int = 2000,
        max_bytes: int = 51200,
        truncate_direction: str = "head",
        output_dir: str = "memory/tool-output"
    ):
        """Initialize the truncator
        
        Args:
            max_lines: Maximum number of lines to keep
            max_bytes: Maximum number of bytes to keep
            truncate_direction: Truncation direction (head/tail/head_tail)
            output_dir: Directory to save the full output
        """
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.truncate_direction = truncate_direction
        self.output_dir = output_dir
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def truncate(
        self,
        tool_name: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Truncate tool output
        
        Args:
            tool_name: Tool name
            output: Original output
            metadata: Metadata (optional)
        
        Returns:
            Truncation result dictionary, containing:
            - truncated: bool - Whether it was truncated
            - preview: str - Preview content
            - full_output_path: str - Full output path (if truncated)
            - stats: dict - Statistics
        """
        start = time.time()
        lines = output.splitlines()
        bytes_size = len(output.encode('utf-8'))
        
        # Check if truncation is needed
        if len(lines) <= self.max_lines and bytes_size <= self.max_bytes:
            # No truncation needed
            return {
                "truncated": False,
                "preview": output,
                "full_output_path": None,
                "stats": {
                    "original_lines": len(lines),
                    "original_bytes": bytes_size,
                    "time_ms": int((time.time() - start) * 1000)
                }
            }
        
        # Truncation needed
        truncated_lines = self._truncate_lines(lines)
        preview = "\n".join(truncated_lines)
        truncated_bytes = len(preview.encode('utf-8'))
        
        # Save full output
        output_path = self._save_full_output(tool_name, output, metadata)
        
        return {
            "truncated": True,
            "preview": preview,
            "full_output_path": output_path,
            "stats": {
                "direction": self.truncate_direction,
                "original_lines": len(lines),
                "original_bytes": bytes_size,
                "kept_lines": len(truncated_lines),
                "kept_bytes": truncated_bytes,
                "time_ms": int((time.time() - start) * 1000)
            }
        }
    
    def _truncate_lines(self, lines: list) -> list:
        """Truncate lines based on direction
        
        Args:
            lines: Original list of lines
        
        Returns:
            Truncated list of lines
        """
        if self.truncate_direction == "head":
            return lines[:self.max_lines]
        elif self.truncate_direction == "tail":
            return lines[-self.max_lines:]
        elif self.truncate_direction == "head_tail":
            half = self.max_lines // 2
            return lines[:half] + ["...(middle omitted)..."] + lines[-half:]
        else:
            # Default to head
            return lines[:self.max_lines]
    
    def _save_full_output(
        self,
        tool_name: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save full output to a file
        
        Args:
            tool_name: Tool name
            output: Full output
            metadata: Metadata
        
        Returns:
            Saved file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"tool_{timestamp}_{tool_name}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "tool": tool_name,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filepath
