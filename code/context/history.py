"""HistoryManager - Historical Message Manager

Responsibilities:
- Append messages (append-only, no editing, cache-friendly)
- Compress history (generate summary + retain recent rounds)
- Session serialization/deserialization
- Round boundary detection
"""

from typing import List, Dict, Any
from datetime import datetime
from ..core.message import Message


class HistoryManager:
    """History Manager
    
    Features:
    - Append-only, no editing (cache-friendly)
    - Automatically compress history (summary + retain recent rounds)
    - Support session saving/loading
    
    Usage Example:
    ```python
    manager = HistoryManager(min_retain_rounds=10)
    
    # Append messages
    manager.append(Message("hello", "user"))
    manager.append(Message("hi", "assistant"))
    
    # Get history
    history = manager.get_history()
    
    # Compress history
    manager.compress("This is the summary of the previous conversation")
    
    # Serialize
    data = manager.to_dict()
    
    # Deserialize
    manager.load_from_dict(data)
    ```
    """
    
    def __init__(
        self,
        min_retain_rounds: int = 10,
        compression_threshold: float = 0.8
    ):
        """Initialize the history manager
        
        Args:
            min_retain_rounds: Minimum number of complete rounds to retain when compressing
            compression_threshold: Compression threshold (currently unused, reserved for future use)
        """
        self._history: List[Message] = []
        self.min_retain_rounds = min_retain_rounds
        self.compression_threshold = compression_threshold
    
    def append(self, message: Message) -> None:
        """Append a message (append-only, no editing)
        
        Args:
            message: The message to append
        """
        self._history.append(message)
    
    def get_history(self) -> List[Message]:
        """Get a copy of the history
        
        Returns:
            A copy of the list of historical messages
        """
        return self._history.copy()
    
    def clear(self) -> None:
        """Clear history"""
        self._history.clear()
    
    def estimate_rounds(self) -> int:
        """Estimate the number of complete rounds
        
        Definition of a round: 1 user message + N assistant/tool/summary messages
        
        Returns:
            Number of complete rounds
        """
        rounds = 0
        i = 0
        while i < len(self._history):
            if self._history[i].role == "user":
                rounds += 1
                # Skip subsequent messages in this round
                i += 1
                while i < len(self._history) and self._history[i].role != "user":
                    i += 1
            else:
                i += 1
        return rounds
    
    def find_round_boundaries(self) -> List[int]:
        """Find the starting index of each round
        
        Returns:
            List of starting indices for each round, e.g., [0, 3, 7, 10]
        """
        boundaries = []
        for i, msg in enumerate(self._history):
            if msg.role == "user":
                boundaries.append(i)
        return boundaries
    
    def compress(self, summary: str) -> None:
        """Compress history
        
        Replace old history with a summary message, retaining the most recent N complete rounds of conversation
        
        Args:
            summary: History summary text
        """
        # Check if there are enough rounds to require compression
        rounds = self.estimate_rounds()
        if rounds <= self.min_retain_rounds:
            return
        
        # Find all round boundaries
        boundaries = self.find_round_boundaries()
        
        # Calculate the starting position to retain (retain the most recent min_retain_rounds rounds)
        if len(boundaries) > self.min_retain_rounds:
            keep_from_index = boundaries[-self.min_retain_rounds]
        else:
            # Less than the minimum number of rounds, do not compress
            return
        
        # Generate summary message
        summary_msg = Message(
            content=f"## Archived Session Summary\n{summary}",
            role="summary",
            metadata={"compressed_at": datetime.now().isoformat()}
        )
        
        # Replace history: summary + retained recent rounds
        self._history = [summary_msg] + self._history[keep_from_index:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary (for session saving)
        
        Returns:
            Dictionary containing history and metadata
        """
        return {
            "history": [msg.to_dict() for msg in self._history],
            "created_at": datetime.now().isoformat(),
            "rounds": self.estimate_rounds()
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary (for session recovery)
        
        Args:
            data: Serialized historical data
        """
        self._history = [
            Message.from_dict(msg_data)
            for msg_data in data.get("history", [])
        ]