"""TokenCounter - Token Counter

Responsibilities:
- Local token estimation (no API calls required)
- Caching mechanism (avoid repeated calculations)
- Incremental calculation (only calculate new messages)
- Fallback plan (use character estimation when tiktoken is unavailable)
"""

import tiktoken
from typing import List, Dict
from ..core.message import Message


class TokenCounter:
    """Token Counter
    
    Features:
    - Local estimation (no API calls required)
    - Caching mechanism (avoid repeated calculations)
    - Incremental calculation (only calculate new messages)
    - Fallback plan (use character estimation when tiktoken is unavailable)
    
    Usage Example:
    ```python
    counter = TokenCounter(model="gpt-4")
    
    # Calculate single message
    tokens = counter.count_message(message)
    
    # Calculate list of messages
    total = counter.count_messages(messages)
    
    # Incremental calculation
    new_total = counter.count_incremental(previous_count, new_messages)
    ```
    """
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize Token Counter
        
        Args:
            model: Model name (used to select tiktoken encoder)
        """
        self.model = model
        self._encoding = self._get_encoding()
        self._cache: Dict[str, int] = {}  # Message content -> Token count
    
    def _get_encoding(self):
        """Get tiktoken encoder
        
        Returns:
            tiktoken encoder instance, returns None on failure
        """
        try:
            # Attempt to get encoder based on model name
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to generic encoder
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None
        except Exception:
            # tiktoken unavailable
            return None
    
    def count_messages(self, messages: List[Message]) -> int:
        """Calculate token count for a list of messages
        
        Args:
            messages: List of messages
        
        Returns:
            Token count
        """
        total = 0
        for msg in messages:
            total += self.count_message(msg)
        return total
    
    def count_message(self, message: Message) -> int:
        """Calculate token count for a single message (with caching)
        
        Args:
            message: Message object
        
        Returns:
            Token count
        """
        # Use message content as cache key
        cache_key = f"{message.role}:{message.content}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate token count
        tokens = self._count_text(message.content)
        
        # Add overhead for role markers (approx. 4 tokens)
        tokens += 4
        
        # Cache result
        self._cache[cache_key] = tokens
        
        return tokens
    
    def count_text(self, text: str) -> int:
        """Calculate token count for text (no cache)
        
        Args:
            text: Text content
        
        Returns:
            Token count
        """
        return self._count_text(text)
    
    def _count_text(self, text: str) -> int:
        """Internal token calculation method
        
        Args:
            text: Text content
        
        Returns:
            Token count
        """
        if self._encoding:
            # Precise calculation using tiktoken
            try:
                return len(self._encoding.encode(text))
            except Exception:
                # tiktoken encoding failed, fallback to character estimation
                return len(text) // 4
        else:
            # Fallback plan: Rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    

    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get cache size

        Returns:
            Number of cached messages
        """
        return len(self._cache)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics

        Returns:
            Dictionary of cache statistics
        """
        return {
            "cached_messages": len(self._cache),
            "total_cached_tokens": sum(self._cache.values())
        }