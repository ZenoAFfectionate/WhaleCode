"""TokenCounter - Token Counter

Responsibilities:
- Local token estimation (no API calls required)
- Caching mechanism with bounded size (LRU eviction)
- Incremental calculation (only calculate new messages)
- Multi-backend: transformers AutoTokenizer → tiktoken → character estimation
"""

from collections import OrderedDict
from typing import List, Dict, Optional

from ..core.message import Message


class _LRUCache:
    """Simple bounded LRU cache using OrderedDict."""

    def __init__(self, max_size: int = 4096):
        self._data: OrderedDict[str, int] = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[int]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: int) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
        else:
            if len(self._data) >= self.max_size:
                self._data.popitem(last=False)  # evict oldest
            self._data[key] = value

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def values(self):
        return self._data.values()


class TokenCounter:
    """Token Counter

    Encoder priority:
    1. transformers AutoTokenizer (accurate for Qwen / LLaMA / etc.)
    2. tiktoken (accurate for GPT family)
    3. Character estimation (1 token ≈ 4 chars, no dependency)

    The cache is bounded (default 4096 entries, LRU eviction).
    """

    # Singleton cache for tokenizer instances (avoid repeated loading)
    _tokenizer_cache: Dict[str, object] = {}

    def __init__(self, model: str = "gpt-4", cache_max_size: int = 4096):
        """Initialize Token Counter

        Args:
            model: Model name (used to select the best tokenizer)
            cache_max_size: Maximum number of cached token counts (LRU)
        """
        self.model = model
        self._encoding = self._get_encoding()
        self._cache = _LRUCache(max_size=cache_max_size)

    def _get_encoding(self):
        """Get the best available tokenizer for the model.

        Returns:
            A tokenizer object with an `encode(text) -> list` method,
            or None if no tokenizer is available.
        """
        # 1. Try transformers AutoTokenizer (best for Qwen, LLaMA, etc.)
        hf_tokenizer = self._try_transformers_tokenizer()
        if hf_tokenizer is not None:
            return hf_tokenizer

        # 2. Try tiktoken (best for GPT family)
        return self._try_tiktoken()

    def _try_transformers_tokenizer(self):
        """Try to load a HuggingFace tokenizer for the model."""
        if self.model in self._tokenizer_cache:
            return self._tokenizer_cache[self.model]

        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
            self._tokenizer_cache[self.model] = tokenizer
            return tokenizer
        except Exception:
            return None

    def _try_tiktoken(self):
        """Try to get a tiktoken encoder."""
        try:
            import tiktoken
            try:
                return tiktoken.encoding_for_model(self.model)
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def count_messages(self, messages: List[Message]) -> int:
        """Calculate token count for a list of messages"""
        total = 0
        for msg in messages:
            total += self.count_message(msg)
        return total

    def count_message(self, message: Message) -> int:
        """Calculate token count for a single message (with LRU caching)"""
        cache_key = f"{message.role}:{message.content}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        tokens = self._count_text(message.content)
        # Add overhead for role markers (approx. 4 tokens)
        tokens += 4

        self._cache.put(cache_key, tokens)
        return tokens

    def count_text(self, text: str) -> int:
        """Calculate token count for text (no cache)"""
        return self._count_text(text)

    def _count_text(self, text: str) -> int:
        """Internal token calculation method"""
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get cache size"""
        return len(self._cache)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_messages": len(self._cache),
            "total_cached_tokens": sum(self._cache.values())
        }
