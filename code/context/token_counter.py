"""Token counting utilities used by agents and context compaction.

The counter favors fast, local tokenizers. Remote HuggingFace downloads are
disabled by default because token estimation runs during agent initialization
and should not block the critical path.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional, Sequence

from ..core.message import Message


class _LRUCache:
    """Simple bounded LRU cache for per-message token counts."""

    def __init__(self, max_size: int = 4096):
        self._data: OrderedDict[str, int] = OrderedDict()
        self.max_size = max(1, int(max_size))

    def get(self, key: str) -> Optional[int]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: int) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        elif len(self._data) >= self.max_size:
            self._data.popitem(last=False)
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def values(self) -> Iterable[int]:
        return self._data.values()


class TokenCounter:
    """Local token estimator with bounded caching.

    Encoder priority:
    1. Local HuggingFace tokenizer when the model path exists on disk
    2. ``tiktoken`` model encoder / ``cl100k_base`` fallback
    3. Character-based estimation
    """

    _UNAVAILABLE = object()
    _tokenizer_cache: Dict[str, object] = {}

    def __init__(self, model: str = "gpt-4", cache_max_size: int = 4096):
        self.model = model or ""
        self._encoding = self._get_encoding()
        self._cache = _LRUCache(max_size=cache_max_size)

    def _get_encoding(self):
        """Return the best locally-available tokenizer for the current model."""
        local_tokenizer = self._try_local_transformers_tokenizer()
        if local_tokenizer is not None:
            return local_tokenizer
        return self._try_tiktoken()

    def _try_local_transformers_tokenizer(self):
        """Load a local HuggingFace tokenizer without hitting the network."""
        if not self.model:
            return None

        cache_key = f"hf::{self.model}"
        cached = self._tokenizer_cache.get(cache_key)
        if cached is self._UNAVAILABLE:
            return None
        if cached is not None:
            return cached

        model_path = Path(os.path.expanduser(self.model))
        if not model_path.exists():
            self._tokenizer_cache[cache_key] = self._UNAVAILABLE
            return None

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self._tokenizer_cache[cache_key] = self._UNAVAILABLE
            return None

        self._tokenizer_cache[cache_key] = tokenizer
        return tokenizer

    def _try_tiktoken(self):
        """Load a tiktoken encoder when available."""
        cache_key = f"tiktoken::{self.model}"
        cached = self._tokenizer_cache.get(cache_key)
        if cached is self._UNAVAILABLE:
            return None
        if cached is not None:
            return cached

        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer_cache[cache_key] = self._UNAVAILABLE
            return None

        self._tokenizer_cache[cache_key] = encoding
        return encoding

    def count_messages(self, messages: Iterable[Message]) -> int:
        """Calculate the token count for a sequence of messages."""
        return sum(self.count_message(message) for message in messages)

    def count_message(self, message: Message) -> int:
        """Calculate the token count for one message, including role overhead."""
        cache_key = f"{message.role}:{message.content}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        tokens = self._count_text(message.content) + 4
        self._cache.put(cache_key, tokens)
        return tokens

    def count_text(self, text: str) -> int:
        """Calculate the token count for raw text without message overhead."""
        return self._count_text(text)

    def encode_text(self, text: str) -> Optional[Sequence[Any]]:
        """Best-effort tokenization for preview slicing."""
        payload = text or ""
        if not payload:
            return []

        if self._encoding is not None:
            try:
                return self._encoding.encode(payload)
            except Exception:
                pass

        fallback_tokens = re.findall(r"\S+\s*", payload)
        if fallback_tokens:
            return fallback_tokens
        return [payload]

    def decode_tokens(self, tokens: Sequence[Any]) -> str:
        """Best-effort decode for tokens produced by ``encode_text``."""
        if not tokens:
            return ""

        if self._encoding is not None:
            try:
                return self._encoding.decode(list(tokens))
            except Exception:
                pass

        return "".join(str(token) for token in tokens)

    def _count_text(self, text: str) -> int:
        payload = text or ""
        if not payload:
            return 0

        if self._encoding is not None:
            try:
                return len(self._encoding.encode(payload))
            except Exception:
                pass

        return max(1, len(payload) // 4)

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cached_messages": len(self._cache),
            "total_cached_tokens": sum(self._cache.values()),
        }
