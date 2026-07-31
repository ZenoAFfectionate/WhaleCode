"""Token counting utilities used by agents and context compaction.

The counter favors fast, local tokenizers. Remote HuggingFace downloads are
disabled by default because token estimation runs during agent initialization
and should not block the critical path.
"""

from __future__ import annotations

import json as _json
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

            # 建议-13: do NOT execute arbitrary tokenizer code by default. A model
            # id that happens to match a local path could otherwise run custom
            # code via trust_remote_code. Opt in explicitly when the source is
            # trusted (HELLOAGENTS_TRUST_REMOTE_CODE=1).
            trust_remote = os.getenv("HELLOAGENTS_TRUST_REMOTE_CODE", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=trust_remote,
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
        """Calculate the token count for one message, including role overhead.

        重要-7: assistant 工具调用消息的真实负载在 metadata.tool_calls 中的
        arguments 里（例如 Write 会把整份文件塞进 arguments）。不计会低估 prompt
        规模，导致压缩决策偏晚，下一次请求可能超窗。
        含 tool_calls 的消息不缓存（arguments 差异大，命中率低）。
        """
        content_tokens = self._count_text(message.content)
        tool_calls = (message.metadata or {}).get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            cache_key = f"{message.role}:{message.content}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            tokens = content_tokens + 4
            self._cache.put(cache_key, tokens)
            return tokens

        # tool_calls 消息：content + 每个 tool_call 的 name + arguments
        tokens = content_tokens + 4
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name", ""))
            tokens += self._count_text(name) + 4
            arguments = tc.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = _json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            elif not isinstance(arguments, str):
                arguments = str(arguments)
            tokens += self._count_text(arguments or "") + 4
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
