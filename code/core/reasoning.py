"""Helpers for extracting reasoning text from provider-specific payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


_REASONING_FIELDS = ("reasoning_content", "reasoning")
_REASONING_TEXT_KEYS = ("text", "content", "reasoning_content", "reasoning")


@dataclass(frozen=True)
class ReasoningPayload:
    """Normalized reasoning text plus the source field that provided it."""

    content: Optional[str]
    source: Optional[str] = None


def _normalize_reasoning_value(value: Any, *, preserve_whitespace: bool) -> Optional[str]:
    """Convert provider-specific reasoning payloads into plain text."""
    if value is None:
        return None

    if isinstance(value, str):
        if preserve_whitespace:
            return value if value != "" else None
        text = value.strip()
        return text or None

    if isinstance(value, list):
        parts = []
        for item in value:
            text = _normalize_reasoning_value(item, preserve_whitespace=preserve_whitespace)
            if text is not None:
                parts.append(text)
        if not parts:
            return None
        joined = "".join(parts) if preserve_whitespace else "\n".join(parts)
        joined = joined if preserve_whitespace else joined.strip()
        return joined or None

    if isinstance(value, dict):
        for key in _REASONING_TEXT_KEYS:
            if key in value:
                text = _normalize_reasoning_value(value[key], preserve_whitespace=preserve_whitespace)
                if text is not None:
                    return text

        serialized = json.dumps(value, ensure_ascii=False, default=str)
        if preserve_whitespace:
            return serialized if serialized != "" else None
        serialized = serialized.strip()
        return serialized or None

    text = str(value)
    if preserve_whitespace:
        return text if text != "" else None
    text = text.strip()
    return text or None


def _extract_from_mapping(mapping: dict[str, Any], *, preserve_whitespace: bool) -> ReasoningPayload:
    for field_name in _REASONING_FIELDS:
        if field_name not in mapping:
            continue
        text = _normalize_reasoning_value(mapping.get(field_name), preserve_whitespace=preserve_whitespace)
        if text is not None:
            return ReasoningPayload(content=text, source=field_name)
    return ReasoningPayload(content=None, source=None)


def extract_reasoning_payload(
    obj: Any,
    *,
    preserve_whitespace: bool = False,
) -> ReasoningPayload:
    """Extract reasoning text from message/choice/delta objects or plain dicts."""
    if obj is None:
        return ReasoningPayload(content=None, source=None)

    for field_name in _REASONING_FIELDS:
        if hasattr(obj, field_name):
            text = _normalize_reasoning_value(
                getattr(obj, field_name),
                preserve_whitespace=preserve_whitespace,
            )
            if text is not None:
                return ReasoningPayload(content=text, source=field_name)

    if isinstance(obj, dict):
        return _extract_from_mapping(obj, preserve_whitespace=preserve_whitespace)

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump()
        except TypeError:
            payload = model_dump(exclude_none=False)
        if isinstance(payload, dict):
            result = _extract_from_mapping(payload, preserve_whitespace=preserve_whitespace)
            if result.content is not None:
                return result

    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        payload = dict_method()
        if isinstance(payload, dict):
            result = _extract_from_mapping(payload, preserve_whitespace=preserve_whitespace)
            if result.content is not None:
                return result

    return ReasoningPayload(content=None, source=None)
