"""Conversation compaction helpers for long-running agents."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.config import Config
from .token_counter import TokenCounter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUMMARY_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "summary_prompt.md"
COMPACTION_PREFIX = "[Context compacted at "
COMPACTION_SUMMARY_HEADING = "## Conversation Summary"
COMPACTION_ACKNOWLEDGEMENT = (
    "Understood. I have the conversation summary and will continue from the current state."
)
_TOOL_RESULT_PLACEHOLDER = "[Previous tool result: {tool_name} - truncated to save context]"


def _extract_section(text: str, name: str) -> str:
    """Extract content between ``START`` and ``END`` HTML comment markers."""
    pattern = rf"<!--\s*{re.escape(name)}_START\s*-->\s*\n(.*?)\n\s*<!--\s*{re.escape(name)}_END\s*-->"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Section '{name}' not found in summary_prompt.md")
    return match.group(1).strip()


@lru_cache(maxsize=1)
def _load_summary_prompts() -> Tuple[str, str]:
    prompt_markdown = _SUMMARY_PROMPT_FILE.read_text(encoding="utf-8")
    return (
        _extract_section(prompt_markdown, "SUMMARY_SYSTEM_PROMPT"),
        _extract_section(prompt_markdown, "SUMMARY_USER_TEMPLATE"),
    )


SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE = _load_summary_prompts()


def _stringify_content(content: Any) -> str:
    """Normalize OpenAI-style message content into a plain text string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


class ContextCompactor:
    """Compaction pipeline for OpenAI-style chat transcripts."""

    def __init__(self, config: Config, token_counter: Optional[TokenCounter] = None):
        self.config = config
        self.token_counter = token_counter or TokenCounter()
        self._tool_name_map: Dict[str, str] = {}
        self._tool_name_map_msg_count = 0

    def estimate_tokens(self, messages: List[Dict]) -> int:
        """Estimate total prompt tokens for a message list."""
        total = 0
        for msg in messages:
            total += self.token_counter.count_text(_stringify_content(msg.get("content")))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += self.token_counter.count_text(json.dumps(tool_calls, ensure_ascii=False, default=str))
            total += 4
        return total

    def micro_compact(self, messages: List[Dict]) -> List[Dict]:
        """Replace older, verbose tool results with small placeholders in place."""
        keep_recent = max(0, int(self.config.compact_keep_recent_tool_results))
        tool_indices = [index for index, msg in enumerate(messages) if msg.get("role") == "tool"]
        if len(tool_indices) <= keep_recent:
            return messages

        tool_name_map = self._update_tool_name_map(messages)
        for reverse_index, message_index in enumerate(reversed(tool_indices)):
            if reverse_index < keep_recent:
                continue

            message = messages[message_index]
            content = _stringify_content(message.get("content"))
            if len(content) <= 200 or content.startswith("[Previous tool result:"):
                continue

            tool_call_id = message.get("tool_call_id", "")
            tool_name = tool_name_map.get(tool_call_id, "unknown")
            message["content"] = _TOOL_RESULT_PLACEHOLDER.format(tool_name=tool_name)

        return messages

    def auto_compact(self, messages: List[Dict], llm) -> List[Dict]:
        """Compact history using the default summary prompt."""
        return self._compact_with_llm(messages, llm, focus=None)

    def manual_compact(self, messages: List[Dict], llm, focus: Optional[str] = None) -> List[Dict]:
        """Compact history with an optional summary focus."""
        return self._compact_with_llm(messages, llm, focus=focus)

    def _compact_with_llm(self, messages: List[Dict], llm, focus: Optional[str]) -> List[Dict]:
        """Persist transcript, summarize non-system history, and rebuild messages."""
        self._save_transcript(messages)

        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        conversation_text = self._serialize_messages(non_system_messages, max_chars=80000)
        summary = self._summarize_messages(non_system_messages, conversation_text, llm, focus)

        compacted: List[Dict] = []
        if system_messages:
            merged_system = "\n\n".join(_stringify_content(msg.get("content")) for msg in system_messages)
            compacted.append({"role": "system", "content": merged_system})
        compacted.append({"role": "user", "content": self._build_summary_message(summary)})
        compacted.append({"role": "assistant", "content": COMPACTION_ACKNOWLEDGEMENT})
        return compacted

    def _summarize_messages(
        self,
        messages: List[Dict],
        conversation_text: str,
        llm,
        focus: Optional[str],
    ) -> str:
        summary_system_prompt, summary_user_template = _load_summary_prompts()
        focus_instruction = f"Pay special attention to: {focus}" if focus else ""
        summary_prompt = summary_user_template.format(
            conversation=conversation_text,
            focus_instruction=focus_instruction,
            max_tokens=self.config.summary_max_tokens,
        )

        try:
            response = llm.invoke(
                messages=[
                    {"role": "system", "content": summary_system_prompt},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=self.config.summary_temperature,
                max_tokens=self.config.summary_max_tokens,
            )
            summary = getattr(response, "content", str(response)).strip()
            return summary or self._generate_fallback_summary(messages)
        except Exception as exc:
            print(f"[compact] LLM summary failed ({exc}), using fallback")
            return self._generate_fallback_summary(messages)

    def _update_tool_name_map(self, messages: List[Dict]) -> Dict[str, str]:
        """Incrementally update the ``tool_call_id -> tool_name`` mapping."""
        start = self._tool_name_map_msg_count
        if start > len(messages):
            self._tool_name_map.clear()
            start = 0

        for msg in messages[start:]:
            if msg.get("role") != "assistant":
                continue
            for tool_call in msg.get("tool_calls") or []:
                tool_call_id = tool_call.get("id", "")
                function = tool_call.get("function", {}) or {}
                tool_name = function.get("name", "unknown")
                if tool_call_id:
                    self._tool_name_map[tool_call_id] = tool_name

        self._tool_name_map_msg_count = len(messages)
        return self._tool_name_map

    def _serialize_messages(
        self,
        messages: List[Dict],
        max_chars: int = 80000,
        max_message_chars: int = 2000,
    ) -> str:
        """Serialize the newest relevant messages first within a fixed character budget."""
        serialized = [
            self._serialize_single_message(message, max_message_chars=max_message_chars)
            for message in messages
        ]
        if not serialized:
            return ""

        selected: List[str] = []
        total_chars = 0
        for line in reversed(serialized):
            line_chars = len(line) + (2 if selected else 0)
            if total_chars + line_chars > max_chars:
                break
            selected.append(line)
            total_chars += line_chars

        selected.reverse()
        if len(selected) < len(serialized):
            selected.insert(0, "... [earlier messages truncated]")
        return "\n\n".join(selected)

    def _serialize_single_message(self, message: Dict, max_message_chars: int) -> str:
        """Serialize one message, preserving tool-call metadata when present."""
        role = message.get("role", "?")
        content = _stringify_content(message.get("content"))
        if len(content) > max_message_chars:
            content = content[:max_message_chars].rstrip() + "... [truncated]"

        suffix_parts: List[str] = []
        if role == "assistant" and message.get("tool_calls"):
            tool_names = [
                tool_call.get("function", {}).get("name", "unknown")
                for tool_call in message.get("tool_calls", [])
            ]
            suffix_parts.append("tool_calls=" + ", ".join(tool_names))
        if role == "tool" and message.get("tool_call_id"):
            suffix_parts.append(f"tool_call_id={message['tool_call_id']}")

        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        return f"[{role}{suffix}] {content}"

    def _generate_fallback_summary(self, messages: Iterable[Dict]) -> str:
        """Generate a deterministic summary when the LLM summary step fails."""
        role_counts: Dict[str, int] = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1

        parts = [f"Conversation contained {sum(role_counts.values())} messages:"]
        for role, count in sorted(role_counts.items()):
            parts.append(f"- {role}: {count}")
        return "\n".join(parts)

    def _build_summary_message(self, summary: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{COMPACTION_PREFIX}{timestamp}]\n\n{COMPACTION_SUMMARY_HEADING}\n{summary}"

    def _save_transcript(self, messages: List[Dict]) -> Optional[str]:
        """Persist the full transcript as JSONL using an atomic write."""
        transcript_dir = Path(self.config.compact_transcript_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = transcript_dir / f"transcript_{timestamp}.jsonl"

        lines = []
        for message in messages:
            serializable = {
                key: value
                for key, value in message.items()
                if isinstance(value, (str, int, float, bool, list, dict, type(None)))
            }
            lines.append(json.dumps(serializable, ensure_ascii=False, default=str))

        try:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write(filepath, "\n".join(lines) + "\n")
            return str(filepath.resolve())
        except Exception as exc:
            print(f"[compact] Failed to save transcript: {exc}")
            return None

    @staticmethod
    def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
        """Write a file atomically to avoid partially-written transcripts."""
        temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temp_path.write_text(content, encoding=encoding)
        os.replace(temp_path, path)
