"""Conversation compaction helpers for long-running agents."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..core.config import Config
from .history import HistoryManager
from .token_counter import TokenCounter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUMMARY_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "summary_prompt.md"
COMPACTION_PREFIX = "[Context compacted at "
COMPACTION_SUMMARY_HEADING = "## Conversation Summary"
ESSENTIAL_CONTEXT_PREFIX = "[Preserved context]"
ESSENTIAL_CONTEXT_HEADING = "## Essential Context Snapshot"
COMPACTION_ACKNOWLEDGEMENT = (
    "Understood. I have the conversation summary and will continue from the current state."
)
COMPACTION_TRANSCRIPT_LABEL = "Transcript path:"
COMPACTION_RECENT_TAIL_NOTE = "Recent messages are preserved verbatim below."
_TOOL_RESULT_PLACEHOLDER = "[Previous tool result: {tool_name} - truncated to save context]"
_MICRO_COMPACTABLE_TOOLS = {
    "Read",
    "Bash",
    "Grep",
    "Glob",
    "WebSearch",
    "WebFetch",
    "Edit",
    "Write",
}
_WORKING_FILE_TOOLS = {"Read", "Edit", "Write", "Delete"}
_FULL_OUTPUT_PATH_PATTERNS = (
    re.compile(r"^full_output_path:\s*(.+)$", re.MULTILINE),
    re.compile(r"^Full output saved to:\s*(.+)$", re.MULTILINE),
)
_ARCHIVED_CONTEXT_PREFIXES = (
    COMPACTION_PREFIX,
    ESSENTIAL_CONTEXT_PREFIX,
    HistoryManager.SUMMARY_NOTE_PREFIX,
    HistoryManager.SYSTEM_NOTE_PREFIX,
)
_ANALYSIS_TAG_PATTERN = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)
_SUMMARY_TAG_PATTERN = re.compile(r"<summary>([\s\S]*?)</summary>", re.IGNORECASE)
_SUMMARY_BUDGET_MARKER = "... [earlier messages omitted to fit the summary input budget]"
_ESSENTIAL_USER_REQUEST_TOKENS = 800
_ESSENTIAL_COMMAND_TOKENS = 200
_WORKING_FILE_LIMIT = 12
_RECENT_COMMAND_LIMIT = 5
_RECENT_SKILL_LIMIT = 5
_RECOVERABLE_OUTPUT_LIMIT = 8


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


def format_compact_summary(summary: str) -> str:
    """Remove scratchpad tags from a compaction summary response."""
    formatted = (summary or "").strip()
    if not formatted:
        return ""

    formatted = _ANALYSIS_TAG_PATTERN.sub("", formatted).strip()
    summary_match = _SUMMARY_TAG_PATTERN.search(formatted)
    if summary_match:
        formatted = (summary_match.group(1) or "").strip()

    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


@dataclass
class _PreservedContextSnapshot:
    latest_user_request: Optional[str] = None
    todo_lines: List[str] = field(default_factory=list)
    working_files: List[str] = field(default_factory=list)
    recent_commands: List[str] = field(default_factory=list)
    loaded_skills: List[str] = field(default_factory=list)
    recoverable_paths: List[str] = field(default_factory=list)


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

    def get_effective_prompt_tokens(
        self,
        messages: List[Dict],
        latest_prompt_tokens: Optional[int] = None,
    ) -> int:
        """Prefer real prompt usage, but never ignore current message growth."""
        estimated_tokens = self.estimate_tokens(messages)
        if latest_prompt_tokens is None or int(latest_prompt_tokens) <= 0:
            return estimated_tokens
        return max(int(latest_prompt_tokens), estimated_tokens)

    def should_compact(
        self,
        messages: List[Dict],
        latest_prompt_tokens: Optional[int] = None,
    ) -> bool:
        """Return whether the prompt is at or above the compact trigger limit."""
        return (
            self.get_effective_prompt_tokens(
                messages,
                latest_prompt_tokens=latest_prompt_tokens,
            )
            >= self.config.get_compact_trigger_limit()
        )

    def micro_compact(self, messages: List[Dict]) -> List[Dict]:
        """Replace older, bulky tool results with recoverable placeholders in place."""
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
            if tool_name not in _MICRO_COMPACTABLE_TOOLS:
                continue

            placeholder = _TOOL_RESULT_PLACEHOLDER.format(tool_name=tool_name)
            full_output_path = self._extract_full_output_path(content)
            if full_output_path:
                placeholder = f"{placeholder}\nfull_output_path: {full_output_path}"
            message["content"] = placeholder

        return messages

    def auto_compact(self, messages: List[Dict], llm) -> List[Dict]:
        """Compact history using the default summary prompt."""
        return self._compact_with_llm(messages, llm, focus=None)

    def manual_compact(self, messages: List[Dict], llm, focus: Optional[str] = None) -> List[Dict]:
        """Compact history with an optional summary focus."""
        return self._compact_with_llm(messages, llm, focus=focus)

    def _compact_with_llm(self, messages: List[Dict], llm, focus: Optional[str]) -> List[Dict]:
        """Persist transcript, summarize older history, and keep the recent raw tail."""
        transcript_path = self._save_transcript(messages)

        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        if not non_system_messages:
            return self._build_compacted_messages(
                system_messages,
                "",
                [],
                transcript_path=transcript_path,
            )

        round_starts = self._round_start_indices(non_system_messages)
        max_preserved_rounds = self._max_preserved_rounds(
            non_system_messages,
            round_starts=round_starts,
        )
        trigger_limit = self.config.get_compact_trigger_limit()
        best_candidate: Optional[List[Dict]] = None
        best_tokens: Optional[int] = None
        summary_input_budget = self._summary_input_token_budget(
            focus=focus,
            transcript_path=transcript_path,
        )

        for preserved_rounds in range(max_preserved_rounds, -1, -1):
            summary_source, preserved_tail = self._split_for_recent_rounds(
                non_system_messages,
                preserved_rounds,
                round_starts=round_starts,
            )
            essential_context = self._build_essential_context_message(summary_source)
            summary_input_messages = self._messages_for_summary(summary_source)
            conversation_text = self._serialize_messages(
                summary_input_messages,
                max_tokens=summary_input_budget,
            )
            summary = self._summarize_messages(
                summary_input_messages,
                conversation_text,
                llm,
                focus,
                transcript_path=transcript_path,
            )
            candidate = self._build_compacted_messages(
                system_messages,
                summary,
                preserved_tail,
                essential_context=essential_context,
                transcript_path=transcript_path,
            )
            candidate_tokens = self.estimate_tokens(candidate)

            if best_tokens is None or candidate_tokens < best_tokens:
                best_candidate = candidate
                best_tokens = candidate_tokens

            if candidate_tokens <= trigger_limit:
                return candidate

        return best_candidate or self._build_compacted_messages(
            system_messages,
            self._generate_fallback_summary(non_system_messages),
            [],
            essential_context=self._build_essential_context_message(non_system_messages),
            transcript_path=transcript_path,
        )

    def _summarize_messages(
        self,
        messages: List[Dict],
        conversation_text: str,
        llm,
        focus: Optional[str],
        transcript_path: Optional[str] = None,
    ) -> str:
        summary_system_prompt, summary_user_template = _load_summary_prompts()
        focus_instruction = f"Pay special attention to: {focus}" if focus else ""
        summary_prompt = summary_user_template.format(
            conversation=conversation_text,
            focus_instruction=focus_instruction,
            max_tokens=self.config.summary_max_tokens,
            transcript_path=transcript_path or "[unavailable]",
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
            summary = format_compact_summary(getattr(response, "content", str(response)))
            return summary or self._generate_fallback_summary(messages)
        except Exception as exc:
            print(f"[compact] LLM summary failed ({exc}), using fallback")
            return self._generate_fallback_summary(messages)

    def _summary_input_token_budget(
        self,
        *,
        focus: Optional[str],
        transcript_path: Optional[str],
    ) -> int:
        """Return the available token budget for serialized conversation text."""
        summary_system_prompt, summary_user_template = _load_summary_prompts()
        focus_instruction = f"Pay special attention to: {focus}" if focus else ""
        prompt_without_conversation = summary_user_template.format(
            conversation="",
            focus_instruction=focus_instruction,
            max_tokens=self.config.summary_max_tokens,
            transcript_path=transcript_path or "[unavailable]",
        )
        prompt_overhead = self.estimate_tokens(
            [
                {"role": "system", "content": summary_system_prompt},
                {"role": "user", "content": prompt_without_conversation},
            ]
        )
        safety_margin = max(128, min(2048, int(self.config.summary_max_tokens)))
        return max(
            256,
            int(self.config.context_window)
            - int(self.config.summary_max_tokens)
            - prompt_overhead
            - safety_margin,
        )

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
        max_tokens: Optional[int] = None,
    ) -> str:
        """Serialize messages for summary input, preferring the full transcript."""
        serialized = [self._serialize_single_message(message) for message in messages]
        if not serialized:
            return ""

        full_text = "\n\n".join(serialized)
        if max_tokens is None or self.token_counter.count_text(full_text) <= max_tokens:
            return full_text

        selected: List[str] = []
        total_tokens = 0
        separator_tokens = self.token_counter.count_text("\n\n")
        marker_tokens = 0
        for line in reversed(serialized):
            line_tokens = self.token_counter.count_text(line)
            if selected:
                line_tokens += separator_tokens
            if total_tokens + line_tokens > max_tokens:
                break
            selected.append(line)
            total_tokens += line_tokens

        selected.reverse()
        if not selected:
            return self._truncate_to_token_budget(serialized[-1], max_tokens)

        if len(selected) < len(serialized):
            marker_tokens = self.token_counter.count_text(_SUMMARY_BUDGET_MARKER) + separator_tokens
            while selected and total_tokens + marker_tokens > max_tokens:
                removed = selected.pop(0)
                total_tokens -= self.token_counter.count_text(removed)
                if selected:
                    total_tokens -= separator_tokens
        if selected and total_tokens + marker_tokens <= max_tokens:
            selected.insert(0, _SUMMARY_BUDGET_MARKER)
        return "\n\n".join(selected)

    def _messages_for_summary(self, messages: List[Dict]) -> List[Dict]:
        """Remove deterministic preserved-context notes before LLM summarization."""
        filtered: List[Dict] = []
        for message in messages:
            content = _stringify_content(message.get("content")).strip()
            if message.get("role") == "assistant" and content == COMPACTION_ACKNOWLEDGEMENT:
                continue
            if message.get("role") == "user" and content.startswith(ESSENTIAL_CONTEXT_PREFIX):
                continue
            filtered.append(message)
        return filtered

    def _build_essential_context_message(self, messages: List[Dict]) -> Optional[str]:
        """Extract structured state that should survive compaction verbatim."""
        snapshot = self._collect_preserved_context_snapshot(messages)

        sections: List[str] = [ESSENTIAL_CONTEXT_PREFIX, "", ESSENTIAL_CONTEXT_HEADING]

        if snapshot.latest_user_request:
            sections.extend(
                [
                    "### Latest Summarized User Request",
                    snapshot.latest_user_request,
                    "",
                ]
            )

        if snapshot.todo_lines:
            sections.extend(["### Todo State", *snapshot.todo_lines, ""])

        if snapshot.working_files:
            sections.extend(["### Working Set Files", *[f"- {path}" for path in snapshot.working_files], ""])

        if snapshot.recent_commands:
            sections.extend(["### Recent Commands", *[f"- {command}" for command in snapshot.recent_commands], ""])

        if snapshot.loaded_skills:
            sections.extend(["### Loaded Skills", *[f"- {skill}" for skill in snapshot.loaded_skills], ""])

        if snapshot.recoverable_paths:
            sections.extend(
                [
                    "### Recoverable Output Paths",
                    *[f"- {path}" for path in snapshot.recoverable_paths],
                    "",
                ]
            )

        while sections and not sections[-1]:
            sections.pop()

        if sections == [ESSENTIAL_CONTEXT_PREFIX, "", ESSENTIAL_CONTEXT_HEADING]:
            return None
        return "\n".join(sections)

    def _collect_preserved_context_snapshot(self, messages: List[Dict]) -> _PreservedContextSnapshot:
        snapshot = _PreservedContextSnapshot()
        seen_working_files: Set[str] = set()
        seen_commands: Set[str] = set()
        seen_skills: Set[str] = set()
        seen_paths: Set[str] = set()
        todo_captured = False

        for message in reversed(messages):
            role = message.get("role")

            if role == "user" and snapshot.latest_user_request is None:
                content = _stringify_content(message.get("content")).strip()
                if content and not any(content.startswith(prefix) for prefix in _ARCHIVED_CONTEXT_PREFIXES):
                    snapshot.latest_user_request = self._truncate_note_text(
                        content,
                        max_tokens=_ESSENTIAL_USER_REQUEST_TOKENS,
                    )

            if role == "tool" and len(snapshot.recoverable_paths) < _RECOVERABLE_OUTPUT_LIMIT:
                content = _stringify_content(message.get("content"))
                full_output_path = self._extract_full_output_path(content)
                if full_output_path:
                    self._append_unique(
                        snapshot.recoverable_paths,
                        seen_paths,
                        full_output_path,
                        limit=_RECOVERABLE_OUTPUT_LIMIT,
                    )

            if role != "assistant":
                continue
            for tool_call in reversed(message.get("tool_calls") or []):
                tool_name, arguments = self._parse_tool_call(tool_call)

                if not todo_captured and tool_name == "TodoWrite":
                    snapshot.todo_lines = self._todo_lines_from_arguments(arguments)
                    todo_captured = True

                if tool_name in _WORKING_FILE_TOOLS:
                    path = arguments.get("path")
                    if isinstance(path, str) and path.strip():
                        self._append_unique(
                            snapshot.working_files,
                            seen_working_files,
                            path.strip(),
                            limit=_WORKING_FILE_LIMIT,
                        )

                if tool_name == "Bash":
                    command = arguments.get("command")
                    if isinstance(command, str) and command.strip():
                        working_directory = arguments.get("working_directory") or "."
                        entry = f"[{working_directory}] {command.strip()}"
                        entry = self._truncate_note_text(
                            entry,
                            max_tokens=_ESSENTIAL_COMMAND_TOKENS,
                        )
                        self._append_unique(
                            snapshot.recent_commands,
                            seen_commands,
                            entry,
                            limit=_RECENT_COMMAND_LIMIT,
                        )

                if tool_name == "Skill":
                    skill = arguments.get("skill")
                    if isinstance(skill, str) and skill.strip():
                        self._append_unique(
                            snapshot.loaded_skills,
                            seen_skills,
                            skill.strip(),
                            limit=_RECENT_SKILL_LIMIT,
                        )

        return snapshot

    @staticmethod
    def _append_unique(items: List[str], seen: Set[str], value: str, *, limit: int) -> None:
        if value in seen or len(items) >= limit:
            return
        seen.add(value)
        items.append(value)

    def _todo_lines_from_arguments(self, arguments: Dict[str, Any]) -> List[str]:
        todos = arguments.get("todos")
        if not isinstance(todos, list):
            return []

        lines: List[str] = []
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            content = str(todo.get("content") or "").strip()
            status = str(todo.get("status") or "").strip()
            if not content or not status:
                continue
            priority = str(todo.get("priority") or "").strip().lower()
            priority_suffix = f" [{priority}]" if priority and priority != "medium" else ""
            lines.append(f"- {status}: {content}{priority_suffix}")
        return lines

    @staticmethod
    def _parse_tool_call(tool_call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        function = tool_call.get("function", {}) or {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        elif not isinstance(arguments, dict):
            arguments = {}
        return function.get("name", "unknown"), arguments

    def _serialize_single_message(self, message: Dict) -> str:
        """Serialize one message, preserving tool-call metadata when present."""
        role = message.get("role", "?")
        content = _stringify_content(message.get("content"))

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

    def _truncate_to_token_budget(self, text: str, max_tokens: int) -> str:
        """Emergency fallback when even one serialized message exceeds the budget."""
        if max_tokens <= 0:
            return ""
        marker = "... [message truncated to fit the summary input budget]"
        return self._truncate_note_text(text, max_tokens=max_tokens, marker=marker)

    def _truncate_note_text(
        self,
        text: str,
        *,
        max_tokens: int,
        marker: str = "... [truncated]",
    ) -> str:
        """Trim structured-note fields without breaking the surrounding format."""
        if max_tokens <= 0:
            return marker
        if self.token_counter.count_text(text) <= max_tokens:
            return text
        marker_tokens = self.token_counter.count_text(marker)
        available_tokens = max(1, max_tokens - marker_tokens)
        tokens = self.token_counter.encode_text(text)
        if not tokens:
            return marker
        truncated = self.token_counter.decode_tokens(tokens[:available_tokens]).rstrip()
        if not truncated:
            return marker
        return f"{truncated}\n{marker}"

    @staticmethod
    def _extract_full_output_path(content: str) -> Optional[str]:
        for pattern in _FULL_OUTPUT_PATH_PATTERNS:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        return None

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

    def _max_preserved_rounds(
        self,
        messages: List[Dict],
        *,
        round_starts: Optional[List[int]] = None,
    ) -> int:
        """Preserve up to N recent rounds, but still summarize older history when possible."""
        if round_starts is None:
            round_starts = self._round_start_indices(messages)
        if not round_starts:
            return 0

        configured_rounds = max(0, int(self.config.compact_preserve_recent_rounds))
        if len(round_starts) <= 1:
            return 0
        return min(configured_rounds, len(round_starts) - 1)

    def _split_for_recent_rounds(
        self,
        messages: List[Dict],
        preserved_rounds: int,
        *,
        round_starts: Optional[List[int]] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split messages into ``(summary_source, preserved_raw_tail)``."""
        if preserved_rounds <= 0:
            return messages, []

        if round_starts is None:
            round_starts = self._round_start_indices(messages)
        if not round_starts:
            return messages, []

        preserved_rounds = min(preserved_rounds, len(round_starts))
        keep_from_index = round_starts[-preserved_rounds]
        return messages[:keep_from_index], messages[keep_from_index:]

    def _round_start_indices(self, messages: List[Dict]) -> List[int]:
        """Return indices whose user message starts a real conversational round."""
        round_starts: List[int] = []
        for index, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            content = _stringify_content(message.get("content")).strip()
            if any(content.startswith(prefix) for prefix in _ARCHIVED_CONTEXT_PREFIXES):
                continue
            round_starts.append(index)
        return round_starts

    def _build_compacted_messages(
        self,
        system_messages: List[Dict],
        summary: str,
        preserved_tail: List[Dict],
        *,
        essential_context: Optional[str] = None,
        transcript_path: Optional[str] = None,
    ) -> List[Dict]:
        compacted: List[Dict] = []
        merged_system = self._merge_system_messages(system_messages)
        if merged_system:
            compacted.append({"role": "system", "content": merged_system})
        compacted.append(
            {
                "role": "user",
                "content": self._build_summary_message(
                    summary,
                    transcript_path=transcript_path,
                    preserved_tail=preserved_tail,
                ),
            }
        )
        compacted.append({"role": "assistant", "content": COMPACTION_ACKNOWLEDGEMENT})
        if essential_context:
            compacted.append({"role": "user", "content": essential_context})
        compacted.extend(preserved_tail)
        return compacted

    @staticmethod
    def _merge_system_messages(system_messages: List[Dict]) -> str:
        parts: List[str] = []
        for message in system_messages:
            content = _stringify_content(message.get("content")).strip()
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    def _build_summary_message(
        self,
        summary: str,
        *,
        transcript_path: Optional[str] = None,
        preserved_tail: Optional[List[Dict]] = None,
    ) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        lines = [f"{COMPACTION_PREFIX}{timestamp}]", "", COMPACTION_SUMMARY_HEADING]
        if transcript_path:
            lines.append(f"{COMPACTION_TRANSCRIPT_LABEL} {transcript_path}")
        if preserved_tail:
            lines.append(COMPACTION_RECENT_TAIL_NOTE)
        if transcript_path or preserved_tail:
            lines.append("")
        if summary:
            lines.append(summary)
        return "\n".join(lines)

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
