"""Append-only conversation history with compression helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.message import Message


class HistoryManager:
    """Manage persisted chat history and project it back into model messages."""

    SUMMARY_HEADING = "## Archived Session Summary"
    SUMMARY_NOTE_PREFIX = "[Conversation summary]"
    SYSTEM_NOTE_PREFIX = "[System note]"
    TOOL_RESULT_PREFIX = "Previous tool result:"

    def __init__(self, min_retain_rounds: int = 10, compression_threshold: float = 0.8):
        self._history: List[Message] = []
        self.min_retain_rounds = min_retain_rounds
        self.compression_threshold = compression_threshold

    def append(self, message: Message) -> None:
        self._history.append(message)

    def get_history(self) -> List[Message]:
        return self._history.copy()

    def clear(self) -> None:
        self._history.clear()

    def estimate_rounds(self, history: Optional[Sequence[Message]] = None) -> int:
        """Estimate complete rounds, where each user message starts a new round."""
        source = self._resolve_history(history)
        rounds = 0
        index = 0
        while index < len(source):
            if source[index].role != "user":
                index += 1
                continue

            rounds += 1
            index += 1
            while index < len(source) and source[index].role != "user":
                index += 1
        return rounds

    def find_round_boundaries(self, history: Optional[Sequence[Message]] = None) -> List[int]:
        """Return the starting index of each round."""
        source = self._resolve_history(history)
        return [index for index, message in enumerate(source) if message.role == "user"]

    def get_compression_split(
        self,
        history: Optional[Sequence[Message]] = None,
    ) -> Optional[Tuple[List[Message], List[Message]]]:
        """Split history into ``(to_compress, retained_recent_rounds)`` when eligible."""
        source = self._resolve_history(history)
        boundaries = self.find_round_boundaries(source)
        if len(boundaries) <= self.min_retain_rounds:
            return None

        keep_from_index = boundaries[-self.min_retain_rounds]
        return source[:keep_from_index], source[keep_from_index:]

    def compress(self, summary: str) -> None:
        """Replace older history with a single durable summary message."""
        split = self.get_compression_split()
        if split is None:
            return

        _, retained_history = split
        summary_message = self.build_summary_message(
            summary,
            metadata={"compressed_at": datetime.now().isoformat()},
        )
        self._history = [summary_message, *retained_history]

    def build_summary_message(
        self,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Create a normalized ``summary`` role message for persisted history."""
        return Message(
            content=self.normalize_summary_content(summary),
            role="summary",
            metadata=metadata or {},
        )

    def normalize_summary_content(self, summary: str) -> str:
        """Ensure persisted summaries share one stable heading."""
        content = (summary or "").strip()
        if not content:
            return self.SUMMARY_HEADING
        if content.startswith(self.SUMMARY_HEADING):
            return content
        return f"{self.SUMMARY_HEADING}\n{content}"

    def build_llm_messages(
        self,
        system_prompt: Optional[str] = None,
        latest_user_input: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Project stored history into model-compatible chat messages."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for message in self._history:
            projected = self._project_message_for_llm(message)
            if projected is not None:
                messages.append(projected)

        if latest_user_input is not None:
            messages.append({"role": "user", "content": latest_user_input})
        return messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": [message.to_dict() for message in self._history],
            "created_at": datetime.now().isoformat(),
            "rounds": self.estimate_rounds(),
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self._history = [Message.from_dict(item) for item in data.get("history", [])]

    def _project_message_for_llm(self, message: Message) -> Optional[Dict[str, str]]:
        """Map persisted roles into legal chat roles for the next model call."""
        if message.role in {"user", "assistant"}:
            return {"role": message.role, "content": message.content}
        if message.role == "summary":
            return {"role": "user", "content": f"{self.SUMMARY_NOTE_PREFIX}\n{message.content}"}
        if message.role == "system":
            return {"role": "user", "content": f"{self.SYSTEM_NOTE_PREFIX}\n{message.content}"}
        if message.role == "tool":
            return {"role": "assistant", "content": f"{self.TOOL_RESULT_PREFIX}\n{message.content}"}
        return None

    def _resolve_history(self, history: Optional[Sequence[Message]]) -> List[Message]:
        if history is None:
            return self._history
        if isinstance(history, list):
            return history
        return list(history)
