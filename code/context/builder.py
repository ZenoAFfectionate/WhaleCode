"""Structured context building helpers.

The builder follows a small Gather-Select-Structure-Compress pipeline and is
kept intentionally lightweight so it can be reused by agents without forcing
extra model calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.message import Message

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
_DEFAULT_OUTPUT_SECTION = "\n".join(
    [
        "[Output]",
        "Please answer in the following format:",
        "1. Conclusion (concise and clear)",
        "2. Basis (list supporting evidence and sources)",
        "3. Risks and Assumptions (if any)",
        "4. Next Steps (if applicable)",
    ]
)


@lru_cache(maxsize=1)
def _get_default_encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Estimate token count using ``tiktoken`` when available."""
    payload = text or ""
    if not payload:
        return 0

    encoding = _get_default_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(payload))
        except Exception:
            pass
    return max(1, len(payload) // 4)


def _tokenize_for_relevance(text: str) -> Set[str]:
    """Tokenize English words and individual CJK characters for simple relevance scoring."""
    return {token.lower() for token in _TOKEN_PATTERN.findall(text or "")}


@dataclass
class ContextPacket:
    """A candidate context block used by the builder."""

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """Configuration for the lightweight context builder."""

    max_tokens: int = 8000
    reserve_ratio: float = 0.15
    min_relevance: float = 0.3
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    enable_compression: bool = True
    history_limit: int = 10

    def get_available_tokens(self) -> int:
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """Build a structured context string from history and auxiliary packets."""

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None,
    ) -> str:
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or [],
        )
        selected_packets = self._select(packets, user_query)
        structured_context = self._structure(selected_packets=selected_packets, user_query=user_query)
        return self._compress(structured_context)

    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket],
    ) -> List[ContextPacket]:
        packets: List[ContextPacket] = []

        if system_instructions:
            packets.append(
                ContextPacket(content=system_instructions, metadata={"type": "instructions", "priority": 0})
            )

        if conversation_history:
            recent_history = conversation_history[-self.config.history_limit :]
            history_text = "\n".join(message.to_text() for message in recent_history)
            packets.append(
                ContextPacket(
                    content=history_text,
                    metadata={"type": "history", "count": len(recent_history), "priority": 3},
                )
            )

        packets.extend(additional_packets)
        return packets

    def _select(self, packets: List[ContextPacket], user_query: str) -> List[ContextPacket]:
        if not packets:
            return []

        query_tokens = _tokenize_for_relevance(user_query)
        scored_packets = [self._score_packet(packet, query_tokens) for packet in packets]

        system_packets = [packet for packet in scored_packets if packet.metadata.get("type") == "instructions"]
        candidate_packets = [
            packet
            for packet in scored_packets
            if packet.metadata.get("type") != "instructions" and packet.relevance_score >= self.config.min_relevance
        ]
        candidate_packets.sort(key=self._packet_rank, reverse=True)

        selected: List[ContextPacket] = []
        used_tokens = 0
        available_tokens = self.config.get_available_tokens()

        for packet in system_packets:
            if used_tokens + packet.token_count > available_tokens:
                continue
            selected.append(packet)
            used_tokens += packet.token_count

        remaining_budget = max(0, available_tokens - used_tokens)
        if self.config.enable_mmr:
            selected.extend(self._select_with_mmr(candidate_packets, remaining_budget, query_tokens))
        else:
            for packet in candidate_packets:
                if packet.token_count > remaining_budget:
                    continue
                selected.append(packet)
                remaining_budget -= packet.token_count

        return selected

    def _score_packet(self, packet: ContextPacket, query_tokens: Set[str]) -> ContextPacket:
        content_tokens = _tokenize_for_relevance(packet.content)
        relevance = self._compute_relevance(query_tokens, content_tokens)
        score = 0.7 * relevance + 0.3 * self._compute_recency(packet.timestamp)
        return replace(
            packet,
            relevance_score=relevance,
            metadata={**packet.metadata, "_content_tokens": content_tokens, "_rank_score": score},
        )

    def _compute_relevance(self, query_tokens: Set[str], content_tokens: Set[str]) -> float:
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        return overlap / len(query_tokens)

    @staticmethod
    def _compute_recency(timestamp: datetime) -> float:
        delta_seconds = max((datetime.now() - timestamp).total_seconds(), 0.0)
        return math.exp(-delta_seconds / 3600.0)

    @staticmethod
    def _packet_rank(packet: ContextPacket) -> Tuple[float, float]:
        rank_score = packet.metadata.get("_rank_score", packet.relevance_score)
        return (rank_score, -packet.token_count)

    def _select_with_mmr(
        self,
        packets: Sequence[ContextPacket],
        available_tokens: int,
        query_tokens: Set[str],
    ) -> List[ContextPacket]:
        selected: List[ContextPacket] = []
        selected_token_sets: List[Set[str]] = []
        remaining = list(packets)

        while remaining and available_tokens > 0:
            best_index = -1
            best_score = float("-inf")
            for index, packet in enumerate(remaining):
                if packet.token_count > available_tokens:
                    continue

                content_tokens = self._packet_tokens(packet)
                diversity_penalty = 0.0
                if selected_token_sets:
                    diversity_penalty = max(
                        self._token_similarity(content_tokens, selected_tokens)
                        for selected_tokens in selected_token_sets
                    )

                relevance = self._compute_relevance(query_tokens, content_tokens)
                mmr_score = self.config.mmr_lambda * relevance - (1 - self.config.mmr_lambda) * diversity_penalty
                mmr_score += 0.1 * self._compute_recency(packet.timestamp)

                if mmr_score > best_score:
                    best_index = index
                    best_score = mmr_score

            if best_index < 0:
                break

            chosen = remaining.pop(best_index)
            selected.append(chosen)
            chosen_tokens = self._packet_tokens(chosen)
            selected_token_sets.append(chosen_tokens)
            available_tokens -= chosen.token_count

        return selected

    @staticmethod
    def _packet_tokens(packet: ContextPacket) -> Set[str]:
        cached = packet.metadata.get("_content_tokens")
        if isinstance(cached, set):
            return cached
        return _tokenize_for_relevance(packet.content)

    @staticmethod
    def _token_similarity(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        union = len(left | right)
        return intersection / union if union else 0.0

    def _structure(self, selected_packets: List[ContextPacket], user_query: str) -> str:
        sections: List[str] = []

        instructions = [packet.content for packet in selected_packets if packet.metadata.get("type") == "instructions"]
        if instructions:
            sections.append("[Role & Policies]\n" + "\n".join(instructions))

        sections.append(f"[Task]\nUser Query: {user_query}")

        task_state = [packet.content for packet in selected_packets if packet.metadata.get("type") == "task_state"]
        if task_state:
            sections.append("[State]\nKey Progress and Pending Issues:\n" + "\n".join(task_state))

        evidence = [
            packet.content
            for packet in selected_packets
            if packet.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if evidence:
            sections.append("[Evidence]\nFacts and References:\n" + "\n\n".join(evidence))

        history = [packet.content for packet in selected_packets if packet.metadata.get("type") == "history"]
        if history:
            sections.append("[Context]\nConversation History and Background:\n" + "\n".join(history))

        sections.append(_DEFAULT_OUTPUT_SECTION)
        return "\n\n".join(section.strip() for section in sections if section.strip())

    def _compress(self, context: str) -> str:
        if not self.config.enable_compression:
            return context

        available_tokens = self.config.get_available_tokens()
        if count_tokens(context) <= available_tokens:
            return context

        kept_lines: List[str] = []
        used_tokens = 0
        for line in context.splitlines():
            line_tokens = count_tokens(line)
            if kept_lines and used_tokens + line_tokens > available_tokens:
                break
            kept_lines.append(line)
            used_tokens += line_tokens
        return "\n".join(kept_lines)
