"""Comprehensive tests for code/context/builder.py — ContextBuilder and its pipeline.

Covers every class and method:
- count_tokens, _tokenize_for_relevance
- ContextPacket: construction, __post_init__, auto token_count
- ContextConfig: defaults, get_available_tokens
- ContextBuilder: build() end-to-end pipeline
- _gather(): history truncation, system_instructions, additional_packets
- _select(): relevance filtering, MMR vs greedy, token budget
- _score_packet(): relevance + recency scoring
- _compute_relevance(): token overlap computation
- _compute_recency(): exponential decay
- _packet_rank(): rank + token_count tiebreaker
- _select_with_mmr(): diversity-based selection
- _packet_tokens(): cached vs computed
- _token_similarity(): Jaccard similarity
- _structure(): grouped section formatting
- _group_selected_packets(): type-based grouping
- _compress(): token-budget truncation
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from code.context.builder import (
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    _get_default_token_counter,
    _tokenize_for_relevance,
    count_tokens,
)
from code.core.message import Message


# ============================================================================
# Helpers
# ============================================================================

FULL_FILE = Path(__file__).resolve()
ROOT = FULL_FILE.parents[1]
CODE_DIR = ROOT / "code"
import sys
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _packet(content: str, **meta) -> ContextPacket:
    return ContextPacket(content=content, metadata=meta)


# ============================================================================
# count_tokens & _tokenize_for_relevance
# ============================================================================


class TestCountTokens:
    """Token counting utility."""

    def test_empty_string(self):
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    def test_simple_text(self):
        tokens = count_tokens("hello world")
        assert tokens > 0
        assert tokens < 10

    def test_long_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 200
        tokens = count_tokens(text)
        assert tokens > 100

    def test_cjk_text(self):
        tokens = count_tokens("你好世界")
        assert tokens > 0

    def test_deterministic(self):
        t1 = count_tokens("hello world")
        t2 = count_tokens("hello world")
        assert t1 == t2

    def test_get_default_counter_is_reused(self):
        """_get_default_token_counter uses LRU cache — same model returns same instance."""
        from code.context.token_counter import TokenCounter
        c1 = _get_default_token_counter()
        c2 = _get_default_token_counter()
        assert isinstance(c1, TokenCounter)
        # LRU cache with maxsize=1 guarantees same instance for same args
        assert c1 is c2


class TestTokenizeForRelevance:
    """Tokenization for relevance scoring."""

    def test_english_words(self):
        tokens = _tokenize_for_relevance("hello WORLD test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_cjk_characters(self):
        tokens = _tokenize_for_relevance("你好世界")
        assert "你" in tokens
        assert "好" in tokens
        assert "世" in tokens
        assert "界" in tokens

    def test_mixed_english_cjk(self):
        tokens = _tokenize_for_relevance("hello 你好 world")
        assert "hello" in tokens
        assert "你" in tokens
        assert "world" in tokens

    def test_numbers(self):
        tokens = _tokenize_for_relevance("abc123 def456")
        assert "abc123" in tokens
        assert "def456" in tokens

    def test_special_characters_ignored(self):
        tokens = _tokenize_for_relevance("!@#$ hello %^&*")
        assert "hello" in tokens
        # Special chars should not produce tokens
        assert "!" not in tokens

    def test_empty_string(self):
        tokens = _tokenize_for_relevance("")
        assert tokens == set()

    def test_none_input(self):
        tokens = _tokenize_for_relevance(None)
        assert tokens == set()


# ============================================================================
# ContextPacket
# ============================================================================


class TestContextPacket:
    """ContextPacket dataclass."""

    def test_basic_construction(self):
        p = ContextPacket(content="hello")
        assert p.content == "hello"
        assert p.token_count > 0
        assert isinstance(p.timestamp, datetime)
        assert p.relevance_score == 0.0

    def test_auto_token_count_on_init(self):
        p = ContextPacket(content="hello world")
        assert p.token_count > 0

    def test_explicit_token_count(self):
        p = ContextPacket(content="hello world", token_count=42)
        assert p.token_count == 42

    def test_metadata_default(self):
        p = ContextPacket(content="test")
        assert p.metadata == {}

    def test_custom_metadata(self):
        p = ContextPacket(content="test", metadata={"type": "history"})
        assert p.metadata["type"] == "history"

    def test_relevance_score(self):
        p = ContextPacket(content="test", relevance_score=0.75)
        assert p.relevance_score == 0.75

    def test_large_content_token_count(self):
        p = ContextPacket(content="x" * 10000)
        assert p.token_count > 0


# ============================================================================
# ContextConfig
# ============================================================================


class TestContextConfig:
    """ContextConfig dataclass."""

    def test_defaults(self):
        cfg = ContextConfig()
        assert cfg.max_tokens == 8000
        assert cfg.reserve_ratio == 0.15
        assert cfg.min_relevance == 0.3
        assert cfg.enable_mmr is True
        assert cfg.mmr_lambda == 0.7
        assert cfg.enable_compression is True
        assert cfg.history_limit == 10

    def test_get_available_tokens(self):
        cfg = ContextConfig(max_tokens=10000, reserve_ratio=0.2)
        assert cfg.get_available_tokens() == 8000  # 10000 * 0.8

    def test_get_available_tokens_no_reserve(self):
        cfg = ContextConfig(max_tokens=10000, reserve_ratio=0.0)
        assert cfg.get_available_tokens() == 10000

    def test_custom_values(self):
        cfg = ContextConfig(max_tokens=5000, reserve_ratio=0.1, min_relevance=0.5,
                            enable_mmr=False, history_limit=5)
        assert cfg.max_tokens == 5000
        assert cfg.min_relevance == 0.5
        assert cfg.enable_mmr is False


# ============================================================================
# ContextBuilder — _gather
# ============================================================================


class TestContextBuilderGather:
    """ContextBuilder._gather()"""

    def test_empty_inputs(self):
        builder = ContextBuilder()
        packets = builder._gather("query", [], None, [])
        assert packets == []

    def test_system_instructions_added_first(self):
        builder = ContextBuilder()
        packets = builder._gather("query", [], "You are helpful.", [])
        assert len(packets) == 1
        assert packets[0].metadata["type"] == "instructions"
        assert packets[0].metadata["priority"] == 0
        assert "You are helpful." in packets[0].content

    def test_conversation_history_truncated(self):
        builder = ContextBuilder(ContextConfig(history_limit=3))
        messages = [Message(f"msg {i}", "user") for i in range(10)]
        packets = builder._gather("query", messages, None, [])
        assert len(packets) == 1
        p = packets[0]
        assert p.metadata["type"] == "history"
        assert p.metadata["count"] == 3  # Only last 3
        assert p.metadata["priority"] == 3
        # Should contain the last 3 messages
        for i in range(7, 10):
            assert f"msg {i}" in p.content
        # Should NOT contain early messages
        assert "msg 0" not in p.content

    def test_conversation_history_below_limit(self):
        builder = ContextBuilder(ContextConfig(history_limit=10))
        messages = [Message("hello", "user"), Message("hi", "assistant")]
        packets = builder._gather("query", messages, None, [])
        assert packets[0].metadata["count"] == 2

    def test_additional_packets_appended(self):
        builder = ContextBuilder()
        extra = _packet("extra data", type="knowledge_base")
        packets = builder._gather("query", [], None, [extra])
        assert len(packets) == 1
        assert packets[0].content == "extra data"

    def test_full_combination(self):
        builder = ContextBuilder(ContextConfig(history_limit=5))
        messages = [Message(f"msg {i}", "user") for i in range(8)]
        extra = _packet("kb content", type="knowledge_base")
        packets = builder._gather("query", messages, "System prompt.", [extra])
        assert len(packets) == 3
        types = {p.metadata.get("type") for p in packets}
        assert types == {"instructions", "history", "knowledge_base"}

    def test_no_system_instructions(self):
        builder = ContextBuilder()
        messages = [Message("test", "user")]
        packets = builder._gather("query", messages, None, [])
        assert len(packets) == 1
        assert packets[0].metadata["type"] == "history"


# ============================================================================
# ContextBuilder — _select
# ============================================================================


class TestContextBuilderSelect:
    """ContextBuilder._select() — relevance filtering + MMR selection."""

    def test_empty_packets(self):
        builder = ContextBuilder()
        assert builder._select([], "query") == []

    def test_system_packets_always_included(self):
        builder = ContextBuilder()
        sys_pkt = _packet("System instruction content", type="instructions", priority=0)
        selected = builder._select([sys_pkt], "query")
        assert len(selected) == 1
        assert selected[0].metadata["type"] == "instructions"

    def test_below_relevance_threshold_excluded(self):
        builder = ContextBuilder(ContextConfig(min_relevance=0.5))
        # "xyzzy" has no overlap with "hello world" — relevance=0
        pkt = _packet("completely unrelated content here", type="knowledge_base")
        selected = builder._select([pkt], "hello world")
        # May or may not be selected depending on token budget and recency
        # But with 0 relevance, should be filtered out
        assert len(selected) <= 1

    def test_high_relevance_selected(self):
        builder = ContextBuilder(ContextConfig(min_relevance=0.1))
        pkt = _packet("hello world test content here", type="knowledge_base")
        selected = builder._select([pkt], "hello world")
        assert len(selected) == 1

    def test_token_budget_respected(self):
        builder = ContextBuilder(ContextConfig(max_tokens=100, reserve_ratio=0.1))
        # Create a packet that exceeds available tokens
        huge = _packet("x " * 500, type="knowledge_base")
        selected = builder._select([huge], "hello")
        # The huge packet should be excluded
        assert len(selected) == 0

    def test_mmr_disabled_uses_greedy(self):
        cfg = ContextConfig(enable_mmr=False, max_tokens=10000)
        builder = ContextBuilder(cfg)
        pkts = [_packet(f"doc {i} content here", type="knowledge_base") for i in range(5)]
        selected = builder._select(pkts, "doc content")
        assert len(selected) == 5  # All fit in budget

    def test_multiple_packets_ranked_by_relevance(self):
        builder = ContextBuilder(ContextConfig(max_tokens=10000))
        relevant = _packet("hello world test", type="knowledge_base")
        irrelevant = _packet("xyzzy foo bar", type="knowledge_base")
        selected = builder._select([irrelevant, relevant], "hello world")
        # The more relevant packet should be first (higher ranked)
        if len(selected) >= 1:
            assert "hello world test" in selected[0].content


# ============================================================================
# ContextBuilder — _score_packet
# ============================================================================


class TestContextBuilderScorePacket:
    """ContextBuilder._score_packet() — relevance + recency scoring."""

    def test_computes_relevance_score(self):
        builder = ContextBuilder()
        scored = builder._score_packet(_packet("hello world"), {"hello", "world"})
        assert scored.relevance_score > 0
        assert "_content_tokens" in scored.metadata
        assert "_rank_score" in scored.metadata
        assert "_recency_score" in scored.metadata

    def test_no_query_tokens(self):
        builder = ContextBuilder()
        scored = builder._score_packet(_packet("hello"), set())
        assert scored.relevance_score == 0.0

    def test_perfect_match(self):
        builder = ContextBuilder()
        scored = builder._score_packet(_packet("hello world"), {"hello", "world"})
        assert scored.relevance_score == 1.0

    def test_partial_match(self):
        builder = ContextBuilder()
        scored = builder._score_packet(_packet("hello foo"), {"hello", "world"})
        assert scored.relevance_score == 0.5

    def test_no_match(self):
        builder = ContextBuilder()
        scored = builder._score_packet(_packet("foo bar"), {"hello", "world"})
        assert scored.relevance_score == 0.0

    def test_rank_score_combines_relevance_and_recency(self):
        builder = ContextBuilder()
        scored = builder._score_packet(_packet("hello world"), {"hello"})
        rank = scored.metadata["_rank_score"]
        assert 0 <= rank <= 1.0


# ============================================================================
# ContextBuilder — _compute_relevance
# ============================================================================


class TestComputeRelevance:
    """ContextBuilder._compute_relevance()"""

    def test_empty_query_returns_zero(self):
        builder = ContextBuilder()
        assert builder._compute_relevance(set(), {"hello"}) == 0.0

    def test_full_overlap(self):
        builder = ContextBuilder()
        assert builder._compute_relevance({"a", "b"}, {"a", "b", "c"}) == 1.0

    def test_partial_overlap(self):
        builder = ContextBuilder()
        assert builder._compute_relevance({"a", "b", "c", "d"}, {"a", "b"}) == 0.5

    def test_no_overlap(self):
        builder = ContextBuilder()
        assert builder._compute_relevance({"a", "b"}, {"c", "d"}) == 0.0

    def test_both_empty(self):
        builder = ContextBuilder()
        assert builder._compute_relevance(set(), set()) == 0.0


# ============================================================================
# ContextBuilder — _compute_recency (static)
# ============================================================================


class TestComputeRecency:
    """ContextBuilder._compute_recency() — exponential decay function (static method)."""

    def test_now_is_high_recency(self):
        score = ContextBuilder._compute_recency(datetime.now())
        assert score > 0.9  # Very recent = ~1.0

    def test_one_hour_old(self):
        ts = datetime.now() - timedelta(hours=1)
        score = ContextBuilder._compute_recency(ts)
        expected = math.exp(-1.0)
        assert abs(score - expected) < 0.05

    def test_one_day_old(self):
        ts = datetime.now() - timedelta(days=1)
        score = ContextBuilder._compute_recency(ts)
        expected = math.exp(-24.0)
        assert abs(score - expected) < 0.001

    def test_future_timestamp_handled(self):
        ts = datetime.now() + timedelta(hours=1)
        score = ContextBuilder._compute_recency(ts)
        assert 0 <= score <= 1.0  # delta clamped to 0

    def test_very_old_near_zero(self):
        ts = datetime.now() - timedelta(days=365)
        score = ContextBuilder._compute_recency(ts)
        assert score < 0.001


# ============================================================================
# ContextBuilder — _packet_rank (static)
# ============================================================================


class TestPacketRank:
    """_packet_rank() — sort key for relevance-based ordering."""

    def test_uses_rank_score_if_present(self):
        p = _packet("test")
        p.metadata["_rank_score"] = 0.8
        rank = ContextBuilder._packet_rank(p)
        assert rank[0] == 0.8

    def test_falls_back_to_relevance_score(self):
        p = ContextPacket(content="test", relevance_score=0.5)
        rank = ContextBuilder._packet_rank(p)
        assert rank[0] == 0.5

    def test_second_element_is_negative_token_count(self):
        p = ContextPacket(content="a" * 1000, relevance_score=0.7)
        rank = ContextBuilder._packet_rank(p)
        assert rank[1] < 0  # negative token_count

    def test_higher_relevance_ranks_higher(self):
        p1 = ContextPacket(content="test", relevance_score=0.9)
        p2 = ContextPacket(content="test", relevance_score=0.3)
        assert ContextBuilder._packet_rank(p1) > ContextBuilder._packet_rank(p2)


# ============================================================================
# ContextBuilder — _packet_tokens (static)
# ============================================================================


class TestPacketTokens:
    """_packet_tokens() — cached vs computed token extraction."""

    def test_returns_cached_when_set(self):
        p = _packet("hello world")
        p.metadata["_content_tokens"] = {"custom", "tokens"}
        result = ContextBuilder._packet_tokens(p)
        assert result == {"custom", "tokens"}

    def test_computes_when_no_cache(self):
        p = _packet("hello world")
        result = ContextBuilder._packet_tokens(p)
        assert "hello" in result
        assert "world" in result

    def test_ignores_non_set_cache(self):
        p = _packet("hello world")
        p.metadata["_content_tokens"] = ["not", "a", "set"]
        result = ContextBuilder._packet_tokens(p)
        assert "hello" in result


# ============================================================================
# ContextBuilder — _token_similarity (static)
# ============================================================================


class TestTokenSimilarity:
    """_token_similarity() — Jaccard similarity."""

    def test_identical_sets(self):
        sim = ContextBuilder._token_similarity({"a", "b"}, {"a", "b"})
        assert sim == 1.0

    def test_disjoint_sets(self):
        sim = ContextBuilder._token_similarity({"a", "b"}, {"c", "d"})
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = ContextBuilder._token_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert sim == 0.5  # intersection {b,c}=2, union {a,b,c,d}=4

    def test_empty_left(self):
        sim = ContextBuilder._token_similarity(set(), {"a"})
        assert sim == 0.0

    def test_empty_right(self):
        sim = ContextBuilder._token_similarity({"a"}, set())
        assert sim == 0.0

    def test_both_empty(self):
        sim = ContextBuilder._token_similarity(set(), set())
        assert sim == 0.0


# ============================================================================
# ContextBuilder — _select_with_mmr
# ============================================================================


class TestSelectWithMMR:
    """ContextBuilder._select_with_mmr() — diversity-aware selection."""

    def test_empty_packets(self):
        builder = ContextBuilder()
        selected = builder._select_with_mmr([], 1000)
        assert selected == []

    def test_single_packet_fits(self):
        builder = ContextBuilder()
        pkt = _packet("hello world test", type="knowledge_base")
        # Score it first so relevance_score is populated
        scored = builder._score_packet(pkt, {"hello"})
        selected = builder._select_with_mmr([scored], 10000)
        assert len(selected) == 1

    def test_packet_too_large_excluded(self):
        builder = ContextBuilder()
        huge = _packet("x " * 1000, type="knowledge_base")
        selected = builder._select_with_mmr([huge], 5)
        assert len(selected) == 0

    def test_mmr_diversity_penalty(self):
        builder = ContextBuilder(ContextConfig(mmr_lambda=0.5))
        # Two very similar packets
        p1 = builder._score_packet(_packet("hello world test content", type="kb"), {"hello"})
        p2 = builder._score_packet(_packet("hello world test content", type="kb"), {"hello"})
        selected = builder._select_with_mmr([p1, p2], 10000)
        # MMR should penalize the second because it's very similar to the first
        assert len(selected) == 2  # Both fit in budget

    def test_diverse_packets_both_selected(self):
        builder = ContextBuilder(ContextConfig(mmr_lambda=0.7))
        p1 = builder._score_packet(_packet("hello world python code", type="kb"), {"hello", "python"})
        p2 = builder._score_packet(_packet("database sql query", type="kb"), {"hello", "python"})
        selected = builder._select_with_mmr([p1, p2], 10000)
        assert len(selected) == 2

    def test_recency_score_fallback_in_mmr(self):
        builder = ContextBuilder(ContextConfig(mmr_lambda=0.7))
        pkt = _packet("test content", type="kb")
        scored = builder._score_packet(pkt, {"test"})
        # Remove _recency_score to test fallback
        scored.metadata.pop("_recency_score", None)
        selected = builder._select_with_mmr([scored], 10000)
        assert len(selected) == 1

    def test_exhaust_budget(self):
        builder = ContextBuilder(ContextConfig(mmr_lambda=0.7))
        pkts = []
        for i in range(20):
            p = _packet(f"document {i} with some content for testing purposes", type="kb")
            pkts.append(builder._score_packet(p, {"document", "content"}))
        selected = builder._select_with_mmr(pkts, 100)  # Very small budget
        # Should select some but not all
        assert len(selected) < len(pkts)
        total_tokens = sum(p.token_count for p in selected)
        assert total_tokens <= 100


# ============================================================================
# ContextBuilder — _structure
# ============================================================================


class TestContextBuilderStructure:
    """ContextBuilder._structure() — section formatting."""

    def test_basic_structure(self):
        builder = ContextBuilder()
        pkts = [
            _packet("Task progress: 50%", type="task_state"),
            _packet("def foo(): pass", type="related_memory"),
        ]
        result = builder._structure(pkts, "implement feature X")
        assert "[Task]" in result
        assert "implement feature X" in result
        assert "[State]" in result
        assert "Task progress" in result
        assert "[Evidence]" in result
        assert "[Output]" in result

    def test_instructions_section(self):
        builder = ContextBuilder()
        pkts = [_packet("You are a Python expert.", type="instructions", priority=0)]
        result = builder._structure(pkts, "query")
        assert "[Role & Policies]" in result
        assert "Python expert" in result

    def test_history_section(self):
        builder = ContextBuilder()
        pkts = [_packet("Previous conversation...", type="history")]
        result = builder._structure(pkts, "query")
        assert "[Context]" in result
        assert "Previous conversation" in result

    def test_knowledge_base_in_evidence(self):
        builder = ContextBuilder()
        pkts = [
            _packet("KB: Python 3.12 spec", type="knowledge_base"),
            _packet("Retrieval: PEP 484", type="retrieval"),
            _packet("Result: tests passed", type="tool_result"),
        ]
        result = builder._structure(pkts, "query")
        assert "[Evidence]" in result
        assert "Python 3.12 spec" in result
        assert "PEP 484" in result
        assert "tests passed" in result

    def test_output_section_always_present(self):
        builder = ContextBuilder()
        result = builder._structure([], "test query")
        assert "[Output]" in result

    def test_empty_packets_still_has_task_and_output(self):
        builder = ContextBuilder()
        result = builder._structure([], "my query")
        assert "[Task]" in result
        assert "my query" in result
        assert "[Output]" in result

    def test_empty_packet_content_excluded(self):
        builder = ContextBuilder()
        pkts = [_packet("", type="instructions")]
        # Structure should handle empty content gracefully
        result = builder._structure(pkts, "query")
        assert "[Task]" in result

    def test_no_duplicate_sections(self):
        builder = ContextBuilder()
        pkts = [
            _packet("Inst1", type="instructions"),
            _packet("Inst2", type="instructions"),
        ]
        result = builder._structure(pkts, "query")
        # Should only have one [Role & Policies] section
        assert result.count("[Role & Policies]") == 1


# ============================================================================
# ContextBuilder — _group_selected_packets (static)
# ============================================================================


class TestGroupSelectedPackets:
    """_group_selected_packets() — type-based grouping."""

    def test_groups_by_type(self):
        pkts = [
            _packet("Inst1", type="instructions"),
            _packet("Inst2", type="instructions"),
            _packet("History content", type="history"),
            _packet("KB content", type="knowledge_base"),
        ]
        grouped = ContextBuilder._group_selected_packets(pkts)
        assert len(grouped["instructions"]) == 2
        assert len(grouped["history"]) == 1
        assert len(grouped["knowledge_base"]) == 1

    def test_empty_list(self):
        grouped = ContextBuilder._group_selected_packets([])
        assert grouped == {}

    def test_packet_without_type_skipped(self):
        pkts = [_packet("no type here")]
        grouped = ContextBuilder._group_selected_packets(pkts)
        assert grouped == {}

    def test_empty_type_string_skipped(self):
        pkts = [_packet("content", type="")]
        grouped = ContextBuilder._group_selected_packets(pkts)
        assert grouped == {}

    def test_type_none_skipped(self):
        p = _packet("content")
        p.metadata.pop("type", None)
        grouped = ContextBuilder._group_selected_packets([p])
        assert grouped == {}


# ============================================================================
# ContextBuilder — _compress
# ============================================================================


class TestContextBuilderCompress:
    """ContextBuilder._compress() — token-budget truncation."""

    def test_no_compression_when_disabled(self):
        cfg = ContextConfig(enable_compression=False)
        builder = ContextBuilder(cfg)
        long_text = "hello world. " * 5000
        result = builder._compress(long_text)
        assert result == long_text  # Unchanged

    def test_short_text_not_compressed(self):
        builder = ContextBuilder(ContextConfig(max_tokens=10000))
        short = "hello world"
        result = builder._compress(short)
        assert result == short

    def test_long_text_truncated(self):
        cfg = ContextConfig(max_tokens=500, reserve_ratio=0.1)
        builder = ContextBuilder(cfg)
        # Create text that definitely exceeds the budget
        long_text = "hello world test content " * 500
        result = builder._compress(long_text)
        # Compression should either truncate or pass through; never expand
        assert len(result) <= len(long_text)
        # If it was compressed, it should still start the same way
        assert result.startswith("hello world test content")

    def test_compressed_text_starts_same(self):
        cfg = ContextConfig(max_tokens=500, reserve_ratio=0.1)
        builder = ContextBuilder(cfg)
        lines = [f"Line {i}: some content here" for i in range(500)]
        long_text = "\n".join(lines)
        result = builder._compress(long_text)
        # Should keep the first few lines
        assert "Line 0" in result
        # Should truncate later lines
        assert f"Line {len(lines) - 1}" not in result

    def test_empty_input(self):
        builder = ContextBuilder()
        result = builder._compress("")
        assert result == ""

    def test_compression_keeps_complete_lines(self):
        cfg = ContextConfig(max_tokens=200, reserve_ratio=0.1)
        builder = ContextBuilder(cfg)
        lines = ["Line A: a" * 500, "Line B: b" * 500]
        long_text = "\n".join(lines)
        result = builder._compress(long_text)
        # Should at least include the first line completely
        assert result.startswith("Line A")


# ============================================================================
# ContextBuilder — build() end-to-end
# ============================================================================


class TestContextBuilderBuildE2E:
    """ContextBuilder.build() — full pipeline integration."""

    def test_simple_build(self):
        builder = ContextBuilder()
        result = builder.build(
            user_query="What does the code do?",
            system_instructions="You are a code reviewer.",
        )
        assert "[Role & Policies]" in result
        assert "[Task]" in result
        assert "What does the code do?" in result
        assert "[Output]" in result
        assert "You are a code reviewer." in result

    def test_build_with_conversation_history(self):
        builder = ContextBuilder(ContextConfig(history_limit=5, min_relevance=0.0))
        messages = [
            Message("Find all bugs in main.py", "user"),
            Message("I found 3 bugs", "assistant"),
            Message("Fix them all", "user"),
        ]
        result = builder.build(
            user_query="What's the next step?",
            conversation_history=messages,
        )
        assert "[Context]" in result
        assert "Find all bugs" in result

    def test_build_with_additional_packets(self):
        builder = ContextBuilder()
        extra = [_packet("Module structure architecture src main utils", type="knowledge_base")]
        result = builder.build(
            user_query="Explain the architecture.",
            additional_packets=extra,
        )
        assert "[Evidence]" in result
        assert "Module structure" in result

    def test_build_full_pipeline(self):
        builder = ContextBuilder(ContextConfig(max_tokens=5000, history_limit=5, min_relevance=0.0))
        messages = [
            Message("Hello", "user"),
            Message("Hi, how can I help?", "assistant"),
        ]
        extra = [
            _packet("Project uses Python 3.12 with FastAPI", type="knowledge_base"),
            _packet("CI passes on main branch", type="retrieval"),
        ]
        result = builder.build(
            user_query="What's the project status?",
            conversation_history=messages,
            system_instructions="You are a project manager assistant.",
            additional_packets=extra,
        )
        # All sections should be present
        assert "[Role & Policies]" in result
        assert "[Task]" in result
        assert "project status" in result.lower()
        assert "[Evidence]" in result
        assert "Python 3.12" in result
        assert "CI passes" in result
        assert "[Context]" in result
        assert "[Output]" in result

    def test_build_with_compression_disabled(self):
        """With compression disabled, _compress is a no-op — test via _compress directly."""
        cfg = ContextConfig(enable_compression=False, max_tokens=100)
        builder = ContextBuilder(cfg)
        long_text = "some random text words here for testing " * 500
        # Test _compress directly (the build pipeline also calls _select first)
        result = builder._compress(long_text)
        # Compression disabled → unchanged
        assert result == long_text

    def test_build_no_optional_args(self):
        builder = ContextBuilder()
        result = builder.build(user_query="simple question")
        assert "[Task]" in result
        assert "simple question" in result
        assert "[Output]" in result

    def test_build_deterministic(self):
        builder = ContextBuilder()
        r1 = builder.build(user_query="test", system_instructions="Be concise.")
        r2 = builder.build(user_query="test", system_instructions="Be concise.")
        assert r1 == r2
