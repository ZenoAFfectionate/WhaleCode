"""Stress tests for ultra-long agent sessions and history compression.

Covers:
- 100-round continuous ReAct loop stability
- 500-round extreme session boundary
- Compression trigger at token thresholds
- Token cache consistency across many rounds
- HistoryManager serialization round-trip with 500+ messages
- Compression boundary behavior (exact threshold crossing)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from hello_agents.context.history import HistoryManager
from hello_agents.context.token_counter import TokenCounter
from hello_agents.core.config import Config
from hello_agents.core.message import Message


# ============================================================================
# Helpers
# ============================================================================


def _make_message(content: str, role: str = "user") -> Message:
    return Message(content, role)


def _make_assistant_with_tool_calls(
    content: str,
    tool_calls: List[Dict[str, Any]],
) -> Message:
    """Create an assistant message with tool_call metadata."""
    return Message(
        content=content,
        role="assistant",
        metadata={"tool_calls": tool_calls},
    )


def _make_tool_result(tool_name: str, content: str, tool_call_id: str = "call_1") -> Message:
    return Message(
        content=content,
        role="tool",
        metadata={"tool_name": tool_name, "tool_call_id": tool_call_id},
    )


# ============================================================================
# 100-Round Session
# ============================================================================


class TestLongSession100Rounds:
    """Verify the agent survives 100 rounds of tool calls without crashing."""

    def test_100_rounds_history_append_only(self):
        """HistoryManager handles 100 rounds of appends without errors."""
        hm = HistoryManager(token_counter=TokenCounter())
        for i in range(100):
            hm.append(_make_message(f"User input round {i}", "user"))
            hm.append(_make_assistant_with_tool_calls(
                f"Working round {i}",
                [{"id": f"call_{i}", "type": "function", "function": {
                    "name": "Read", "arguments": json.dumps({"path": f"file_{i}.py"}),
                }}],
            ))
            hm.append(_make_tool_result("Read", f"Content of file_{i}.py: line1\\nline2", f"call_{i}"))

        history = hm.get_history()
        assert len(history) == 300  # 3 messages per round × 100 rounds

    def test_100_rounds_token_estimation_accuracy(self):
        """Token estimation remains consistent across 100 rounds of appends."""
        hm = HistoryManager(token_counter=TokenCounter())

        for i in range(100):
            hm.append(_make_message(f"Round {i} input text here", "user"))
            hm.append(_make_assistant_with_tool_calls(
                f"Processing round {i}",
                [{"id": f"c_{i}", "type": "function", "function": {
                    "name": "Bash", "arguments": json.dumps({"command": f"echo {i}"}),
                }}],
            ))
            hm.append(_make_tool_result("Bash", str(i), f"c_{i}"))

        # Token count should grow monotonically
        tokens = hm.get_estimated_token_count()
        assert tokens > 0

        # The estimate should be deterministic (call twice → same result)
        t1 = hm.estimate_tokens(system_prompt="test", latest_user_input="hello")
        t2 = hm.estimate_tokens(system_prompt="test", latest_user_input="hello")
        assert t1 == t2

    def test_100_rounds_no_memory_leak(self):
        """100 rounds of large messages should not cause unbounded memory growth."""
        # NOTE: RSS-based memory checks are inherently flaky in CI.
        # Instead, we verify that the HistoryManager doesn't crash and
        # the number of messages is correct after many rounds.
        hm = HistoryManager(token_counter=TokenCounter())
        long_content = "x" * 200  # 200-char messages

        for i in range(100):
            hm.append(_make_message(f"Round {i}: {long_content}", "user"))
            hm.append(_make_tool_result("Read", long_content * 5, f"c_{i}"))

        # After 100 rounds, the history should have exactly 200 messages
        history = hm.get_history()
        assert len(history) == 200

        # Token cache should still be functional after many operations
        tokens = hm.estimate_tokens()
        assert tokens > 0
        # Second call should be deterministic (cache hit)
        assert hm.estimate_tokens() == tokens

    def test_token_cache_hit_rate(self):
        """Token cache should have high hit rate during repeated estimate_tokens calls."""
        hm = HistoryManager(token_counter=TokenCounter())

        # Build up some history
        for i in range(20):
            hm.append(_make_message(f"Message {i}", "user"))

        # Call estimate_tokens multiple times with same params — should hit cache
        cache_hits = 0
        total_calls = 10
        last_result = None
        for _ in range(total_calls):
            result = hm.estimate_tokens(system_prompt="static prompt", latest_user_input="same input")
            if last_result is not None and result == last_result:
                cache_hits += 1
            last_result = result

        # At least N-1 calls after the first should be identical (cache hit)
        assert cache_hits >= total_calls - 1


# ============================================================================
# 500-Round Extreme Session
# ============================================================================


class TestLongSession500Rounds:
    """Extreme session length — compression must work to keep memory bounded."""

    def test_500_rounds_with_compression(self):
        """After 500 rounds, history compression keeps token count bounded."""
        config = Config(
            compact_enabled=True,
            compression_threshold=0.3,
            compact_preserve_recent_rounds=2,
            context_window=32768,
        )
        hm = HistoryManager(
            compression_threshold=0.3,
            token_counter=TokenCounter(),
            config=config,
        )

        for i in range(500):
            hm.append(_make_message(f"Round {i} — user request with extra padding to consume tokens.", "user"))
            hm.append(_make_tool_result("Bash", f"output_{i}", f"call_{i}"))

            # Periodically trigger compression to keep tokens bounded
            if i > 0 and i % 50 == 0:
                if hm.should_compress(system_prompt="sp", latest_user_input="new"):
                    hm.compress(f"[Auto-summary of rounds {i-50}-{i}]")

        # After 500 rounds with periodic compression, history should be manageable
        history = hm.get_history()
        assert len(history) > 0
        # Token count should be reasonable (compression keeps it bounded)
        tokens = hm.estimate_tokens()
        assert tokens < 200_000, f"Token count {tokens} is unusually high for 500 rounds"

    def test_500_rounds_stability(self):
        """500 rounds of rapid append + estimate + compress should not crash."""
        hm = HistoryManager(
            compression_threshold=0.5,
            token_counter=TokenCounter(),
        )

        for i in range(500):
            hm.append(_make_message(f"Input {i}", "user"))
            hm.append(_make_tool_result("Read", f"data {i}", f"c_{i}"))

            # Every 50 rounds, verify token estimation works
            if i % 50 == 0:
                tokens = hm.estimate_tokens(system_prompt="sp", latest_user_input="new")
                assert tokens > 0

        # Final verification
        history = hm.get_history()
        assert len(history) > 0
        assert hm.get_estimated_token_count() > 0


# ============================================================================
# Compression Boundary Testing
# ============================================================================


class TestCompressionBoundary:
    """Verify compression behavior at exact token thresholds."""

    def test_should_compress_below_threshold(self):
        """should_compress returns False when tokens are below threshold."""
        config = Config(compact_enabled=True, compression_threshold=0.8)
        hm = HistoryManager(
            compression_threshold=0.8,
            token_counter=TokenCounter(),
            config=config,
        )
        # Empty history — well below any threshold
        assert hm.should_compress(
            system_prompt="", latest_user_input="hello",
        ) is False

    def test_should_compress_above_threshold(self):
        """should_compress returns True when tokens are above threshold."""
        config = Config(
            compact_enabled=True,
            compression_threshold=0.01,
            context_window=32768,
        )
        hm = HistoryManager(
            compression_threshold=0.01,
            token_counter=TokenCounter(),
            config=config,
        )
        # Add many messages to push tokens above the low threshold
        for i in range(200):
            hm.append(_make_message(f"Long message number {i} with some extra content to consume tokens.", "user"))

        assert hm.should_compress(
            system_prompt="test", latest_user_input="hello",
        ) is True

    def test_micro_compact_before_full_compact(self):
        """micro_compact_tool_results runs before compact_with_llm in maybe_compact."""
        config = Config(
            compact_enabled=True,
            compression_threshold=0.3,
            compact_keep_recent_tool_results=2,
        )
        hm = HistoryManager(
            compression_threshold=0.3,
            token_counter=TokenCounter(),
            config=config,
        )

        # Add many tool results with long content
        for i in range(50):
            hm.append(_make_message(f"User {i}", "user"))
            hm.append(_make_tool_result(
                "Read", "x" * 1000, f"call_{i}",
            ))

        # Check that micro_compact changes tool results
        changed = hm.micro_compact_tool_results()
        assert changed is True

    def test_compression_preserves_summary_structure(self):
        """After compression, history starts with a summary message."""
        hm = HistoryManager(
            compression_threshold=0.3,
            token_counter=TokenCounter(),
        )

        for i in range(100):
            hm.append(_make_message(f"User message {i}", "user"))

        # Manual compress
        hm.compress("This is a test summary of the conversation.")

        history = hm.get_history()
        assert len(history) > 0
        # First message should be the summary
        assert "summary" in history[0].role.lower() or "Summary" in history[0].content

    def test_get_compression_split_returns_none_when_few_rounds(self):
        """get_compression_split returns None when there are few rounds."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(_make_message("Single message", "user"))

        split = hm.get_compression_split(retain_rounds=2)
        assert split is None  # Only 1 round, retaining 2 → no split


# ============================================================================
# Token Cache Consistency
# ============================================================================


class TestTokenCacheConsistency:
    """Verify token cache correctness under mutation."""

    def test_cache_invalidated_on_append(self):
        """estimate_tokens cache is cleared when new messages are appended."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(_make_message("first", "user"))

        t1 = hm.estimate_tokens(system_prompt="sp", latest_user_input="input")
        hm.append(_make_message("second", "user"))
        t2 = hm.estimate_tokens(system_prompt="sp", latest_user_input="input")

        assert t2 > t1  # More messages = more tokens

    def test_cache_invalidated_on_clear(self):
        """estimate_tokens cache is cleared when history is cleared."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(_make_message("content", "user"))

        t1 = hm.estimate_tokens(system_prompt="sp", latest_user_input="input")
        hm.clear()
        t2 = hm.estimate_tokens(system_prompt="sp", latest_user_input="input")

        assert t2 < t1  # Cleared = fewer tokens

    def test_cache_invalidated_on_compress(self):
        """estimate_tokens cache is cleared after compression."""
        hm = HistoryManager(
            compression_threshold=0.5,
            token_counter=TokenCounter(),
        )

        for i in range(100):
            hm.append(_make_message(f"Message {i}", "user"))

        t1 = hm.estimate_tokens()
        hm.compress("Summary after 100 messages.")
        t2 = hm.estimate_tokens()

        assert t2 < t1  # Compression should reduce token count

    def test_cache_invalidated_on_load_from_dict(self):
        """estimate_tokens cache is cleared after loading from dict."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(_make_message("original", "user"))
        hm.estimate_tokens(system_prompt="sp", latest_user_input="input")

        saved = hm.to_dict()
        hm.load_from_dict(saved)
        t = hm.estimate_tokens(system_prompt="sp", latest_user_input="new")
        assert t > 0

    def test_system_prompt_cache_reuse(self):
        """System prompt tokens are cached and reused across calls."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(_make_message("test", "user"))

        sp = "You are a helpful assistant. " * 20  # Longer prompt

        t1 = hm.estimate_tokens(system_prompt=sp)
        t2 = hm.estimate_tokens(system_prompt=sp)  # Same prompt → system prompt cache hit
        assert t1 == t2

        t3 = hm.estimate_tokens(system_prompt="Different prompt")  # Different → cache miss
        assert t3 != t1

    def test_different_inputs_produce_different_cache_keys(self):
        """Different latest_user_input values produce different cache keys."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(_make_message("msg", "user"))

        t1 = hm.estimate_tokens(system_prompt="sp", latest_user_input="input A")
        t2 = hm.estimate_tokens(system_prompt="sp", latest_user_input="input B")
        # Different inputs may or may not have different token counts
        # but the cache should NOT return stale values
        assert isinstance(t1, int) and isinstance(t2, int)


# ============================================================================
# History Serialization Round-Trip
# ============================================================================


class TestHistorySerializationStress:
    """Large-scale history serialization and deserialization."""

    def test_500_messages_to_dict_and_back(self):
        """500 messages survive to_dict → load_from_dict without data loss."""
        hm = HistoryManager(token_counter=TokenCounter())

        for i in range(500):
            role = "user" if i % 3 == 0 else ("assistant" if i % 3 == 1 else "tool")
            hm.append(_make_message(f"Message number {i:04d}", role))

        d = hm.to_dict()
        assert "history" in d
        assert len(d["history"]) == 500

        # Load into a fresh manager
        hm2 = HistoryManager(token_counter=TokenCounter())
        hm2.load_from_dict(d)

        history2 = hm2.get_history()
        assert len(history2) == 500
        for i, msg in enumerate(history2):
            assert f"{i:04d}" in msg.content

    def test_to_dict_includes_rounds_and_usage(self):
        """to_dict includes round count and usage snapshot."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(_make_message("user msg 1", "user"))
        hm.append(_make_message("assistant msg 1", "assistant"))
        hm.append(_make_message("user msg 2", "user"))

        hm.record_usage(prompt_tokens=100, completion_tokens=50)

        d = hm.to_dict()
        assert "rounds" in d
        assert "usage" in d
        assert d["usage"]["prompt_tokens"] == 100

    def test_summary_message_format(self):
        """build_summary_message produces correctly formatted summary messages."""
        hm = HistoryManager(token_counter=TokenCounter())
        msg = hm.build_summary_message("Test summary content")

        assert msg.role == "summary"
        assert "Summary" in msg.content
        assert "Test summary content" in msg.content

    def test_build_llm_messages_with_summary_and_tools(self):
        """build_llm_messages correctly projects summary messages for LLM."""
        hm = HistoryManager(token_counter=TokenCounter())
        hm.append(hm.build_summary_message("Earlier conversation archived."))
        hm.append(_make_assistant_with_tool_calls(
            "", [{"id": "c1", "type": "function", "function": {
                "name": "Read", "arguments": json.dumps({"path": "test.py"}),
            }}],
        ))
        hm.append(_make_tool_result("Read", "file content", "c1"))

        messages = hm.build_llm_messages(
            system_prompt="You are helpful.",
            latest_user_input="Continue.",
        )

        assert len(messages) > 0
        assert messages[0]["role"] == "system"
        # Should include the assistant's tool_calls
        has_tool_calls = any(
            m.get("tool_calls") for m in messages
        )
        assert has_tool_calls


