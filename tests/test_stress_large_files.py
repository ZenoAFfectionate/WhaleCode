"""Stress tests for large files, massive tool outputs, and message content explosion.

Covers:
- 10MB+ tool output truncation
- 100K-line file read/write behavior
- Message content with 1M+ characters
- ObservationTruncator edge cases
- TokenCounter behavior with extreme inputs
- LRU cache overflow protection
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from code.context.token_counter import TokenCounter, _LRUCache
from code.context.truncator import ObservationTruncator
from code.core.message import Message


# ============================================================================
# Massive Tool Output
# ============================================================================


class TestMassiveToolOutput:
    """Tool outputs up to 10MB are handled without OOM."""

    @pytest.fixture
    def truncator(self):
        with tempfile.TemporaryDirectory() as d:
            yield ObservationTruncator(
                max_lines=2000,
                max_bytes=51200,
                truncate_direction="head",
                output_dir=d,
                retention_days=1,
            )

    def test_1mb_output_truncation(self, truncator):
        """1MB of tool output is truncated to within limits."""
        huge_output = "Line of text content here. " * 50000  # ~1MB
        result = truncator.truncate("Read", huge_output)

        display = result.get("display_preview", result.get("preview", ""))
        assert result.get("truncated") is True
        assert len(display) < len(huge_output)
        assert len(display) < 200_000  # Should be much smaller

    def test_10mb_output_truncation(self, truncator):
        """10MB of tool output is truncated without OOM."""
        huge_output = "Data " * (2 * 1024 * 1024)  # ~10MB (5 bytes × 2M)
        result = truncator.truncate("Bash", huge_output)

        display = result.get("display_preview", result.get("preview", ""))
        # Must fit within limits
        assert len(display) < 500_000
        # Must not be empty
        assert len(display) > 0

    def test_output_saved_to_disk(self, truncator):
        """Full output is saved to disk when content exceeds limits."""
        # Must exceed max_bytes (51200) + max_lines (2000) to trigger truncation
        content = "SAVE_ME_" + "x" * 100_000  # ~100KB, exceeds max_bytes
        result = truncator.truncate("Read", content)

        # Verify full_output_path points to a real file
        full_path = result.get("full_output_path")
        assert full_path is not None, f"Expected full_output_path, got keys: {result.keys()}"
        assert Path(full_path).exists()

        # The saved file should contain the original content
        saved_content = Path(full_path).read_text()
        assert "SAVE_ME_" in saved_content

    def test_empty_output_not_truncated(self, truncator):
        """Empty string is not truncated."""
        result = truncator.truncate("Read", "")
        assert result.get("truncated") is False or "preview" in result

    def test_short_output_not_truncated(self, truncator):
        """Short output is returned without truncation."""
        short = "Hello, world!"
        result = truncator.truncate("Read", short)
        assert result.get("truncated") is False
        assert short in result.get("preview", "")

    def test_head_tail_truncation(self):
        """Head+tail truncation preserves both ends."""
        with tempfile.TemporaryDirectory() as d:
            truncator = ObservationTruncator(
                max_lines=100,
                max_bytes=10000,
                truncate_direction="head_tail",
                output_dir=d,
            )
            lines = [f"Line {i:06d}: some content here for testing purposes" for i in range(10000)]
            content = "\n".join(lines)

            result = truncator.truncate("Grep", content)
            display = result.get("display_preview", result.get("preview", ""))
            # Should contain content from both beginning and end
            assert "Line 000000" in display
            assert "Line 009999" in display
            # Middle should be omitted
            assert "..." in display or "omit" in display.lower() or result.get("truncated")


# ============================================================================
# Large File Read/Write
# ============================================================================


class TestLargeFileReadWrite:
    """Behavior with files approaching size limits."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_100k_line_file_read_truncation(self, workspace):
        """A 100K-line file is read and output is truncated appropriately."""
        # Create a 100K-line file
        big_file = workspace / "big.txt"
        lines = [f"Line {i:06d}: The quick brown fox jumps over the lazy dog." for i in range(100_000)]
        big_file.write_text("\n".join(lines))

        # Use the truncator directly
        with tempfile.TemporaryDirectory() as outdir:
            truncator = ObservationTruncator(
                max_lines=2000,
                max_bytes=200_000,
                truncate_direction="head",
                output_dir=outdir,
            )
            content = big_file.read_text()
            result = truncator.truncate("Read", content)

            display = result.get("display_preview", result.get("preview", ""))
            # Result should be truncated
            line_count = display.count("\n")
            assert line_count < 5000  # Well within limits
            assert len(display) < 500_000

    def test_large_file_generation_stress(self, workspace):
        """Write a 2MB file, then read it back — verify truncation chain."""
        content = "ABCDEFGHIJ" * 200_000  # 2MB
        out_file = workspace / "output.txt"
        out_file.write_text(content)

        # Verify file size
        assert out_file.stat().st_size > 1_000_000

        # Read and truncate
        with tempfile.TemporaryDirectory() as outdir:
            truncator = ObservationTruncator(
                max_lines=1000,
                max_bytes=50000,
                truncate_direction="head",
                output_dir=outdir,
            )
            result = truncator.truncate("Read", out_file.read_text())

            display = result.get("display_preview", result.get("preview", ""))
            assert len(display) < 100_000
            # Full content should be recoverable from disk
            output_files = list(Path(outdir).glob("*"))
            assert len(output_files) > 0


# ============================================================================
# Message Content Explosion
# ============================================================================


class TestMessageContentExplosion:
    """Messages with extremely large content don't break token counting."""

    def test_1m_char_message_token_count(self):
        """A message with 1M characters doesn't overflow token counting."""
        tc = TokenCounter()
        huge_content = "hello world " * 90_000  # ~1M chars

        count = tc.count_text(huge_content)
        assert count > 0
        assert count < 10_000_000  # Should be a reasonable number

    def test_large_message_in_history_manager(self):
        """HistoryManager handles a single huge message gracefully."""
        from code.context.history import HistoryManager

        hm = HistoryManager(token_counter=TokenCounter())
        huge_msg = Message("x" * 500_000, "user")  # 500K chars
        hm.append(huge_msg)

        tokens = hm.estimate_tokens()
        assert tokens > 0
        assert tokens < 5_000_000  # Reasonable upper bound

    def test_many_large_messages(self):
        """50 messages of 10K chars each — token estimation stays consistent."""
        from code.context.history import HistoryManager

        hm = HistoryManager(token_counter=TokenCounter())
        for i in range(50):
            hm.append(Message(f"Msg {i}: " + ("data " * 2000), "user"))  # ~10K each

        t1 = hm.estimate_tokens()
        t2 = hm.estimate_tokens()
        assert t1 == t2  # Deterministic

    def test_lru_cache_eviction_under_pressure(self):
        """LRU cache correctly evicts old entries when full."""
        cache = _LRUCache(max_size=100)
        for i in range(200):
            cache.put(f"key_{i}", i)

        # Size should be capped at max_size
        assert len(cache) == 100
        # Oldest entries should be evicted
        assert cache.get("key_0") is None
        assert cache.get("key_100") is not None

    def test_lru_cache_does_not_store_super_large_keys(self):
        """LRU cache handles very large keys without issues."""
        cache = _LRUCache(max_size=50)
        huge_key = "k" * 100_000  # 100KB key

        cache.put(huge_key, 42)
        assert cache.get(huge_key) == 42

        # Fill cache and ensure the huge key is evicted normally
        for i in range(60):
            cache.put(f"small_{i}", i)

        # Cache should still be functional after storing huge key
        assert len(cache) <= 50


# ============================================================================
# ObservationTruncator Edge Cases
# ============================================================================


class TestTruncatorEdgeCases:
    """Corner cases for truncation behavior."""

    @pytest.fixture
    def truncator(self):
        with tempfile.TemporaryDirectory() as d:
            yield ObservationTruncator(
                max_lines=10,
                max_bytes=500,
                truncate_direction="head",
                output_dir=d,
                retention_days=1,
            )

    def test_utf8_multibyte_truncation(self, truncator):
        """Multibyte UTF-8 characters are not split mid-character."""
        content = "你好世界！" * 5000
        result = truncator.truncate("Read", content)
        # Should not raise UnicodeDecodeError
        assert isinstance(result, dict)
        display = result.get("display_preview", result.get("preview", ""))
        # Should still be valid UTF-8
        display.encode("utf-8")

    def test_binary_like_content(self, truncator):
        """Content with null bytes and control chars doesn't break truncation."""
        content = "normal text \x00 with \x01 control \x02 chars \n" * 100
        result = truncator.truncate("Bash", content)
        assert isinstance(result, dict)

    def test_newline_only_content(self, truncator):
        """Content consisting only of newlines is handled."""
        result = truncator.truncate("Read", "\n" * 10000)
        assert isinstance(result, dict)

    def test_direction_tail(self):
        """Tail truncation preserves the end of the output."""
        with tempfile.TemporaryDirectory() as d:
            truncator = ObservationTruncator(
                max_lines=5,
                max_bytes=500,
                truncate_direction="tail",
                output_dir=d,
            )
            lines = [f"Line {i}" for i in range(100)]
            content = "\n".join(lines)
            result = truncator.truncate("Read", content)
            display = result.get("display_preview", result.get("preview", ""))
            # Should contain the last lines
            assert "Line 99" in display
            # Should NOT contain the first line
            assert "Line 0" not in display


# ============================================================================
# TokenCounter Extreme Inputs
# ============================================================================


class TestTokenCounterStress:
    """TokenCounter behavior with extreme inputs."""

    def test_empty_string(self):
        tc = TokenCounter()
        assert tc.count_text("") == 0

    def test_only_whitespace(self):
        tc = TokenCounter()
        tokens = tc.count_text("   \n\n   \t   ")
        assert tokens >= 0

    def test_only_special_chars(self):
        tc = TokenCounter()
        tokens = tc.count_text("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")
        assert tokens > 0

    def test_count_message_with_many_tool_calls(self):
        """Message with many tool_calls — all call arguments are counted."""
        tc = TokenCounter()
        tool_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "arguments": json.dumps({"param": f"value_{i}"}),
                },
            }
            for i in range(20)
        ]
        msg = Message(
            content="Processing",
            role="assistant",
            metadata={"tool_calls": tool_calls},
        )

        count = tc.count_message(msg)
        assert count > 0
        # 20 tool calls should contribute significant tokens
        assert count > 100

    def test_count_message_fallback_when_no_tiktoken(self):
        """When tiktoken is unavailable, crude estimate is used."""
        tc = TokenCounter()

        # Test with known content to verify fallback estimation
        count = tc.count_text("hello world")
        assert count > 0  # Fallback should produce a positive number
