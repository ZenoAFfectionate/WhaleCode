"""Unit tests for code/context/ — HistoryManager, ObservationTruncator, TokenCounter.

Covers:
- HistoryManager: build_llm_messages, compression, round boundaries, serialization,
  summary construction, preserved context, micro_compact, should_compress
- ObservationTruncator: truncate modes, token-based truncation, save/load, cleanup
- TokenCounter: count_text, count_message, encode/decode, cache
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hello_agents.context.history import (
    HistoryManager,
    format_compact_summary,
)
from hello_agents.context.token_counter import TokenCounter
from hello_agents.context.truncator import ObservationTruncator
from hello_agents.core.message import Message


# ────────────────────────────────────────────────────────────────────────────
# TokenCounter
# ────────────────────────────────────────────────────────────────────────────


class TestTokenCounter:
    def test_count_text_empty(self):
        assert TokenCounter().count_text("") == 0

    def test_count_text_nonempty(self):
        assert TokenCounter().count_text("hello world") > 0

    def test_count_text_estimate_fallback(self):
        tc = TokenCounter(model="nonexistent-model-xyz")
        assert tc.count_text("hello world test") >= 1

    def test_count_message_no_tool_calls(self):
        tc = TokenCounter()
        assert tc.count_message(Message("hello", "user")) >= 5

    def test_count_message_with_tool_calls(self):
        tc = TokenCounter()
        msg = Message(
            "", "assistant",
            metadata={"tool_calls": [{"id": "1", "name": "Write", "arguments": {"path": "a.py", "content": "x" * 200}}]},
        )
        assert tc.count_message(msg) > 20

    def test_count_message_tool_calls_non_list(self):
        tc = TokenCounter()
        assert tc.count_message(Message("hello", "assistant", metadata={"tool_calls": "bad"})) >= 5

    def test_count_messages_batch(self):
        tc = TokenCounter()
        msgs = [Message("a", "user"), Message("b", "assistant")]
        total = tc.count_messages(msgs)
        assert total == sum(tc.count_message(m) for m in msgs)

    def test_encode_decode_roundtrip(self):
        tc = TokenCounter()
        tokens = tc.encode_text("hello world test")
        assert tokens is not None and len(tokens) > 0
        decoded = tc.decode_tokens(tokens)
        assert "hello" in decoded

    def test_cache_and_clear(self):
        tc = TokenCounter()
        tc.count_message(Message("hello", "user"))
        assert tc.get_cache_size() >= 1
        tc.clear_cache()
        assert tc.get_cache_size() == 0

    def test_count_message_arguments_as_string(self):
        tc = TokenCounter()
        msg = Message("", "assistant", metadata={
            "tool_calls": [{"id": "c1", "name": "Read", "arguments": '{"path": "a.py"}'}],
        })
        n = tc.count_message(msg)
        assert n > 5

    def test_count_message_multiple_tool_calls(self):
        tc = TokenCounter()
        msg = Message("", "assistant", metadata={
            "tool_calls": [
                {"id": "1", "name": "Read", "arguments": {"path": "a.py"}},
                {"id": "2", "name": "Grep", "arguments": {"pattern": "def test"}},
            ],
        })
        n = tc.count_message(msg)
        assert n > tc.count_message(Message("", "assistant", metadata={
            "tool_calls": [{"id": "1", "name": "Read", "arguments": {"path": "a.py"}}],
        }))

    def test_count_messages_empty(self):
        assert TokenCounter().count_messages([]) == 0

    def test_decode_tokens_empty(self):
        assert TokenCounter().decode_tokens([]) == ""

    def test_get_cache_stats(self):
        tc = TokenCounter()
        tc.count_message(Message("hello", "user"))
        stats = tc.get_cache_stats()
        assert stats["cached_messages"] >= 1
        assert stats["total_cached_tokens"] > 0


# ────────────────────────────────────────────────────────────────────────────
# ObservationTruncator
# ────────────────────────────────────────────────────────────────────────────


class TestObservationTruncator:

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def _make(self, tmp_dir, **kw):
        return ObservationTruncator(
            output_dir=f"{tmp_dir}/output",
            token_counter=TokenCounter(model="gpt-4"),
            **kw,
        )

    def test_passthrough_small(self, tmp_dir):
        r = self._make(tmp_dir, max_lines=100, max_bytes=10000).truncate("Read", "small\n")
        assert r["truncated"] is False

    def test_truncate_head(self, tmp_dir):
        trunc = self._make(tmp_dir, max_lines=5, max_bytes=500)
        r = trunc.truncate("Bash", "\n".join(f"line {i:04d}" for i in range(100)))
        assert r["truncated"] is True
        assert r["full_output_path"] is not None

    def test_truncate_tail(self, tmp_dir):
        trunc = self._make(tmp_dir, max_lines=3, max_bytes=500, truncate_direction="tail")
        r = trunc.truncate("Bash", "\n".join(f"line {i:04d}" for i in range(50)))
        assert "line 0049" in r["preview"]

    def test_truncate_head_tail(self, tmp_dir):
        trunc = self._make(tmp_dir, max_lines=20, max_bytes=2000, truncate_direction="head_tail")
        r = trunc.truncate("Read", "\n".join(f"line {i:04d}" for i in range(100)))
        assert "line 0000" in r["preview"]
        assert "line 0099" in r["preview"]

    def test_truncate_for_context_passthrough(self, tmp_dir):
        r = self._make(tmp_dir).truncate_for_context("Read", "tiny")
        assert r["truncated"] is False

    def test_truncate_for_context_large(self, tmp_dir):
        trunc = self._make(tmp_dir)
        r = trunc.truncate_for_context("Read", ". " * 3000)
        assert r["truncated"] is True
        assert r["full_output_path"] is not None

    def test_save_and_load(self, tmp_dir):
        trunc = self._make(tmp_dir, max_lines=2, max_bytes=200)
        r = trunc.truncate("Grep", "line1\nline2\nline3\nline4\nline5")
        loaded = ObservationTruncator.load_saved_output(r["full_output_path"])
        assert loaded is not None and loaded["tool"] == "Grep"

    def test_load_nonexistent(self):
        assert ObservationTruncator.load_saved_output("/nonexistent/x.json") is None

    def test_per_call_overrides(self, tmp_dir):
        trunc = self._make(tmp_dir, max_lines=3, max_bytes=300)
        r = trunc.truncate("Bash", "\n".join(f"line {i}" for i in range(20)), max_lines=100, max_bytes=100000)
        assert r["truncated"] is False

    def test_empty_output(self, tmp_dir):
        assert self._make(tmp_dir).truncate("Read", "")["truncated"] is False

    def test_reuse_output_path(self, tmp_dir):
        trunc = self._make(tmp_dir, max_lines=1, max_bytes=100)
        r1 = trunc.truncate("Read", "a\nb\nc\nd")
        r2 = trunc.truncate("Read", "a\nb\nc\nd", metadata={"full_output_path": r1["full_output_path"]})
        assert r2["stats"]["reused_output_path"] is True

    def test_cleanup_old_outputs(self, tmp_dir):
        trunc = self._make(tmp_dir, retention_days=0)
        old = Path(trunc.output_dir) / "tool_20000101_000000_000000_Bash.json"
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text('{"tool":"Bash","output":"test"}')
        # set mtime to year 2000
        import os
        os.utime(str(old), (946684800, 946684800))
        trunc._cleanup_old_outputs(force=True)
        assert not old.exists()

    def test_text_metrics(self):
        lines, nbytes = ObservationTruncator._text_metrics("a\nb\nc")
        assert lines == ["a", "b", "c"]
        assert nbytes > 0


# ────────────────────────────────────────────────────────────────────────────
# HistoryManager — message projection
# ────────────────────────────────────────────────────────────────────────────


class TestHistoryManagerBuildMessages:

    @pytest.fixture
    def hm(self):
        return HistoryManager(token_counter=TokenCounter(model="gpt-4"))

    def test_system_prompt(self, hm):
        assert hm.build_llm_messages(system_prompt="Bot.")[0]["content"] == "Bot."

    def test_user_projection(self, hm):
        hm.append(Message("hello", "user"))
        assert hm.build_llm_messages()[0]["content"] == "hello"

    def test_assistant_with_tool_calls(self, hm):
        hm.append(Message("", "assistant", metadata={
            "tool_calls": [{"id": "c1", "name": "Read", "arguments": {"path": "a.py"}}],
        }))
        msgs = hm.build_llm_messages()
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "Read"

    def test_tool_with_call_id(self, hm):
        hm.append(Message("result", "tool", metadata={"tool_call_id": "c1"}))
        assert hm.build_llm_messages()[0]["role"] == "tool"

    def test_tool_without_call_id(self, hm):
        hm.append(Message("result", "tool"))
        assert "Previous tool result" in hm.build_llm_messages()[0]["content"]

    def test_summary_role(self, hm):
        hm.append(Message("## Summary\nDone.", "summary"))
        assert "[Conversation summary]" in hm.build_llm_messages()[0]["content"]

    def test_system_role(self, hm):
        hm.append(Message("Note", "system"))
        assert "[System note]" in hm.build_llm_messages()[0]["content"]

    def test_preserved_context(self, hm):
        hm.append(Message("[Preserved context]\n\n## Essential Context Snapshot", "system",
                          metadata={"kind": "preserved_context"}))
        assert hm.build_llm_messages()[0]["role"] == "user"

    def test_latest_input(self, hm):
        hm.append(Message("earlier", "user"))
        assert hm.build_llm_messages(latest_user_input="new")[-1]["content"] == "new"

    def test_empty_assistant_placeholder(self, hm):
        hm.append(Message("", "assistant"))
        assert hm.build_llm_messages()[0]["content"] == "[no output]"

    def test_ordering(self, hm):
        for role in ["user", "assistant", "tool"]:
            hm.append(Message("m", role, metadata={"tool_call_id": "c1"} if role == "tool" else None))
        roles = [m["role"] for m in hm.build_llm_messages()]
        assert roles == ["user", "assistant", "tool"]

    def test_unaffected_by_side_effects(self, hm):
        hm.append(Message("original", "user"))
        msgs = hm.build_llm_messages()
        msgs[0]["content"] = "modified"
        assert hm.build_llm_messages()[0]["content"] == "original"


# ────────────────────────────────────────────────────────────────────────────
# HistoryManager — compression & rounds
# ────────────────────────────────────────────────────────────────────────────


class TestHistoryManagerCompression:

    @pytest.fixture
    def hm(self):
        return HistoryManager(token_counter=TokenCounter(model="gpt-4"))

    def test_round_boundaries(self, hm):
        for i in range(3):
            hm.append(Message(f"u{i}", "user"))
            hm.append(Message(f"a{i}", "assistant"))
        assert hm.find_round_boundaries() == [0, 2, 4]

    def test_estimate_rounds(self, hm):
        hm.append(Message("u1", "user")); hm.append(Message("a1", "assistant"))
        hm.append(Message("u2", "user"))
        assert hm.estimate_rounds() == 2

    def test_skip_internal_user(self, hm):
        hm.append(Message("u1", "user"))
        hm.append(Message("retry", "user", metadata={"kind": "retry_reminder"}))
        hm.append(Message("a1", "assistant"))
        assert hm.estimate_rounds() == 1

    def test_compression_split(self, hm):
        for i in range(10):
            hm.append(Message(f"u{i}", "user")); hm.append(Message(f"a{i}", "assistant"))
        split = hm.get_compression_split(retain_rounds=2)
        assert split is not None
        assert len(split[0]) + len(split[1]) == 20

    def test_split_insufficient(self, hm):
        hm.append(Message("u1", "user")); hm.append(Message("a1", "assistant"))
        assert hm.get_compression_split(retain_rounds=5) is None

    def test_compress_replaces(self, hm):
        for i in range(10):
            hm.append(Message(f"u{i}", "user")); hm.append(Message(f"a{i}", "assistant"))
        original = len(hm.get_history())
        hm.compress("Summary text.")
        assert len(hm.get_history()) < original
        assert hm.get_history()[0].role == "summary"


# ────────────────────────────────────────────────────────────────────────────
# HistoryManager — serialization
# ────────────────────────────────────────────────────────────────────────────


class TestHistoryManagerSerialization:

    def test_roundtrip(self):
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("u1", "user"))
        hm.append(Message("", "assistant", metadata={
            "tool_calls": [{"id": "c1", "name": "Read", "arguments": {"path": "x.py"}}],
        }))
        hm.append(Message("result", "tool", metadata={"tool_call_id": "c1"}))
        hm2 = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm2.load_from_dict(hm.to_dict())
        assert len(hm2.get_history()) == 3
        assert hm2.estimate_rounds() == 1

    def test_restores_usage(self):
        hm = HistoryManager()
        hm.record_usage(prompt_tokens=100, completion_tokens=50)
        hm2 = HistoryManager()
        hm2.load_from_dict(hm.to_dict())
        assert hm2.get_usage_snapshot()["prompt_tokens"] == 100

    def test_to_dict_empty(self):
        data = HistoryManager().to_dict()
        assert data["history"] == []
        assert data["rounds"] == 0


# ────────────────────────────────────────────────────────────────────────────
# HistoryManager — summary / preserved context / usage
# ────────────────────────────────────────────────────────────────────────────


class TestHistoryManagerMessages:

    def test_summary_message_new_heading(self):
        msg = HistoryManager().build_summary_message("Content")
        assert msg.role == "summary"
        assert "Content" in msg.content

    def test_build_tool_call_message(self):
        msg = HistoryManager().build_assistant_tool_call_message(
            tool_calls=[{"id": "c1", "name": "Read", "arguments": {"path": "x.py"}}],
            content="Reading",
        )
        assert msg.role == "assistant" and msg.metadata["tool_calls"] is not None

    def test_preserved_context_message(self):
        msg = HistoryManager().build_preserved_context_message("Ctx\nData")
        assert msg.role == "system" and msg.metadata["kind"] == "preserved_context"

    def test_format_summary_strips_tags(self):
        raw = "<analysis>x</analysis>\n<summary>Real</summary>"
        assert format_compact_summary(raw) == "Real"

    def test_format_summary_plain(self):
        assert format_compact_summary("Plain text.") == "Plain text."

    def test_record_usage(self):
        hm = HistoryManager()
        hm.record_usage(prompt_tokens=42, completion_tokens=10)
        snap = hm.get_usage_snapshot()
        assert snap["prompt_tokens"] == 42 and snap["total_tokens"] == 52
        assert hm.get_last_api_prompt_tokens() == 42

    def test_stale_on_append(self):
        hm = HistoryManager()
        hm.record_usage(prompt_tokens=10, completion_tokens=5)
        hm.append(Message("new", "user"))
        assert hm.get_usage_snapshot()["stale"] is True


# ────────────────────────────────────────────────────────────────────────────
# HistoryManager — mutation
# ────────────────────────────────────────────────────────────────────────────


class TestHistoryManagerMutation:

    def test_append_counts(self):
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        assert hm.get_estimated_token_count() > 0

    def test_clear_resets(self):
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        hm.record_usage(prompt_tokens=10, completion_tokens=5)
        hm.clear()
        assert len(hm.get_history()) == 0
        assert hm.get_estimated_token_count() == 0
        assert hm.get_last_api_prompt_tokens() == 0

    def test_get_history_is_copy(self):
        hm = HistoryManager()
        hm.append(Message("hello", "user"))
        h = hm.get_history()
        h.pop()
        assert len(hm.get_history()) == 1

    def test_tool_name_map(self):
        hm = HistoryManager()
        hm.append(Message("", "assistant", metadata={
            "tool_calls": [{"id": "t1", "name": "Read", "arguments": {}}],
        }))
        assert hm._build_tool_name_map(hm.get_history()) == {"t1": "Read"}


# ────────────────────────────────────────────────────────────────────────────
# HistoryManager — normalize / arguments / content detection
# ────────────────────────────────────────────────────────────────────────────


class TestHistoryManagerEdgeCases:

    def test_normalize_summary_empty(self):
        result = HistoryManager().normalize_summary_content("")
        assert result == HistoryManager.SUMMARY_HEADING

    def test_normalize_summary_already_has_heading(self):
        content = HistoryManager.SUMMARY_HEADING + "\nSome content"
        result = HistoryManager().normalize_summary_content(content)
        assert result == content

    def test_normalize_summary_new_content(self):
        result = HistoryManager().normalize_summary_content("New summary")
        assert result.startswith(HistoryManager.SUMMARY_HEADING)
        assert "New summary" in result

    def test_tool_call_arguments_string(self):
        assert HistoryManager._tool_call_arguments_for_llm('{"x":1}') == '{"x":1}'

    def test_tool_call_arguments_dict(self):
        result = HistoryManager._tool_call_arguments_for_llm({"x": 1})
        assert isinstance(result, str)
        assert '"x":1' in result

    def test_tool_call_arguments_none(self):
        result = HistoryManager._tool_call_arguments_for_llm(None)
        assert result == "{}"

    def test_is_archived_content_summary_prefix(self):
        hm = HistoryManager()
        assert hm._is_archived_user_content(hm.SUMMARY_NOTE_PREFIX + " ...") is True

    def test_is_archived_content_essential_prefix(self):
        hm = HistoryManager()
        assert hm._is_archived_user_content(hm.ESSENTIAL_CONTEXT_PREFIX + " ...") is True

    def test_is_archived_content_system_prefix(self):
        hm = HistoryManager()
        assert hm._is_archived_user_content(hm.SYSTEM_NOTE_PREFIX + " ...") is True

    def test_is_archived_content_empty(self):
        hm = HistoryManager()
        assert hm._is_archived_user_content("") is False

    def test_is_archived_content_normal(self):
        hm = HistoryManager()
        assert hm._is_archived_user_content("normal user message") is False

    def test_is_preserved_context_by_kind(self):
        hm = HistoryManager()
        msg = Message("anything", "system", metadata={"kind": "preserved_context"})
        assert hm._is_preserved_context_message(msg) is True

    def test_is_preserved_context_by_prefix(self):
        hm = HistoryManager()
        msg = Message(hm.ESSENTIAL_CONTEXT_PREFIX + " data", "system")
        assert hm._is_preserved_context_message(msg) is True

    def test_is_preserved_context_normal(self):
        hm = HistoryManager()
        assert hm._is_preserved_context_message(Message("normal", "system")) is False

    def test_tool_result_prompt_prefix_with_name(self):
        hm = HistoryManager()
        msg = Message("result", "tool", metadata={"tool_name": "Read"})
        prefix = hm._tool_result_prompt_prefix(msg)
        assert "Read" in prefix

    def test_tool_result_prompt_prefix_without_name(self):
        hm = HistoryManager()
        msg = Message("result", "tool")
        prefix = hm._tool_result_prompt_prefix(msg)
        assert "Previous tool result" in prefix

    def test_estimate_tokens_explicit_history(self):
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        # With explicit history, the cache should be bypassed
        custom = [Message("completely different content", "user")]
        v_default = hm.estimate_tokens(system_prompt="SP")
        v_custom = hm.estimate_tokens(system_prompt="SP", history=custom)
        # Different histories → different token counts
        assert v_default != v_custom

    def test_compression_split_retain_zero(self):
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        for i in range(5):
            hm.append(Message(f"u{i}", "user"))
            hm.append(Message(f"a{i}", "assistant"))
        # retain_rounds=0: all messages go to "old" part
        split = hm.get_compression_split(retain_rounds=0)
        assert split is not None
        old, recent = split
        assert len(old) == 10
        assert len(recent) == 0

    def test_compression_split_insufficient_rounds(self):
        hm = HistoryManager()
        hm.append(Message("only", "user"))
        assert hm.get_compression_split(retain_rounds=2) is None
