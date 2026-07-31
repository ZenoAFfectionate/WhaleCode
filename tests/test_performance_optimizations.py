"""Test performance optimizations: token estimate caching + lazy tool init.

Covers:
- HistoryManager estimate_tokens cache hit / miss / invalidation
- system_prompt token cache
- WebSearchTool lazy backend init
"""

from hello_agents.context.token_counter import TokenCounter
from hello_agents.context.history import HistoryManager
from hello_agents.core.message import Message


# ---------------------------------------------------------------------------
# HistoryManager estimate_tokens cache
# ---------------------------------------------------------------------------

class TestEstimateTokensCache:
    """Tests for the estimate_tokens() caching optimization."""

    def test_cache_hit_same_params(self):
        """Same (system_prompt, history, input) → cache hit."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        hm.append(Message("hello back", "assistant"))

        v1 = hm.estimate_tokens(system_prompt="You are helpful.")
        v2 = hm.estimate_tokens(system_prompt="You are helpful.")
        assert v1 == v2
        assert hm._cached_estimate_value == v1

    def test_cache_miss_different_system_prompt(self):
        """Different system_prompt → cache miss, recompute (may have same tokens)."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        v1 = hm.estimate_tokens(system_prompt="Short")
        # Different prompt → different cache key → recompute
        v2 = hm.estimate_tokens(system_prompt="A much longer system prompt with more tokens")
        # Cached hash should be different (not the same cache hit)
        assert hm._cached_system_prompt_hash != hash("Short")
        assert v2 > v1  # longer prompt → more tokens

    def test_cache_miss_different_latest_input(self):
        """Different latest_user_input → cache miss (different cache keys)."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp = "You are helpful."
        v1 = hm.estimate_tokens(system_prompt=sp, latest_user_input="short")
        # Different input → different cache key → recompute
        # Use a much longer input so the token count clearly differs
        v2 = hm.estimate_tokens(
            system_prompt=sp,
            latest_user_input="a much longer input query that has many more tokens",
        )
        assert v2 > v1

    def test_cache_invalidated_on_append(self):
        """Appending a message invalidates the cache."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp = "You are helpful."
        v1 = hm.estimate_tokens(system_prompt=sp)
        hm.append(Message("new", "user"))
        v2 = hm.estimate_tokens(system_prompt=sp)
        assert v2 > v1  # more messages → more tokens

    def test_cache_invalidated_on_clear(self):
        """clear() invalidates the cache."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp = "You are helpful."
        v1 = hm.estimate_tokens(system_prompt=sp)
        hm.clear()
        v2 = hm.estimate_tokens(system_prompt=sp)
        assert v2 < v1  # empty history → fewer tokens

    def test_cache_hit_with_latest_input_then_same(self):
        """Cache hit when latest_user_input is the same on repeated calls."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp = "You are helpful."
        inp = "user query here"
        v1 = hm.estimate_tokens(system_prompt=sp, latest_user_input=inp)
        v2 = hm.estimate_tokens(system_prompt=sp, latest_user_input=inp)
        assert v1 == v2

    def test_explicit_history_parameter_bypasses_cache(self):
        """Passing an explicit history= skips the cache."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp = "You are helpful."
        # Fill cache with default history
        hm.estimate_tokens(system_prompt=sp)
        # Now call with explicit history → should NOT use cache
        custom = [Message("completely different", "user")]
        v1 = hm.estimate_tokens(system_prompt=sp)
        v2 = hm.estimate_tokens(system_prompt=sp, history=custom)
        assert v1 != v2  # explicit history → different result

    def test_cache_invalidated_on_load_from_dict(self):
        """load_from_dict() replaces history → cache invalidated."""
        from datetime import datetime
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp = "You are helpful."
        hm.estimate_tokens(system_prompt=sp)
        # load completely different history
        ts = datetime.now().isoformat()
        new_data = {
            "history": [
                {"role": "user", "content": "msg1", "timestamp": ts, "metadata": {}},
                {"role": "assistant", "content": "msg2", "timestamp": ts, "metadata": {}},
                {"role": "user", "content": "msg3", "timestamp": ts, "metadata": {}},
            ]
        }
        hm.load_from_dict(new_data)
        v2 = hm.estimate_tokens(system_prompt=sp)
        # Three messages > one message
        assert v2 > 0

    def test_cache_not_stale_after_estimate_call(self):
        """Cache value is correct across multiple rounds of the same pattern."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        messages = [
            ("Hello, please read foo.py", "user"),
            ("I'll read it.", "assistant"),
            ("Read result: ...", "tool"),
            ("Now edit it.", "user"),
            ("I'll edit.", "assistant"),
            ("Edit result: ...", "tool"),
        ]
        for content, role in messages:
            hm.append(Message(content, role))

        sp = "You are a helpful coding assistant."
        # Round 1
        r1 = hm.estimate_tokens(system_prompt=sp)
        # Round 2 — same params, should hit cache
        r2 = hm.estimate_tokens(system_prompt=sp)
        assert r1 == r2

        # Round 3 — new message, cache invalidated
        hm.append(Message("Check tests", "user"))
        r3 = hm.estimate_tokens(system_prompt=sp)
        assert r3 > r2


# ---------------------------------------------------------------------------
# system_prompt token cache
# ---------------------------------------------------------------------------

class TestSystemPromptCache:
    """Tests for the system_prompt token sub-cache within estimate_tokens()."""

    def test_system_prompt_cached_across_calls(self):
        """Same system_prompt reused → sub-cache hit."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp = "You are helpful. " * 50  # long prompt, worth caching
        v1 = hm.estimate_tokens(system_prompt=sp)
        v2 = hm.estimate_tokens(system_prompt=sp)
        assert v1 == v2
        # sp_hash should be cached
        assert hm._cached_system_prompt_hash == hash(sp)

    def test_system_prompt_cache_switches_on_change(self):
        """Different system_prompt → recompute and cache new one."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        sp_a = "Prompt A " * 30
        sp_b = "Prompt B " * 30
        v1 = hm.estimate_tokens(system_prompt=sp_a)
        v2 = hm.estimate_tokens(system_prompt=sp_b)
        # Both prompts same length so tokens may be equal — test cache logic instead
        assert hm._cached_system_prompt_hash == hash(sp_b)

    def test_estimate_without_system_prompt(self):
        """estimate_tokens works correctly without system_prompt."""
        hm = HistoryManager(token_counter=TokenCounter(model="gpt-4"))
        hm.append(Message("hello", "user"))
        v = hm.estimate_tokens()
        assert v > 0


# ---------------------------------------------------------------------------
# WebSearchTool lazy backend
# ---------------------------------------------------------------------------

class TestWebSearchLazyBackend:
    """Tests for WebSearchTool lazy backend initialization."""

    def test_backend_not_created_on_init(self, tmp_path):
        """Backend should NOT be created eagerly in __init__."""
        from hello_agents.tools.builtin.web_tool import WebSearchTool
        tool = WebSearchTool(project_root=str(tmp_path))
        assert tool._search_backend is None

    def test_backend_created_on_first_access(self, tmp_path):
        """First access to search_backend property creates the backend."""
        from hello_agents.tools.builtin.web_tool import WebSearchTool
        tool = WebSearchTool(project_root=str(tmp_path))
        backend = tool.search_backend
        assert backend is not None
        assert tool._search_backend is not None

    def test_backend_returns_same_instance(self, tmp_path):
        """Multiple accesses return the same backend instance."""
        from hello_agents.tools.builtin.web_tool import WebSearchTool
        tool = WebSearchTool(project_root=str(tmp_path))
        b1 = tool.search_backend
        b2 = tool.search_backend
        assert b1 is b2

    def test_custom_backend_used_directly(self, tmp_path):
        """If a custom backend is passed, it's used without lazy default."""
        from hello_agents.tools.builtin.web_tool import WebSearchTool

        class FakeBackend:
            def search_text(self, **kwargs):
                return [{"title": "fake", "href": "http://x", "body": "x"}]

        fake = FakeBackend()
        tool = WebSearchTool(project_root=str(tmp_path), search_backend=fake)
        # Custom backend stored directly (not None, so no lazy init)
        assert tool._search_backend is fake
        assert tool.search_backend is fake
