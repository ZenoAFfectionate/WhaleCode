"""A2: stream_invoke / astream_invoke 首 chunk 重试测试（2026-08-19）.

策略：用可编程的假 client 直接驱动三个 provider 的 OpenAIAdapter/
AnthropicAdapter/GeminiAdapter.stream_invoke——首个（或前 N 个）连接抛出
可重试异常（429），之后成功。断言：
1. 首 chunk 前的瞬时失败被自动重试，最终拿到完整流（内容不重复）；
2. 已开始产出后（首 chunk 已 yield）的中段失败不会重放（避免内容重复）；
3. 不可重试错误（401/404）不触发重试。
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

# conftest.py（同目录，pytest 自动加载）已把 code/ 暴露为 hello_agents 包。
from hello_agents.core.llm_adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
)


def _make(adapter_cls, model, base_url="http://x"):
    """构造 adapter（BaseLLMAdapter 的 api_key/base_url/timeout/model 均为必填）。"""
    return _fast_adapter(
        adapter_cls(model=model, api_key="sk-test", base_url=base_url, timeout=30)
    )


class _TransientError(Exception):
    """模拟 429 限流——类型名与状态码均命中可重试分类。"""

    def __init__(self, status_code=429):
        super().__init__(f"transient {status_code}")
        self.status_code = status_code


class _FatalError(Exception):
    """模拟 401 鉴权失败——不可重试。"""

    def __init__(self):
        super().__init__("unauthorized")
        self.status_code = 401


def _fast_adapter(adapter: OpenAIAdapter) -> OpenAIAdapter:
    """压缩退避延迟，让重试在毫秒级完成。"""
    adapter.retry_base_delay = 0.001
    adapter.retry_max_delay = 0.002
    return adapter


def _openai_chunk(text):
    chunk = mock.MagicMock()
    delta = mock.MagicMock()
    delta.content = text
    chunk.choices = [mock.MagicMock(delta=delta)]
    chunk.usage = None
    return chunk


class _FlakyOpenAIStream:
    """前 fail_times 次"建立流并取首 chunk"抛瞬时异常，之后返回 3 个 chunk。"""

    def __init__(self, fail_times=1, mid_stream_error=False):
        self.fail_times = fail_times
        self.mid_stream_error = mid_stream_error
        self.open_count = 0

    def create(self, **kwargs):
        self.open_count += 1
        if self.open_count <= self.fail_times:
            # SDK 惰性：请求在首次迭代时才发出，因此异常也在迭代时抛
            def _err_iter():
                raise _TransientError()
                yield  # pragma: no cover
            return _err_iter()
        chunks = [_openai_chunk("你"), _openai_chunk("好"), _openai_chunk("!")]
        if self.mid_stream_error:
            def _mid_iter():
                yield chunks[0]
                raise _TransientError()
            return _mid_iter()
        return iter(chunks)


class TestOpenAIStreamRetry:
    def _adapter(self, client):
        adapter = _make(OpenAIAdapter, model="test")
        adapter._client = client
        return adapter

    def test_first_chunk_transient_failure_is_retried(self):
        flaky = _FlakyOpenAIStream(fail_times=2)
        client = mock.MagicMock()
        client.chat.completions.create = flaky.create
        adapter = self._adapter(client)

        out = list(adapter.stream_invoke([{"role": "user", "content": "hi"}]))
        assert out == ["你", "好", "!"]
        assert flaky.open_count == 3  # 失败 2 次 + 成功 1 次

    def test_mid_stream_failure_is_not_replayed(self):
        """首 chunk 已产出后的失败不得重放（重放 = 内容重复）。"""
        flaky = _FlakyOpenAIStream(fail_times=0, mid_stream_error=True)
        client = mock.MagicMock()
        client.chat.completions.create = flaky.create
        adapter = self._adapter(client)

        collected = []
        with pytest.raises(Exception):
            for text in adapter.stream_invoke([{"role": "user", "content": "hi"}]):
                collected.append(text)
        assert collected == ["你"]  # 只有已产出的首 chunk，没有重复
        assert flaky.open_count == 1  # 未重试

    def test_fatal_error_not_retried(self):
        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            def _err_iter():
                raise _FatalError()
                yield  # pragma: no cover
            return _err_iter()

        client = mock.MagicMock()
        client.chat.completions.create = _create
        adapter = self._adapter(client)

        with pytest.raises(Exception):
            list(adapter.stream_invoke([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 1  # 无重试

    def test_empty_stream_completes_without_retry(self):
        client = mock.MagicMock()
        client.chat.completions.create = lambda **k: iter([])
        adapter = self._adapter(client)
        assert list(adapter.stream_invoke([{"role": "user", "content": "hi"}])) == []


class TestOpenAIAsyncStreamRetry:
    def test_async_first_chunk_retry(self):
        """asyncio.run 驱动（环境无 pytest-asyncio，不依赖其 mark）。"""
        flaky = _FlakyOpenAIStream(fail_times=1)

        class _AsyncIter:
            def __init__(self, it):
                self._it = it
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration

        async def _create(**kwargs):
            return _AsyncIter(flaky.create(**kwargs))

        client = mock.MagicMock()
        client.chat.completions.create = _create
        adapter = _make(OpenAIAdapter, model="t")
        adapter._async_client = client

        async def _collect():
            out = []
            async for text in adapter.astream_invoke([{"role": "user", "content": "hi"}]):
                out.append(text)
            return out

        out = asyncio.run(_collect())
        assert out == ["你", "好", "!"]
        assert flaky.open_count == 2


class TestAnthropicStreamRetry:
    def test_first_chunk_transient_failure_is_retried(self):
        """SDK 形态：messages.stream() 返回 CM，__enter__ 返回 stream 对象，
        text_stream 是其可暂停/继续的生成器属性，get_final_message 在流尾。"""
        attempts = {"n": 0}

        def _final_message():
            msg = mock.MagicMock()
            msg.usage.input_tokens = 10
            msg.usage.output_tokens = 5
            return msg

        def _err_text_gen():
            raise _TransientError()
            yield  # pragma: no cover

        class _ErrStream:
            """首次迭代 text_stream 即抛瞬时异常的 stream 对象。"""
            def __init__(self):
                self.text_stream = _err_text_gen()

        class _OkStream:
            def __init__(self):
                self.text_stream = iter(["hi", "!"])
            def get_final_message(self):
                return _final_message()

        class _FakeCM:
            def __init__(self, stream):
                self._stream = stream
                self.exited = 0
            def __enter__(self):
                return self._stream
            def __exit__(self, *exc):
                self.exited += 1
                return False

        bad_cm = _FakeCM(_ErrStream())  # 首次 CM：text_stream 抛错
        ok_stream = _OkStream()
        ok_cm = _FakeCM(ok_stream)

        def _stream(**kwargs):
            attempts["n"] += 1
            return bad_cm if attempts["n"] == 1 else ok_cm

        client = mock.MagicMock()
        client.messages.stream = _stream
        adapter = _make(AnthropicAdapter, model="claude", base_url=None)
        adapter._client = client

        out = list(adapter.stream_invoke([{"role": "user", "content": "hi"}]))
        assert out == ["hi", "!"]
        assert attempts["n"] == 2
        # 失败的 cm 被立即释放，成功的 cm 也被关闭
        assert bad_cm.exited == 1
        assert ok_cm.exited == 1


class TestGeminiStreamRetry:
    def test_first_chunk_transient_failure_is_retried(self):
        attempts = {"n": 0}

        def _generate(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                def _err_iter():
                    raise _TransientError()
                    yield  # pragma: no cover
                return _err_iter()

            def _chunk(text):
                c = mock.MagicMock()
                c.text = text
                c.usage_metadata = None
                return c
            return iter([_chunk("a"), _chunk("b")])

        client = mock.MagicMock()
        client.GenerativeModel.return_value.generate_content = _generate
        adapter = _make(GeminiAdapter, model="gemini-pro", base_url=None)
        adapter._client = client

        out = list(adapter.stream_invoke([{"role": "user", "content": "hi"}]))
        assert out == ["a", "b"]
        assert attempts["n"] == 2
