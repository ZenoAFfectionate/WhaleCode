"""LLM适配器 - 支持OpenAI、Anthropic、Gemini等不同接口格式"""

import json as _json
import os
import random
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Dict, Any, Union, AsyncIterator

from .llm_response import LLMResponse, StreamStats
from .exceptions import HelloAgentsException
from .reasoning import extract_reasoning_payload
from .env_utils import env_int, env_float


# --- Cross-provider tool-call normalization (重要-3) --------------------------
# Agents consume ``response.choices[0].message.tool_calls`` (OpenAI shape).
# Anthropic/Gemini return native structures without ``.choices`` and would crash
# the ReAct loop. These light shims wrap those responses into an OpenAI-like
# object so a single agent code path works across providers.

class _NToolFunction:
    """Lightweight OpenAI-compatible ``function`` shim for tool-call normalization (重要-3)."""
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _NToolCall:
    """Lightweight OpenAI-compatible ``tool_call`` shim."""
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = _NToolFunction(name, arguments)


class _NMessage:
    """Lightweight OpenAI-compatible ``message`` shim."""
    def __init__(self, content: str, tool_calls: Optional[List["_NToolCall"]] = None):
        self.content = content or ""
        self.tool_calls = tool_calls or None


class _NChoice:
    """Lightweight OpenAI-compatible ``choice`` shim."""
    def __init__(self, message: "_NMessage"):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _NResponse:
    """OpenAI-compatible response shim (only the attributes agents read)."""

    def __init__(self, message: "_NMessage", *, usage: Any = None, usage_metadata: Any = None):
        self.choices = [_NChoice(message)]
        self.usage = usage
        self.usage_metadata = usage_metadata


def normalize_anthropic_tool_response(response: Any) -> _NResponse:
    """Convert an Anthropic Messages response into the OpenAI-compatible shim."""
    content_text = ""
    tool_calls: List[_NToolCall] = []
    for block in getattr(response, "content", None) or []:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use" or (getattr(block, "name", None) and getattr(block, "input", None) is not None):
            tool_calls.append(
                _NToolCall(
                    id=str(getattr(block, "id", "") or ""),
                    name=str(getattr(block, "name", "") or ""),
                    arguments=_json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                )
            )
        elif getattr(block, "text", None) is not None:
            content_text += getattr(block, "text", "") or ""
    return _NResponse(_NMessage(content_text, tool_calls), usage=getattr(response, "usage", None))


def normalize_gemini_tool_response(response: Any) -> _NResponse:
    """Convert a Gemini generate_content response into the OpenAI-compatible shim."""
    content_text = ""
    tool_calls: List[_NToolCall] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            function_call = getattr(part, "function_call", None)
            if function_call is not None and getattr(function_call, "name", None):
                args = getattr(function_call, "args", {}) or {}
                try:
                    args = dict(args)
                except (TypeError, ValueError):
                    args = {}
                tool_calls.append(
                    _NToolCall(
                        id=str(getattr(function_call, "name", "")),
                        name=str(getattr(function_call, "name", "")),
                        arguments=_json.dumps(args, ensure_ascii=False),
                    )
                )
            elif getattr(part, "text", None):
                content_text += getattr(part, "text", "") or ""
    return _NResponse(
        _NMessage(content_text, tool_calls),
        usage_metadata=getattr(response, "usage_metadata", None),
    )


# --- Retry with exponential backoff (重要-12) --------------------------------
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_NAME_HINTS = (
    "timeout",
    "connection",
    "apiconnection",
    "serviceunavailable",
    "internalserver",
    "ratelimit",
    "overloaded",
    "temporarilyunavailable",
)


class BaseLLMAdapter(ABC):
    """LLM适配器基类"""

    def __init__(self, api_key: str, base_url: Optional[str], timeout: int, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self._client = None
        self._async_client = None
        # 重要-12: bounded exponential-backoff retry for transient API failures.
        self.max_retries = env_int("LLM_MAX_RETRIES", 2)
        self.retry_base_delay = env_float("LLM_RETRY_BASE_DELAY", 0.5)
        self.retry_max_delay = env_float("LLM_RETRY_MAX_DELAY", 8.0)

    @abstractmethod
    def create_client(self) -> Any:
        """创建客户端实例"""
        pass

    def create_async_client(self) -> Any:
        """创建异步客户端实例（子类可选实现）"""
        return None

    @abstractmethod
    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        pass

    @abstractmethod
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用，返回生成器"""
        pass

    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """异步流式调用（子类可选实现真正的异步）

        默认实现：使用队列 + 线程池包装同步流式方法
        """
        queue = asyncio.Queue()
        try:
            loop = asyncio.get_running_loop()  # 建议-15: prefer running loop
        except RuntimeError:
            loop = asyncio.get_event_loop()

        def _stream_to_queue():
            try:
                for chunk in self.stream_invoke(messages, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # 在线程池中运行同步流式方法
        loop.run_in_executor(None, _stream_to_queue)

        # 从队列中逐个取出 chunk
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    @abstractmethod
    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """工具调用（Function Calling）"""
        pass

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Classify an exception as a transient (retryable) API failure (重要-12)."""
        name = type(exc).__name__.lower()
        if any(hint in name for hint in _RETRYABLE_NAME_HINTS):
            return True
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
            return True
        return False

    def _retry_call(self, fn, *, op: str = "llm"):
        """Call ``fn`` with bounded exponential backoff + jitter on transient errors."""
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if attempt >= self.max_retries or not self._is_retryable_error(exc):
                    raise
                delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** attempt))
                delay *= 0.5 + random.random() * 0.5  # jitter to avoid thundering herds
                time.sleep(delay)
                attempt += 1

    async def _aretry_call(self, fn, *, op: str = "llm"):
        """Async counterpart of ``_retry_call`` (A2): same policy, ``asyncio.sleep`` backoff.

        ``fn`` must be an awaitable-returning callable (e.g. an async function).
        """
        attempt = 0
        while True:
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if attempt >= self.max_retries or not self._is_retryable_error(exc):
                    raise
                delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** attempt))
                delay *= 0.5 + random.random() * 0.5
                await asyncio.sleep(delay)
                attempt += 1


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI兼容接口适配器（默认）

    支持：
    - OpenAI官方API
    - 所有OpenAI兼容接口（DeepSeek、Qwen、Kimi、智谱等）
    - Thinking Models（o1、deepseek-reasoner等）
    """

    def create_client(self) -> Any:
        """创建OpenAI客户端"""
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def create_async_client(self) -> Any:
        """创建OpenAI异步客户端"""
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()
        
        start_time = time.time()
        
        try:
            response = self._retry_call(
                lambda: self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                ),
                op="openai.invoke",
            )

            latency_ms = int((time.time() - start_time) * 1000)
            
            # 提取内容和推理过程
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning_content = extract_reasoning_payload(choice.message).content
            if reasoning_content is None:
                reasoning_content = extract_reasoning_payload(choice).content
            
            # 提取usage信息
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )
            
        except Exception as e:
            raise HelloAgentsException(f"OpenAI API调用失败: {str(e)}")
    
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()

        try:
            # A2: 首 chunk 前的瞬时失败可安全重放——此时尚未向调用方产出任何
            # 内容；一旦开始 yield 则不再重试（重放会导致内容重复）。
            holder: Dict[str, Any] = {}

            def _open_and_take_first():
                stream_obj = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **kwargs
                )
                holder["stream"] = stream_obj
                for first in stream_obj:
                    holder["first"] = first
                    return

            self._retry_call(_open_and_take_first, op="openai.stream")
            stream = holder["stream"]

            collected_content = []
            reasoning_content = None
            usage = {}

            def _chunks():
                if "first" in holder:
                    yield holder["first"]
                yield from stream

            for chunk in _chunks():
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    # 提取内容
                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content
                    
                    reasoning_piece = extract_reasoning_payload(
                        delta,
                        preserve_whitespace=True,
                    ).content
                    if reasoning_piece is not None:
                        if reasoning_content is None:
                            reasoning_content = ""
                        reasoning_content += reasoning_piece

                # 提取usage（流式最后一个chunk可能包含）
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            if reasoning_content is not None:
                reasoning_content = reasoning_content.strip() or None

            # 建议-9: many OpenAI-compatible servers omit usage on streamed
            # responses; fall back to a rough local estimate so token accounting
            # is non-zero instead of silently 0.
            if not usage:
                approx_prompt = sum(len(str(m.get("content", ""))) for m in messages) // 4
                approx_completion = len("".join(collected_content)) // 4
                usage = {
                    "prompt_tokens": approx_prompt,
                    "completion_tokens": approx_completion,
                    "total_tokens": approx_prompt + approx_completion,
                    "estimated": True,
                }

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )

        except Exception as e:
            raise HelloAgentsException(f"OpenAI API流式调用失败: {str(e)}")

    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """真正的异步流式调用（使用 OpenAI 原生异步客户端）"""
        if not self._async_client:
            self._async_client = self.create_async_client()

        start_time = time.time()

        try:
            # A2: 与同步版同构——首 chunk 前的瞬时失败经 _aretry_call 重放。
            holder: Dict[str, Any] = {}

            async def _open_and_take_first():
                stream_obj = await self._async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **kwargs
                )
                holder["stream"] = stream_obj
                async for first in stream_obj:
                    holder["first"] = first
                    return

            await self._aretry_call(_open_and_take_first, op="openai.astream")

            collected_content = []
            reasoning_content = None
            usage = {}

            async def _achunks():
                if "first" in holder:
                    yield holder["first"]
                async for c in holder["stream"]:
                    yield c

            async for chunk in _achunks():
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # 提取内容
                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content

                    reasoning_piece = extract_reasoning_payload(
                        delta,
                        preserve_whitespace=True,
                    ).content
                    if reasoning_piece is not None:
                        if reasoning_content is None:
                            reasoning_content = ""
                        reasoning_content += reasoning_piece

                # 提取usage（流式最后一个chunk可能包含）
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            if reasoning_content is not None:
                reasoning_content = reasoning_content.strip() or None

            # 建议-9: many OpenAI-compatible servers omit usage on streamed
            # responses; fall back to a rough local estimate so token accounting
            # is non-zero instead of silently 0.
            if not usage:
                approx_prompt = sum(len(str(m.get("content", ""))) for m in messages) // 4
                approx_completion = len("".join(collected_content)) // 4
                usage = {
                    "prompt_tokens": approx_prompt,
                    "completion_tokens": approx_completion,
                    "total_tokens": approx_prompt + approx_completion,
                    "estimated": True,
                }

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )

        except Exception as e:
            raise HelloAgentsException(f"OpenAI API异步流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict],
                         tool_choice: Union[str, Dict] = "auto", **kwargs) -> Any:
        """工具调用（Function Calling）"""
        if not self._client:
            self._client = self.create_client()

        try:
            response = self._retry_call(
                lambda: self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                ),
                op="openai.tools",
            )
            return response

        except Exception as e:
            raise HelloAgentsException(f"OpenAI Function Calling调用失败: {str(e)}")


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude适配器

    处理Claude特有的消息格式：
    - system参数独立（不在messages中）
    - 消息格式转换
    """

    def create_client(self) -> Any:
        """创建Anthropic客户端"""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise HelloAgentsException(
                "使用Anthropic需要安装: pip install anthropic"
            )

        return Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _convert_messages(self, messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
        """转换消息格式，提取system消息"""
        system_content = None
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                converted_messages.append(msg)

        return system_content, converted_messages

    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            response = self._client.messages.create(**request_params)

            latency_ms = int((time.time() - start_time) * 1000)

            # 提取内容
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text

            # 提取usage
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Anthropic API调用失败: {str(e)}")

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "stream": True,
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            usage = {}

            # A2: 首 chunk 前的瞬时失败可安全重放。Anthropic 的 MessageStream
            # 在 __enter__ 时由后台线程发起请求，失败会在迭代 text_stream 时
            # 冒出——因此重试粒度必须覆盖"进入 + 取首个 text"。上下文管理器
            # 手动管理：失败时立即 __exit__ 释放，成功后由外层 finally 收尾。
            # text_stream 为生成器语义：取首个后再迭代会从暂停处继续（不重复）。
            holder: Dict[str, Any] = {}

            def _open_and_take_first():
                cm = self._client.messages.stream(**request_params)
                stream_obj = cm.__enter__()
                try:
                    for text in stream_obj.text_stream:
                        holder["first"] = text
                        break
                except BaseException:
                    cm.__exit__(None, None, None)
                    raise
                holder["cm"] = cm
                holder["stream"] = stream_obj

            self._retry_call(_open_and_take_first, op="anthropic.stream")

            try:
                if "first" in holder:
                    yield holder["first"]
                for text in holder["stream"].text_stream:
                    yield text

                # 获取最终消息以提取usage（get_final_message 属于 stream 对象，
                # 与真实 SDK 的 `with ... as stream: stream.get_final_message()` 一致）
                final_message = holder["stream"].get_final_message()
                if hasattr(final_message, 'usage') and final_message.usage:
                    usage = {
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                    }
            except BaseException:
                holder["cm"].__exit__(None, None, None)
                raise
            else:
                holder["cm"].__exit__(None, None, None)

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Anthropic API流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """工具调用（Anthropic格式）"""
        if not self._client:
            self._client = self.create_client()

        system_content, converted_messages = self._convert_messages(messages)

        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "tools": tools,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            response = self._retry_call(
                lambda: self._client.messages.create(**request_params), op="anthropic.tools"
            )
            return normalize_anthropic_tool_response(response)

        except Exception as e:
            raise HelloAgentsException(f"Anthropic工具调用失败: {str(e)}")


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini适配器

    处理Gemini特有的API格式
    """

    def create_client(self) -> Any:
        """创建Gemini客户端"""
        try:
            import google.generativeai as genai
        except ImportError:
            raise HelloAgentsException(
                "使用Gemini需要安装: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)
        return genai

    def _convert_messages(self, messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
        """转换消息格式"""
        system_instruction = None
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                # Gemini使用 "user" 和 "model" 作为角色
                role = "model" if msg["role"] == "assistant" else "user"
                converted_messages.append({
                    "role": role,
                    "parts": [msg["content"]]
                })

        return system_instruction, converted_messages

    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 创建生成配置
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

            # 创建模型
            model_params = {"model_name": self.model}
            if system_instruction:
                model_params["system_instruction"] = system_instruction

            model = self._client.GenerativeModel(**model_params)

            # 生成内容
            response = model.generate_content(
                converted_messages,
                generation_config=generation_config if generation_config else None
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # 提取内容
            content = response.text if hasattr(response, 'text') else ""

            # 提取usage
            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Gemini API调用失败: {str(e)}")

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

            model_params = {"model_name": self.model}
            if system_instruction:
                model_params["system_instruction"] = system_instruction

            model = self._client.GenerativeModel(**model_params)

            usage = {}

            # A2: 首 chunk 前的瞬时失败可安全重放（与 OpenAI 同构）。
            holder: Dict[str, Any] = {}

            def _open_and_take_first():
                stream_obj = model.generate_content(
                    converted_messages,
                    generation_config=generation_config if generation_config else None,
                    stream=True
                )
                holder["stream"] = stream_obj
                for first in stream_obj:
                    holder["first"] = first
                    return

            self._retry_call(_open_and_take_first, op="gemini.stream")

            def _chunks():
                if "first" in holder:
                    yield holder["first"]
                yield from holder["stream"]

            for chunk in _chunks():
                if hasattr(chunk, 'text'):
                    yield chunk.text

                # 尝试提取usage（可能在最后一个chunk；非末尾 chunk 的
                # usage_metadata 为 None——A2 测试暴露的既有缺陷，加 None 防护）
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = {
                        "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                        "completion_tokens": chunk.usage_metadata.candidates_token_count,
                        "total_tokens": chunk.usage_metadata.total_token_count,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Gemini API流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """工具调用（Gemini格式）"""
        if not self._client:
            self._client = self.create_client()

        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 转换工具格式为Gemini格式
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    gemini_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    })

            model_params = {"model_name": self.model}
            if system_instruction:
                model_params["system_instruction"] = system_instruction

            model = self._client.GenerativeModel(**model_params, tools=gemini_tools)

            response = self._retry_call(
                lambda: model.generate_content(converted_messages), op="gemini.tools"
            )
            return normalize_gemini_tool_response(response)

        except Exception as e:
            raise HelloAgentsException(f"Gemini工具调用失败: {str(e)}")


def create_adapter(
    api_key: str,
    base_url: Optional[str],
    timeout: int,
    model: str
) -> BaseLLMAdapter:
    """
    根据base_url自动选择适配器

    检测逻辑：
    - anthropic.com -> AnthropicAdapter
    - googleapis.com 或 generativelanguage -> GeminiAdapter
    - 其他 -> OpenAIAdapter（默认）
    """
    if base_url:
        base_url_lower = base_url.lower()

        if "anthropic.com" in base_url_lower:
            return AnthropicAdapter(api_key, base_url, timeout, model)

        if "googleapis.com" in base_url_lower or "generativelanguage" in base_url_lower:
            return GeminiAdapter(api_key, base_url, timeout, model)

    # 默认使用OpenAI适配器（兼容所有OpenAI格式接口）
    return OpenAIAdapter(api_key, base_url, timeout, model)
