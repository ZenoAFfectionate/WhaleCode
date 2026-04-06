"""HistoryManager - 历史消息管理器

职责：
- 消息追加（只追加，不编辑，缓存友好）
- 历史压缩（生成 summary + 保留最近轮次）
- 工具输出微压缩（保留可恢复路径）
- 会话序列化/反序列化
- 轮次边界检测
- 历史到 LLM messages 的统一投影
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.message import Message

if TYPE_CHECKING:
    from .token_counter import TokenCounter


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUMMARY_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "summary_prompt.md"
_FULL_OUTPUT_PATH_PATTERNS = (
    re.compile(r"^full_output_path:\s*(.+)$", re.MULTILINE),
    re.compile(r"^Full output saved to:\s*(.+)$", re.MULTILINE),
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
    """提取 summary_prompt.md 中指定 section 的内容。"""
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


def format_compact_summary(summary: str) -> str:
    """清洗 LLM 返回的摘要文本。"""
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


class HistoryManager:
    """历史管理器

    特性：
    - 只追加，不编辑（缓存友好）
    - 自动压缩历史（summary + 保留最近轮次）
    - 支持会话保存/加载
    - 统一上下文压缩入口

    说明：
    - token 计算统一复用现有 TokenCounter
    - agent 侧只需调用 HistoryManager，不再维护独立 compactor 逻辑
    """

    SUMMARY_HEADING = "## Archived Session Summary"
    SUMMARY_NOTE_PREFIX = "[Conversation summary]"
    SYSTEM_NOTE_PREFIX = "[System note]"
    TOOL_RESULT_PREFIX = "Previous tool result"
    ESSENTIAL_CONTEXT_PREFIX = "[Preserved context]"
    ESSENTIAL_CONTEXT_HEADING = "## Essential Context Snapshot"
    COMPACTION_TRANSCRIPT_LABEL = "Transcript path:"
    COMPACTION_RECENT_TAIL_NOTE = "Recent messages are preserved verbatim below."
    TOOL_RESULT_PLACEHOLDER = "[Previous tool result: {tool_name} - truncated to save context]"

    COMPACTABLE_TOOL_OUTPUTS = {
        "Read",
        "Bash",
        "Grep",
        "Glob",
        "WebSearch",
        "WebFetch",
        "Edit",
        "Write",
    }
    WORKING_FILE_TOOLS = {"Read", "Edit", "Write", "Delete"}
    INTERNAL_USER_KINDS = {"retry_reminder"}

    def __init__(
        self,
        compression_threshold: float = 0.8,
        token_counter: Optional["TokenCounter"] = None,
        config: Optional[Any] = None,
    ):
        """初始化历史管理器。

        Args:
            compression_threshold: 压缩阈值
            token_counter: 现有 TokenCounter 实例
            config: Agent Config，可选
        """
        if config is not None:
            compression_threshold = float(
                getattr(config, "compression_threshold", compression_threshold)
            )

        self._history: List[Message] = []
        self.compression_threshold = compression_threshold
        self.token_counter = token_counter
        self.config = config
        self._estimated_history_token_count = 0
        self._recorded_usage = self._empty_usage_snapshot()

    def set_token_counter(self, token_counter: "TokenCounter") -> None:
        """绑定现有 TokenCounter，并重算缓存 token 数。"""
        self.token_counter = token_counter
        self._estimated_history_token_count = self._count_messages(self._history)

    def set_config(self, config: Any) -> None:
        """绑定 Config，便于直接从 HistoryManager 驱动压缩。"""
        self.config = config
        self.compression_threshold = float(
            getattr(config, "compression_threshold", self.compression_threshold)
        )

    def append(self, message: Message) -> None:
        """追加消息（只追加，不编辑）。"""
        self._history.append(message)
        self._estimated_history_token_count += self._count_message(message)
        self._mark_usage_stale()

    def get_history(self) -> List[Message]:
        """获取历史副本。"""
        return self._history.copy()

    def get_messages(self) -> List[Message]:
        """兼容文档中的 get_messages() 名称。"""
        return self.get_history()

    def clear(self) -> None:
        """清空历史。"""
        self._history.clear()
        self._estimated_history_token_count = 0
        self._recorded_usage = self._empty_usage_snapshot()

    def get_token_count(self) -> int:
        """返回最近一次真实记录到的 prompt token 数。"""
        return int(self._recorded_usage.get("prompt_tokens", 0) or 0)

    def get_estimated_token_count(self) -> int:
        """返回基于本地 TokenCounter 的历史估算值。"""
        return self._estimated_history_token_count

    def get_usage_snapshot(self) -> Dict[str, Any]:
        """返回最近一次记录到的真实 usage 快照。"""
        return dict(self._recorded_usage)

    def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ) -> None:
        """记录最近一次模型调用返回的真实 usage。"""
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        self._recorded_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(total_tokens or 0),
            "stale": False,
            "recorded_at": datetime.now().isoformat(),
        }

    def estimate_rounds(self, history: Optional[Sequence[Message]] = None) -> int:
        """预估完整轮次数。"""
        return len(self._round_boundaries(self._resolve_history(history)))

    def find_round_boundaries(self, history: Optional[Sequence[Message]] = None) -> List[int]:
        """查找每轮的起始索引。"""
        return self._round_boundaries(self._resolve_history(history))

    def get_compression_split(
        self,
        history: Optional[Sequence[Message]] = None,
        retain_rounds: Optional[int] = None,
    ) -> Optional[Tuple[List[Message], List[Message]]]:
        """将历史拆成 (待压缩部分, 保留的最近轮次)。"""
        source = self._resolve_history(history)
        boundaries = self._round_boundaries(source)
        target_rounds = self._default_preserve_recent_rounds() if retain_rounds is None else max(0, int(retain_rounds))
        if len(boundaries) <= target_rounds:
            return None

        keep_from_index = boundaries[-target_rounds] if target_rounds > 0 else len(source)
        return source[:keep_from_index], source[keep_from_index:]

    def compress(self, summary: str) -> None:
        """简单压缩：旧历史替换为 summary，保留最近 N 轮。"""
        split = self.get_compression_split()
        if split is None:
            return

        _, retained_history = split
        summary_msg = self.build_summary_message(
            summary,
            metadata={"compressed_at": datetime.now().isoformat()},
        )
        self._history = [summary_msg, *retained_history]
        self._estimated_history_token_count = self._count_messages(self._history)
        self._mark_usage_stale()

    def build_summary(
        self,
        *,
        llm: Any,
        history: Optional[Sequence[Message]] = None,
        focus: Optional[str] = None,
    ) -> str:
        """仅生成 smart summary 文本，不直接改写历史。"""
        source = self._resolve_history(history)
        split = self.get_compression_split(source)
        if split is None:
            raise ValueError("Not enough completed rounds to summarize.")

        to_compress, _retained = split
        summary_messages = self._messages_for_summary(to_compress)
        if not summary_messages:
            raise ValueError("No eligible history remains after filtering summary input.")

        summary = self._summarize_messages(
            summary_messages,
            self._serialize_messages(summary_messages),
            llm,
            focus,
            transcript_path="[unavailable]",
        )
        return (
            f"{summary}\n\n---\n"
            f"(Compressed, retaining the most recent {self._default_preserve_recent_rounds()} complete rounds)"
        )

    def build_summary_message(
        self,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """构造标准化的 summary 消息。"""
        return Message(
            content=self.normalize_summary_content(summary),
            role="summary",
            metadata=metadata or {},
        )

    def build_preserved_context_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """构造压缩后保留的结构化上下文消息。"""
        payload = dict(metadata or {})
        payload["kind"] = "preserved_context"
        return Message(content=content, role="system", metadata=payload)

    def build_assistant_tool_call_message(
        self,
        tool_calls: Sequence[Dict[str, Any]],
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """构造 assistant 工具调用消息，结构化保存在 metadata 中。"""
        call_entries = self._normalize_tool_calls(tool_calls)
        payload = dict(metadata or {})
        payload["tool_calls"] = call_entries
        return Message(
            content=(content or "").strip(),
            role="assistant",
            metadata=payload,
        )

    def normalize_summary_content(self, summary: str) -> str:
        """确保 summary 使用统一 heading。"""
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
        history: Optional[Sequence[Message]] = None,
    ) -> List[Dict[str, Any]]:
        """将历史投影为 LLM 可直接使用的 chat messages。"""
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for message in self._resolve_history(history):
            projected = self._project_message_for_llm(message)
            if projected is not None:
                messages.append(projected)

        if latest_user_input is not None:
            messages.append({"role": "user", "content": latest_user_input})
        return messages

    def estimate_tokens(
        self,
        history: Optional[Sequence[Message]] = None,
        *,
        system_prompt: Optional[str] = None,
        latest_user_input: Optional[str] = None,
    ) -> int:
        """估算当前 prompt token 数。"""
        messages = self.build_llm_messages(
            system_prompt=system_prompt,
            latest_user_input=latest_user_input,
            history=history,
        )
        total = 0
        counter = self._get_token_counter()
        for message in messages:
            total += counter.count_text(message.get("content", "")) + 4
        return total

    def get_compact_trigger_limit(self) -> int:
        """返回触发压缩的有效 token 阈值。"""
        config = self._require_config()
        context_window = max(1, int(getattr(config, "context_window", 1)))
        compression_threshold = float(
            getattr(config, "compression_threshold", self.compression_threshold) or 0.8
        )
        compression_threshold = min(max(compression_threshold, 0.01), 1.0)

        percentage_limit = max(1, int(context_window * compression_threshold))
        output_buffer = max(0, int(getattr(config, "compact_output_buffer", 0)))
        buffer_limit = max(1, context_window - output_buffer)
        return min(percentage_limit, buffer_limit)

    def should_compress(
        self,
        *,
        latest_prompt_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        latest_user_input: Optional[str] = None,
        history: Optional[Sequence[Message]] = None,
    ) -> bool:
        """判断是否需要压缩。"""
        config = self._require_config()
        if not bool(getattr(config, "compact_enabled", True)):
            return False

        estimated_tokens = self.estimate_tokens(
            history=history,
            system_prompt=system_prompt,
            latest_user_input=latest_user_input,
        )
        prompt_tokens = int(latest_prompt_tokens or 0)
        token_signal = max(prompt_tokens, estimated_tokens)
        if token_signal >= self.get_compact_trigger_limit():
            return True

        return False

    def micro_compact_tool_results(self) -> bool:
        """将较早的大工具结果替换为可恢复占位符。"""
        config = self._require_config()
        keep_recent = max(0, int(getattr(config, "compact_keep_recent_tool_results", 5)))
        tool_indices = [index for index, message in enumerate(self._history) if message.role == "tool"]
        if len(tool_indices) <= keep_recent:
            return False

        tool_name_map = self._build_tool_name_map(self._history)
        changed = False
        for reverse_index, message_index in enumerate(reversed(tool_indices)):
            if reverse_index < keep_recent:
                continue

            message = self._history[message_index]
            content = (message.content or "").strip()
            if len(content) <= 200 or content.startswith("[Previous tool result:"):
                continue

            tool_call_id = self._message_metadata(message).get("tool_call_id", "")
            tool_name = self._message_metadata(message).get("tool_name") or tool_name_map.get(
                tool_call_id,
                "unknown",
            )
            if tool_name not in self.COMPACTABLE_TOOL_OUTPUTS:
                continue

            placeholder = self.TOOL_RESULT_PLACEHOLDER.format(tool_name=tool_name)
            metadata = self._message_metadata(message)
            full_output_path = metadata.get("full_output_path")
            if not isinstance(full_output_path, str) or not full_output_path.strip():
                full_output_path = self._extract_full_output_path(content)
            elif isinstance(full_output_path, str):
                full_output_path = full_output_path.strip()
            if full_output_path:
                placeholder = f"{placeholder}\nfull_output_path: {full_output_path}"

            if placeholder == message.content:
                continue

            if full_output_path:
                metadata["full_output_path"] = full_output_path
            metadata["tool_name"] = tool_name
            self._history[message_index] = self._clone_message(
                message,
                content=placeholder,
                metadata=metadata,
            )
            changed = True

        if changed:
            self._estimated_history_token_count = self._count_messages(self._history)
            self._mark_usage_stale()
        return changed

    def maybe_compact(
        self,
        *,
        llm: Any,
        system_prompt: Optional[str] = None,
        latest_prompt_tokens: Optional[int] = None,
        focus: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """统一压缩入口：先微压缩，再按阈值决定是否生成 summary。"""
        self.micro_compact_tool_results()
        if not self.should_compress(
            latest_prompt_tokens=latest_prompt_tokens,
            system_prompt=system_prompt,
        ):
            return None
        try:
            return self.compact_with_llm(llm=llm, system_prompt=system_prompt, focus=focus)
        except Exception:
            return None

    def compact_with_llm(
        self,
        *,
        llm: Any,
        system_prompt: Optional[str] = None,
        focus: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """使用 LLM 对旧历史做 continuation-ready 压缩。"""
        if not self._history:
            return None

        config = self._require_config()
        before_tokens = self.estimate_tokens(system_prompt=system_prompt)
        transcript_path = self._save_transcript(self._history)
        preferred_rounds = max(
            0,
            int(getattr(config, "compact_preserve_recent_rounds", 0) or 0),
        )
        max_preserved_rounds = self._max_preserved_rounds(self._history, preferred_rounds)
        best_candidate: Optional[List[Message]] = None
        best_tokens: Optional[int] = None
        summary_input_budget = self._summary_input_token_budget(
            focus=focus,
            transcript_path=transcript_path,
        )

        for preserved_rounds in range(max_preserved_rounds, -1, -1):
            summary_source, preserved_tail = self._split_for_recent_rounds(
                self._history,
                preserved_rounds,
            )
            if not summary_source:
                continue

            summary_messages = self._messages_for_summary(summary_source)
            if not summary_messages:
                continue

            essential_context = self._build_essential_context_message(summary_messages)
            conversation_text = self._serialize_messages(
                summary_messages,
                max_tokens=summary_input_budget,
            )
            summary = self._summarize_messages(
                summary_messages,
                conversation_text,
                llm,
                focus,
                transcript_path=transcript_path,
            )
            candidate = self._build_compacted_history(
                summary=summary,
                preserved_tail=preserved_tail,
                essential_context=essential_context,
                transcript_path=transcript_path,
            )
            candidate_tokens = self.estimate_tokens(history=candidate, system_prompt=system_prompt)

            if best_tokens is None or candidate_tokens < best_tokens:
                best_candidate = candidate
                best_tokens = candidate_tokens

            if candidate_tokens <= self.get_compact_trigger_limit():
                break

        if best_candidate is None:
            return None

        self._history = best_candidate
        self._estimated_history_token_count = self._count_messages(self._history)
        self._mark_usage_stale()
        after_tokens = self.estimate_tokens(system_prompt=system_prompt)
        return {
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "saved_tokens": before_tokens - after_tokens,
            "transcript_path": transcript_path,
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于会话保存）。"""
        return {
            "history": [msg.to_dict() for msg in self._history],
            "created_at": datetime.now().isoformat(),
            "rounds": self.estimate_rounds(),
            "usage": self.get_usage_snapshot(),
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载（用于会话恢复）。"""
        self._history = [
            Message.from_dict(msg_data)
            for msg_data in data.get("history", [])
        ]
        self._estimated_history_token_count = self._count_messages(self._history)
        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = usage.get("total_tokens")
            if total_tokens is not None:
                total_tokens = int(total_tokens or 0)
            self.record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            if bool(usage.get("stale")):
                self._mark_usage_stale()
        else:
            self._recorded_usage = self._empty_usage_snapshot()

    def _tool_result_prompt_prefix(self, message: Message) -> str:
        """为投喂给模型的工具结果构造更明确的前缀。"""
        metadata = self._message_metadata(message)
        tool_name = metadata.get("tool_name")
        if isinstance(tool_name, str):
            tool_name = tool_name.strip()
        else:
            tool_name = ""

        if tool_name:
            return f"{self.TOOL_RESULT_PREFIX} [{tool_name}]:"
        return f"{self.TOOL_RESULT_PREFIX}:"

    @staticmethod
    def _tool_call_arguments_for_llm(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False, separators=(",", ":"))

    def _project_message_for_llm(self, message: Message) -> Optional[Dict[str, Any]]:
        """将持久化历史映射为下一次模型调用的合法 chat messages。"""
        content = message.content or ""
        if message.role == "user":
            return {"role": "user", "content": content}
        if message.role == "assistant":
            tool_calls = self._assistant_tool_calls(message)
            if tool_calls:
                return {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": str(tool_call.get("id", "") or ""),
                            "type": "function",
                            "function": {
                                "name": str(tool_call.get("name", "unknown") or "unknown"),
                                "arguments": self._tool_call_arguments_for_llm(tool_call.get("arguments")),
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            if not content.strip():
                return None
            return {"role": "assistant", "content": content}
        if message.role == "summary":
            return {"role": "user", "content": f"{self.SUMMARY_NOTE_PREFIX}\n{content}"}
        if message.role == "system":
            if self._is_preserved_context_message(message):
                return {"role": "user", "content": content}
            return {"role": "user", "content": f"{self.SYSTEM_NOTE_PREFIX}\n{content}"}
        if message.role == "tool":
            tool_call_id = self._message_metadata(message).get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id.strip(),
                    "content": content,
                }
            return {
                "role": "assistant",
                "content": f"{self._tool_result_prompt_prefix(message)}\n{content}",
            }
        return None

    def _require_config(self) -> Any:
        if self.config is None:
            raise RuntimeError("HistoryManager requires config to perform context compaction.")
        return self.config

    def _get_token_counter(self) -> "TokenCounter":
        if self.token_counter is None:
            from .token_counter import TokenCounter

            self.token_counter = TokenCounter(model="gpt-4")
        return self.token_counter

    def _count_message(self, message: Message) -> int:
        return self._get_token_counter().count_message(message)

    def _count_messages(self, messages: Sequence[Message]) -> int:
        if not messages:
            return 0
        return self._get_token_counter().count_messages(messages)

    @staticmethod
    def _empty_usage_snapshot() -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "stale": False,
            "recorded_at": None,
        }

    def _mark_usage_stale(self) -> None:
        if self._recorded_usage.get("prompt_tokens", 0):
            self._recorded_usage["stale"] = True

    def _resolve_history(self, history: Optional[Sequence[Message]]) -> List[Message]:
        if history is None:
            return self._history
        if isinstance(history, list):
            return history
        return list(history)

    def _normalize_tool_calls(self, tool_calls: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {}) or {}
            arguments = function.get("arguments", tool_call.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            normalized.append(
                {
                    "id": tool_call.get("id", ""),
                    "name": function.get("name", tool_call.get("name", "unknown")),
                    "arguments": arguments if arguments is not None else {},
                }
            )
        return normalized

    def _assistant_tool_calls(self, message: Message) -> List[Dict[str, Any]]:
        metadata_calls = self._message_metadata(message).get("tool_calls")
        if isinstance(metadata_calls, list):
            return self._normalize_tool_calls(metadata_calls)
        return []

    def _build_tool_name_map(self, history: Sequence[Message]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for message in history:
            if message.role != "assistant":
                continue
            for tool_call in self._assistant_tool_calls(message):
                tool_call_id = str(tool_call.get("id", "")).strip()
                tool_name = str(tool_call.get("name", "unknown")).strip() or "unknown"
                if tool_call_id:
                    mapping[tool_call_id] = tool_name
        return mapping

    def _build_compacted_history(
        self,
        *,
        summary: str,
        preserved_tail: List[Message],
        essential_context: Optional[Message] = None,
        transcript_path: Optional[str] = None,
    ) -> List[Message]:
        lines: List[str] = []
        if transcript_path:
            lines.append(f"{self.COMPACTION_TRANSCRIPT_LABEL} {transcript_path}")
        if preserved_tail:
            lines.append(self.COMPACTION_RECENT_TAIL_NOTE)
        if lines:
            lines.append("")
        if summary:
            lines.append(summary)

        summary_message = self.build_summary_message(
            "\n".join(lines).strip(),
            metadata={
                "compressed_at": datetime.now().isoformat(),
                "transcript_path": transcript_path,
            },
        )
        compacted = [summary_message]
        if essential_context is not None:
            compacted.append(essential_context)
        compacted.extend(preserved_tail)
        return compacted

    def _summarize_messages(
        self,
        messages: List[Message],
        conversation_text: str,
        llm: Any,
        focus: Optional[str],
        transcript_path: Optional[str] = None,
    ) -> str:
        summary_system_prompt, summary_user_template = _load_summary_prompts()
        config = self._require_config()
        focus_instruction = f"Pay special attention to: {focus}" if focus else ""
        summary_prompt = summary_user_template.format(
            conversation=conversation_text,
            focus_instruction=focus_instruction,
            max_tokens=int(getattr(config, "summary_max_tokens", 2048)),
            transcript_path=transcript_path or "[unavailable]",
        )

        response = llm.invoke(
            messages=[
                {"role": "system", "content": summary_system_prompt},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=float(getattr(config, "summary_temperature", 0.3)),
            max_tokens=int(getattr(config, "summary_max_tokens", 2048)),
        )
        summary = format_compact_summary(getattr(response, "content", str(response)))
        if not summary:
            raise ValueError("Smart summary response was empty.")
        return summary

    def _summary_input_token_budget(
        self,
        *,
        focus: Optional[str],
        transcript_path: Optional[str],
    ) -> int:
        summary_system_prompt, summary_user_template = _load_summary_prompts()
        config = self._require_config()
        focus_instruction = f"Pay special attention to: {focus}" if focus else ""
        prompt_without_conversation = summary_user_template.format(
            conversation="",
            focus_instruction=focus_instruction,
            max_tokens=int(getattr(config, "summary_max_tokens", 2048)),
            transcript_path=transcript_path or "[unavailable]",
        )
        counter = self._get_token_counter()
        prompt_overhead = (
            counter.count_text(summary_system_prompt)
            + counter.count_text(prompt_without_conversation)
            + 8
        )
        summary_max_tokens = int(getattr(config, "summary_max_tokens", 2048))
        context_window = int(getattr(config, "context_window", 32768))
        safety_margin = max(128, min(2048, summary_max_tokens))
        return max(
            256,
            context_window - summary_max_tokens - prompt_overhead - safety_margin,
        )

    def _serialize_messages(
        self,
        messages: List[Message],
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        serialized = [self._serialize_single_message(message) for message in messages]
        if not serialized:
            return ""

        counter = self._get_token_counter()
        full_text = "\n\n".join(serialized)
        if max_tokens is None or counter.count_text(full_text) <= max_tokens:
            return full_text

        selected: List[str] = []
        total_tokens = 0
        separator_tokens = counter.count_text("\n\n")
        marker_tokens = 0
        for line in reversed(serialized):
            line_tokens = counter.count_text(line)
            if selected:
                line_tokens += separator_tokens
            if total_tokens + line_tokens > max_tokens:
                break
            selected.append(line)
            total_tokens += line_tokens

        selected.reverse()
        if not selected:
            return self._truncate_note_text(serialized[-1], max_tokens=max_tokens)

        if len(selected) < len(serialized):
            marker_tokens = counter.count_text(_SUMMARY_BUDGET_MARKER) + separator_tokens
            while selected and total_tokens + marker_tokens > max_tokens:
                removed = selected.pop(0)
                total_tokens -= counter.count_text(removed)
                if selected:
                    total_tokens -= separator_tokens
        if selected and total_tokens + marker_tokens <= max_tokens:
            selected.insert(0, _SUMMARY_BUDGET_MARKER)
        return "\n\n".join(selected)

    def _messages_for_summary(self, messages: List[Message]) -> List[Message]:
        filtered: List[Message] = []
        for message in messages:
            if self._is_preserved_context_message(message):
                continue
            if self._is_internal_user_message(message):
                continue
            filtered.append(message)
        return filtered

    def _build_essential_context_message(self, messages: Sequence[Message]) -> Optional[Message]:
        snapshot = self._collect_preserved_context_snapshot(messages)
        sections: List[str] = [self.ESSENTIAL_CONTEXT_PREFIX, "", self.ESSENTIAL_CONTEXT_HEADING]

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
            sections.extend(
                ["### Working Set Files", *[f"- {path}" for path in snapshot.working_files], ""]
            )
        if snapshot.recent_commands:
            sections.extend(
                ["### Recent Commands", *[f"- {command}" for command in snapshot.recent_commands], ""]
            )
        if snapshot.loaded_skills:
            sections.extend(
                ["### Loaded Skills", *[f"- {skill}" for skill in snapshot.loaded_skills], ""]
            )
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

        if sections == [self.ESSENTIAL_CONTEXT_PREFIX, "", self.ESSENTIAL_CONTEXT_HEADING]:
            return None

        return self.build_preserved_context_message(
            "\n".join(sections),
            metadata={"generated_at": datetime.now().isoformat()},
        )

    def _collect_preserved_context_snapshot(
        self,
        messages: Sequence[Message],
    ) -> _PreservedContextSnapshot:
        snapshot = _PreservedContextSnapshot()
        seen_working_files: Set[str] = set()
        seen_commands: Set[str] = set()
        seen_skills: Set[str] = set()
        seen_paths: Set[str] = set()
        todo_captured = False

        for message in reversed(list(messages)):
            if message.role == "user" and snapshot.latest_user_request is None:
                content = (message.content or "").strip()
                if content and not self._is_archived_user_content(content) and not self._is_internal_user_message(message):
                    snapshot.latest_user_request = self._truncate_note_text(
                        content,
                        max_tokens=_ESSENTIAL_USER_REQUEST_TOKENS,
                    )

            if message.role == "tool" and len(snapshot.recoverable_paths) < _RECOVERABLE_OUTPUT_LIMIT:
                metadata = self._message_metadata(message)
                full_output_path = metadata.get("full_output_path") or self._extract_full_output_path(
                    message.content or ""
                )
                if isinstance(full_output_path, str) and full_output_path.strip():
                    self._append_unique(
                        snapshot.recoverable_paths,
                        seen_paths,
                        full_output_path.strip(),
                        limit=_RECOVERABLE_OUTPUT_LIMIT,
                    )

            if message.role != "assistant":
                continue

            for tool_call in reversed(self._assistant_tool_calls(message)):
                tool_name = tool_call.get("name", "unknown")
                arguments = tool_call.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}

                if not todo_captured and tool_name == "TodoWrite":
                    snapshot.todo_lines = self._todo_lines_from_arguments(arguments)
                    todo_captured = True

                if tool_name in self.WORKING_FILE_TOOLS:
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

    def _serialize_single_message(self, message: Message) -> str:
        role = message.role
        content = message.content or ""

        suffix_parts: List[str] = []
        if role == "assistant":
            tool_calls = self._assistant_tool_calls(message)
            if tool_calls:
                suffix_parts.append(
                    "tool_calls=" + ", ".join(tool_call.get("name", "unknown") for tool_call in tool_calls)
                )
        if role == "tool":
            tool_call_id = self._message_metadata(message).get("tool_call_id")
            if tool_call_id:
                suffix_parts.append(f"tool_call_id={tool_call_id}")

        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        serialized = f"[{role}{suffix}]"
        if content:
            serialized += f" {content}"
        return serialized

    def _truncate_note_text(
        self,
        text: str,
        *,
        max_tokens: int,
        marker: str = "... [truncated]",
    ) -> str:
        counter = self._get_token_counter()
        if max_tokens <= 0:
            return marker
        if counter.count_text(text) <= max_tokens:
            return text

        marker_tokens = counter.count_text(marker)
        available_tokens = max(1, max_tokens - marker_tokens)
        tokens = list(counter.encode_text(text) or [])
        if not tokens:
            return marker
        truncated = counter.decode_tokens(tokens[:available_tokens]).rstrip()
        if not truncated:
            return marker
        return f"{truncated}\n{marker}"

    def _max_preserved_rounds(self, messages: Sequence[Message], preferred_rounds: int) -> int:
        round_starts = self._round_boundaries(messages)
        if not round_starts:
            return 0
        if len(round_starts) <= 1:
            return 0
        return min(max(0, preferred_rounds), len(round_starts) - 1)

    def _split_for_recent_rounds(
        self,
        messages: Sequence[Message],
        preserved_rounds: int,
    ) -> Tuple[List[Message], List[Message]]:
        source = list(messages)
        if preserved_rounds <= 0:
            return source, []

        round_starts = self._round_boundaries(source)
        if not round_starts:
            return source, []

        preserved_rounds = min(preserved_rounds, len(round_starts))
        keep_from_index = round_starts[-preserved_rounds]
        return source[:keep_from_index], source[keep_from_index:]

    def _save_transcript(self, messages: Sequence[Message]) -> Optional[str]:
        config = self._require_config()
        transcript_dir = Path(getattr(config, "compact_transcript_dir", "memory/transcripts"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = transcript_dir / f"transcript_{timestamp}.jsonl"
        lines = [json.dumps(message.to_dict(), ensure_ascii=False, default=str) for message in messages]

        try:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write(filepath, "\n".join(lines) + "\n")
            return str(filepath.resolve())
        except Exception:
            return None

    def _is_internal_user_message(self, message: Message) -> bool:
        metadata = self._message_metadata(message)
        return message.role == "user" and metadata.get("kind") in self.INTERNAL_USER_KINDS

    def _is_preserved_context_message(self, message: Message) -> bool:
        metadata = self._message_metadata(message)
        if metadata.get("kind") == "preserved_context":
            return True
        return bool((message.content or "").startswith(self.ESSENTIAL_CONTEXT_PREFIX))

    def _is_archived_user_content(self, content: str) -> bool:
        if not content:
            return False
        return content.startswith(self.SUMMARY_NOTE_PREFIX) or content.startswith(
            self.ESSENTIAL_CONTEXT_PREFIX
        ) or content.startswith(self.SYSTEM_NOTE_PREFIX)

    def _message_metadata(self, message: Message) -> Dict[str, Any]:
        return dict(message.metadata or {})

    def _clone_message(
        self,
        message: Message,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
    ) -> Message:
        return Message(
            content=message.content if content is None else content,
            role=message.role if role is None else role,
            timestamp=message.timestamp,
            metadata=self._message_metadata(message) if metadata is None else metadata,
        )

    @staticmethod
    def _append_unique(items: List[str], seen: Set[str], value: str, *, limit: int) -> None:
        if value in seen or len(items) >= limit:
            return
        seen.add(value)
        items.append(value)

    @staticmethod
    def _extract_full_output_path(content: str) -> Optional[str]:
        for pattern in _FULL_OUTPUT_PATH_PATTERNS:
            match = pattern.search(content or "")
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
        temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temp_path.write_text(content, encoding=encoding)
        os.replace(temp_path, path)

    def _round_boundaries(self, history: Sequence[Message]) -> List[int]:
        boundaries: List[int] = []
        for index, message in enumerate(history):
            if message.role != "user":
                continue
            if self._is_internal_user_message(message):
                continue
            content = (message.content or "").strip()
            if self._is_archived_user_content(content):
                continue
            boundaries.append(index)
        return boundaries

    def _default_preserve_recent_rounds(self) -> int:
        if self.config is None:
            return 5
        return max(0, int(getattr(self.config, "compact_preserve_recent_rounds", 0) or 0))
