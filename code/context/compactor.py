"""ContextCompactor - 三层上下文压缩引擎

三层压缩策略（适配 OpenAI function-calling 消息格式）：
- Layer 1 (micro_compact): 每轮调用，截断旧工具结果
- Layer 2 (auto_compact): 超过 token 阈值时，LLM 生成摘要替换全部历史
- Layer 3 (manual_compact): 手动触发，支持可选的 focus 参数引导摘要

消息格式（OpenAI function calling）：
- {"role": "system", "content": "..."}
- {"role": "user", "content": "..."}
- {"role": "assistant", "content": "...", "tool_calls": [...]}
- {"role": "tool", "tool_call_id": "...", "content": "..."}
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import Config
from .token_counter import TokenCounter

# Ensure the project root is on sys.path so we can import from the `prompts` package.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE


class ContextCompactor:
    """三层上下文压缩引擎

    复用已有组件：
    - TokenCounter: token 预估
    - ObservationTruncator 的截断思路用于单条结果的大小限制

    用法示例::

        compactor = ContextCompactor(config=config, token_counter=counter)

        # 每轮自动调用
        compactor.micro_compact(messages)

        # 超过阈值时调用
        if compactor.estimate_tokens(messages) > config.compact_token_threshold:
            messages[:] = compactor.auto_compact(messages, llm)

        # 手动调用
        messages[:] = compactor.manual_compact(messages, llm, focus="authentication module")
    """

    def __init__(self, config: Config, token_counter: Optional[TokenCounter] = None):
        self.config = config
        self.token_counter = token_counter or TokenCounter()

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: List[Dict]) -> int:
        """估算消息列表的总 token 数

        遍历所有消息的 content 字段，累加 token 计数。
        对 tool_calls 等结构也做估算。
        """
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += self.token_counter.count_text(content)
            elif isinstance(content, list):
                # 多模态内容列表
                for part in content:
                    if isinstance(part, dict):
                        total += self.token_counter.count_text(part.get("text", ""))
                    elif isinstance(part, str):
                        total += self.token_counter.count_text(part)

            # tool_calls 也占 token
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += self.token_counter.count_text(json.dumps(tool_calls, ensure_ascii=False))

            # 角色标记开销
            total += 4
        return total

    # ------------------------------------------------------------------
    # Layer 1: micro_compact
    # ------------------------------------------------------------------

    def micro_compact(self, messages: List[Dict]) -> List[Dict]:
        """微压缩：截断旧工具结果（原地修改）

        从最新的消息向前扫描 tool role 消息，
        保留最近 N 个工具结果的完整内容，
        将更旧的工具结果替换为简短占位符。

        Args:
            messages: 消息列表（原地修改）

        Returns:
            同一个消息列表（方便链式调用）
        """
        keep_recent = self.config.compact_keep_recent_tool_results

        # 收集所有 tool 消息的索引（从新到旧）
        tool_indices = [
            i for i, msg in enumerate(messages) if msg.get("role") == "tool"
        ]
        tool_indices.reverse()  # 最新的在前

        # 构建 tool_call_id -> tool_name 映射
        tool_name_map = self._build_tool_name_map(messages)

        # 跳过最近 keep_recent 个，截断其余
        for idx_pos, msg_idx in enumerate(tool_indices):
            if idx_pos < keep_recent:
                continue  # 保留最近的

            msg = messages[msg_idx]
            content = msg.get("content", "")
            if len(content) > 200:
                tool_call_id = msg.get("tool_call_id", "")
                tool_name = tool_name_map.get(tool_call_id, "unknown")
                msg["content"] = (
                    f"[Previous tool result: {tool_name} — truncated to save context]"
                )

        return messages

    # ------------------------------------------------------------------
    # Layer 2: auto_compact
    # ------------------------------------------------------------------

    def auto_compact(self, messages: List[Dict], llm) -> List[Dict]:
        """自动压缩：保存转录 + LLM 摘要替换历史

        Args:
            messages: 当前消息列表
            llm: HelloAgentsLLM 实例，用于生成摘要

        Returns:
            新的（压缩后的）消息列表
        """
        return self._compact_with_llm(messages, llm, focus=None)

    # ------------------------------------------------------------------
    # Layer 3: manual_compact
    # ------------------------------------------------------------------

    def manual_compact(self, messages: List[Dict], llm, focus: Optional[str] = None) -> List[Dict]:
        """手动压缩：与 auto_compact 相同，但可指定 focus

        Args:
            messages: 当前消息列表
            llm: HelloAgentsLLM 实例
            focus: 可选的关注点描述，引导摘要侧重

        Returns:
            新的（压缩后的）消息列表
        """
        return self._compact_with_llm(messages, llm, focus=focus)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compact_with_llm(self, messages: List[Dict], llm, focus: Optional[str]) -> List[Dict]:
        """核心压缩逻辑：保存转录、生成摘要、重建消息列表"""

        # 1. 保存完整转录
        self._save_transcript(messages)

        # 2. 分离系统消息（合并为最多一条）
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        # 3. 序列化对话文本（限制长度避免 prompt 过大）
        conversation_text = self._serialize_messages(non_system_msgs, max_chars=80000)

        # 4. 构建摘要 prompt
        focus_instruction = ""
        if focus:
            focus_instruction = f"Pay special attention to: {focus}"

        summary_prompt = SUMMARY_USER_TEMPLATE.format(
            conversation=conversation_text,
            focus_instruction=focus_instruction,
            max_tokens=self.config.compact_summary_max_tokens,
        )

        # 5. 调用 LLM 生成摘要
        try:
            summary_response = llm.invoke(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=self.config.summary_temperature,
                max_tokens=self.config.compact_summary_max_tokens,
            )
            # LLMResponse 对象 - 取 content 属性
            summary = getattr(summary_response, "content", str(summary_response))
        except Exception as e:
            # 降级为简单统计摘要
            summary = self._generate_fallback_summary(non_system_msgs)
            print(f"[compact] LLM summary failed ({e}), using fallback")

        # 6. 重建消息列表（保证只有一条 system 消息在最前）
        compacted: List[Dict] = []
        if system_msgs:
            merged_system = "\n\n".join(m.get("content", "") for m in system_msgs)
            compacted.append({"role": "system", "content": merged_system})
        compacted.append({
            "role": "user",
            "content": (
                f"[Context compacted at {datetime.now().strftime('%H:%M:%S')}]\n\n"
                f"## Conversation Summary\n{summary}"
            ),
        })
        compacted.append({
            "role": "assistant",
            "content": (
                "Understood. I have the conversation summary and will continue from the current state."
            ),
        })

        return compacted

    def _build_tool_name_map(self, messages: List[Dict]) -> Dict[str, str]:
        """从 assistant 消息的 tool_calls 中构建 tool_call_id -> tool_name 映射"""
        mapping: Dict[str, str] = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                if tc_id:
                    mapping[tc_id] = name
        return mapping

    def _serialize_messages(self, messages: List[Dict], max_chars: int = 80000) -> str:
        """将消息列表序列化为文本（用于摘要 prompt）"""
        lines = []
        total_chars = 0
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")

            # 截断单条过长内容
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"

            line = f"[{role}] {content}"
            if total_chars + len(line) > max_chars:
                lines.append("... [earlier messages truncated]")
                break
            lines.append(line)
            total_chars += len(line)

        return "\n\n".join(lines)

    def _generate_fallback_summary(self, messages: List[Dict]) -> str:
        """降级摘要：纯统计信息（不调用 LLM）"""
        role_counts: Dict[str, int] = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1

        parts = [f"Conversation contained {len(messages)} messages:"]
        for role, count in sorted(role_counts.items()):
            parts.append(f"- {role}: {count}")

        return "\n".join(parts)

    def _save_transcript(self, messages: List[Dict]) -> Optional[str]:
        """保存完整转录到 JSONL 文件"""
        transcript_dir = self.config.compact_transcript_dir
        try:
            os.makedirs(transcript_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(transcript_dir, f"transcript_{timestamp}.jsonl")

            with open(filepath, "w", encoding="utf-8") as f:
                for msg in messages:
                    # 过滤掉不可序列化的字段
                    serializable = {
                        k: v for k, v in msg.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }
                    f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

            return filepath
        except Exception as e:
            print(f"[compact] Failed to save transcript: {e}")
            return None
