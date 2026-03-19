"""ContextCompactor - Three-layer context compression engine

Three-layer compression strategy (adapts to OpenAI function-calling message format):
- Layer 1 (micro_compact): Called every round, truncates old tool results.
- Layer 2 (auto_compact): When the token threshold is exceeded, the LLM generates a summary to replace all history.
- Layer 3 (manual_compact): Manually triggered, supports an optional focus parameter to guide the summary.

Message format (OpenAI function calling):
- {"role": "system", "content": "..."}
- {"role": "user", "content": "..."}
- {"role": "assistant", "content": "...", "tool_calls": [...]}
- {"role": "tool", "tool_call_id": "...", "content": "..."}
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from ..core.config import Config
from .token_counter import TokenCounter

# Ensure the project root is on sys.path so we can import from the `prompts` package.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE


class ContextCompactor:
    """Three-layer context compression engine

    Reuses existing components:
    - TokenCounter: token estimation
    - ObservationTruncator's truncation concept used for single result size limits

    Usage example::

        compactor = ContextCompactor(config=config, token_counter=counter)

        # Automatically called every round
        compactor.micro_compact(messages)

        # Called when the threshold is exceeded
        if compactor.estimate_tokens(messages) > config.compact_token_threshold:
            messages[:] = compactor.auto_compact(messages, llm)

        # Called manually
        messages[:] = compactor.manual_compact(messages, llm, focus="authentication module")
    """

    def __init__(self, config: Config, token_counter: Optional[TokenCounter] = None):
        self.config = config
        self.token_counter = token_counter or TokenCounter()
        # Cached tool_call_id -> tool_name mapping (incrementally updated)
        self._tool_name_map: Dict[str, str] = {}
        self._tool_name_map_msg_count: int = 0  # messages scanned so far


    def estimate_tokens(self, messages: List[Dict]) -> int:
        """Estimate the total number of tokens in the message list

        Iterates over the content field of all messages, accumulating the token count.
        Also estimates structures like tool_calls.
        """
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += self.token_counter.count_text(content)
            elif isinstance(content, list):
                # Multimodal content list
                for part in content:
                    if isinstance(part, dict):
                        total += self.token_counter.count_text(part.get("text", ""))
                    elif isinstance(part, str):
                        total += self.token_counter.count_text(part)

            # tool_calls also consume tokens
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += self.token_counter.count_text(json.dumps(tool_calls, ensure_ascii=False))

            # Role marker overhead
            total += 4
        return total


    def micro_compact(self, messages: List[Dict]) -> List[Dict]:
        """Micro-compression: Truncate old tool results (in-place modification)

        Scans tool role messages backward from the newest message,
        retains the full content of the N most recent tool results,
        and replaces older tool results with short placeholders.

        Args:
            messages: List of messages (modified in-place)

        Returns:
            The same list of messages (convenient for chaining)
        """
        keep_recent = self.config.compact_keep_recent_tool_results

        # Collect indices of all tool messages (from newest to oldest)
        tool_indices = [
            i for i, msg in enumerate(messages) if msg.get("role") == "tool"
        ]
        tool_indices.reverse()  # Newest first

        # Build tool_call_id -> tool_name mapping (incremental update)
        tool_name_map = self._update_tool_name_map(messages)

        # Skip the most recent keep_recent, truncate the rest
        for idx_pos, msg_idx in enumerate(tool_indices):
            if idx_pos < keep_recent:
                continue  # Keep the recent ones

            msg = messages[msg_idx]
            content = msg.get("content", "")
            if len(content) > 200:
                tool_call_id = msg.get("tool_call_id", "")
                tool_name = tool_name_map.get(tool_call_id, "unknown")
                msg["content"] = (
                    f"[Previous tool result: {tool_name} — truncated to save context]"
                )

        return messages


    def auto_compact(self, messages: List[Dict], llm) -> List[Dict]:
        """Automatic compression: Save transcript + LLM summary replaces history

        Args:
            messages: Current list of messages
            llm: HelloAgentsLLM instance, used to generate the summary

        Returns:
            A new (compressed) list of messages
        """
        return self._compact_with_llm(messages, llm, focus=None)


    def manual_compact(self, messages: List[Dict], llm, focus: Optional[str] = None) -> List[Dict]:
        """Manual compression: Same as auto_compact, but allows specifying a focus

        Args:
            messages: Current list of messages
            llm: HelloAgentsLLM instance
            focus: Optional focus description to guide the summary's emphasis

        Returns:
            A new (compressed) list of messages
        """
        return self._compact_with_llm(messages, llm, focus=focus)


    def _compact_with_llm(self, messages: List[Dict], llm, focus: Optional[str]) -> List[Dict]:
        """Core compression logic: Save transcript, generate summary, rebuild message list"""

        # 1. Save full transcript
        self._save_transcript(messages)

        # 2. Separate system messages (merge into at most one)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        # 3. Serialize conversation text (limit length to avoid overly large prompts)
        conversation_text = self._serialize_messages(non_system_msgs, max_chars=80000)

        # 4. Build summary prompt
        focus_instruction = ""
        if focus:
            focus_instruction = f"Pay special attention to: {focus}"

        summary_prompt = SUMMARY_USER_TEMPLATE.format(
            conversation=conversation_text,
            focus_instruction=focus_instruction,
            max_tokens=self.config.summary_max_tokens,
        )

        # 5. Call LLM to generate summary
        try:
            summary_response = llm.invoke(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=self.config.summary_temperature,
                max_tokens=self.config.summary_max_tokens,
            )
            # LLMResponse object - get the content attribute
            summary = getattr(summary_response, "content", str(summary_response))
        except Exception as e:
            # Fallback to a simple statistical summary
            summary = self._generate_fallback_summary(non_system_msgs)
            print(f"[compact] LLM summary failed ({e}), using fallback")

        # 6. Rebuild message list (ensure only one system message is at the front)
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

    def _update_tool_name_map(self, messages: List[Dict]) -> Dict[str, str]:
        """Incrementally update the tool_call_id -> tool_name mapping.

        Only scans messages added since the last call, avoiding O(n) full scan
        on every micro_compact invocation.
        """
        start = self._tool_name_map_msg_count
        # Handle list shrinking after compaction
        if start > len(messages):
            self._tool_name_map.clear()
            start = 0

        for msg in messages[start:]:
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
                    self._tool_name_map[tc_id] = name

        self._tool_name_map_msg_count = len(messages)
        return self._tool_name_map

    def _build_tool_name_map(self, messages: List[Dict]) -> Dict[str, str]:
        """Build tool_call_id -> tool_name mapping from assistant message's tool_calls"""
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
        """Serialize the message list to text (for the summary prompt)"""
        lines = []
        total_chars = 0
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")

            # Truncate a single excessively long content
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
        """Fallback summary: Pure statistical info (no LLM call)"""
        role_counts: Dict[str, int] = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1

        parts = [f"Conversation contained {len(messages)} messages:"]
        for role, count in sorted(role_counts.items()):
            parts.append(f"- {role}: {count}")

        return "\n".join(parts)

    def _save_transcript(self, messages: List[Dict]) -> Optional[str]:
        """Save full transcript to a JSONL file"""
        transcript_dir = self.config.compact_transcript_dir
        try:
            os.makedirs(transcript_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(transcript_dir, f"transcript_{timestamp}.jsonl")

            with open(filepath, "w", encoding="utf-8") as f:
                for msg in messages:
                    # Filter out non-serializable fields
                    serializable = {
                        k: v for k, v in msg.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }
                    f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

            return filepath
        except Exception as e:
            print(f"[compact] Failed to save transcript: {e}")
            return None