"""Coding-focused ReAct agent built on top of the current HelloAgent framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..context.compactor import (
    COMPACTION_ACKNOWLEDGEMENT,
    COMPACTION_PREFIX,
    COMPACTION_SUMMARY_HEADING,
)
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..observability.trace_logger import TraceLogger
from ..tools.registry import ToolRegistry
from .react_agent import ReActAgent

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CODE_AGENT_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "system_prompt.md"
CODE_AGENT_SYSTEM_PROMPT: str = _CODE_AGENT_PROMPT_FILE.read_text(encoding="utf-8")


def _copy_config(config: Optional[Config]) -> Config:
    base = config or Config()
    if hasattr(base, "model_copy"):
        return base.model_copy(deep=True)
    return base.copy(deep=True)


class CodeAgent(ReActAgent):
    """Coding agent with repository-aware prompts and built-in coding tools."""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 0,
        register_default_tools: bool = True,
        enable_task_tool: bool = True,
        interactive: bool = True,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        initial_working_dir = (
            Path(working_dir).expanduser().resolve()
            if working_dir
            else self.project_root
        )
        initial_working_dir.relative_to(self.project_root)
        self.working_dir = initial_working_dir

        effective_config = _copy_config(config)
        self._trace_enabled = bool(effective_config.trace_enabled)
        effective_config.trace_enabled = False
        effective_config.subagent_enabled = False
        effective_config.todowrite_enabled = bool(
            register_default_tools and enable_task_tool and effective_config.todowrite_enabled
        )

        registry = tool_registry or ToolRegistry()

        super().__init__(
            name=name,
            llm=llm,
            tool_registry=registry,
            system_prompt=system_prompt or CODE_AGENT_SYSTEM_PROMPT,
            config=effective_config,
            max_steps=max_steps,
        )

        self.config.trace_enabled = self._trace_enabled
        self.max_steps = max_steps

        self.interactive = interactive

        if register_default_tools:
            self.register_default_tools(enable_task_tool=enable_task_tool)

    def register_default_tools(self, enable_task_tool: bool = True) -> None:
        """Register the coding-oriented tool set for this agent instance."""
        from ..tools.builtin.ask_user import AskUserTool
        from ..tools.builtin.bash import BashTool
        from ..tools.builtin.file_tools import DeleteTool, EditTool, ListFilesTool, ReadTool, WriteTool
        from ..tools.builtin.glob_tool import GlobTool
        from ..tools.builtin.grep_tool import GrepTool
        from ..tools.builtin.web_tool import WebSearchTool, WebFetchTool

        self.tool_registry.register_tool(
            ListFilesTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                registry=self.tool_registry,
            )
        )
        self.tool_registry.register_tool(
            GlobTool(project_root=str(self.project_root), working_dir=str(self.working_dir))
        )
        self.tool_registry.register_tool(
            GrepTool(project_root=str(self.project_root), working_dir=str(self.working_dir))
        )
        self.tool_registry.register_tool(
            ReadTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                registry=self.tool_registry,
            )
        )
        self.tool_registry.register_tool(
            WriteTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                registry=self.tool_registry,
            )
        )
        self.tool_registry.register_tool(
            DeleteTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                registry=self.tool_registry,
            )
        )
        self.tool_registry.register_tool(
            EditTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                registry=self.tool_registry,
            )
        )
        self.tool_registry.register_tool(
            BashTool(project_root=str(self.project_root), working_dir=str(self.working_dir))
        )
        self.tool_registry.register_tool(AskUserTool(interactive=self.interactive))

        if WebSearchTool.is_enabled_by_default():
            self.tool_registry.register_tool(
                WebSearchTool(project_root=str(self.project_root))
            )
        if WebFetchTool.is_enabled_by_default():
            self.tool_registry.register_tool(
                WebFetchTool(project_root=str(self.project_root))
            )
        if enable_task_tool and self.config.todowrite_enabled and self.tool_registry.get_tool("TodoWrite") is None:
            self._register_todowrite_tool()

    def set_working_dir(self, working_dir: str) -> None:
        """Update the agent and file tools to a new working directory."""
        new_working_dir = Path(working_dir).expanduser().resolve()
        new_working_dir.relative_to(self.project_root)
        self.working_dir = new_working_dir

        if not self.tool_registry:
            return

        for tool in self.tool_registry.get_all_tools():
            if hasattr(tool, "working_dir"):
                tool.working_dir = new_working_dir

    def _parallel_user_tool_execution_enabled(self) -> bool:
        """CodeAgent uses the shared concurrent tool executor in sync run()."""
        return True

    # ------------------------------------------------------------------
    # Context compaction (public API)
    # ------------------------------------------------------------------

    def compact(self, focus: str = None) -> str:
        """Manually compact the conversation context.

        Reconstructs messages from the HistoryManager, runs manual_compact,
        and replaces the stored history with the compressed version.

        Args:
            focus: Optional focus string to guide the summary.

        Returns:
            Status message.
        """
        messages = self._build_workspace_messages()
        if not messages:
            return "Nothing to compact."

        before_tokens = self._compactor.estimate_tokens(messages)
        compacted = self._compactor.manual_compact(messages, self.llm, focus=focus)
        after_tokens = self._compactor.estimate_tokens(compacted)

        # Replace history with the compacted summary
        self._replace_history_from_messages(compacted)

        return (
            f"Context compacted: {before_tokens} -> {after_tokens} tokens "
            f"(saved {before_tokens - after_tokens})"
        )

    def _build_workspace_messages(self, input_text: Optional[str] = None) -> List[Dict[str, str]]:
        """Build a single-system-message transcript with optional new user input."""
        messages: List[Dict[str, str]] = []

        system_parts: List[str] = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        system_parts.append(
            f"Workspace root: {self.project_root}\n"
            f"Current working directory: {self.working_dir}\n"
            "All file paths must stay within the workspace root."
        )
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for message in self.get_history():
            if message.role == "summary":
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Conversation summary]\n{message.content}",
                    }
                )
            elif message.role == "system":
                messages.append(
                    {
                        "role": "user",
                        "content": f"[System note]\n{message.content}",
                    }
                )
            elif message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
            elif message.role == "tool":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Previous tool result:\n{message.content}",
                    }
                )

        if input_text is not None:
            messages.append({"role": "user", "content": input_text})

        return messages

    def _replace_history_from_messages(self, messages: List[Dict]) -> None:
        """Replace HistoryManager history with messages from a compacted list."""
        from ..core.message import Message

        self.history_manager.clear()
        self._history_token_count = 0

        for msg in messages:
            role = msg.get("role", "user")
            content = self._normalize_compacted_message_content(msg)
            if role == "system":
                continue
            if role == "assistant" and content == COMPACTION_ACKNOWLEDGEMENT:
                continue
            if role == "user" and isinstance(content, str) and content.startswith(COMPACTION_PREFIX):
                content = self._extract_compaction_summary(content)
                role = "summary"

            if role not in {"user", "assistant", "summary", "tool"}:
                continue

            message = Message(content, role)
            self.history_manager.append(message)
            self._history_token_count += self.token_counter.count_message(message)

    @staticmethod
    def _extract_compaction_summary(content: str) -> str:
        """Strip compaction wrappers and keep only the durable summary text."""
        if COMPACTION_SUMMARY_HEADING not in content:
            return content
        _, _, summary = content.partition(COMPACTION_SUMMARY_HEADING)
        return summary.lstrip("\n").strip()

    @staticmethod
    def _normalize_compacted_message_content(message: Dict) -> str:
        """Convert compacted chat messages into durable text-only history entries."""
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
                if item
            )
        else:
            content = str(content)

        if message.get("role") != "assistant":
            return content

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return content

        call_lines: List[str] = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) or {}
            tool_name = function.get("name", "unknown")
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            call_lines.append(f"- {tool_name}({arguments})")

        tool_call_text = "[Assistant tool calls]\n" + "\n".join(call_lines)
        if content.strip():
            return f"{content.strip()}\n\n{tool_call_text}"
        return tool_call_text

    def run(self, input_text: str, **kwargs) -> str:
        """Run with the shared ReActAgent contract and CodeAgent tracing metadata."""
        self.trace_logger = None

        if self._trace_enabled:
            self.trace_logger = TraceLogger(
                output_dir=self.config.trace_dir,
                sanitize=self.config.trace_sanitize,
                html_include_raw_response=self.config.trace_html_include_raw_response,
            )
            self.trace_logger.log_event(
                "session_start",
                {
                    "agent_name": self.name,
                    "agent_type": self.__class__.__name__,
                    "project_root": str(self.project_root),
                    "working_dir": str(self.working_dir),
                },
            )

        try:
            return super().run(input_text, **kwargs)
        except KeyboardInterrupt:
            if self.trace_logger and not self.trace_logger.jsonl_file.closed:
                try:
                    self.trace_logger.log_event(
                        "session_end",
                        {"status": "interrupted", "message": "run aborted by interruption"},
                    )
                    self.trace_logger.finalize()
                except Exception:
                    pass
            raise
        except Exception:
            if self.trace_logger and not self.trace_logger.jsonl_file.closed:
                try:
                    self.trace_logger.log_event(
                        "session_end",
                        {"status": "error", "message": "run aborted by exception"},
                    )
                    self.trace_logger.finalize()
                except Exception:
                    pass
            raise
        finally:
            self.trace_logger = None

    def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        """Build messages with workspace context and preserved conversation history.

        All system-level content is merged into a single system message at
        position 0 so that models whose chat template forbids mid-conversation
        system messages (e.g. Qwen served via vLLM) work correctly.
        """
        return self._build_workspace_messages(input_text)

    def _create_subagent(self, agent_type: str = "code") -> "CodeAgent":
        """Create a fresh sub-agent with isolated tool state."""
        sub_config = _copy_config(self.config)
        sub_config.trace_enabled = False
        sub_config.subagent_enabled = False

        return CodeAgent(
            name=f"{self.name}-{agent_type}-subagent",
            llm=self.llm,
            tool_registry=ToolRegistry(),
            project_root=str(self.project_root),
            working_dir=str(self.working_dir),
            config=sub_config,
            max_steps=sub_config.subagent_max_steps,
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
        )
