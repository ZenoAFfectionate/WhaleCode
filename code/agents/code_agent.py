"""Coding-focused ReAct agent built on top of the current HelloAgent framework."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message
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
        self.background_manager = None

        effective_config = _copy_config(config)
        self._trace_enabled = bool(effective_config.trace_enabled)
        effective_config.trace_enabled = False
        effective_config.subagent_enabled = False

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
        from ..tools.builtin.background import BackgroundTool, get_background_manager
        from ..tools.builtin.bash import BashTool
        from ..tools.builtin.file_tools import EditTool, ListFilesTool, MultiEditTool, ReadTool, WriteTool
        from ..tools.builtin.glob_tool import GlobTool
        from ..tools.builtin.grep_tool import GrepTool
        from ..tools.builtin.task_tool import TaskTool
        from ..tools.builtin.web_tool import WebSearchTool, WebFetchTool

        self.background_manager = get_background_manager(self.project_root)

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
            EditTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                registry=self.tool_registry,
            )
        )
        self.tool_registry.register_tool(
            MultiEditTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                registry=self.tool_registry,
            )
        )
        self.tool_registry.register_tool(
            BashTool(project_root=str(self.project_root), working_dir=str(self.working_dir))
        )
        self.tool_registry.register_tool(
            BackgroundTool(project_root=str(self.project_root), working_dir=str(self.working_dir))
        )
        self.tool_registry.register_tool(AskUserTool(interactive=self.interactive))
        self.tool_registry.register_tool(WebSearchTool())
        self.tool_registry.register_tool(WebFetchTool())

        if enable_task_tool:
            self.tool_registry.register_tool(
                TaskTool(
                    config=self.config,
                    persistence_dir=str(self.project_root / "memory" / "tasks"),
                )
            )

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

    def _before_model_call(self, messages: List[Dict[str, object]], current_step: int) -> None:
        super()._before_model_call(messages, current_step)
        if not self.background_manager:
            return

        notifications = self.background_manager.drain_notifications()
        if not notifications:
            return

        notification_text = self.background_manager.format_notifications(notifications)
        self._render_background_update(current_step, notification_text, notifications=notifications)
        messages.append(
            {
                "role": "user",
                "content": f"<background-results>\n{notification_text}\n</background-results>",
            }
        )

        if self.trace_logger:
            self.trace_logger.log_event(
                "background_notifications",
                {
                    "count": len(notifications),
                    "notifications": notifications,
                },
                step=current_step,
            )

    def _render_background_update(
        self,
        current_step: int,
        notification_text: str,
        *,
        notifications: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        self._render_event(
            "background_update",
            {
                "step": current_step,
                "notification_text": notification_text,
                "notifications": notifications or [],
            },
        )

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
        messages = self._build_messages_from_history()
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

    def _build_messages_from_history(self) -> List[Dict]:
        """Reconstruct an OpenAI-format message list from HistoryManager.

        Mirrors _build_messages: single system message at position 0,
        summaries and system history entries use user role.
        """
        messages: List[Dict] = []

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

        return messages

    def _replace_history_from_messages(self, messages: List[Dict]) -> None:
        """Replace HistoryManager history with messages from a compacted list."""
        from ..core.message import Message

        self.history_manager.clear()
        self._history_token_count = 0

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Map system summary back to summary role
            if role == "system" and content.startswith("[Context compacted"):
                role = "summary"
            elif role == "system":
                continue  # skip system prompts (reconstructed on each turn)

            if role in {"user", "assistant", "summary"}:
                m = Message(content, role)
                self.history_manager.append(m)
                self._history_token_count += self.token_counter.count_message(m)

    def run(self, input_text: str, **kwargs) -> str:
        """Run with a fresh trace logger per turn for interactive CLI usage."""
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
        messages: List[Dict[str, str]] = []

        # --- single system message (merge prompt + workspace info) ---
        system_parts: List[str] = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        system_parts.append(
            f"Workspace root: {self.project_root}\n"
            f"Current working directory: {self.working_dir}\n"
            "All file paths must stay within the workspace root."
        )
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # --- conversation history ---
        for message in self.get_history():
            if message.role == "summary":
                # Summaries are injected as user messages to avoid
                # system messages appearing after non-system messages.
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Conversation summary]\n{message.content}",
                    }
                )
            elif message.role == "system":
                # Historic system notes -> user role for compatibility
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

        messages.append({"role": "user", "content": input_text})
        return messages

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
