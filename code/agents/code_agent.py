"""Coding-focused ReAct agent built on top of the current HelloAgent framework."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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
        effective_config.todowrite_enabled = bool(
            register_default_tools and enable_task_tool and effective_config.todowrite_enabled
        )

        registry = tool_registry or ToolRegistry(config=effective_config)

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
        """Manually compact the conversation context via HistoryManager."""
        if not self.get_history():
            return "Nothing to compact."

        result = self.history_manager.compact_with_llm(
            llm=self.llm,
            system_prompt=self._get_context_system_prompt(),
            focus=focus,
        )
        self._sync_history_token_count()
        self._estimated_next_prompt_tokens = self.history_manager.estimate_tokens(
            system_prompt=self._get_context_system_prompt(),
        )
        if result is None:
            return "Nothing to compact."

        return (
            f"Context compacted: estimated next prompt {result['before_tokens']} -> {result['after_tokens']} tokens "
            f"(saved {result['saved_tokens']})"
        )

    def _get_context_system_prompt(self) -> str:
        """Build the workspace-aware system prompt for all model calls."""
        system_parts: List[str] = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        system_parts.append(
            f"Workspace root: {self.project_root}\n"
            f"Current working directory: {self.working_dir}\n"
            "All file paths must stay within the workspace root."
        )
        return "\n\n".join(system_parts)

    def _build_workspace_messages(self, input_text: Optional[str] = None):
        """兼容旧调用点，实际复用共享的 HistoryManager 投影逻辑。"""
        return self._build_messages(input_text)

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

    def _create_subagent(self, agent_type: str = "code") -> "CodeAgent":
        """Create a fresh sub-agent with isolated tool state."""
        sub_config = _copy_config(self.config)
        sub_config.trace_enabled = False

        return CodeAgent(
            name=f"{self.name}-{agent_type}-subagent",
            llm=self.llm,
            tool_registry=ToolRegistry(config=sub_config),
            project_root=str(self.project_root),
            working_dir=str(self.working_dir),
            config=sub_config,
            max_steps=sub_config.subagent_max_steps,
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
        )
