"""Coding-focused ReAct agent built on top of the current HelloAgent framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..tools.registry import ToolRegistry
from .react_agent import ReActAgent

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CODE_AGENT_PROMPT_FILE = _PROJECT_ROOT / "code" / "prompts" / "system_prompt.md"
CODE_AGENT_SYSTEM_PROMPT: str = _CODE_AGENT_PROMPT_FILE.read_text(encoding="utf-8")

# 重要-4: give the coding agent a finite default step budget so a model that
# keeps issuing productive-looking tool calls cannot loop unbounded and rack up
# unbounded cost. Pass max_steps=0 explicitly to opt into unlimited mode.
DEFAULT_CODE_AGENT_MAX_STEPS = 100


def _copy_config(config: Optional[Config]) -> Config:
    base = config or Config()
    if hasattr(base, "model_copy"):
        return base.model_copy(deep=True)
    return base.copy(deep=True)


class CodeAgent(ReActAgent):
    """Coding agent with repository-aware prompts and built-in coding tools."""

    #: Finite default step budget (see 重要-4). ``max_steps=0`` stays as an
    #: explicit opt-in to unlimited stepping for advanced callers.
    DEFAULT_MAX_STEPS = DEFAULT_CODE_AGENT_MAX_STEPS

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = DEFAULT_CODE_AGENT_MAX_STEPS,
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

        # 将标配的 truncator output_dir 修正为 project_root 下的绝对路径，
        # 确保所有工具共用同一个截断器输出目录（truncate_for_context 的
        # 第二层截断能正确找到工具第一层保存的全量输出）。
        shared_output_dir = self.project_root / "memory" / "tool-output"
        self.truncator.output_dir = shared_output_dir
        shared_output_dir.mkdir(parents=True, exist_ok=True)

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
                config=self.config,
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
                config=self.config,
            )
        )
        self.tool_registry.register_tool(
            BashTool(
                project_root=str(self.project_root),
                working_dir=str(self.working_dir),
                config=self.config,
                output_truncator=self.truncator,
            )
        )
        self.tool_registry.register_tool(AskUserTool(interactive=self.interactive))

        if WebSearchTool.is_enabled_by_default():
            self.tool_registry.register_tool(
                WebSearchTool(
                    project_root=str(self.project_root),
                    output_truncator=self.truncator,
                )
            )
        if WebFetchTool.is_enabled_by_default():
            self.tool_registry.register_tool(
                WebFetchTool(
                    project_root=str(self.project_root),
                    config=self.config,
                    output_truncator=self.truncator,
                )
            )
        if enable_task_tool and self.config.todowrite_enabled and self.tool_registry.get_tool("TodoWrite") is None:
            self._register_todowrite_tool()

        # LSP tools — always available (graceful degradation when server absent)
        from ..tools.lsp import (
            LSPDefinitionTool,
            LSPDiagnosticsTool,
            LSPHoverTool,
            LSPReferencesTool,
            LSPManager,
        )
        manager = LSPManager(self.project_root)
        self.tool_registry.register_tool(
            LSPDefinitionTool(workspace_root=str(self.project_root), manager=manager)
        )
        self.tool_registry.register_tool(
            LSPReferencesTool(workspace_root=str(self.project_root), manager=manager)
        )
        self.tool_registry.register_tool(
            LSPHoverTool(workspace_root=str(self.project_root), manager=manager)
        )
        self.tool_registry.register_tool(
            LSPDiagnosticsTool(workspace_root=str(self.project_root), manager=manager)
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

    # ------------------------------------------------------------------
    # Trace customization — 仅覆盖 hook，无需重写 run/arun/arun_stream
    # ------------------------------------------------------------------

    def _trace_session_metadata(self) -> Dict[str, Any]:
        """向 session_start 事件中注入 workspace 路径信息。"""
        return {
            "project_root": str(self.project_root),
            "working_dir": str(self.working_dir),
        }

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
