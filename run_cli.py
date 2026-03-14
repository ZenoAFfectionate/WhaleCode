"""Interactive CLI for running the coding agent in a terminal."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"

try:
    from rich.align import Align
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    Align = Console = Markdown = Panel = Rule = Table = Text = None
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.styles import Style as PromptStyle

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PromptSession = HTML = FileHistory = KeyBindings = Keys = PromptStyle = None
    PROMPT_TOOLKIT_AVAILABLE = False


def _get_version() -> str:
    """Try to read __version__ from code/version.py."""
    version_file = CODE_DIR / "version.py"
    if version_file.exists():
        try:
            ns: dict = {}
            exec(version_file.read_text(encoding="utf-8"), ns)
            return f"v{ns.get('__version__', '2.0')}"
        except Exception:
            pass
    return "v2.0"


def _detect_provider(base_url: str, model: str) -> str:
    """Detect LLM provider from base URL or model name."""
    url_lower = base_url.lower()
    model_lower = model.lower()

    url_map = {
        "bigmodel.cn": "ZHIPU",
        "openai.com": "OPENAI",
        "deepseek.com": "DEEPSEEK",
        "anthropic.com": "ANTHROPIC",
        "dashscope.aliyuncs.com": "ALIBABA",
        "moonshot.cn": "MOONSHOT",
        "minimax.chat": "MINIMAX",
        "baichuan-ai.com": "BAICHUAN",
        "localhost": "LOCAL",
        "127.0.0.1": "LOCAL",
    }
    for domain, name in url_map.items():
        if domain in url_lower:
            return name

    model_map = {
        "glm": "ZHIPU",
        "gpt": "OPENAI",
        "o1": "OPENAI",
        "o3": "OPENAI",
        "claude": "ANTHROPIC",
        "deepseek": "DEEPSEEK",
        "qwen": "ALIBABA",
        "moonshot": "MOONSHOT",
        "llama": "META",
        "gemini": "GOOGLE",
    }
    for prefix, name in model_map.items():
        if model_lower.startswith(prefix):
            return name

    return "LLM"


def bootstrap_package() -> None:
    """Expose the local `code/` directory as the `hello_agents` package."""
    if "hello_agents" in sys.modules:
        return

    package = types.ModuleType("hello_agents")
    package.__path__ = [str(CODE_DIR)]
    package.__file__ = str(CODE_DIR / "__init__.py")
    sys.modules["hello_agents"] = package


class CLIUI:
    """Small rendering helper with optional Rich support."""

    def __init__(self, use_rich: bool = True):
        self.use_rich = bool(use_rich and RICH_AVAILABLE)
        self.console = Console(record=True) if self.use_rich else None

    def print(self, message: str = "") -> None:
        if self.use_rich:
            self.console.print(message)
        else:
            print(message)

    def info(self, message: str) -> None:
        if self.use_rich:
            self.console.print(f"[cyan]{message}[/cyan]")
        else:
            print(message)

    def success(self, message: str) -> None:
        if self.use_rich:
            self.console.print(f"[green]{message}[/green]")
        else:
            print(message)

    def warning(self, message: str) -> None:
        if self.use_rich:
            self.console.print(f"[yellow]{message}[/yellow]")
        else:
            print(message)

    def error(self, message: str) -> None:
        if self.use_rich:
            self.console.print(f"[bold red]{message}[/bold red]")
        else:
            print(message, file=sys.stderr)

    def render_banner(self, agent, workspace: Path) -> None:
        model = getattr(agent.llm, "model", "[unknown]")
        agent_name = getattr(agent, "name", "code-agent")
        base_url = str(getattr(agent.llm, "base_url", "") or "")

        display_name = getattr(agent, "display_name", None) or "Whale Code"
        version = _get_version()
        provider = _detect_provider(base_url, str(model))

        def _pretty_path(path: Path) -> str:
            try:
                return str(path).replace(str(Path.home()), "~", 1)
            except Exception:
                return str(path)

        if self.use_rich:
            # 3-column right side: label (fixed) + value (flexible)
            layout = Table.grid(padding=(0, 3), expand=True)
            layout.add_column(width=35)   # whale art (with indent)
            layout.add_column(width=14)   # label
            layout.add_column(ratio=1)    # value

            P = "    "  # 4-space indent for whale
            layout.add_row(
                Text(f"{P}      .", style="bold bright_blue"),
                Text(""), Text(""),
            )
            layout.add_row(
                Text(f'{P}      ":"', style="bold bright_blue"),
                Text(""), Text(""),
            )
            layout.add_row(
                Text(f'{P}    ___:____     |"\\/"|', style="bold bright_blue"),
                Text(display_name, style="bold white"),
                Text(version, style="dim"),
            )
            layout.add_row(
                Text(f"{P}  ,'        `.    \\  /", style="bold bright_blue"),
                Text(provider, style="bold bright_green"),
                Text(str(model), style="bold bright_cyan"),
            )
            layout.add_row(
                Text(f"{P}  |  O    _   \\___/  |", style="bold bright_blue"),
                Text("Workspace", style="dim"),
                Text(_pretty_path(workspace), style="bold bright_green"),
            )
            layout.add_row(
                Text(f"{P}~^~^~^~^~^~^~", style="bold bright_blue"),
                Text(""), Text(""),
            )

            banner = Panel(
                layout,
                border_style="bright_blue",
                padding=(1, 2),
                width=self.console.width,
            )

            self.console.print(banner)
            self.console.print("[dim]Type `/help` for commands, or `exit` to quit.[/dim]")
        else:
            P = "    "
            W = 38
            L = 14
            print(f"{P}      .")
            print(f'{P}      ":"')
            art3 = f'{P}    ___:____     |"\\/"|'
            art4 = f"{P}  ,'        `.    \\  /"
            art5 = f"{P}  |  O    _   \\___/  |"
            print(f"{art3:<{W}}{display_name:<{L}}{version}")
            print(f"{art4:<{W}}{provider:<{L}}{model}")
            print(f"{art5:<{W}}{'Workspace':<{L}}{_pretty_path(workspace)}")
            print(f"{P}~^~^~^~^~^~^~")
            print("Type `/help` for commands, or `exit` to quit.")

    def render_assistant(self, text: str) -> None:
        if self.use_rich:
            self.console.print(Panel(Markdown(text), title="Assistant", border_style="blue", width=self.console.width))
        else:
            print(text)

    def render_summary(self, elapsed_s: float, agent=None) -> None:
        if self.use_rich:
            time_part = f"⌚ Completed in {elapsed_s:.2f}s"

            if agent is not None:
                prompt_t = getattr(agent, '_turn_prompt_tokens', 0)
                comp_t = getattr(agent, '_turn_completion_tokens', 0)
                total_t = getattr(agent, '_total_tokens', 0)
                ctx_window = getattr(agent.config, 'context_window', 0)
                ctx_used = getattr(agent, '_last_prompt_tokens', 0)

                if total_t > 0:
                    token_parts = [
                        f"[green]⬆{prompt_t:,}[/green]",
                        f"[yellow]⬇{comp_t:,}[/yellow]",
                    ]
                    if ctx_window > 0 and ctx_used > 0:
                        pct = ctx_used * 100 / ctx_window
                        token_parts.append(
                            f"[cyan]📊 {ctx_used:,} / {ctx_window:,} ({pct:.0f}%)[/cyan]"
                        )
                    self.console.print(
                        f"[dim]{time_part}    |    {' · '.join(token_parts)}[/dim]"
                    )
                else:
                    self.console.print(f"[dim cyan]{time_part}[/dim cyan]")
            else:
                self.console.print(f"[dim cyan]{time_part}[/dim cyan]")
        else:
            print(f"[completed in {elapsed_s:.2f}s]")

    def render_rule(self, title: str) -> None:
        if self.use_rich:
            self.console.print(Rule(title, style="dim"))
        else:
            print(title)

    def render_log_block(self, kind: str, content: str) -> None:
        content = content.rstrip()
        if not content:
            return

        if not self.use_rich:
            print(content)
            return

        # Wrap in Text() to prevent Rich from interpreting [] as markup tags.
        # Tool call IDs like [call_abc123] would otherwise be silently consumed.
        safe = Text(content)

        if kind == "action":
            self.console.print(Panel(safe, title="Tool Call", border_style="bright_green", width=self.console.width))
        elif kind == "thinking":
            self.console.print(Panel(safe, title="Thinking", border_style="bright_blue", width=self.console.width))
        elif kind == "observation":
            self.console.print(Panel(safe, title="Observation", border_style="bright_cyan", width=self.console.width))
        elif kind == "background":
            self.console.print(Panel(safe, title="Background Update", border_style="magenta", width=self.console.width))
        elif kind == "warning":
            self.console.print(Panel(safe, title="Warning", border_style="yellow", width=self.console.width))
        elif kind == "error":
            self.console.print(Panel(safe, title="Error", border_style="red", width=self.console.width))
        else:
            self.console.print(Text.assemble(("", "dim"), safe))

    def render_tools(self, agent) -> None:
        tools = sorted(agent.tool_registry.get_all_tools(), key=lambda item: item.name)
        if self.use_rich:
            table = Table(title="Registered Tools", border_style="cyan")
            table.add_column("Tool", style="bold cyan")
            table.add_column("Description", style="white")
            for tool in tools:
                description = (tool.description or "").strip().replace("\n", " ")
                table.add_row(tool.name, description)
            self.console.print(table)
            return

        print("Registered tools:")
        for tool in tools:
            description = (tool.description or "").strip().replace("\n", " ")
            print(f"- {tool.name}: {description}")

    def render_history(self, history: Iterable, limit: Optional[int] = None) -> None:
        all_items = list(history)
        items = list(all_items)
        if limit is not None and limit > 0:
            items = items[-limit:]
        if not items:
            self.warning("History is empty.")
            return

        if self.use_rich:
            table = Table(title="Conversation History", border_style="cyan")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Role", style="magenta", width=10)
            table.add_column("Content", style="white")
            start_index = max(1, len(all_items) - len(items) + 1)
            for index, message in enumerate(items, start=start_index):
                content = str(message.content).replace("\n", " ")
                if len(content) > 160:
                    content = content[:160] + " ..."
                table.add_row(str(index), message.role, content)
            self.console.print(table)
            return

        for index, message in enumerate(items, start=1):
            print(f"{index}. [{message.role}] {message.content}")

    def render_sessions(self, sessions: list[dict]) -> None:
        if not sessions:
            self.warning("No saved sessions found.")
            return

        if self.use_rich:
            table = Table(title="Saved Sessions", border_style="cyan")
            table.add_column("File", style="bold cyan")
            table.add_column("Saved At", style="white")
            table.add_column("Steps", style="magenta")
            table.add_column("Tokens", style="green")
            for item in sessions:
                metadata = item.get("metadata", {}) or {}
                table.add_row(
                    item.get("filename", ""),
                    item.get("saved_at", ""),
                    str(metadata.get("total_steps", "-")),
                    str(metadata.get("total_tokens", "-")),
                )
            self.console.print(table)
            return

        print("Saved sessions:")
        for item in sessions:
            metadata = item.get("metadata", {}) or {}
            print(
                f"- {item.get('filename')} | saved_at={item.get('saved_at')} "
                f"steps={metadata.get('total_steps', '-')} tokens={metadata.get('total_tokens', '-')}"
            )


class CLICodeAgentMixin:
    """UI-aware render overrides for the terminal CLI."""

    ui: CLIUI

    def _console(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        if end == "" and hasattr(self, "_streaming_line_buffer"):
            self._streaming_line_buffer += message
            return

        pending = getattr(self, "_streaming_line_buffer", "")
        if pending:
            message = pending + message
            self._streaming_line_buffer = ""

        if message:
            self.ui.print(message)
        elif end == "\n":
            self.ui.print("")

    def _render_event(self, event_type: str, payload: dict) -> None:
        if event_type == "agent_start":
            self.ui.render_log_block("info", f"🤖 {self.name} processing: {payload.get('input_text', '')}")
        elif event_type == "step_start":
            pass
        elif event_type == "compaction_notice":
            self.ui.render_log_block("warning", "[auto-compact triggered]")
        elif event_type == "llm_error":
            self.ui.render_log_block("error", f"LLM call failed: {payload.get('error', '')}")
        elif event_type == "direct_response":
            return None
        elif event_type == "builtin_tool":
            tool_name = payload.get("tool_name")
            result_content = str(payload.get("result_content", ""))
            if tool_name == "Thought":
                content = result_content.removeprefix("Reasoning: ") if result_content.startswith("Reasoning: ") else result_content
                self.ui.render_log_block("thinking", content)
            elif tool_name != "Finish":
                self.ui.render_log_block("info", f"{tool_name}: {result_content}")
        elif event_type == "tool_call":
            tool_name = payload.get("tool_name")
            arguments = payload.get("arguments", {})
            tool_call_id = payload.get("tool_call_id")
            prefix = f"[{tool_call_id}] " if tool_call_id else ""
            self.ui.render_log_block("action", f"{prefix}{tool_name}({arguments})")
        elif event_type == "tool_result":
            kind = "error" if payload.get("status") == "error" else "observation"
            tool_name = payload.get("tool_name")
            tool_call_id = payload.get("tool_call_id")
            header = ""
            if tool_name or tool_call_id:
                left = f"[{tool_call_id}] " if tool_call_id else ""
                right = f"{tool_name}\n" if tool_name else ""
                header = left + right
            self.ui.render_log_block(kind, f"{header}{payload.get('result_content', '')}")
        elif event_type == "final_answer":
            return None
        elif event_type == "timeout":
            pass
        elif event_type == "stream_chunk":
            self._streaming_line_buffer += payload.get("chunk", "")
        elif event_type == "stream_newline":
            pending = getattr(self, "_streaming_line_buffer", "")
            if pending:
                self.ui.render_log_block("info", pending)
                self._streaming_line_buffer = ""
        elif event_type == "agent_error":
            self.ui.render_log_block("error", payload.get("message", ""))
        elif event_type == "background_update":
            step = payload.get("step")
            notification_text = payload.get("notification_text", "")
            self.ui.render_log_block("background", f"Before step {step}\n{notification_text}")
        elif event_type == "console":
            self._console(
                payload.get("message", ""),
                end=payload.get("end", "\n"),
                flush=payload.get("flush", False),
            )
        else:
            super()._render_event(event_type, payload)

    def render_tools(self, agent) -> None:
        tools = sorted(agent.tool_registry.get_all_tools(), key=lambda item: item.name)
        if self.use_rich:
            table = Table(title="Registered Tools", border_style="cyan")
            table.add_column("Tool", style="bold cyan")
            table.add_column("Description", style="white")
            for tool in tools:
                description = (tool.description or "").strip().replace("\n", " ")
                table.add_row(tool.name, description)
            self.console.print(table)
            return

        print("Registered tools:")
        for tool in tools:
            description = (tool.description or "").strip().replace("\n", " ")
            print(f"- {tool.name}: {description}")

    def render_history(self, history: Iterable, limit: Optional[int] = None) -> None:
        all_items = list(history)
        items = list(all_items)
        if limit is not None and limit > 0:
            items = items[-limit:]
        if not items:
            self.warning("History is empty.")
            return

        if self.use_rich:
            table = Table(title="Conversation History", border_style="cyan")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Role", style="magenta", width=10)
            table.add_column("Content", style="white")
            start_index = max(1, len(all_items) - len(items) + 1)
            for index, message in enumerate(items, start=start_index):
                content = str(message.content).replace("\n", " ")
                if len(content) > 160:
                    content = content[:160] + " ..."
                table.add_row(str(index), message.role, content)
            self.console.print(table)
            return

        for index, message in enumerate(items, start=1):
            print(f"{index}. [{message.role}] {message.content}")

    def render_sessions(self, sessions: list[dict]) -> None:
        if not sessions:
            self.warning("No saved sessions found.")
            return

        if self.use_rich:
            table = Table(title="Saved Sessions", border_style="cyan")
            table.add_column("File", style="bold cyan")
            table.add_column("Saved At", style="white")
            table.add_column("Steps", style="magenta")
            table.add_column("Tokens", style="green")
            for item in sessions:
                metadata = item.get("metadata", {}) or {}
                table.add_row(
                    item.get("filename", ""),
                    item.get("saved_at", ""),
                    str(metadata.get("total_steps", "-")),
                    str(metadata.get("total_tokens", "-")),
                )
            self.console.print(table)
            return

        print("Saved sessions:")
        for item in sessions:
            metadata = item.get("metadata", {}) or {}
            print(
                f"- {item.get('filename')} | saved_at={item.get('saved_at')} "
                f"steps={metadata.get('total_steps', '-')} tokens={metadata.get('total_tokens', '-')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the repository coding agent.")
    parser.add_argument("prompt", nargs="?", help="Single-turn prompt to run.")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace root for file tools and Bash. Defaults to the current directory.",
    )
    parser.add_argument("--model", help="Override `LLM_MODEL_ID`.")
    parser.add_argument("--api-key", help="Override `LLM_API_KEY`.")
    parser.add_argument("--base-url", help="Override `LLM_BASE_URL`.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override model temperature.",
    )
    parser.add_argument(
        "--resume",
        help="Optional session file path to load before running.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the registered tool names and descriptions, then exit.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable trace logging for this run.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even when a prompt is supplied.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable Rich rendering even if `rich` is installed.",
    )
    parser.add_argument(
        "--history-file",
        help="Custom prompt history file for interactive mode.",
    )
    parser.add_argument(
        "--session-name",
        default="session-latest",
        help="Default session name for auto-save and `/save` with no argument.",
    )
    parser.add_argument(
        "--no-auto-save",
        action="store_true",
        help="Disable automatic save-on-exit for interactive mode.",
    )
    return parser


def create_agent(args, ui: CLIUI):
    bootstrap_package()

    from hello_agents.agents.code_agent import CodeAgent as BaseCodeAgent
    from hello_agents.core.config import Config
    from hello_agents.core.llm import HelloAgentsLLM
    from hello_agents.tools.registry import ToolRegistry

    class CLICodeAgent(CLICodeAgentMixin, BaseCodeAgent):
        def __init__(self, *inner_args, ui: CLIUI, **inner_kwargs):
            self.ui = ui
            self._streaming_line_buffer = ""
            super().__init__(*inner_args, **inner_kwargs)

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    config = Config.from_env()
    config.trace_enabled = not args.no_trace

    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    if args.api_key:
        llm_kwargs["api_key"] = args.api_key
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    if args.temperature is not None:
        llm_kwargs["temperature"] = args.temperature

    llm = HelloAgentsLLM(**llm_kwargs)
    registry = ToolRegistry(verbose=False)
    agent = CLICodeAgent(
        name="code-agent",
        llm=llm,
        tool_registry=registry,
        project_root=str(workspace),
        working_dir=str(workspace),
        config=config,
        register_default_tools=True,
        enable_task_tool=True,
        ui=ui,
    )

    if args.resume:
        agent.load_session(args.resume)

    return agent, workspace


def default_history_file(workspace: Path) -> Path:
    return workspace / ".codeagent_cli_history"


def default_session_path(agent, session_name: str) -> Optional[Path]:
    if not getattr(agent, "session_store", None):
        return None
    return Path(agent.session_store.session_dir) / f"{session_name}.json"


def normalize_session_name(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return "session-latest"
    return Path(value).stem


def resolve_session_to_load(agent, raw_value: Optional[str], fallback_name: str) -> Optional[Path]:
    if not getattr(agent, "session_store", None):
        return None

    if raw_value:
        candidate = Path(raw_value).expanduser()
        if candidate.exists():
            return candidate.resolve()

        session_dir = Path(agent.session_store.session_dir)
        in_dir = session_dir / candidate.name
        if in_dir.exists():
            return in_dir.resolve()

        if not candidate.suffix:
            with_suffix = session_dir / f"{candidate.name}.json"
            if with_suffix.exists():
                return with_suffix.resolve()

        return candidate.resolve()

    return default_session_path(agent, fallback_name)


def maybe_auto_save(agent, session_name: str, enabled: bool, ui: CLIUI, reason: str) -> None:
    if not enabled or not getattr(agent, "session_store", None):
        return
    try:
        saved_path = agent.save_session(normalize_session_name(session_name))
        ui.info(f"Auto-saved session ({reason}): {saved_path}")
    except Exception as exc:
        ui.warning(f"Auto-save failed: {exc}")


def run_agent_turn(agent, prompt: str, ui: CLIUI) -> str:
    start_time = time.time()
    response = agent.run(prompt)
    ui.render_assistant(response or "")
    ui.render_summary(time.time() - start_time, agent=agent)
    return response or ""


def print_help(ui: CLIUI) -> None:
    lines = [
        "Commands:",
        "- /help                 Show this help message",
        "- /info | /model        Show workspace, model, and runtime info",
        "- /tools                Show registered tools and descriptions",
        "- /pwd                  Show the current working directory",
        "- /cd <path>            Change the agent working directory within the workspace",
        "- /history [n]          Show recent conversation turns",
        "- /log                  View all terminal output in a scrollable pager",
        "- /clear                Clear in-memory conversation history",
        "- /save [name]          Save a session snapshot into the session directory",
        "- /load [path|name]     Load a saved session (default: session-latest)",
        "- /sessions             List saved sessions",
        "- /compact [focus]      Compact conversation context",
        "- exit | quit | q       Exit the CLI",
    ]
    if ui.use_rich:
        ui.console.print(Panel("\n".join(lines), title="Help", border_style="cyan", width=ui.console.width))
    else:
        print("\n".join(lines))


def show_runtime_info(agent, workspace: Path, ui: CLIUI) -> None:
    lines = [
        f"Workspace root: {workspace}",
        f"Current working directory: {getattr(agent, 'working_dir', workspace)}",
        f"Model: {getattr(agent.llm, 'model', '[unknown]')}",
        f"Base URL: {getattr(agent.llm, 'base_url', '[unknown]')}",
        f"Temperature: {getattr(agent.llm, 'temperature', '[unknown]')}",
        f"Trace enabled: {getattr(agent.config, 'trace_enabled', False)}",
        f"Session enabled: {bool(getattr(agent, 'session_store', None))}",
        f"Registered tools: {len(agent.tool_registry.list_tools()) if agent.tool_registry else 0}",
    ]
    if ui.use_rich:
        ui.console.print(Panel("\n".join(lines), title="Runtime Info", border_style="cyan", width=ui.console.width))
    else:
        print("\n".join(lines))


def build_prompt_reader(history_file: Path):
    if PROMPT_TOOLKIT_AVAILABLE and sys.stdin.isatty():
        # Timing-based paste detection.
        # Human typing: >50ms between last keystroke and Enter.
        # Pasted text:  ~0ms between characters (arrives as a burst).
        #
        # When Enter arrives within _PASTE_THRESHOLD of the last text
        # change, we treat it as part of a paste and insert a newline.
        # Otherwise we submit.  This works regardless of whether the
        # terminal supports bracket paste mode.
        _PASTE_THRESHOLD = 0.05  # 50 ms
        _last_text_change = [0.0]

        def _on_text_changed(buf):
            _last_text_change[0] = time.monotonic()

        bindings = KeyBindings()

        @bindings.add(Keys.Enter, eager=True)
        def _smart_enter(event):
            """Submit on Enter, unless rapid input indicates a paste."""
            delta = time.monotonic() - _last_text_change[0]
            if delta < _PASTE_THRESHOLD and event.current_buffer.text:
                # Rapid input → likely a paste → insert newline
                event.current_buffer.insert_text("\n")
            else:
                event.current_buffer.validate_and_handle()

        @bindings.add(Keys.Escape, Keys.Enter)
        def _newline(event):
            """Insert a literal newline on Esc+Enter (manual multi-line)."""
            event.current_buffer.insert_text("\n")

        session = PromptSession(
            history=FileHistory(str(history_file)),
            style=PromptStyle.from_dict(
                {
                    "user": "#00ff99 bold",
                    "arrow": "#00aaff bold",
                }
            ),
            multiline=True,
            mouse_support=False,
            key_bindings=bindings,
            prompt_continuation=lambda width, line_number, wrap_count: " " * width,
        )

        # Track buffer text changes for paste detection.
        session.default_buffer.on_text_changed += _on_text_changed

        def read_prompt() -> str:
            return session.prompt(HTML("<user>user</user> <arrow>➜</arrow> ")).strip()

        return read_prompt

    def read_prompt() -> str:
        return input("\n> ").strip()

    return read_prompt


def run_once(agent, prompt: str, ui: CLIUI) -> int:
    try:
        run_agent_turn(agent, prompt, ui)
        return 0
    except Exception as exc:
        ui.error(f"Run failed: {exc}")
        return 1


def run_interactive(agent, workspace: Path, args, ui: CLIUI) -> int:
    history_file = Path(args.history_file).expanduser() if args.history_file else default_history_file(workspace)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    read_prompt = build_prompt_reader(history_file)

    ui.render_banner(agent, workspace)

    while True:
        try:
            user_input = read_prompt()
        except EOFError:
            ui.print()
            maybe_auto_save(agent, args.session_name, not args.no_auto_save, ui, "eof")
            return 0
        except KeyboardInterrupt:
            ui.warning("Interrupted. Type `exit` to quit.")
            continue

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit", "q"}:
            maybe_auto_save(agent, args.session_name, not args.no_auto_save, ui, "exit")
            return 0

        if lowered == "/help":
            print_help(ui)
            continue
        if lowered in {"/info", "/model"}:
            show_runtime_info(agent, workspace, ui)
            continue
        if lowered == "/tools":
            ui.render_tools(agent)
            continue
        if lowered == "/pwd":
            ui.info(f"Current working directory: {getattr(agent, 'working_dir', workspace)}")
            continue
        if lowered.startswith("/cd"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                ui.warning("Usage: /cd <path>")
                continue
            new_dir = parts[1].strip()
            try:
                requested = Path(new_dir).expanduser()
                current_dir = Path(getattr(agent, "working_dir", workspace))
                resolved = requested.resolve() if requested.is_absolute() else (current_dir / requested).resolve()
                agent.set_working_dir(str(resolved))
                ui.success(f"Working directory updated: {agent.working_dir}")
            except Exception as exc:
                ui.error(f"cd failed: {exc}")
            continue
        if lowered.startswith("/history"):
            parts = user_input.split(maxsplit=1)
            limit = None
            if len(parts) > 1:
                try:
                    limit = int(parts[1].strip())
                except ValueError:
                    ui.warning("Usage: /history [n]")
                    continue
            ui.render_history(agent.get_history(), limit=limit)
            continue
        if lowered == "/log":
            if ui.use_rich:
                text = ui.console.export_text(clear=False)
                if not text.strip():
                    ui.info("No output recorded yet.")
                    continue
                pager = shutil.which("less") or shutil.which("more")
                if pager:
                    proc = subprocess.Popen(
                        [pager, "-R"] if "less" in pager else [pager],
                        stdin=subprocess.PIPE,
                    )
                    try:
                        proc.communicate(input=text.encode("utf-8", errors="replace"))
                    except BrokenPipeError:
                        pass
                else:
                    print(text)
            else:
                ui.warning("/log requires rich mode (run without --plain).")
            continue
        if lowered == "/clear":
            agent.clear_history()
            ui.success("History cleared.")
            continue
        if lowered.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            session_name = normalize_session_name(parts[1].strip()) if len(parts) > 1 else normalize_session_name(args.session_name)
            try:
                saved_path = agent.save_session(session_name)
                ui.success(f"Saved session: {saved_path}")
            except Exception as exc:
                ui.error(f"Save failed: {exc}")
            continue
        if lowered.startswith("/load"):
            parts = user_input.split(maxsplit=1)
            session_path = resolve_session_to_load(agent, parts[1].strip() if len(parts) > 1 else None, args.session_name)
            if session_path is None:
                ui.error("Session persistence is not enabled.")
                continue
            if not session_path.exists():
                ui.error(f"Session not found: {session_path}")
                continue
            try:
                agent.load_session(str(session_path))
                ui.success(f"Loaded session: {session_path}")
            except Exception as exc:
                ui.error(f"Load failed: {exc}")
            continue
        if lowered == "/sessions":
            if not getattr(agent, "session_store", None):
                ui.error("Session persistence is not enabled.")
                continue
            try:
                ui.render_sessions(agent.session_store.list_sessions())
            except Exception as exc:
                ui.error(f"Failed to list sessions: {exc}")
            continue
        if lowered.startswith("/compact"):
            parts = user_input.split(maxsplit=1)
            focus = parts[1].strip() if len(parts) > 1 else None
            try:
                result = agent.compact(focus=focus)
                ui.success(result)
            except Exception as exc:
                ui.error(f"Compact failed: {exc}")
            continue

        try:
            ui.print()
            run_agent_turn(agent, user_input, ui)
        except KeyboardInterrupt:
            ui.warning("Interrupted.")
        except Exception as exc:
            ui.error(f"Error: {exc}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    ui = CLIUI(use_rich=not args.plain)

    try:
        agent, workspace = create_agent(args, ui)
    except Exception as exc:
        ui.error(f"Failed to initialize agent: {exc}")
        return 1

    if args.list_tools:
        ui.render_tools(agent)
        return 0

    if args.prompt and not args.interactive:
        ui.render_banner(agent, workspace)
        exit_code = run_once(agent, args.prompt, ui)
        maybe_auto_save(agent, args.session_name, not args.no_auto_save, ui, "single-turn")
        return exit_code

    return run_interactive(agent, workspace, args, ui)


if __name__ == "__main__":
    raise SystemExit(main())
