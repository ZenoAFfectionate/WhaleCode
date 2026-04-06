"""Interactive CLI for running the coding agent in a terminal."""

from __future__ import annotations

import argparse
import json
import selectors
import shutil
import subprocess
import sys
import threading
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


TODO_MUTATING_ACTIONS = {"create", "update", "bulk_create", "delete"}
INTERACTIVE_EXIT_WORDS = frozenset({"exit"})
INTERACTIVE_EXACT_COMMANDS = (
    "/help",
    "/info",
    "/tools",
    "/pwd",
    "/log",
    "/clear",
    "/sessions",
)
INTERACTIVE_PREFIX_COMMANDS = (
    "/cd",
    "/history",
    "/save",
    "/resume",
    "/compact",
)


class InputBuffer:
    """Thread-safe buffer for user input during agent execution."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: list[str] = []

    def add(self, text: str) -> None:
        with self._lock:
            self._items.append(text)

    def drain(self) -> list[str]:
        """Return and clear all buffered messages."""
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items

    def has_pending(self) -> bool:
        with self._lock:
            return len(self._items) > 0

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def _collect_buffered_input(
    thread: threading.Thread, buffer: InputBuffer, ui: "CLIUI"
) -> None:
    """Collect user input lines from stdin while *thread* is alive.

    Uses ``selectors`` for non-blocking reads so the main thread can
    periodically check whether the agent thread has finished.
    """
    sel = selectors.DefaultSelector()
    try:
        sel.register(sys.stdin, selectors.EVENT_READ)
    except (ValueError, OSError):
        # stdin is not selectable (e.g. redirected / not a real FD)
        thread.join()
        return

    try:
        while thread.is_alive():
            events = sel.select(timeout=0.3)
            if events:
                try:
                    line = sys.stdin.readline()
                except (EOFError, OSError):
                    break
                if not line:  # EOF
                    break
                text = line.strip()
                if text:
                    buffer.add(text)
                    ui.info(
                        f"  [queued for next turn] {text[:60]}{'...' if len(text) > 60 else ''}"
                    )
    except KeyboardInterrupt:
        pass  # User interrupted; we still join the thread below
    finally:
        try:
            sel.unregister(sys.stdin)
        except Exception:
            pass
        sel.close()
    # Ensure thread is fully joined before returning
    thread.join(timeout=5)


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

    TASK_MARKERS = {
        "completed": ("✔", "green"),
        "in_progress": ("►", "bold yellow"),
        "pending": ("◻", "dim"),
        "cancelled": ("✘", "dim red"),
    }

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
            import rich.box as box

            # --- Left panel content (whale art, centered) ---
            whale_lines = [
                '      .',
                '      ":"',
                '    ___:____     |"\\/"|',
                "  ,'        `.    \\  /",
                "  |  O    _   \\___/  |",
                "~^~^~^~^~^~^~",
            ]
            # Find the widest whale line for centering reference
            whale_width = max(len(line) for line in whale_lines)

            # Pad "Welcome back!" to center-align with whale art
            welcome_text = "Welcome back!"
            welcome_pad = max(0, (whale_width - len(welcome_text)) // 2)
            model_text = str(model)
            model_pad = max(0, (whale_width - len(model_text)) // 2)

            left_parts = Text()
            left_parts.append("\n")
            left_parts.append(" " * welcome_pad + welcome_text + "\n", style="bold white")
            left_parts.append("\n")
            for line in whale_lines:
                left_parts.append(f"{line}\n", style="bold bright_blue")
            left_parts.append("\n")
            left_parts.append(" " * model_pad + model_text + "\n", style="bold bright_cyan")

            left_aligned = Align(left_parts, align="center")

            # --- Right panel content ---
            tool_count = len(agent.tool_registry.list_tools()) if agent.tool_registry else 0
            right_parts = Text()
            right_parts.append("\n")
            right_parts.append("  Runtime\n", style="bold white")
            right_parts.append(f"  Provider    {provider}\n", style="dim")
            right_parts.append(f"  Workspace   {_pretty_path(workspace)}\n", style="dim")
            right_parts.append(f"  Tools       {tool_count} registered\n", style="dim")
            session_on = bool(getattr(agent, "session_store", None))
            right_parts.append(f"  Session     {'enabled' if session_on else 'disabled'}\n", style="dim")

            # --- Two-column table with vertical divider ---
            # Use a custom box that only draws inner vertical lines
            INNER_VERT = box.Box(
                "    \n"
                "  │ \n"
                "    \n"
                "  │ \n"
                "    \n"
                "  │ \n"
                "    \n"
                "    \n"
            )
            layout = Table(
                box=INNER_VERT,
                show_header=False,
                show_edge=False,
                expand=True,
                border_style="dim bright_blue",
                padding=(0, 2),
            )
            layout.add_column(ratio=1)   # left: whale
            layout.add_column(ratio=1)   # right: info

            layout.add_row(left_aligned, right_parts)

            title = f"  {display_name} {version}  "
            banner = Panel(
                layout,
                title=title,
                title_align="left",
                border_style="bright_blue",
                padding=(0, 1),
                width=self.console.width,
            )

            self.console.print(banner)
            self.console.print("[dim]Type `/help` for commands, or `exit` to quit.[/dim]")
        else:
            P = "    "
            W = 38
            L = 14
            print(f"--- {display_name} {version} ---")
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

    def render_task_status(self, agent) -> None:
        """Render a compact task status bar from the TodoWrite tool's task files."""
        tasks = self._get_task_list(agent)
        if not tasks:
            return

        max_label_len = 30
        lines = []
        for t in tasks:
            label = t["subject"]
            if len(label) > max_label_len:
                label = label[: max_label_len - 1] + "…"
            marker, style = self.TASK_MARKERS.get(t["status"], ("?", "dim"))
            if self.use_rich:
                lines.append(f"  [{style}]{marker} {label}[/{style}]")
            else:
                lines.append(f"  {marker} {label}")

        if not lines:
            return

        total = len(tasks)
        done = sum(1 for t in tasks if t["status"] == "completed")
        header = f"Tasks [{done}/{total}]"

        if self.use_rich:
            self.console.print(Rule(header, style="dim cyan"))
            for line in lines:
                self.console.print(line)
            self.console.print(Rule(style="dim cyan"))
        else:
            print(f"--- {header} ---")
            for line in lines:
                print(line)
            print("---")

    def render_inline_task_progress(self, agent) -> None:
        """Print a compact one-line task progress during agent execution."""
        tasks = self._get_task_list(agent)
        if not tasks:
            return

        total = len(tasks)
        done = sum(1 for t in tasks if t["status"] == "completed")
        in_progress = [t for t in tasks if t["status"] == "in_progress"]
        current = in_progress[0]["subject"][:40] if in_progress else ""

        if self.use_rich:
            line = f"[dim cyan]Tasks [{done}/{total}][/dim cyan]"
            if current:
                line += f" [bold yellow]► {current}[/bold yellow]"
            self.console.print(line)
        else:
            line = f"Tasks [{done}/{total}]"
            if current:
                line += f" ► {current}"
            print(line)

    def all_tasks_completed(self, agent) -> bool:
        """Return True when tasks exist and every one is completed or cancelled."""
        tasks = self._get_task_list(agent)
        return bool(tasks) and all(
            t["status"] in {"completed", "cancelled"} for t in tasks
        )

    def has_active_tasks(self, agent) -> bool:
        """Return True when there are tasks that are not all completed."""
        tasks = self._get_task_list(agent)
        return bool(tasks) and not all(
            t["status"] in {"completed", "cancelled"} for t in tasks
        )

    @staticmethod
    def _get_task_list(agent) -> list[dict]:
        """Return the current task list from the agent's TodoWrite tool, or []."""
        registry = getattr(agent, "tool_registry", None)
        if not registry:
            return []
        todo_tool = registry.get_tool("TodoWrite")
        if not todo_tool or not hasattr(todo_tool, "task_manager"):
            return []
        try:
            return todo_tool.task_manager.list_all()
        except Exception:
            return []

    def render_assistant(self, text: str) -> None:
        if self.use_rich:
            self.console.print(Panel(Markdown(text), title="Assistant", border_style="blue", width=self.console.width))
        else:
            print(text)

    def render_summary(self, elapsed_s: float, agent=None) -> None:
        if not self.use_rich:
            print(f"[completed in {elapsed_s:.2f}s]")
            return

        time_part = f"⌚ Completed in {elapsed_s:.2f}s"
        total_t = getattr(agent, "_total_tokens", 0) if agent else 0

        if total_t > 0:
            prompt_t = getattr(agent, "_turn_prompt_tokens", 0)
            comp_t = getattr(agent, "_turn_completion_tokens", 0)
            ctx_window = getattr(agent.config, "context_window", 0)
            ctx_used = getattr(agent, "_last_prompt_tokens", 0)

            token_parts = [
                f"[green]⬆Σ{prompt_t:,}[/green]",
                f"[yellow]⬇Σ{comp_t:,}[/yellow]",
            ]
            if ctx_window > 0 and ctx_used > 0:
                pct = ctx_used * 100 / ctx_window
                token_parts.append(
                    f"[cyan]📊 last {ctx_used:,} / {ctx_window:,} ({pct:.0f}%)[/cyan]"
                )
            self.console.print(
                f"[dim]{time_part}    |    {' · '.join(token_parts)}[/dim]"
            )
        else:
            self.console.print(f"[dim cyan]{time_part}[/dim cyan]")

    def render_rule(self, title: str) -> None:
        if self.use_rich:
            self.console.print(Rule(title, style="dim"))
        else:
            print(title)

    _LOG_BLOCK_STYLES = {
        "action":     ("Tool Call",        "bright_green", "🎬"),
        "thinking":   ("Thinking",         "bright_blue",  "💭"),
        "observation":("Observation",      "bright_cyan",  "👀"),
        "info":       ("Info",             "cyan",         "ℹ️"),
        "background": ("Background Update","magenta",      "📬"),
        "warning":    ("Warning",          "yellow",       "⚠️"),
        "error":      ("Error",            "red",          "❌"),
    }

    def render_log_block(self, kind: str, content: str) -> None:
        content = content.rstrip()
        if not content:
            return

        title, border, icon = self._LOG_BLOCK_STYLES.get(kind, (None, None, ""))

        if not self.use_rich:
            prefix = f"[{title or kind}] " if title else ""
            print(f"{prefix}{content}")
            return

        safe = Text(content)
        if title:
            self.console.print(Panel(safe, title=title, border_style=border, width=self.console.width))
        else:
            self.console.print(safe)

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
        items = all_items[-limit:] if limit and limit > 0 else all_items
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

    # Events that are intentionally suppressed in the CLI.
    _IGNORED_EVENTS = frozenset({
        "agent_start",      # Don't echo user input back
        "step_start",       # Rendered implicitly by tool calls
        "direct_response",  # Rendered via render_assistant
        "final_answer",     # Rendered via render_assistant
        "timeout",          # Rendered by the agent loop itself
    })

    def _reset_todo_turn_tracking(self) -> None:
        self._todo_changed_this_turn = False
        self._todo_mutating_call_ids = set()
        self._todo_mutating_call_without_id = False

    def _console(self, message: str = "", *, end: str = "\n") -> None:
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
        if event_type in self._IGNORED_EVENTS:
            return

        if event_type == "compaction_notice":
            self.ui.render_log_block("warning", "[auto-compact triggered]")
        elif event_type == "llm_error":
            self.ui.render_log_block("error", f"LLM call failed: {payload.get('error', '')}")
        elif event_type == "builtin_tool":
            tool_name = payload.get("tool_name")
            result_content = str(payload.get("result_content", ""))
            if tool_name == "Thought":
                self.ui.render_log_block("thinking", result_content.removeprefix("Reasoning: "))
            elif tool_name != "Finish":
                self.ui.render_log_block("info", f"{tool_name}: {result_content}")
        elif event_type == "tool_call":
            tool_name = payload.get("tool_name")
            arguments = payload.get("arguments", {})
            tool_call_id = payload.get("tool_call_id")
            prefix = f"[{tool_call_id}] " if tool_call_id else ""
            self.ui.render_log_block("action", f"{prefix}{tool_name}({arguments})")

            if tool_name == "TodoWrite" and isinstance(arguments, dict):
                action = str(arguments.get("action", "")).strip().lower()
                has_todos_payload = isinstance(arguments.get("todos"), list)
                if action in TODO_MUTATING_ACTIONS or has_todos_payload:
                    if tool_call_id:
                        if not hasattr(self, "_todo_mutating_call_ids"):
                            self._todo_mutating_call_ids = set()
                        self._todo_mutating_call_ids.add(tool_call_id)
                    else:
                        self._todo_mutating_call_without_id = True
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

            if tool_name == "TodoWrite":
                tracked_ids = getattr(self, "_todo_mutating_call_ids", set())
                tracked_wo_id = getattr(self, "_todo_mutating_call_without_id", False)
                is_mutating = (tool_call_id in tracked_ids) or tracked_wo_id
                if is_mutating:
                    status = str(payload.get("status", "")).lower()
                    content = str(payload.get("result_content", ""))
                    is_success = status != "error" and not content.startswith("❌")
                    if is_success:
                        self._todo_changed_this_turn = True
                        self.ui.render_inline_task_progress(self)

                if tool_call_id and tool_call_id in tracked_ids:
                    tracked_ids.discard(tool_call_id)
                elif tracked_wo_id:
                    self._todo_mutating_call_without_id = False
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
            )
        else:
            super()._render_event(event_type, payload)


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
    registry = ToolRegistry(config=config, verbose=False)
    agent = CLICodeAgent(
        name="code-agent",
        llm=llm,
        tool_registry=registry,
        project_root=str(workspace),
        working_dir=str(workspace),
        config=config,
        max_steps=0,
        register_default_tools=True,
        enable_task_tool=True,
        ui=ui,
    )

    if args.resume:
        resume_path = resolve_session_to_load(agent, args.resume, args.session_name)
        if resume_path is None:
            raise RuntimeError("Session persistence is not enabled.")
        if not resume_path.exists():
            raise FileNotFoundError(f"Session not found: {resume_path}")
        agent.load_session(str(resume_path))
        maybe_restore_task_snapshot(agent, resume_path, ui=ui)
    else:
        # New chat starts with fresh tasks instead of stale persisted ones.
        clear_todo_tasks(agent)

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
        maybe_save_task_snapshot(agent, Path(saved_path), ui=ui)
        ui.info(f"Auto-saved session ({reason}): {saved_path}")
    except Exception as exc:
        ui.warning(f"Auto-save failed: {exc}")


def _task_snapshot_path(session_path: Path) -> Path:
    return session_path.with_name(f"{session_path.stem}.tasks.json")


def _get_todo_task_dir(agent) -> Optional[Path]:
    try:
        if not hasattr(agent, "tool_registry") or not agent.tool_registry:
            return None
        todo_tool = agent.tool_registry.get_tool("TodoWrite")
        if not todo_tool or not hasattr(todo_tool, "task_manager"):
            return None
        task_dir = Path(todo_tool.task_manager.dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir
    except Exception:
        return None


def clear_todo_tasks(agent, ui: Optional[CLIUI] = None) -> None:
    """Clear persisted TodoWrite tasks for starting a fresh conversation."""
    task_dir = _get_todo_task_dir(agent)
    if not task_dir:
        return

    removed = 0
    for path in task_dir.glob("task_*.json"):
        try:
            if path.is_file():
                path.unlink()
                removed += 1
        except Exception:
            continue

    if ui and removed:
        ui.info(f"Cleared {removed} task file(s) for fresh conversation.")


def maybe_save_task_snapshot(agent, session_path: Path, ui: Optional[CLIUI] = None) -> None:
    """Persist current TodoWrite tasks beside the session file."""
    task_dir = _get_todo_task_dir(agent)
    if not task_dir:
        return

    tasks = []
    for path in sorted(task_dir.glob("task_*.json")):
        try:
            tasks.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

    snapshot_path = _task_snapshot_path(session_path)
    payload = {"session_file": str(session_path), "tasks": tasks}
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if ui:
        ui.info(f"Task snapshot saved: {snapshot_path}")


def maybe_restore_task_snapshot(agent, session_path: Path, ui: Optional[CLIUI] = None) -> bool:
    """Restore TodoWrite tasks from sidecar snapshot file.

    Returns True if a snapshot file is found and restored.
    """
    task_dir = _get_todo_task_dir(agent)
    if not task_dir:
        return False

    snapshot_path = _task_snapshot_path(session_path)
    if not snapshot_path.exists():
        return False

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return False

        clear_todo_tasks(agent)
        for task in tasks:
            task_id = task.get("id")
            if task_id is None:
                continue
            target = task_dir / f"task_{int(task_id)}.json"
            target.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if ui:
            ui.info(f"Restored {len(tasks)} task(s) from snapshot.")
        return True
    except Exception as exc:
        if ui:
            ui.warning(f"Task snapshot restore failed: {exc}")
        return False


def load_session_and_tasks(agent, raw_value: Optional[str], fallback_name: str, ui: CLIUI) -> bool:
    """Load session history and matching TodoWrite task snapshot."""
    session_path = resolve_session_to_load(agent, raw_value, fallback_name)
    if session_path is None:
        ui.error("Session persistence is not enabled.")
        return False
    if not session_path.exists():
        ui.error(f"Session not found: {session_path}")
        return False

    try:
        agent.load_session(str(session_path))
        maybe_restore_task_snapshot(agent, session_path, ui=ui)
        ui.success(f"Loaded session: {session_path}")
        return True
    except Exception as exc:
        ui.error(f"Load failed: {exc}")
        return False


def run_agent_turn(
    agent, prompt: str, ui: CLIUI, input_buffer: InputBuffer | None = None
) -> str:
    if hasattr(agent, "_reset_todo_turn_tracking"):
        agent._reset_todo_turn_tracking()

    start_time = time.time()

    # Run agent in a background thread so the main thread can collect
    # buffered user input without blocking.
    result_holder: list[str | None] = [None]
    error_holder: list[BaseException | None] = [None]

    def _agent_work():
        try:
            result_holder[0] = agent.run(prompt)
        except BaseException as exc:
            error_holder[0] = exc

    thread = threading.Thread(target=_agent_work, daemon=True)
    thread.start()

    if input_buffer is not None and sys.stdin.isatty():
        _collect_buffered_input(thread, input_buffer, ui)
    else:
        thread.join()

    if error_holder[0] is not None:
        raise error_holder[0]

    response = result_holder[0] or ""

    # Render final task status (static) after the turn.
    ui.render_task_status(agent)

    # If all tasks are now completed, clear them so they won't show again.
    if ui.all_tasks_completed(agent):
        clear_todo_tasks(agent)

    ui.render_assistant(response)
    ui.render_summary(time.time() - start_time, agent=agent)
    return response


def print_help(ui: CLIUI) -> None:
    lines = [
        "Commands:",
        "- /help                 Show this help message",
        "- /info                 Show workspace, model, and runtime info",
        "- /tools                Show registered tools and descriptions",
        "- /pwd                  Show the current working directory",
        "- /cd <path>            Change the agent working directory within the workspace",
        "- /history [n]          Show recent conversation turns",
        "- /log                  View all terminal output in a scrollable pager",
        "- /clear                Clear in-memory conversation history",
        "- /save [name]          Save a session snapshot into the session directory",
        "- /resume [path|name]   Load a saved session (default: session-latest)",
        "- /sessions             List saved sessions",
        "- /compact [focus]      Compact conversation context",
        "- exit                  Exit the CLI",
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
        from prompt_toolkit.filters import in_paste_mode

        # Timing-based paste detection (fallback for terminals without
        # bracketed paste support).  Human typing has >150ms between the
        # last keystroke and Enter; pasted text arrives as a burst (~0ms).
        _PASTE_THRESHOLD = 0.15  # 150 ms
        _last_text_change = [0.0]

        def _on_text_changed(_buf):
            _last_text_change[0] = time.monotonic()

        bindings = KeyBindings()

        # --- Bracketed paste mode: always insert newline (never submit) ---
        @bindings.add(Keys.Enter, eager=True, filter=in_paste_mode)
        def _paste_enter(event):
            event.current_buffer.insert_text("\n")

        # --- Normal mode: timing heuristic ---
        @bindings.add(Keys.Enter, eager=True, filter=~in_paste_mode)
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

    input_buffer = InputBuffer()

    # -- Command handlers --

    def _cmd_help(raw, lowered):
        print_help(ui)

    def _cmd_info(raw, lowered):
        show_runtime_info(agent, workspace, ui)

    def _cmd_tools(raw, lowered):
        ui.render_tools(agent)

    def _cmd_pwd(raw, lowered):
        ui.info(f"Current working directory: {getattr(agent, 'working_dir', workspace)}")

    def _cmd_cd(raw, lowered):
        parts = raw.split(maxsplit=1)
        if len(parts) < 2:
            ui.warning("Usage: /cd <path>")
            return
        try:
            requested = Path(parts[1].strip()).expanduser()
            current_dir = Path(getattr(agent, "working_dir", workspace))
            resolved = requested.resolve() if requested.is_absolute() else (current_dir / requested).resolve()
            agent.set_working_dir(str(resolved))
            ui.success(f"Working directory updated: {agent.working_dir}")
        except Exception as exc:
            ui.error(f"cd failed: {exc}")

    def _cmd_history(raw, lowered):
        parts = raw.split(maxsplit=1)
        limit = None
        if len(parts) > 1:
            try:
                limit = int(parts[1].strip())
            except ValueError:
                ui.warning("Usage: /history [n]")
                return
        ui.render_history(agent.get_history(), limit=limit)

    def _cmd_log(raw, lowered):
        if not ui.use_rich:
            ui.warning("/log requires rich mode (run without --plain).")
            return
        text = ui.console.export_text(clear=False)
        if not text.strip():
            ui.info("No output recorded yet.")
            return
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

    def _cmd_clear(raw, lowered):
        agent.clear_history()
        ui.success("History cleared.")

    def _cmd_save(raw, lowered):
        parts = raw.split(maxsplit=1)
        name = normalize_session_name(parts[1].strip()) if len(parts) > 1 else normalize_session_name(args.session_name)
        try:
            saved_path = agent.save_session(name)
            maybe_save_task_snapshot(agent, Path(saved_path), ui=ui)
            ui.success(f"Saved session: {saved_path}")
        except Exception as exc:
            ui.error(f"Save failed: {exc}")

    def _cmd_resume(raw, lowered):
        parts = raw.split(maxsplit=1)
        load_session_and_tasks(
            agent,
            parts[1].strip() if len(parts) > 1 else None,
            args.session_name,
            ui,
        )

    def _cmd_sessions(raw, lowered):
        if not getattr(agent, "session_store", None):
            ui.error("Session persistence is not enabled.")
            return
        try:
            ui.render_sessions(agent.session_store.list_sessions())
        except Exception as exc:
            ui.error(f"Failed to list sessions: {exc}")

    def _cmd_compact(raw, lowered):
        parts = raw.split(maxsplit=1)
        focus = parts[1].strip() if len(parts) > 1 else None
        try:
            result = agent.compact(focus=focus)
            ui.success(result)
        except Exception as exc:
            ui.error(f"Compact failed: {exc}")

    # Exact-match commands (lowered == key)
    _exact_cmds = {
        "/help": _cmd_help,
        "/info": _cmd_info,
        "/tools": _cmd_tools,
        "/pwd": _cmd_pwd,
        "/log": _cmd_log,
        "/clear": _cmd_clear,
        "/sessions": _cmd_sessions,
    }

    # Prefix-match commands (lowered.startswith(key))
    _prefix_cmds = [
        ("/cd", _cmd_cd),
        ("/history", _cmd_history),
        ("/save", _cmd_save),
        ("/resume", _cmd_resume),
        ("/compact", _cmd_compact),
    ]

    def _dispatch_command(raw: str, lowered: str) -> bool:
        """Try to handle *raw* as a slash command. Returns True if handled."""
        handler = _exact_cmds.get(lowered)
        if handler:
            handler(raw, lowered)
            return True
        for prefix, handler in _prefix_cmds:
            if lowered.startswith(prefix):
                handler(raw, lowered)
                return True
        return False

    # -- Main loop --

    ui.render_banner(agent, workspace)

    while True:
        try:
            if ui.has_active_tasks(agent):
                ui.render_task_status(agent)
            user_input = read_prompt()
        except EOFError:
            ui.print()
            maybe_auto_save(agent, args.session_name, not args.no_auto_save, ui, "eof")
            clear_todo_tasks(agent)
            return 0
        except KeyboardInterrupt:
            ui.warning("Interrupted. Type `exit` to quit.")
            continue

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in INTERACTIVE_EXIT_WORDS:
            maybe_auto_save(agent, args.session_name, not args.no_auto_save, ui, "exit")
            clear_todo_tasks(agent)
            return 0

        if _dispatch_command(user_input, lowered):
            continue

        # Not a command — send to agent.
        try:
            ui.print()
            run_agent_turn(agent, user_input, ui, input_buffer=input_buffer)
        except KeyboardInterrupt:
            ui.warning("Interrupted.")
        except Exception as exc:
            ui.error(f"Error: {exc}")

        # Drain any input that was buffered while the agent was running
        # and auto-send each one as a new turn.
        while input_buffer.has_pending():
            buffered = input_buffer.drain()
            combined = "\n".join(buffered)
            if not combined.strip():
                break
            ui.info(f"[auto-sending buffered input]")
            try:
                ui.print()
                run_agent_turn(agent, combined, ui, input_buffer=input_buffer)
            except KeyboardInterrupt:
                ui.warning("Interrupted.")
                input_buffer.clear()
                break
            except Exception as exc:
                ui.error(f"Error: {exc}")
                break


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
        clear_todo_tasks(agent)
        return exit_code

    return run_interactive(agent, workspace, args, ui)


if __name__ == "__main__":
    raise SystemExit(main())
