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
    from rich.box import HEAVY, ROUNDED
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme

    RICH_AVAILABLE = True
except ImportError:
    Align = Console = Markdown = Panel = Rule = Table = Text = Theme = None
    HEAVY = ROUNDED = None
    RICH_AVAILABLE = False


class Palette:
    """Central CLI color palette — mirrors the project Web showcase
    (``asserts/whalecode_showcase.html``) so the terminal matches the brand:
    a deep-navy dark theme with blue/cyan accents and slate muted text.
    """

    BG_PANEL = "#0b1220"     # panel fill (thinking / tool cards)
    BORDER_DIM = "#243047"   # thin separators / step rule
    ACCENT = "#3b82f6"       # primary (assistant / step)
    CYAN = "#22d3ee"         # tools / links
    THINKING = "#a78bfa"     # reasoning
    SUCCESS = "#22c55e"      # success state
    WARNING = "#f59e0b"      # warnings / truncation
    ERROR = "#ef4444"        # errors
    MUTED = "#64748b"        # metadata (elapsed, sizes, paths)
    TEXT = "#f1f5f9"         # body text


# A Rich theme built from the same palette so markup can use semantic names
# (e.g. ``[muted]``) and console styling stays centralized (VR-6).
if RICH_AVAILABLE:
    WHALE_THEME = Theme(
        {
            "accent": Palette.ACCENT,
            "accent.cyan": Palette.CYAN,
            "thinking": Palette.THINKING,
            "success": Palette.SUCCESS,
            "warning": Palette.WARNING,
            "error": Palette.ERROR,
            "muted": Palette.MUTED,
            "text": Palette.TEXT,
            "border.dim": Palette.BORDER_DIM,
            "panel.bg": f"on {Palette.BG_PANEL}",
        }
    )
else:
    WHALE_THEME = None

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
    "/pwd",
    "/log",
    "/trace",
    "/clear",
    "/sessions",
)
INTERACTIVE_PREFIX_COMMANDS = (
    "/tools",
    "/cd",
    "/history",
    "/save",
    "/resume",
    "/compact",
)
REASONING_MODES = ("off", "summary", "preview", "full")
DEFAULT_CLI_ARTIFACT_DIRNAME = "cli_artifacts"


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
        self.console = (
            Console(record=True, theme=WHALE_THEME) if self.use_rich else None
        )

    def spacer(self, n: int = 1) -> None:
        """Emit *n* blank lines to keep a consistent vertical rhythm (VR-5)."""
        for _ in range(max(0, n)):
            if self.use_rich:
                self.console.print("")
            else:
                print("")

    def render_step_header(self, step, ctx_info: str = "") -> None:
        """Render a ReAct step boundary (VR-5).

        Rich mode draws a left-aligned dim rule with a leading blank line so
        each step reads as its own paragraph; plain mode keeps the ``✦ Step N``
        marker string that the transcript tests assert on.
        """
        label = f"✦ Step {step}"
        if ctx_info:
            label += f"  {ctx_info}"
        if self.use_rich:
            self.spacer(1)
            # Pass a Text (not a str) so a bracketed ctx snapshot like
            # "[ctx 3,200 / 100,000  3%]" is not eaten by Rich markup parsing.
            self.console.print(
                Rule(Text(label, style=Palette.MUTED), style=Palette.BORDER_DIM, align="left")
            )
        else:
            print(label)

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
        """Return the current task list from the agent's TodoWrite tool, or [].

        重要-5: TodoWrite 已重构为 session-scoped TodoSessionStore，旧的
        ``task_manager``/``task_*.json`` 集成早已失效。改为读取工具的
        ``export_state()`` 快照，并把字段归一化为渲染层期望的 subject/status。
        """
        registry = getattr(agent, "tool_registry", None)
        if not registry:
            return []
        todo_tool = registry.get_tool("TodoWrite")
        if not todo_tool or not hasattr(todo_tool, "export_state"):
            return []
        try:
            state = todo_tool.export_state()
        except Exception:
            return []
        todos = state.get("todos", []) if isinstance(state, dict) else []
        normalized: list[dict] = []
        for item in todos:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            normalized.append(
                {
                    "subject": content,
                    "content": content,
                    "status": str(item.get("status", "pending")),
                    "priority": str(item.get("priority", "medium")),
                }
            )
        return normalized

    def render_assistant(self, text: str) -> None:
        """CLI-6 / VR-3: render the model's final answer.

        Rich mode boxes the answer in an accent-bordered panel so it reads as
        the highest-weight block on screen; plain mode keeps the ``── Assistant ──``
        rule text that the render tests assert on.
        """
        if not text.strip():
            return
        if self.use_rich:
            self.spacer(1)
            self.console.print(
                Panel(
                    Markdown(text),
                    title="Whale ▸ Assistant",
                    title_align="left",
                    border_style=Palette.ACCENT,
                    box=ROUNDED,
                    padding=(1, 2),
                )
            )
        else:
            print(f"\n── Assistant ──")
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

    # CLI-10: three-tier visual hierarchy.
    # Tier 1 (primary)   — model text output: no Panel, full weight.
    # Tier 2 (secondary) — tool calls / thinking: dim compact line, thin border.
    # Tier 3 (tertiary)  — observations / info: truncated, dim, small.
    _LOG_BLOCK_STYLES = {
        "action":      ("",                 "dim green",      ""),   # compact line
        "thinking":    ("",                 "dim blue",       ""),   # compact line
        "observation": ("",                 "dim",            ""),   # compact line
        "info":        ("",                 "cyan",           ""),   # compact line
        "background":  ("Background Update","magenta",       "📬"),
        "warning":     ("Warning",          "yellow",        "⚠️"),
        "error":       ("Error",            "red",           "❌"),
    }

    def render_log_block(self, kind: str, content: str) -> None:
        """CLI-10: tiered rendering — secondary items are compact dim lines;
        only warnings/errors get a full Panel.
        """
        content = content.rstrip()
        if not content:
            return

        title, border, icon = self._LOG_BLOCK_STYLES.get(kind, (None, None, ""))
        icon_str = f"{icon} " if icon else ""

        # VR-2: reasoning gets its own low-saturation panel in rich mode so the
        # "what the model is thinking" block is clearly bounded and separable.
        if kind == "thinking" and self.use_rich:
            self.spacer(1)
            self.console.print(
                Panel(
                    Text(content, style=f"italic {Palette.MUTED}"),
                    title="🧠 thinking",
                    title_align="left",
                    border_style=Palette.THINKING,
                    box=ROUNDED,
                    padding=(0, 1),
                    style=f"on {Palette.BG_PANEL}",
                )
            )
            return

        # Secondary tier: compact dim line.
        if kind in ("action", "thinking", "observation", "info"):
            if self.use_rich:
                self.console.print(f"[{border}]{icon_str}{content}[/{border}]")
            else:
                print(f"{icon_str}{content}")
            return

        # Primary tier: warning / error — keep the Panel for visibility.
        if not self.use_rich:
            prefix = f"[{title or kind}] " if title else ""
            print(f"{prefix}{content}")
            return

        # Primary tier: warning / error / background — boxed for visibility.
        safe = Text(content)
        if title:
            border_map = {
                "Warning": Palette.WARNING,
                "Error": Palette.ERROR,
                "Background Update": Palette.CYAN,
            }
            box_style = border_map.get(title, border)
            self.spacer(1)
            self.console.print(
                Panel(
                    safe,
                    title=f"{icon_str}{title}".strip(),
                    title_align="left",
                    border_style=box_style,
                    box=HEAVY,
                    padding=(0, 1),
                    width=self.console.width,
                )
            )
        else:
            self.console.print(safe)

    def render_tool_card(
        self,
        tool_name: str,
        arg_summary: str = "",
        *,
        is_error: bool = False,
        elapsed: float = 0.0,
        meta: str = "",
        body: str = "",
    ) -> None:
        """VR-4: render one tool call+result as a single bounded card.

        The title carries the tool name and a compact argument summary; the
        right-aligned subtitle carries the status marker, elapsed time and size;
        the body holds the (already head/tail-truncated) output preview. A
        leading blank line keeps cards visually separated.
        """
        marker = "✗" if is_error else "✓"
        border = Palette.ERROR if is_error else Palette.SUCCESS

        if not self.use_rich:
            line = f"{marker} {tool_name} {elapsed:.1f}s {meta}".rstrip()
            print(line)
            if body:
                print(body)
            return

        title = Text()
        title.append(f"▸ {tool_name}", style=Palette.CYAN)
        if arg_summary:
            title.append(f"  {arg_summary}", style=Palette.MUTED)

        subtitle = Text()
        subtitle.append(marker, style=border)
        tail = f" {elapsed:.1f}s"
        if meta:
            tail += f" · {meta}"
        subtitle.append(tail, style=Palette.MUTED)

        card_body = Text(body if body else "(no output)", style=Palette.MUTED)
        self.spacer(1)
        self.console.print(
            Panel(
                card_body,
                title=title,
                title_align="left",
                subtitle=subtitle,
                subtitle_align="right",
                border_style=border,
                box=ROUNDED,
                padding=(0, 1),
                style=f"on {Palette.BG_PANEL}",
            )
        )

    def status(self, message: str) -> None:
        """Render a compact current-phase hint."""
        if self.use_rich:
            self.console.print(f"[dim cyan]{message}[/dim cyan]")
        else:
            print(message)

    # ── VR-4b: /tools categorization ────────────────────────────────
    _TOOL_CATEGORY = {
        "Read": "File", "Write": "File", "Edit": "File", "MultiEdit": "File",
        "LS": "File", "Delete": "File", "Glob": "File", "Grep": "File",
        "Bash": "Shell",
        "WebFetch": "Web", "WebSearch": "Web",
        "TodoWrite": "Planning", "Task": "Planning", "AskUser": "Planning",
    }
    _TOOL_TAGS = {
        "Read": ("read",), "LS": ("read",), "Glob": ("read",), "Grep": ("read",),
        "Write": ("write",), "Edit": ("write",), "MultiEdit": ("write",),
        "Delete": ("write", "risk"),
        "Bash": ("shell", "risk"),
        "WebFetch": ("net",), "WebSearch": ("net",),
        "TodoWrite": ("plan",), "Task": ("plan",), "AskUser": ("ask",),
    }
    _TOOL_CATEGORY_ORDER = ("File", "Shell", "Web", "Planning", "Benchmark", "Other")

    @classmethod
    def _tool_category(cls, name: str) -> str:
        return cls._TOOL_CATEGORY.get(name, "Other")

    @classmethod
    def _tool_tags(cls, name: str) -> tuple:
        return cls._TOOL_TAGS.get(name, ())

    def render_tools(self, agent, full: bool = False) -> None:
        tools = sorted(agent.tool_registry.get_all_tools(), key=lambda item: item.name)

        # Full view: complete schema-style table with descriptions.
        if full:
            if self.use_rich:
                table = Table(title="Registered Tools (full)", border_style=Palette.CYAN)
                table.add_column("Tool", style=f"bold {Palette.CYAN}")
                table.add_column("Category", style=Palette.MUTED)
                table.add_column("Description", style=Palette.TEXT)
                for tool in tools:
                    description = (tool.description or "").strip().replace("\n", " ")
                    table.add_row(tool.name, self._tool_category(tool.name), description)
                self.console.print(table)
                return
            print("Registered tools (full):")
            for tool in tools:
                description = (tool.description or "").strip().replace("\n", " ")
                print(f"- [{self._tool_category(tool.name)}] {tool.name}: {description}")
            return

        # Default view: compact, category-grouped, scannable.
        from collections import defaultdict

        groups = defaultdict(list)
        for tool in tools:
            groups[self._tool_category(tool.name)].append(tool)
        ordered = [c for c in self._TOOL_CATEGORY_ORDER if c in groups]

        if not self.use_rich:
            print("Registered tools  (/tools --full for schemas):")
            for cat in ordered:
                names = ", ".join(t.name for t in groups[cat])
                print(f"  {cat}: {names}")
            return

        self.console.print(
            Rule(
                Text("Registered Tools  ·  /tools --full for schemas", style=Palette.MUTED),
                style=Palette.BORDER_DIM,
                align="left",
            )
        )
        for cat in ordered:
            header = Text(f"{cat}", style=f"bold {Palette.ACCENT}")
            self.console.print(header)
            row = Text("  ")
            for i, tool in enumerate(groups[cat]):
                if i:
                    row.append("   ")
                row.append(tool.name, style=Palette.TEXT)
                tags = self._tool_tags(tool.name)
                if tags:
                    row.append(f" [{'/'.join(tags)}]", style=Palette.MUTED)
            self.console.print(row)
            self.spacer(1)

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

    def render_event_history(self, events: Iterable[dict], limit: Optional[int] = None) -> None:
        """Render the structured CLI event timeline for debugging a turn."""
        all_items = list(events)
        items = all_items[-limit:] if limit and limit > 0 else all_items
        if not items:
            self.warning("Event trace is empty.")
            return

        start_index = max(1, len(all_items) - len(items) + 1)
        if self.use_rich:
            table = Table(title="Event Trace", border_style="cyan")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Event", style="magenta", width=14)
            table.add_column("Summary", style="white")
            for index, event in enumerate(items, start=start_index):
                table.add_row(
                    str(index),
                    str(event.get("event", "")),
                    str(event.get("summary", "")),
                )
            self.console.print(table)
            return

        for index, event in enumerate(items, start=start_index):
            print(f"{index}. [{event.get('event', '')}] {event.get('summary', '')}")

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
        "direct_response",  # Rendered via render_assistant
        "final_answer",     # Rendered via render_assistant
        "timeout",          # Rendered by the agent loop itself
    })

    # ── CLI-2: step visibility ──────────────────────────────────────
    # Compact step prefix shown once per model turn.
    _last_step_shown: int = 0

    def _reset_todo_turn_tracking(self) -> None:
        self._todo_changed_this_turn = False
        self._todo_mutating_call_ids = set()
        self._todo_mutating_call_without_id = False
        self._ensure_cli_runtime_state()

    # ── CLI-5: streaming line buffer ────────────────────────────────
    _streaming_line_buffer: str = ""

    def _ensure_cli_runtime_state(self) -> None:
        """Initialize per-CLI runtime state lazily for tests and resumed sessions."""
        if not hasattr(self, "_cli_events"):
            self._cli_events = []
        if not hasattr(self, "_tool_call_state"):
            self._tool_call_state = {}
        if not hasattr(self, "reasoning_mode"):
            self.reasoning_mode = "preview"

    def _record_cli_event(self, event: str, summary: str, **fields) -> dict:
        self._ensure_cli_runtime_state()
        item = {"ts": time.time(), "event": event, "summary": summary}
        item.update(fields)
        self._cli_events.append(item)
        if len(self._cli_events) > 500:
            del self._cli_events[:-500]
        return item

    def get_cli_events(self) -> list[dict]:
        self._ensure_cli_runtime_state()
        return list(self._cli_events)

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

    # ── CLI-3: compact tool-call argument display ───────────────────
    @staticmethod
    def _compact_args(arguments, max_len: int = 120) -> str:
        """Render tool arguments as a compact one-line summary."""
        if not isinstance(arguments, dict) or not arguments:
            return ""
        # Prefer semantically-meaningful keys first.
        for key in ("path", "command", "pattern", "query", "name", "url"):
            val = arguments.get(key)
            if isinstance(val, str) and val.strip():
                return val[:max_len] + ("…" if len(val) > max_len else "")
        # Fallback: flatten first value.
        first_val = next(iter(arguments.values()), None)
        text = str(first_val) if isinstance(first_val, str) else repr(first_val)
        return text[:max_len] + ("…" if len(text) > max_len else "")

    # ── VR-4: per-tool argument summarizer ──────────────────────────
    @staticmethod
    def _summarize_tool_args(tool_name: str, arguments, max_len: int = 120) -> str:
        """Render a tool-specific, human-scannable argument summary.

        Falls back to the generic :meth:`_compact_args` for unknown tools.
        Only covers tools that actually exist in this repo's registry.
        """
        if not isinstance(arguments, dict) or not arguments:
            return ""

        def _clip(value: str) -> str:
            value = str(value)
            return value[:max_len] + ("…" if len(value) > max_len else "")

        name = (tool_name or "").strip()

        if name == "Bash":
            cmd = str(arguments.get("command", "")).strip()
            if not cmd:
                return ""
            lines = cmd.splitlines()
            summary = _clip(lines[0])
            if len(lines) > 1 and not summary.endswith("…"):
                summary += "…"
            return summary
        if name in ("Read", "LS", "Delete"):
            path = str(arguments.get("path", "")).strip()
            offset = arguments.get("offset")
            return f"{path}:{offset}" if path and offset else path
        if name == "Write":
            path = str(arguments.get("path", "")).strip()
            content = arguments.get("content", "")
            size = len(content) if isinstance(content, str) else 0
            return f"{path} ({size} chars)" if path else f"{size} chars"
        if name == "Edit":
            path = str(arguments.get("path", "")).strip()
            return f"{path} (replace all)" if arguments.get("replace_all") else path
        if name in ("Grep", "Glob"):
            pattern = str(arguments.get("pattern", "")).strip()
            path = str(arguments.get("path", "")).strip()
            include = str(arguments.get("include", "")).strip()
            summary = pattern
            if path:
                summary += f" in {path}"
            if include:
                summary += f" ({include})"
            return _clip(summary)
        if name == "WebSearch":
            return _clip(str(arguments.get("query", "")).strip())
        if name == "WebFetch":
            return _clip(str(arguments.get("url", "")).strip())
        if name == "TodoWrite":
            todos = arguments.get("todos")
            if isinstance(todos, list):
                total = len(todos)
                done = sum(
                    1 for t in todos if isinstance(t, dict) and t.get("status") == "completed"
                )
                active = sum(
                    1 for t in todos if isinstance(t, dict) and t.get("status") == "in_progress"
                )
                summary = f"{total} tasks · {done} done"
                if active:
                    summary += f" · {active} active"
                return summary
            return ""

        # Unknown tool → generic summary.
        return CLICodeAgentMixin._compact_args(arguments, max_len)

    @staticmethod
    def _artifact_safe_name(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
        return safe[:40] or "artifact"

    def _cli_artifact_root(self) -> Optional[Path]:
        if getattr(self, "save_cli_artifacts", True) is False:
            return None
        base = getattr(self, "working_dir", None) or getattr(self, "project_root", None)
        try:
            root = Path(base).expanduser().resolve() if base else PROJECT_ROOT
            configured = getattr(self, "cli_artifact_dir", None)
            if configured:
                artifact_root = Path(configured).expanduser()
                if not artifact_root.is_absolute():
                    artifact_root = root / artifact_root
                artifact_root = artifact_root.resolve()
            else:
                artifact_root = root / "memory" / DEFAULT_CLI_ARTIFACT_DIRNAME
            return artifact_root
        except Exception:
            return None

    def _cli_artifact_dir(self, subdir: str) -> Optional[Path]:
        try:
            artifact_root = self._cli_artifact_root()
            if artifact_root is None:
                return None
            path = artifact_root / subdir
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            return None

    def _save_cli_artifact(self, subdir: str, stem: str, content: str) -> Optional[Path]:
        if not content:
            return None
        directory = self._cli_artifact_dir(subdir)
        if directory is None:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = len(getattr(self, "_cli_events", [])) + 1
        path = directory / f"{stamp}-{suffix:04d}-{self._artifact_safe_name(stem)}.txt"
        try:
            path.write_text(content, encoding="utf-8")
            return path
        except Exception:
            return None

    @staticmethod
    def _short_path(path: Optional[Path]) -> str:
        if path is None:
            return ""
        try:
            return str(path.relative_to(Path.cwd()))
        except Exception:
            return str(path)

    @staticmethod
    def _reasoning_display_text(text: str, mode: str) -> str:
        mode = mode if mode in REASONING_MODES else "preview"
        if not text or mode == "off":
            return ""
        if mode == "full":
            return text
        if mode == "summary":
            first_lines = [line.strip() for line in text.splitlines() if line.strip()]
            summary = first_lines[0] if first_lines else text.strip()
            return summary[:180] + ("…" if len(summary) > 180 or len(text) > len(summary) else "")
        return text[:400] + ("…" if len(text) > 400 else "")

    # ── CLI-4: tool-result truncation ───────────────────────────────
    @staticmethod
    def _truncate_observation(text: str, lines: int = 10, cols: int = 200) -> str:
        """Return a head+tail preview of long tool output (CLI-4)."""
        return CLICodeAgentMixin._truncate_observation_info(text, lines, cols)["display"]

    @staticmethod
    def _truncate_observation_info(text: str, lines: int = 10, cols: int = 200) -> dict:
        """Return a head+tail preview of long tool output with metadata."""
        if not text:
            return {"display": text, "truncated": False, "omitted": 0, "line_count": 0}
        raw_lines = text.splitlines()
        if len(raw_lines) <= lines:
            return {
                "display": text,
                "truncated": False,
                "omitted": 0,
                "line_count": len(raw_lines),
            }
        head = raw_lines[: lines - 2]
        tail = raw_lines[-2:]
        head_joined = "\n".join(line[:cols] for line in head)
        tail_joined = "\n".join(line[-cols:] for line in tail)
        omitted = len(raw_lines) - len(head) - len(tail)
        return {
            "display": f"{head_joined}\n  ⋯ {omitted} lines omitted ⋯\n{tail_joined}",
            "truncated": True,
            "omitted": omitted,
            "line_count": len(raw_lines),
        }

    # ── CLI-8: context pressure snapshot ────────────────────────────
    @staticmethod
    def _context_snapshot(agent) -> str:
        """Return a compact ctx-usage string for the step label (CLI-8)."""
        ctx = getattr(getattr(agent, "config", None), "context_window", 0)
        used = getattr(agent, "_last_prompt_tokens", 0)
        if ctx > 0 and used > 0:
            pct = used * 100 / ctx
            return f"[ctx {used:,} / {ctx:,}  {pct:.0f}%]"
        return ""

    # ═══════════════════════════════════════════════════════════════
    # Main render dispatch
    # ═══════════════════════════════════════════════════════════════

    def _render_event(self, event_type: str, payload: dict) -> None:
        self._ensure_cli_runtime_state()
        # ── CLI-2: compact step header ──────────────────────
        if event_type == "step_start":
            step = payload.get("step", 0)
            ctx_info = self._context_snapshot(self) if isinstance(step, int) and step > 0 else ""
            if hasattr(self.ui, "render_step_header"):
                self.ui.render_step_header(step, ctx_info)
            else:
                self.ui.info(f"✦ Step {step}  {ctx_info}")
            if hasattr(self.ui, "status"):
                self.ui.status("Thinking...")
            self._record_cli_event("step", f"Step {step} {ctx_info}".strip(), step=step)
            return

        if event_type in self._IGNORED_EVENTS:
            return

        # ── CLI-1: model reasoning / thinking ───────────────
        if event_type == "model_output":
            rc = str(payload.get("reasoning_content", "") or "")
            if rc:
                path = self._save_cli_artifact(
                    "reasoning",
                    f"step-{payload.get('step', 'model')}",
                    rc,
                )
                display = self._reasoning_display_text(
                    rc,
                    getattr(self, "reasoning_mode", "preview"),
                )
                if display:
                    self.ui.render_log_block("thinking", display)
                    # When the preview was truncated, point to the saved full
                    # reasoning artifact (mirrors the tool-output "full output"
                    # hint) so nothing is silently lost.
                    if display != rc and path:
                        self.ui.render_log_block(
                            "observation",
                            f"  (full reasoning: {self._short_path(path)})",
                        )
                self._record_cli_event(
                    "model_output",
                    f"reasoning {len(rc)} chars",
                    reasoning=rc,
                    reasoning_path=str(path) if path else None,
                    mode=getattr(self, "reasoning_mode", "preview"),
                )
            return

        # ── CLI-5: streaming chunks → real-time ─────────────
        if event_type == "stream_chunk":
            chunk = str(payload.get("chunk", ""))
            # Write directly so the user sees the model typing in real time.
            self.ui.print(chunk, end="", flush=True) if hasattr(self.ui, "print") else print(chunk, end="", flush=True)
            return

        if event_type == "stream_newline":
            return  # the chunk rendering already includes newlines

        if event_type == "compaction_notice":
            if hasattr(self.ui, "status"):
                self.ui.status("Compacting context...")
            self._record_cli_event("status", "Compacting context")
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
            compact = self._summarize_tool_args(tool_name, arguments)
            tool_call_id = payload.get("tool_call_id") or f"__no_id_{len(self._tool_call_state) + 1}"
            self._tool_call_state[tool_call_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "compact": compact,
                "started_at": time.time(),
            }
            if hasattr(self.ui, "status"):
                self.ui.status(f"Running {tool_name}...")
            label = f"▸ {tool_name}"
            if compact:
                label += f": {compact}"
            # Rich mode defers this header into the unified tool card rendered
            # at tool_result (VR-4); plain mode prints it now (transcript contract).
            if not getattr(self.ui, "use_rich", False):
                self.ui.render_log_block("action", label)
            self._record_cli_event(
                "tool_call",
                label,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
            )

            if tool_name == "TodoWrite" and isinstance(arguments, dict):
                has_todos_payload = isinstance(arguments.get("todos"), list)
                if has_todos_payload:
                    tool_call_id = payload.get("tool_call_id")
                    if tool_call_id:
                        if not hasattr(self, "_todo_mutating_call_ids"):
                            self._todo_mutating_call_ids = set()
                        self._todo_mutating_call_ids.add(tool_call_id)
                    else:
                        self._todo_mutating_call_without_id = True
        elif event_type == "tool_result":
            kind = "error" if payload.get("status") == "error" else "observation"
            raw = str(payload.get("result_content", ""))
            info = self._truncate_observation_info(raw)
            display = info["display"]
            tool_name = payload.get("tool_name")
            tool_call_id = payload.get("tool_call_id")
            state = self._tool_call_state.pop(tool_call_id, None) if tool_call_id else None
            if state is None and len(self._tool_call_state) == 1:
                fallback_id, state = next(iter(self._tool_call_state.items()))
                self._tool_call_state.pop(fallback_id, None)

            started_at = state.get("started_at") if state else None
            elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
            display_tool_name = tool_name or (state or {}).get("tool_name") or "Tool"
            status_text = str(payload.get("status", "") or "success").lower()
            is_error = status_text == "error"
            marker = "✗" if is_error else "✓"
            line_count = info["line_count"]
            size_text = f"{line_count} lines, {len(raw)} chars"
            completion = f"{marker} {display_tool_name} {elapsed:.1f}s {size_text}"
            output_path = None
            if info["truncated"]:
                output_path = self._save_cli_artifact("tool_outputs", str(display_tool_name), raw)
                full_hint = (
                    f"full output: {self._short_path(output_path)}"
                    if output_path
                    else "full output available in /log"
                )
                completion += f" (truncated; {full_hint})"
                display = f"{display}\n  ({full_hint}; /trace shows event metadata)"
            reason = str(payload.get("error") or payload.get("message") or "").strip()
            if is_error and reason:
                completion += f" - {reason[:120]}"

            # VR-4: rich mode merges call + result into one bounded card; plain
            # mode keeps the two-line completion + body contract for the tests.
            rich_card = getattr(self.ui, "use_rich", False) and hasattr(
                self.ui, "render_tool_card"
            )
            if rich_card:
                arg_summary = (state or {}).get("compact", "") if state else ""
                card_meta = size_text + ("  (truncated)" if info["truncated"] else "")
                self.ui.render_tool_card(
                    display_tool_name,
                    arg_summary,
                    is_error=is_error,
                    elapsed=elapsed,
                    meta=card_meta,
                    body=display,
                )
            else:
                self.ui.render_log_block("error" if is_error else "action", completion)
            self._record_cli_event(
                "tool_result",
                completion,
                tool_call_id=tool_call_id,
                tool_name=display_tool_name,
                status=status_text,
                elapsed_s=elapsed,
                line_count=line_count,
                char_count=len(raw),
                truncated=info["truncated"],
                full_output_path=str(output_path) if output_path else None,
            )

            if tool_name == "TodoWrite":
                tracked_ids = getattr(self, "_todo_mutating_call_ids", set())
                tracked_wo_id = getattr(self, "_todo_mutating_call_without_id", False)
                is_mutating = (payload.get("tool_call_id") in tracked_ids) or tracked_wo_id
                if is_mutating:
                    status = str(payload.get("status", "")).lower()
                    is_success = status != "error" and not raw.startswith("❌")
                    if is_success:
                        self._todo_changed_this_turn = True
                        self.ui.render_inline_task_progress(self)
                if payload.get("tool_call_id") in tracked_ids:
                    tracked_ids.discard(payload.get("tool_call_id"))
                elif tracked_wo_id:
                    self._todo_mutating_call_without_id = False

            # Plain mode prints the observation body separately (rich already
            # folded it into the card above).
            if not rich_card:
                # For observation, only show head. For errors, keep full.
                self.ui.render_log_block(kind, display)
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
        "--reasoning",
        choices=REASONING_MODES,
        default="preview",
        help=(
            "Control terminal reasoning display: off, summary, preview, or full. "
            "Full reasoning is still saved to CLI artifacts when present."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        help=(
            "Directory for full CLI artifacts such as reasoning and truncated tool outputs. "
            "Relative paths are resolved under the workspace. Default: memory/cli_artifacts."
        ),
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not write full CLI reasoning/tool-output artifacts to disk.",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help=(
            "Maximum ReAct steps per turn before the agent stops (default 100). "
            "Set to 0 for unlimited stepping (advanced; no hard cost ceiling)."
        ),
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
        max_steps=args.max_steps,
        register_default_tools=True,
        enable_task_tool=True,
        ui=ui,
    )
    agent.reasoning_mode = args.reasoning
    agent.save_cli_artifacts = not args.no_artifacts
    agent.cli_artifact_dir = args.artifact_dir

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


def _get_todo_tool(agent):
    """Return the agent's TodoWrite tool if it exposes the session-state API."""
    try:
        registry = getattr(agent, "tool_registry", None)
        if not registry:
            return None
        todo_tool = registry.get_tool("TodoWrite")
        if (
            todo_tool
            and hasattr(todo_tool, "export_state")
            and hasattr(todo_tool, "import_state")
        ):
            return todo_tool
    except Exception:
        return None
    return None


def clear_todo_tasks(agent, ui: Optional[CLIUI] = None) -> None:
    """Reset TodoWrite tasks for starting a fresh conversation (重要-5)."""
    todo_tool = _get_todo_tool(agent)
    if not todo_tool:
        return
    try:
        state = todo_tool.export_state()
        had = bool(state.get("todos")) if isinstance(state, dict) else False
        todo_tool.import_state({"todos": []})
    except Exception:
        return
    if ui and had:
        ui.info("Cleared tasks for fresh conversation.")


def maybe_save_task_snapshot(agent, session_path: Path, ui: Optional[CLIUI] = None) -> None:
    """Persist current TodoWrite tasks beside the session file (重要-5).

    The session JSON already embeds ``todo_state`` via ``save_session``; this
    sidecar keeps the CLI's explicit /save + task-restore flow self-contained.
    """
    todo_tool = _get_todo_tool(agent)
    if not todo_tool:
        return
    try:
        state = todo_tool.export_state()
    except Exception:
        return
    todos = state.get("todos", []) if isinstance(state, dict) else []
    snapshot_path = _task_snapshot_path(session_path)
    payload = {"session_file": str(session_path), "todos": todos}
    try:
        snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        return
    if ui:
        ui.info(f"Task snapshot saved: {snapshot_path}")


def maybe_restore_task_snapshot(agent, session_path: Path, ui: Optional[CLIUI] = None) -> bool:
    """Restore TodoWrite tasks from the sidecar snapshot file (重要-5).

    Returns True if a snapshot file is found and restored.
    """
    todo_tool = _get_todo_tool(agent)
    if not todo_tool:
        return False

    snapshot_path = _task_snapshot_path(session_path)
    if not snapshot_path.exists():
        return False

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        # Accept both the new {"todos": [...]} and legacy {"tasks": [...]} shapes.
        todos = data.get("todos")
        if not isinstance(todos, list):
            legacy = data.get("tasks")
            todos = legacy if isinstance(legacy, list) else []
        todo_tool.import_state({"todos": todos})
        if ui:
            ui.info(f"Restored {len(todos)} task(s) from snapshot.")
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

    # CLI-6: show answer *before* task status (answer is primary content).
    ui.render_assistant(response)
    ui.render_summary(time.time() - start_time, agent=agent)

    # Render task status below the answer (secondary).
    ui.render_task_status(agent)

    # If all tasks are now completed, clear them so they won't show again.
    if ui.all_tasks_completed(agent):
        clear_todo_tasks(agent)
    return response


def print_help(ui: CLIUI) -> None:
    lines = [
        "Commands:",
        "- /help                 Show this help message",
        "- /info                 Show workspace, model, and runtime info",
        "- /tools [--full]       Show registered tools grouped by category (--full for schemas)",
        "- /pwd                  Show the current working directory",
        "- /cd <path>            Change the agent working directory within the workspace",
        "- /history [n]          Show recent conversation turns",
        "- /history --events [n] Show recent structured step/tool/model events",
        "- /trace [n]            Show the structured event timeline",
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
    cfg = agent.config
    llm = agent.llm
    tool_count = len(agent.tool_registry.list_tools()) if agent.tool_registry else 0
    total_tokens = getattr(agent, "_total_tokens", 0)
    max_steps = getattr(agent, "max_steps", "?")
    current_step = getattr(agent, "_current_step", 0)
    ctx_window = getattr(cfg, "context_window", 0)
    ctx_used = getattr(agent, "_last_prompt_tokens", 0)
    # Bash sandbox parameters
    bash_tool = agent.tool_registry.get_tool("Bash") if agent.tool_registry else None
    bash_cpu = getattr(bash_tool, "max_cpu_seconds", "?") if bash_tool else "?"
    bash_net = getattr(bash_tool, "allow_network", False) if bash_tool else False
    artifact_root = agent._cli_artifact_root() if hasattr(agent, "_cli_artifact_root") else None

    lines = [
        f"Workspace:        {workspace}",
        f"Working dir:      {getattr(agent, 'working_dir', workspace)}",
        f"Model:            {getattr(llm, 'model', '[?]')}",
        f"Base URL:         {getattr(llm, 'base_url', '[?]')}",
        f"Temperature:      {getattr(llm, 'temperature', '[?]')}",
        f"Max steps:        {max_steps}  (current: {current_step})",
        f"Tokens used:      {total_tokens:,}",
        f"Context:          {ctx_used:,} / {ctx_window:,}  ({ctx_used*100/max(ctx_window,1):.0f}%)" if ctx_window else "Context:          N/A",
        f"Reasoning view:   {getattr(agent, 'reasoning_mode', 'preview')}",
        f"Artifacts:        {artifact_root if artifact_root else 'off'}",
        f"Trace:            {'on' if getattr(cfg, 'trace_enabled', False) else 'off'}",
        f"Session:          {'on' if getattr(agent, 'session_store', None) else 'off'}",
        f"Tools registered: {tool_count}",
        f"Bash CPU limit:   {bash_cpu}s  |  network: {'on' if bash_net else 'off'}",
    ]
    if ui.use_rich:
        ui.console.print(Panel("\n".join(lines), title="Runtime Info", border_style="cyan", width=ui.console.width))
    else:
        print("\n".join(lines))


def _input_style_tokens() -> dict:
    """prompt_toolkit style tokens for the input zone (VR-1).

    The ``chip`` (prompt glyph) and ``toolbar`` (status bar) both carry a
    ``bg:`` so the input area is visibly backed by a dark surface — the signal
    that "this is where you type".
    """
    return {
        "chip": f"bg:{Palette.BG_PANEL} {Palette.CYAN} bold",
        "toolbar": f"bg:#0f172a {Palette.MUTED}",
        "placeholder": Palette.MUTED,
        "user": f"{Palette.CYAN} bold",
        "arrow": f"{Palette.ACCENT} bold",
    }


def build_prompt_reader(history_file: Path, status_provider=None):
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

        def _bottom_toolbar():
            """Full-width status bar under the input (VR-1) — gives the input
            area a visible bottom border/background so it reads as a field."""
            if not status_provider:
                return None
            try:
                return HTML(f"<toolbar> {status_provider()} </toolbar>")
            except Exception:
                return None

        session = PromptSession(
            history=FileHistory(str(history_file)),
            style=PromptStyle.from_dict(_input_style_tokens()),
            multiline=True,
            mouse_support=False,
            key_bindings=bindings,
            bottom_toolbar=_bottom_toolbar,
            placeholder=HTML(
                "<placeholder>输入需求，Enter 发送 · Esc+Enter 换行 · /help 查看命令</placeholder>"
            ),
            prompt_continuation=lambda width, line_number, wrap_count: " " * width,
        )

        # Track buffer text changes for paste detection.
        session.default_buffer.on_text_changed += _on_text_changed

        def read_prompt() -> str:
            return session.prompt(HTML("<chip> ▌ › </chip> ")).strip()

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

    def _prompt_status() -> str:
        model = getattr(getattr(agent, "llm", None), "model", "?")
        mode = getattr(agent, "reasoning_mode", "preview")
        wd = getattr(agent, "working_dir", str(workspace))
        try:
            wd = str(wd).replace(str(Path.home()), "~", 1)
        except Exception:
            pass
        return f"{model}  ·  reasoning:{mode}  ·  {wd}"

    read_prompt = build_prompt_reader(history_file, status_provider=_prompt_status)

    input_buffer = InputBuffer()

    # -- Command handlers --

    def _cmd_help(raw, lowered):
        print_help(ui)

    def _cmd_info(raw, lowered):
        show_runtime_info(agent, workspace, ui)

    def _cmd_tools(raw, lowered):
        ui.render_tools(agent, full="--full" in raw)

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
        parts = raw.split()
        limit = None
        if len(parts) > 1 and parts[1] == "--events":
            if len(parts) > 2:
                try:
                    limit = int(parts[2])
                except ValueError:
                    ui.warning("Usage: /history --events [n]")
                    return
            events = agent.get_cli_events() if hasattr(agent, "get_cli_events") else []
            ui.render_event_history(events, limit=limit)
            return
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                ui.warning("Usage: /history [n] or /history --events [n]")
                return
        ui.render_history(agent.get_history(), limit=limit)

    def _cmd_trace(raw, lowered):
        parts = raw.split()
        limit = None
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                ui.warning("Usage: /trace [n]")
                return
        events = agent.get_cli_events() if hasattr(agent, "get_cli_events") else []
        ui.render_event_history(events, limit=limit)

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
        "/pwd": _cmd_pwd,
        "/log": _cmd_log,
        "/trace": _cmd_trace,
        "/clear": _cmd_clear,
        "/sessions": _cmd_sessions,
    }

    # Prefix-match commands (lowered.startswith(key))
    _prefix_cmds = [
        ("/tools", _cmd_tools),
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
