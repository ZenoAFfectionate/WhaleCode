"""Base benchmark runner for Whale Code agent evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _bootstrap_package() -> None:
    """Expose the local ``code/`` directory as the ``hello_agents`` package."""
    import types

    if "hello_agents" in sys.modules:
        return
    code_dir = _PROJECT_ROOT / "code"
    package = types.ModuleType("hello_agents")
    package.__path__ = [str(code_dir)]
    package.__file__ = str(code_dir / "__init__.py")
    sys.modules["hello_agents"] = package


_bootstrap_package()

from hello_agents.agents.code_agent import CodeAgent
from hello_agents.core.config import Config
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.registry import ToolRegistry

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    _RICH_AVAILABLE = True
except Exception:
    Align = None
    Console = None
    Group = None
    Live = None
    Panel = None
    Text = None
    _RICH_AVAILABLE = False


_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "data" / "_results"
_DEFAULT_TRAJECTORY_DIR = _PROJECT_ROOT / "data" / "_trajectory"


def _safe_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text.strip("._") or "task"


def _clip_text(value: Any, limit: int = 240) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _json_safe(value: Any, *, max_depth: int = 4, max_items: int = 40, max_string: int = 12000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip_text(value, max_string)
    if max_depth <= 0:
        return _clip_text(repr(value), max_string)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        data = {
            str(key): _json_safe(val, max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
            for key, val in items
        }
        if len(value) > max_items:
            data["__truncated__"] = f"{len(value) - max_items} additional item(s)"
        return data
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        data = [
            _json_safe(item, max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
            for item in seq[:max_items]
        ]
        if len(seq) > max_items:
            data.append(f"... {len(seq) - max_items} additional item(s)")
        return data
    return _clip_text(repr(value), max_string)


def _read_text_if_exists(path: Path, limit: int = 80000) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    return _clip_text(content, limit)


def _human_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


class BenchmarkProgressManager:
    """Render batch-level progress without streaming full trajectories."""

    def __init__(self, benchmark_name: str, total: int):
        self.benchmark_name = benchmark_name
        self.total = max(int(total), 0)
        self.started_at = time.time()
        self.current_started_at = time.time()
        self.completed = 0
        self.skipped = 0
        self.passed = 0
        self.failed = 0
        self.unfinished = 0
        self.current_index = 0
        self.current_task_id = ""
        self.current_step = 0
        self.current_status = "Idle"
        self.current_detail = ""
        self._last_render = 0.0
        self._fallback_lines = 0
        self._live = None
        self._use_rich = bool(_RICH_AVAILABLE)
        self._console = Console(stderr=True) if self._use_rich else None
        self._ansi = bool(sys.stdout.isatty())
        self._mode = str(os.getenv("WHALE_BENCH_PROGRESS_MODE", "live") or "live").strip().lower()
        self._append_mode = self._mode in {"append", "static", "copyable"}
        self._status_counts: Dict[str, int] = {}
        self._recent_instances: Dict[str, List[str]] = {}

    def start(self) -> None:
        if self._use_rich and not self._append_mode:
            self._live = Live(
                self._renderable(),
                refresh_per_second=6,
                transient=False,
                console=self._console,
            )
            self._live.start()
            self._refresh(force=True)
            return
        self._render_fallback(force=True)

    def begin_task(self, index: int, task_id: str) -> None:
        self.current_index = index
        self.current_task_id = str(task_id)
        self.current_step = 0
        self.current_status = "Running"
        self.current_detail = "starting"
        self.current_started_at = time.time()
        self._refresh(force=True)

    def update(self, *, step: Optional[int] = None, status: Optional[str] = None, detail: Optional[str] = None) -> None:
        if step is not None:
            self.current_step = max(self.current_step, int(step))
        if status:
            self.current_status = status
        if detail is not None:
            self.current_detail = detail
        self._refresh()

    def skip_task(self, index: int, task_id: str) -> None:
        self.current_index = index
        self.current_task_id = str(task_id)
        self.current_step = 0
        self.current_status = "Skip"
        self.current_detail = "already completed"
        self.skipped += 1
        self.completed += 1
        self._refresh(force=True)

    def finish_task(self, result: Dict[str, Any]) -> None:
        self.completed += 1
        passed = result.get("passed")
        if passed is True:
            self.passed += 1
            self.current_status = "Pass"
        elif passed is False:
            self.failed += 1
            self.current_status = "Fail"
        else:
            self.unfinished += 1
            self.current_status = "Pending"
        self.current_detail = _clip_text(result.get("error") or result.get("exit_status") or "done", 80)
        self._record_status(result.get("task_id", self.current_task_id), self._result_label(result))
        self._refresh(force=True)

    def close(self) -> None:
        if self._live is not None:
            self._refresh(force=True)
            self._live.stop()
            self._live = None
        elif self._ansi and self._fallback_lines:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._fallback_lines = 0

    def _refresh(self, force: bool = False) -> None:
        now = time.time()
        min_interval = 2.0 if self._append_mode else 0.12
        if not force and now - self._last_render < min_interval:
            return
        self._last_render = now
        if self._live is not None:
            self._live.update(self._renderable(), refresh=True)
        elif self._use_rich and self._append_mode:
            self._console.print(self._renderable())
        else:
            self._render_fallback(force=force)

    def _renderable(self):
        width = self._target_width()
        header = Group(
            Text(_clip_text(f"Benchmark  {self.benchmark_name}", self._content_width()), style="bold white"),
            Text(
                _clip_text(
                    (
                        f"Completed  {self.completed}/{self.total}    "
                        f"Elapsed  {_human_elapsed(time.time() - self.started_at)}"
                    ),
                    self._content_width(),
                ),
                style="bold bright_blue",
            ),
            Text(self._counts_line(), style="dim"),
        )
        progress_panel = Panel(
            Group(
                Text(self._progress_line(), style="bold bright_blue"),
                Text(self._timing_line(), style="dim"),
                Text(self._status_line(), style="bold white"),
            ),
            title=" Progress ",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            width=width,
        )
        return Group(
            Align.left(
                Panel(
                    header,
                    title=f"  {self.benchmark_name}  ",
                    title_align="left",
                    border_style="bright_blue",
                    padding=(0, 1),
                    width=width,
                )
            ),
            Align.left(progress_panel),
            Align.left(
                Panel(
                    self._status_table(),
                    title=" Recent Outcomes ",
                    title_align="left",
                    border_style="blue",
                    padding=(0, 1),
                    width=width,
                )
            ),
        )

    def _render_fallback(self, force: bool = False) -> None:
        lines = self._fallback_panels()
        if self._ansi and not self._append_mode:
            if self._fallback_lines:
                sys.stdout.write("\r")
                for idx in range(self._fallback_lines):
                    sys.stdout.write("\x1b[2K")
                    if idx < self._fallback_lines - 1:
                        sys.stdout.write("\x1b[1A")
                sys.stdout.write("\r")
            sys.stdout.write("\n".join(lines))
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._fallback_lines = len(lines)
            return
        if force:
            print("\n".join(lines))

    def _status_line(self) -> str:
        inner = self._content_width()
        step_label = f"Step {self.current_step}" if self.current_step else "Init"
        task_elapsed = _human_elapsed(time.time() - self.current_started_at)
        prefix = f"[{self.current_index}/{self.total}] "
        middle = f" | {step_label} | "
        suffix = f" ({task_elapsed})"
        remaining = max(16, inner - len(prefix) - len(middle) - len(suffix))
        task_budget = max(10, min(30, remaining // 2))
        detail_budget = max(10, remaining - task_budget)
        task_label = _clip_text(self.current_task_id or "-", task_budget)
        detail = _clip_text(self.current_detail or self.current_status, detail_budget)
        line = f"{prefix}{task_label}{middle}{detail}{suffix}"
        return _clip_text(line, inner)

    def _counts_line(self) -> str:
        line = (
            f"Passed {self.passed}   Failed {self.failed}   "
            f"Pending {self.unfinished}   Skipped {self.skipped}"
        )
        return _clip_text(line, self._content_width())

    def _progress_line(self) -> str:
        inner = self._content_width()
        total = max(self.total, 1)
        percent = self.completed / total
        prefix = "Overall Progress "
        suffix = f" {self.completed}/{self.total} {percent * 100:>3.0f}%"
        bar_width = max(10, inner - len(prefix) - len(suffix) - 2)
        filled = min(bar_width, max(0, int(round(bar_width * percent))))
        bar = ("#" * filled) + ("-" * (bar_width - filled))
        return _clip_text(f"{prefix}[{bar}]{suffix}", inner)

    def _timing_line(self) -> str:
        elapsed_s = max(0.0, time.time() - self.started_at)
        if self.completed <= 0:
            eta = "--:--"
        else:
            remaining = max(0, self.total - self.completed)
            avg = elapsed_s / max(self.completed, 1)
            eta = _human_elapsed(avg * remaining)
        return _clip_text(
            f"Elapsed {_human_elapsed(elapsed_s)}   ETA {eta}",
            self._content_width(),
        )

    def _result_label(self, result: Dict[str, Any]) -> str:
        exit_status = str(result.get("exit_status") or "").strip().lower()
        if result.get("timeout") or "timeout" in exit_status:
            return "Timeout"

        if result.get("passed") is True:
            return "Passed"

        if result.get("has_diff"):
            return "Passed"

        if exit_status in {"completed", "submitted"}:
            return "Passed"

        error_text = str(result.get("error") or "").strip().lower()
        if error_text:
            if "timeout" in error_text:
                return "Timeout"
        return "Error"

    def _record_status(self, task_id: Any, status: str) -> None:
        label = str(status or "Unknown")
        tid = str(task_id or self.current_task_id or "-")
        self._status_counts[label] = self._status_counts.get(label, 0) + 1
        recent = self._recent_instances.setdefault(label, [])
        if tid in recent:
            recent.remove(tid)
        recent.insert(0, tid)
        del recent[3:]

    def _status_table(self):
        if not self._status_counts:
            return Text("No completed items yet.", style="dim")

        lines = [Text(self._status_header_line(), style="bold cyan")]
        for label, count, recent in self._status_rows():
            lines.append(Text(self._status_row_line(label, count, recent), style="white"))
        return Group(*lines)

    def _status_rows(self) -> List[tuple[str, int, str]]:
        rows: List[tuple[str, int, str]] = []
        items = sorted(self._status_counts.items(), key=lambda item: (-item[1], item[0]))
        for label, count in items[:6]:
            recent = ", ".join(self._recent_instances.get(label, []))
            rows.append((label, count, _clip_text(recent, self._recent_text_width())))
        return rows

    def _fallback_panels(self) -> List[str]:
        try:
            columns = os.get_terminal_size().columns
        except OSError:
            columns = 120
        width = self._target_width(columns)
        left_pad = ""

        def panel(title: str, body_lines: List[str]) -> List[str]:
            inner_width = max(20, width - 4)
            title_text = f" {title} "
            title_segment = f"┌{title_text}".ljust(width - 1, "─") + "┐"
            lines = [left_pad + title_segment]
            for line in body_lines:
                clipped = _clip_text(line, inner_width)
                lines.append(left_pad + f"│ {clipped.ljust(inner_width)} │")
            lines.append(left_pad + "└" + ("─" * (width - 2)) + "┘")
            return lines

        rows = self._status_rows()
        if rows:
            status_lines = [self._status_header_line()] + [
                self._status_row_line(label, count, recent)
                for label, count, recent in rows
            ]
        else:
            status_lines = ["No completed items yet."]

        header_lines = [
            f"Benchmark  {self.benchmark_name}",
            (
                f"Completed  {self.completed}/{self.total}    "
                f"Pass {self.passed}    Fail {self.failed}    "
                f"Pending {self.unfinished}    Skip {self.skipped}"
            ),
        ]
        progress_lines = [
            self._progress_line(),
            self._timing_line(),
            self._status_line(),
        ]
        return (
            panel(self.benchmark_name, header_lines)
            + panel("Progress", progress_lines)
            + panel("Recent Outcomes", status_lines)
        )

    def _terminal_width(self) -> int:
        try:
            if self._console is not None:
                return int(self._console.size.width)
            return os.get_terminal_size().columns
        except OSError:
            return 120

    def _target_width(self, terminal_width: Optional[int] = None) -> int:
        columns = max(60, int(terminal_width or self._terminal_width()))
        if columns <= 90:
            return max(58, columns - 2)
        return max(72, min(columns - 2, int(columns * 0.8)))

    def _content_width(self) -> int:
        return max(24, self._target_width() - 4)

    def _recent_text_width(self) -> int:
        return max(16, self._content_width() - 26)

    def _status_header_line(self) -> str:
        recent_width = self._recent_text_width()
        return (
            f"{'Status':<14} {'Count':>5}  "
            f"{_clip_text('Recent', recent_width):<{recent_width}}"
        )

    def _status_row_line(self, label: str, count: int, recent: str) -> str:
        recent_width = self._recent_text_width()
        return (
            f"{_clip_text(label, 14):<14} {count:>5}  "
            f"{_clip_text(recent, recent_width):<{recent_width}}"
        )


class BenchmarkCodeAgent(CodeAgent):
    """Benchmark-specific agent that records events and suppresses console spam."""

    def __init__(
        self,
        *args,
        task_id: str,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        self.task_id = task_id
        self._event_sink = event_sink
        self.benchmark_events: List[Dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _console(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        return

    def _render_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        safe_payload = _json_safe(payload, max_string=2000)
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "payload": safe_payload,
        }
        self.benchmark_events.append(event_record)
        if self._event_sink is not None:
            try:
                self._event_sink(event_type, safe_payload)
            except Exception:
                pass


def _evaluate_task_in_subprocess(
    runner: "BenchmarkRunner",
    task: Dict[str, Any],
    result_queue: Any,
    progress_queue: Any,
    task_id: str,
) -> None:
    """Execute ``runner._evaluate_task`` and send a serializable payload back."""
    try:
        runner._bind_progress_queue(task_id, progress_queue)
        result_queue.put({"ok": True, "result": runner._evaluate_task(task)})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        runner._bind_progress_queue(None, None)

def _remove_markdown_section(text: str, heading: str) -> str:
    """Remove a markdown section (heading + body) from *text*.

    Matches ``## <heading>`` through to the next same-level heading or EOF.
    """
    pattern = rf"(^|\n)(##\s+\d+\.\s+)?{re.escape(heading)}.*?(?=\n## |\Z)"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def _build_benchmark_system_prompt() -> str:
    """Derive benchmark system prompt from the canonical *system_prompt.md*.

    Removes:
    - AskUser tool definition (### User Interaction block)
    - Web tool definitions (### Web block)
    - Any mention of AskUser / WebSearch / WebFetch in surrounding text
    """
    prompt_file = _PROJECT_ROOT / "prompts" / "system_prompt.md"
    text = prompt_file.read_text(encoding="utf-8")

    # Remove "### User Interaction" section (heading + body until next ###/## or EOF)
    text = re.sub(
        r"\n### User Interaction\n.*?(?=\n###|\n##|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove "### Web" section
    text = re.sub(
        r"\n### Web\n.*?(?=\n###|\n##|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove stray references to AskUser / WebSearch / WebFetch
    text = re.sub(r"- \*\*AskUser\*\*.*\n", "", text)
    text = re.sub(r"- \*\*WebSearch\*\*.*\n", "", text)
    text = re.sub(r"- \*\*WebFetch\*\*.*\n", "", text)

    # Clean up multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


BENCHMARK_BASE_SYSTEM_PROMPT: str = _build_benchmark_system_prompt()


class BenchmarkRunner(ABC):
    """Base class for all benchmark runners.

    Subclasses must implement :meth:`_load_tasks` and :meth:`_evaluate_task`.
    """

    benchmark_name: str = "base"

    @staticmethod
    def add_shared_run_args(
        parser,
        *,
        default_temperature: float = 1.0,
        default_max_steps: int = 64,
        default_timeout: int = 60,
        timeout_help: Optional[str] = None,
        include_task_timeout: bool = False,
        default_task_timeout: int = 1200,
    ) -> None:
        parser.add_argument("--output-dir", default=str(_DEFAULT_RESULTS_DIR))
        parser.add_argument("--temperature", type=float, default=default_temperature)
        parser.add_argument("--max-steps", type=int, default=default_max_steps)
        if timeout_help is None:
            parser.add_argument("--timeout", type=int, default=default_timeout)
        else:
            parser.add_argument("--timeout", type=int, default=default_timeout, help=timeout_help)
        if include_task_timeout:
            parser.add_argument("--task-timeout", type=int, default=default_task_timeout)
        parser.add_argument("--trajectory-dir", default=str(_DEFAULT_TRAJECTORY_DIR))
        parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
        parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
        parser.add_argument("--resume", default=None, help="Resume from a previous .jsonl results file")
        parser.add_argument("--dry-run", action="store_true")

    def __init__(
        self,
        data_path: str,
        output_dir: str = str(_DEFAULT_RESULTS_DIR),
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_steps: int = 30,
        timeout: int = 30,
        task_timeout: int = 1200,
        trajectory_dir: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else _DEFAULT_TRAJECTORY_DIR
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_steps = max_steps
        self.timeout = timeout  # seconds for sandboxed code execution
        self.task_timeout = task_timeout  # seconds for one end-to-end benchmark task
        self._progress_manager: Optional[BenchmarkProgressManager] = None
        self._progress_queue = None
        self._current_task_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> Optional[str]:
        """Return a custom system prompt for this benchmark, or None for default."""
        return None

    def _load_jsonl_tasks(
        self,
        *,
        task_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                if task_transform is not None:
                    task = task_transform(task)
                tasks.append(task)
        return tasks

    def _make_workspace(self, prefix: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=prefix))

    def _build_result(
        self,
        task_id: str,
        *,
        passed: Optional[bool],
        error: Optional[str],
        agent_response: str = "",
        start_time: Optional[float] = None,
        elapsed_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if elapsed_s is None:
            elapsed_s = 0.0 if start_time is None else round(time.time() - start_time, 2)
        result: Dict[str, Any] = {
            "task_id": task_id,
            "passed": passed,
            "error": error,
            "agent_response": (agent_response or "")[:500],
            "elapsed_s": round(float(elapsed_s), 2),
        }
        if extra:
            result.update(extra)
        return result

    def _run_agent_prompt(
        self,
        *,
        agent: CodeAgent,
        task_id: str,
        prompt_text: str,
        start_time: float,
        run_kwargs: Optional[Dict[str, Any]] = None,
        error_extra: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        try:
            return agent.run(prompt_text, **(run_kwargs or {})), None
        except Exception as exc:
            return "", self._build_result(
                task_id,
                passed=False,
                error=f"Agent error: {exc}",
                start_time=start_time,
                extra=error_extra,
            )

    def _missing_output_result(
        self,
        task_id: str,
        *,
        path_label: str,
        start_time: Optional[float] = None,
        elapsed_s: Optional[float] = None,
        agent_response: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._build_result(
            task_id,
            passed=False,
            error=f"{path_label} not found after agent run",
            agent_response=agent_response,
            start_time=start_time,
            elapsed_s=elapsed_s,
            extra=extra,
        )

    def _create_agent(self, workspace: Path) -> CodeAgent:
        """Create a fresh CodeAgent with coding tools only (no web tools)."""
        from hello_agents.tools.builtin.bash import BashTool
        from hello_agents.tools.builtin.file_tools import (
            DeleteTool,
            EditTool,
            ListFilesTool,
            ReadTool,
            WriteTool,
        )
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool

        llm_kwargs: Dict[str, Any] = {"temperature": self.temperature}
        if self.model:
            llm_kwargs["model"] = self.model
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        if self.api_key:
            llm_kwargs["api_key"] = self.api_key

        llm = HelloAgentsLLM(**llm_kwargs)
        registry = ToolRegistry(verbose=False)

        config = Config.from_env()
        config.trace_enabled = False

        task_id = self._current_task_id or workspace.name
        agent = BenchmarkCodeAgent(
            name="bench-agent",
            llm=llm,
            tool_registry=registry,
            project_root=str(workspace),
            working_dir=str(workspace),
            config=config,
            max_steps=self.max_steps,
            register_default_tools=False,  # We register manually below
            enable_task_tool=False,
            interactive=False,
            system_prompt=self._get_system_prompt() or BENCHMARK_BASE_SYSTEM_PROMPT,
            task_id=task_id,
            event_sink=lambda event_type, payload: self._emit_progress_event(task_id, event_type, payload),
        )

        ws = str(workspace)
        registry.register_tool(ListFilesTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(GlobTool(project_root=ws, working_dir=ws))
        registry.register_tool(GrepTool(project_root=ws, working_dir=ws))
        registry.register_tool(ReadTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(WriteTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(DeleteTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(EditTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(BashTool(project_root=ws, working_dir=ws))
        # Persist session todo state outside the repo to avoid diff pollution.
        _bench_task_dir = Path(tempfile.gettempdir()) / "whale_bench_tasks" / uuid.uuid4().hex[:8]
        _bench_task_dir.mkdir(parents=True, exist_ok=True)
        registry.register_tool(
            TodoWriteTool(
                project_root=ws,
                persistence_dir=str(_bench_task_dir),
                session_id=agent.session_id,
            )
        )

        return agent

    def _bind_progress_queue(self, task_id: Optional[str], progress_queue: Any) -> None:
        self._current_task_id = task_id
        self._progress_queue = progress_queue

    def _emit_progress_event(self, task_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        update = {
            "task_id": str(task_id),
            "event_type": event_type,
            "payload": _json_safe(payload, max_string=1000),
        }
        if self._progress_queue is not None:
            try:
                self._progress_queue.put_nowait(update)
            except Exception:
                pass
            return
        self._handle_progress_update(update)

    def _handle_progress_update(self, update: Dict[str, Any]) -> None:
        if self._progress_manager is None:
            return
        payload = update.get("payload") or {}
        step = payload.get("step")
        event_type = str(update.get("event_type") or "")

        detail = None
        status = None
        if event_type == "agent_start":
            status = "Running"
            detail = "agent started"
        elif event_type == "step_start":
            status = "Running"
            detail = "thinking"
        elif event_type in {"tool_call", "builtin_tool"}:
            tool_name = payload.get("tool_name") or event_type
            status = "Running"
            detail = tool_name
            if tool_name == "Bash":
                command = ((payload.get("arguments") or {}).get("command") or "").strip()
                if command:
                    detail = f"Bash: {_clip_text(command, 56)}"
        elif event_type == "tool_result":
            tool_name = payload.get("tool_name") or "tool"
            tool_status = payload.get("status") or "success"
            status = "Running" if tool_status == "success" else "Error"
            detail = tool_name if tool_status == "success" else f"{tool_name}: error"
        elif event_type == "final_answer":
            status = "Finishing"
            detail = "final answer"
        elif event_type == "timeout":
            status = "Timeout"
            detail = "step limit reached"
        elif event_type == "stagnation_detected":
            status = "Stalled"
            detail = payload.get("reason") or "stagnation detected"
        elif event_type == "llm_error":
            status = "Error"
            detail = payload.get("error") or "LLM error"
        elif event_type == "agent_error":
            status = "Error"
            detail = payload.get("message") or "agent error"
        elif event_type == "background_update":
            status = "Running"
            detail = "background update"
        elif event_type == "compaction_notice":
            status = "Running"
            detail = "compacting context"

        if detail is not None or status is not None or step is not None:
            self._progress_manager.update(step=step, status=status, detail=detail)

    def _drain_progress_queue(self, progress_queue: Any) -> None:
        while progress_queue is not None:
            try:
                update = progress_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            self._handle_progress_update(update)

    # ------------------------------------------------------------------
    # Sandboxed code execution
    # ------------------------------------------------------------------

    def _run_code_in_sandbox(
        self,
        code: str,
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Execute *code* in a subprocess and return ``(passed, output)``."""
        timeout = timeout or self.timeout
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
            )
            passed = result.returncode == 0
            output = (result.stdout + result.stderr).strip()
            return passed, output
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as exc:
            return False, f"Execution error: {exc}"

    def _run_script_in_sandbox(
        self,
        script_path: Path,
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Execute a Python script file in a subprocess."""
        timeout = timeout or self.timeout
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
            )
            passed = result.returncode == 0
            output = (result.stdout + result.stderr).strip()
            return passed, output
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as exc:
            return False, f"Execution error: {exc}"

    # ------------------------------------------------------------------
    # Trajectory persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_terminal_outputs(workspace: Optional[Path]) -> List[Dict[str, Any]]:
        if workspace is None:
            return []
        terminals_dir = workspace / "memory" / "terminals"
        if not terminals_dir.exists():
            return []
        outputs: List[Dict[str, Any]] = []
        for file_path in sorted(terminals_dir.glob("*")):
            if not file_path.is_file():
                continue
            content = _read_text_if_exists(file_path, limit=50000) or ""
            outputs.append(
                {
                    "path": str(file_path.relative_to(workspace)),
                    "content": content,
                    "truncated": len(content) >= 49997,
                }
            )
        return outputs

    @staticmethod
    def _collect_workspace_artifacts(workspace: Optional[Path], artifact_paths: Optional[List[str]]) -> Dict[str, str]:
        if workspace is None or not artifact_paths:
            return {}
        workspace_root = workspace.resolve()
        artifacts: Dict[str, str] = {}
        for rel_path in artifact_paths:
            try:
                target = (workspace_root / rel_path).resolve()
                target.relative_to(workspace_root)
            except Exception:
                continue
            content = _read_text_if_exists(target)
            if content is None:
                continue
            artifacts[str(rel_path)] = content
        return artifacts

    def _trajectory_dir_for_task(self, task_id: str) -> Path:
        task_dir = self.trajectory_dir / self.benchmark_name / _safe_name(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _build_trajectory_readme(self, payload: Dict[str, Any]) -> str:
        result = payload.get("result") or {}
        events = payload.get("agent", {}).get("events") or []
        prompts = payload.get("prompts") or []
        artifacts = payload.get("workspace", {}).get("artifacts") or {}
        terminals = payload.get("workspace", {}).get("terminals") or []

        def format_event(event: Dict[str, Any]) -> str:
            event_type = event.get("event_type") or "event"
            event_payload = event.get("payload") or {}
            step = event_payload.get("step")
            prefix = f"step {step} " if step is not None else ""
            if event_type in {"tool_call", "builtin_tool"}:
                detail = event_payload.get("tool_name") or event_type
            elif event_type == "tool_result":
                detail = event_payload.get("tool_name") or event_type
            elif event_type == "llm_error":
                detail = event_payload.get("error") or event_type
            elif event_type == "agent_error":
                detail = event_payload.get("message") or event_type
            else:
                detail = event_type
            return f"- {prefix}{_clip_text(detail, 120)}"

        lines = [
            f"# {payload.get('benchmark')} / {payload.get('task_id')}",
            "",
            f"- Saved at: {payload.get('saved_at')}",
            f"- Passed: {result.get('passed')}",
            f"- Elapsed: {result.get('elapsed_s')}",
            "",
            "## Result",
            "```json",
            json.dumps(result, indent=2, ensure_ascii=False),
            "```",
        ]

        if prompts:
            lines.extend(["", "## Prompts"])
            for idx, prompt_text in enumerate(prompts, start=1):
                lines.extend(
                    [
                        "",
                        f"### Prompt {idx}",
                        "```text",
                        _clip_text(prompt_text, 16000),
                        "```",
                    ]
                )

        if events:
            lines.extend(["", "## Event Summary"])
            lines.extend(format_event(event) for event in events[-40:])

        if artifacts:
            lines.extend(["", "## Workspace Artifacts"])
            for rel_path, content in artifacts.items():
                lines.extend(
                    [
                        "",
                        f"### {rel_path}",
                        "```text",
                        content,
                        "```",
                    ]
                )

        if terminals:
            lines.extend(["", "## Terminal Outputs"])
            for terminal in terminals[:10]:
                lines.extend(
                    [
                        "",
                        f"### {terminal.get('path')}",
                        "```text",
                        terminal.get("content", ""),
                        "```",
                    ]
                )

        return "\n".join(lines).rstrip() + "\n"

    def _save_task_trajectory(
        self,
        *,
        task: Dict[str, Any],
        workspace: Optional[Path],
        agent: Optional[CodeAgent],
        prompt_texts: Optional[List[str]] = None,
        result: Optional[Dict[str, Any]] = None,
        artifact_paths: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = str(task.get("task_id") or uuid.uuid4().hex)
        task_dir = self._trajectory_dir_for_task(task_id)
        trajectory_path = task_dir / "trajectory.json"
        read_cache = {}
        history: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        if agent is not None:
            history = [msg.to_dict() if hasattr(msg, "to_dict") else _json_safe(msg) for msg in agent.get_history()]
            events = list(getattr(agent, "benchmark_events", []))
            if getattr(agent, "tool_registry", None) is not None:
                read_cache = dict(getattr(agent.tool_registry, "read_metadata_cache", {}))

        payload = {
            "trajectory_format": "whale-code-benchmark-1.0",
            "benchmark": self.benchmark_name,
            "task_id": task_id,
            "saved_at": datetime.now().isoformat(),
            "task": _json_safe(task, max_string=16000),
            "prompts": [_clip_text(text, 16000) for text in (prompt_texts or []) if text],
            "result": _json_safe(result or {}),
            "agent": {
                "history": history,
                "events": events,
                "read_cache": _json_safe(read_cache, max_string=4000),
            },
            "workspace": {
                "root": str(workspace) if workspace else None,
                "artifacts": self._collect_workspace_artifacts(workspace, artifact_paths),
                "terminals": self._collect_terminal_outputs(workspace),
            },
            "extra": _json_safe(extra or {}, max_string=16000),
        }
        trajectory_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (task_dir / "README.md").write_text(self._build_trajectory_readme(payload), encoding="utf-8")
        return str(trajectory_path)

    # ------------------------------------------------------------------
    # Data loading & evaluation (subclass hooks)
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load and return the task list from ``self.data_path``."""

    @abstractmethod
    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on *task* and return a result dict.

        The returned dict must include at least:
        ``task_id``, ``passed`` (bool), ``error`` (str or None),
        ``elapsed_s`` (float).
        """

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _evaluate_task_with_timeout(self, task: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Run ``_evaluate_task`` with an end-to-end wall-clock timeout."""
        if self.task_timeout <= 0:
            return self._evaluate_task(task)

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        result_queue = ctx.Queue(maxsize=1)
        progress_queue = ctx.Queue(maxsize=512)
        process = ctx.Process(
            target=_evaluate_task_in_subprocess,
            args=(self, task, result_queue, progress_queue, task_id),
            name=f"{self.benchmark_name}-{task_id}",
            daemon=True,
        )
        process.start()

        deadline = time.time() + self.task_timeout
        timed_out = False
        while process.is_alive():
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break
            try:
                update = progress_queue.get(timeout=min(0.2, max(0.01, remaining)))
            except queue.Empty:
                update = None
            except Exception:
                update = None
            if update is not None:
                self._handle_progress_update(update)

        self._drain_progress_queue(progress_queue)

        if timed_out and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5)
            return {
                "task_id": task_id,
                "passed": False,
                "error": f"Timeout: problem solving exceeded {self.task_timeout}s",
                "elapsed_s": float(self.task_timeout),
                "timeout": True,
            }

        process.join(timeout=1)
        try:
            payload = result_queue.get_nowait()
        except Exception:
            payload = None

        if not payload:
            return {
                "task_id": task_id,
                "passed": False,
                "error": "Runner process exited without returning a result",
                "elapsed_s": 0.0,
            }

        if not payload.get("ok"):
            raise RuntimeError(payload.get("traceback") or payload.get("error") or "Unknown runner error")

        return payload["result"]

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    @staticmethod
    def _load_completed_ids(resume_file: Path) -> set:
        """Return task_ids whose latest recorded result is a successful completion."""
        completed: set = set()
        if not resume_file.exists():
            return completed

        records = BenchmarkRunner._load_result_records(resume_file)
        for record in BenchmarkRunner._latest_result_records(records):
            if record.get("passed") is not True:
                continue
            tid = record.get("task_id")
            if tid is not None:
                completed.add(str(tid))
        return completed

    @staticmethod
    def _load_result_records(results_file: Path) -> List[Dict[str, Any]]:
        """Read a results JSONL file and return all valid parsed records."""
        records: List[Dict[str, Any]] = []
        if not results_file.exists():
            return records

        with open(results_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    @staticmethod
    def _latest_result_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate records by task_id, keeping the latest entry for each task."""
        latest_by_task: Dict[str, Dict[str, Any]] = {}
        anonymous_records: List[Dict[str, Any]] = []

        for record in records:
            task_id = record.get("task_id")
            if task_id is None:
                anonymous_records.append(record)
            else:
                latest_by_task[str(task_id)] = record

        return list(latest_by_task.values()) + anonymous_records

    @staticmethod
    def _summarize_result_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build aggregate stats from result records, deduplicated by latest task_id."""
        final_records = BenchmarkRunner._latest_result_records(records)

        passed = sum(1 for r in final_records if r.get("passed") is True)
        failed = sum(1 for r in final_records if r.get("passed") is False)
        unfinished = sum(1 for r in final_records if r.get("passed") is None)
        total_time = round(sum(float(r.get("elapsed_s", 0.0) or 0.0) for r in final_records), 2)
        total = len(final_records)
        pass_rate = round((passed / total * 100), 2) if total > 0 else 0.0

        return {
            "records_in_file": len(records),
            "tasks": total,
            "passed": passed,
            "failed": failed,
            "unfinished": unfinished,
            "pass_rate": pass_rate,
            "total_time_s": total_time,
            "avg_time_s": round(total_time / total, 2) if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark and return a summary dict.

        Args:
            limit: Only run the first *limit* tasks.
            task_ids: Only run tasks whose ``task_id`` is in this list.
            dry_run: Print tasks without executing.
            resume: Path to a previous results ``.jsonl`` file.  Already-
                completed task IDs are skipped and new results are **appended**
                to the same file.
        """
        tasks = self._load_tasks()
        if task_ids:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.get("task_id") in id_set]
        if limit and limit > 0:
            tasks = tasks[:limit]

        # --- Resume handling ---
        completed_ids: set = set()
        resume_path: Optional[Path] = Path(resume) if resume else None
        if resume:
            if not resume_path.exists():
                print(f"  ▶ Resume target does not exist yet: {resume}")
                print(f"    A new results file will be created at this path.\n")
            else:
                completed_ids = self._load_completed_ids(resume_path)
                print(f"  ▶ Resuming from: {resume}")
                print(f"    Already completed: {len(completed_ids)} tasks")

        # Decide which file to write results to
        if resume_path is not None:
            results_file = resume_path
            results_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"{self.benchmark_name}_{timestamp}.jsonl"

        print(f"\n{'=' * 60}")
        print(f"  Benchmark: {self.benchmark_name}")
        print(f"  Tasks: {len(tasks)}")
        model_label = self.model or os.getenv("LLM_MODEL_ID") or "(from env)"
        print(f"  Model: {model_label}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Task timeout: {self.task_timeout}s")
        if completed_ids:
            remaining = sum(1 for t in tasks if t.get("task_id", "") not in completed_ids)
            print(f"  Resume: {len(completed_ids)} done, {remaining} remaining")
        print(f"{'=' * 60}\n")

        if dry_run:
            for t in tasks:
                tid = t.get("task_id")
                tag = " [SKIP]" if tid in completed_ids else ""
                print(f"  [dry-run] {tid}{tag}")
            return {"benchmark": self.benchmark_name, "total": len(tasks), "dry_run": True}

        results: List[Dict[str, Any]] = []
        passed_count = 0
        total_time = 0.0
        skipped = 0
        progress = BenchmarkProgressManager(self.benchmark_name, len(tasks))
        self._progress_manager = progress
        progress.start()

        try:
            for i, task in enumerate(tasks):
                task_id = str(task.get("task_id", f"task_{i}"))
                self._current_task_id = task_id

                if task_id in completed_ids:
                    skipped += 1
                    progress.skip_task(i + 1, task_id)
                    continue

                progress.begin_task(i + 1, task_id)

                try:
                    result = self._evaluate_task_with_timeout(task, task_id)
                except Exception as exc:
                    result = {
                        "task_id": task_id,
                        "passed": False,
                        "error": f"Runner exception: {exc}",
                        "elapsed_s": 0.0,
                    }

                self._drain_progress_queue(self._progress_queue)
                results.append(result)
                if result.get("passed") is True:
                    passed_count += 1
                total_time += result.get("elapsed_s", 0)
                progress.finish_task(result)

                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        finally:
            progress.close()
            self._progress_manager = None
            self._progress_queue = None
            self._current_task_id = None

        # Summary
        evaluated = len(results)
        new_pass_rate = (passed_count / evaluated * 100) if evaluated > 0 else 0
        combined_records = self._load_result_records(results_file)
        combined = self._summarize_result_records(combined_records)
        summary = {
            "benchmark": self.benchmark_name,
            "model": self.model or "(from env)",
            "total": len(tasks),
            "evaluated": combined["tasks"],
            "new_evaluated": evaluated,
            "skipped": skipped,
            "passed": combined["passed"],
            "failed": combined["failed"],
            "unfinished": combined["unfinished"],
            "pass_rate": combined["pass_rate"],
            "total_time_s": combined["total_time_s"],
            "avg_time_s": combined["avg_time_s"],
            "records_in_file": combined["records_in_file"],
            "new_passed": passed_count,
            "new_failed": sum(1 for r in results if r.get("passed") is False),
            "new_unfinished": sum(1 for r in results if r.get("passed") is None),
            "new_pass_rate": round(new_pass_rate, 2),
            "new_total_time_s": round(total_time, 2),
            "new_avg_time_s": round(total_time / evaluated, 2) if evaluated > 0 else 0,
            "timestamp": timestamp,
            "results_file": str(results_file),
            "trajectory_dir": str(self.trajectory_dir),
            "resumed_from": resume if resume else None,
        }

        summary_file = self.output_dir / f"{self.benchmark_name}_{timestamp}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        if skipped:
            print(f"  Resumed: {skipped} previously completed")
        print(
            f"  Combined results: {combined['passed']}/{combined['tasks']} passed "
            f"({combined['pass_rate']:.1f}%)"
        )
        if combined["unfinished"]:
            print(f"  Combined unfinished: {combined['unfinished']}")
        print(
            f"  New results: {passed_count}/{evaluated} passed "
            f"({new_pass_rate:.1f}%)"
        )
        print(
            f"  Combined time: {combined['total_time_s']:.1f}s total, "
            f"{combined['avg_time_s']:.1f}s avg"
        )
        print(
            f"  New time: {total_time:.1f}s total, "
            f"{summary['new_avg_time_s']:.1f}s avg"
        )
        print(f"  Output: {results_file}")
        print(f"  Trajectory: {self.trajectory_dir}")
        print(f"  Summary: {summary_file}")
        print(f"{'=' * 60}\n")

        return summary
