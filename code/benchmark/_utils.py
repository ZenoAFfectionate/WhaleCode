"""Shared utility helpers for benchmark runners."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    _RICH_AVAILABLE = True
except Exception:
    Console = None
    Group = None
    Live = None
    Panel = None
    Text = None
    _RICH_AVAILABLE = False


_MINIMAL_CHILD_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP")


def _safe_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text.strip("._") or "task"


def _clip_text(value: Any, limit: int = 240) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _char_width(ch: str) -> int:
    """Return the display width of a single character (1 or 2)."""
    eaw = unicodedata.east_asian_width(ch)
    return 2 if eaw in ("W", "F") else 1


def _display_width(text: str) -> int:
    """Return the total display width of *text* accounting for wide characters."""
    return sum(_char_width(ch) for ch in text)


def _clip_display(text: str, width: int) -> str:
    """Clip *text* so that its display width does not exceed *width*."""
    if _display_width(text) <= width:
        return text
    out: list[str] = []
    used = 0
    for ch in text:
        char_width = _char_width(ch)
        if used + char_width > width - 3:
            break
        out.append(ch)
        used += char_width
    return "".join(out).rstrip() + "..."


def _ljust_display(text: str, width: int, fillchar: str = " ") -> str:
    """Left-justify *text* to *width* display columns using *fillchar*."""
    pad = max(0, width - _display_width(text))
    return text + fillchar * pad


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


def _json_safe_full(value: Any, *, max_depth: int = 16) -> Any:
    """Convert values into JSON-serializable structures without truncating text."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if max_depth <= 0:
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_full(val, max_depth=max_depth - 1) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_full(item, max_depth=max_depth - 1) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe_full(model_dump(), max_depth=max_depth - 1)
        except TypeError:
            return _json_safe_full(model_dump(exclude_none=False), max_depth=max_depth - 1)

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _json_safe_full(dict_method(), max_depth=max_depth - 1)

    return repr(value)


def _read_text_if_exists(path: Path, limit: Optional[int] = 80000) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if limit is None:
        return content
    return _clip_text(content, limit)


def _human_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def build_minimal_child_env() -> Dict[str, str]:
    """Return a small, stable child-process environment for benchmark evaluators."""
    env = {key: os.environ[key] for key in _MINIMAL_CHILD_ENV_KEYS if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def truncate_feedback(
    text: str,
    *,
    max_lines: int,
    max_chars: int,
    marker: str = "[feedback truncated]",
) -> str:
    """Bound feedback size while preserving the leading diagnostic context."""
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [marker]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip() + f"\n{marker}"
    return clipped


def build_benchmark_system_prompt(project_root: Path) -> str:
    """Derive the benchmark system prompt from the canonical system prompt file."""
    prompt_file = project_root / "prompts" / "system_prompt.md"
    text = prompt_file.read_text(encoding="utf-8")

    text = re.sub(r"\n### User Interaction\n.*?(?=\n###|\n##|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"\n### Web\n.*?(?=\n###|\n##|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"- \*\*AskUser\*\*.*\n", "", text)
    text = re.sub(r"- \*\*WebSearch\*\*.*\n", "", text)
    text = re.sub(r"- \*\*WebFetch\*\*.*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    benchmark_runtime_notes = """
---

## Benchmark Runtime Notes

- `TodoWrite` is optional in benchmark runs. Use it only when genuine multi-step planning is worth the step budget.
- When a submission is ready, call `Finish` alone after all other tool work is complete.
"""
    return text.strip() + "\n\n" + benchmark_runtime_notes.strip() + "\n"


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
        self.current_detail = _clip_display(result.get("error") or result.get("exit_status") or "done", 80)
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
        content_w = self._content_width()
        header = Group(
            Text(
                _clip_display(f"Benchmark  {self.benchmark_name}", content_w),
                style="bold white",
                no_wrap=True,
                overflow="ellipsis",
            ),
            Text(
                _clip_display(
                    f"Completed  {self.completed}/{self.total}    Elapsed  {_human_elapsed(time.time() - self.started_at)}",
                    content_w,
                ),
                style="bold bright_blue",
                no_wrap=True,
                overflow="ellipsis",
            ),
            Text(self._counts_line(), style="dim", no_wrap=True, overflow="ellipsis"),
        )
        progress_panel = Panel(
            Group(
                Text(self._progress_line(), style="bold bright_blue", no_wrap=True, overflow="ellipsis"),
                Text(self._timing_line(), style="dim", no_wrap=True, overflow="ellipsis"),
                Text(self._status_line(), style="bold white", no_wrap=True, overflow="ellipsis"),
            ),
            title=f" {self._progress_panel_title()} ",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            width=width,
        )
        return Group(
            Panel(
                header,
                title=f" {self._header_panel_title()} ",
                title_align="left",
                border_style="bright_blue",
                padding=(0, 1),
                width=width,
            ),
            progress_panel,
            Panel(
                self._status_table(),
                title=f" {self._outcomes_panel_title()} ",
                title_align="left",
                border_style="blue",
                padding=(0, 1),
                width=width,
            ),
        )

    def _render_fallback(self, force: bool = False) -> None:
        lines = self._fallback_panels()
        if self._ansi and not self._append_mode:
            try:
                term_cols = os.get_terminal_size().columns
            except OSError:
                term_cols = 120
            if self._fallback_lines:
                sys.stdout.write("\r")
                for idx in range(self._fallback_lines):
                    sys.stdout.write("\x1b[2K")
                    if idx < self._fallback_lines - 1:
                        sys.stdout.write("\x1b[1A")
                sys.stdout.write("\r")
            visible_count = 0
            for line in lines:
                visible_count += max(1, -(-_display_width(line) // term_cols))
            sys.stdout.write("\n".join(lines))
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._fallback_lines = visible_count
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
        fixed_len = _display_width(prefix) + _display_width(middle) + _display_width(suffix)
        remaining = max(0, inner - fixed_len)
        task_budget = min(30, max(0, remaining // 2))
        detail_budget = max(0, remaining - task_budget)
        task_label = _clip_display(self.current_task_id or "-", task_budget) if task_budget > 0 else ""
        detail = _clip_display(self.current_detail or self.current_status, detail_budget) if detail_budget > 0 else ""
        return _clip_display(f"{prefix}{task_label}{middle}{detail}{suffix}", inner)

    def _counts_line(self) -> str:
        if self._is_compact_layout():
            line = f"Pass {self.passed}   Fail {self.failed}   Pend {self.unfinished}   Skip {self.skipped}"
        else:
            line = f"Passed {self.passed}   Failed {self.failed}   Pending {self.unfinished}   Skipped {self.skipped}"
        return _clip_display(line, self._content_width())

    def _progress_line(self) -> str:
        inner = self._content_width()
        total = max(self.total, 1)
        percent = self.completed / total
        prefix = "Progress " if self._is_compact_layout() else "Overall Progress "
        suffix = f" {self.completed}/{self.total} {percent * 100:>3.0f}%"
        bar_width = max(10, inner - _display_width(prefix) - _display_width(suffix) - 2)
        filled = min(bar_width, max(0, int(round(bar_width * percent))))
        bar = ("█" * filled) + ("░" * (bar_width - filled))
        return _clip_display(f"{prefix}[{bar}]{suffix}", inner)

    def _timing_line(self) -> str:
        elapsed_s = max(0.0, time.time() - self.started_at)
        if self.completed <= 0:
            eta = "--:--"
        else:
            remaining = max(0, self.total - self.completed)
            eta = _human_elapsed((elapsed_s / max(self.completed, 1)) * remaining)
        return _clip_display(f"Elapsed {_human_elapsed(elapsed_s)}   ETA {eta}", self._content_width())

    def _result_label(self, result: Dict[str, Any]) -> str:
        exit_status = str(result.get("exit_status") or "").strip().lower()
        if result.get("timeout") or "timeout" in exit_status:
            return "Timeout"
        if result.get("passed") is True or result.get("has_diff") or exit_status in {"completed", "submitted"}:
            return "Passed"
        error_text = str(result.get("error") or "").strip().lower()
        if error_text and "timeout" in error_text:
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
            return Text("No completed items yet.", style="dim", no_wrap=True)
        lines = [Text(self._status_header_line(), style="bold cyan", no_wrap=True, overflow="ellipsis")]
        for label, count, recent in self._status_rows():
            lines.append(Text(self._status_row_line(label, count, recent), style="white", no_wrap=True, overflow="ellipsis"))
        return Group(*lines)

    def _status_rows(self) -> List[tuple[str, int, str]]:
        rows: List[tuple[str, int, str]] = []
        items = sorted(self._status_counts.items(), key=lambda item: (-item[1], item[0]))
        for label, count in items[:6]:
            recent = ", ".join(self._recent_instances.get(label, []))
            rows.append((label, count, _clip_display(recent, self._recent_text_width())))
        return rows

    def _fallback_panels(self) -> List[str]:
        width = self._target_width()

        def panel(title: str, body_lines: List[str]) -> List[str]:
            inner_width = max(20, width - 4)
            top_content = f"┌ {title} "
            top_fill = max(0, width - _display_width(top_content) - 1)
            lines = [top_content + ("─" * top_fill) + "┐"]
            for line in body_lines:
                padded = _ljust_display(_clip_display(line, inner_width), inner_width)
                lines.append(f"│ {padded} │")
            lines.append("└" + ("─" * (width - 2)) + "┘")
            return lines

        rows = self._status_rows()
        status_lines = (
            [self._status_header_line()] + [self._status_row_line(label, count, recent) for label, count, recent in rows]
            if rows
            else ["No completed items yet."]
        )
        header_lines = [
            f"Benchmark  {self.benchmark_name}",
            (
                f"Completed  {self.completed}/{self.total}    "
                f"Pass {self.passed}    Fail {self.failed}    "
                f"{'Pend' if self._is_compact_layout() else 'Pending'} {self.unfinished}    "
                f"Skip {self.skipped}"
            ),
        ]
        progress_lines = [self._progress_line(), self._timing_line(), self._status_line()]
        return (
            panel(self._header_panel_title(), header_lines)
            + panel(self._progress_panel_title(), progress_lines)
            + panel(self._outcomes_panel_title(), status_lines)
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
        gutter = 4 if columns <= 100 else 6
        usable = max(58, columns - gutter)
        if columns <= 90:
            return usable
        return max(72, min(usable, int(columns * 0.82)))

    def _content_width(self) -> int:
        return max(24, self._target_width() - 4)

    def _recent_text_width(self) -> int:
        status_width, count_width = self._status_column_widths()
        return max(12, self._content_width() - status_width - count_width - 4)

    def _is_compact_layout(self) -> bool:
        return self._terminal_width() <= 88

    def _header_panel_title(self) -> str:
        return self.benchmark_name

    def _progress_panel_title(self) -> str:
        return "Progress"

    def _outcomes_panel_title(self) -> str:
        return "Outcomes" if self._is_compact_layout() else "Recent Outcomes"

    def _status_column_widths(self) -> tuple[int, int]:
        return (11, 4) if self._is_compact_layout() else (14, 5)

    def _status_header_line(self) -> str:
        status_width, count_width = self._status_column_widths()
        recent_width = self._recent_text_width()
        return (
            _ljust_display("Status", status_width)
            + " "
            + f"{'Cnt' if self._is_compact_layout() else 'Count':>{count_width}}  "
            + _ljust_display(_clip_display("Recent", recent_width), recent_width)
        )

    def _status_row_line(self, label: str, count: int, recent: str) -> str:
        status_width, count_width = self._status_column_widths()
        recent_width = self._recent_text_width()
        return (
            _ljust_display(_clip_display(label, status_width), status_width)
            + f" {count:>{count_width}}  "
            + _ljust_display(_clip_display(recent, recent_width), recent_width)
        )


def build_progress_update(task_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": str(task_id),
        "event_type": event_type,
        "payload": _json_safe(payload, max_string=1000),
        "payload_full": _json_safe_full(payload),
    }


def describe_progress_update(update: Dict[str, Any]) -> tuple[Optional[int], Optional[str], Optional[str]]:
    payload = update.get("payload") or {}
    step = payload.get("step")
    event_type = str(update.get("event_type") or "")

    if event_type == "agent_start":
        return step, "Running", "Agent init"
    if event_type == "step_start":
        return step, "Running", "Thinking"
    if event_type in {"tool_call", "control_tool", "builtin_tool"}:
        tool_name = payload.get("tool_name") or event_type
        detail = tool_name
        if tool_name == "Bash":
            command = ((payload.get("arguments") or {}).get("command") or "").strip()
            if command:
                detail = f"Bash: {_clip_display(command, 56)}"
        return step, "Running", detail
    if event_type == "tool_result":
        tool_name = payload.get("tool_name") or "tool"
        tool_status = payload.get("status") or "success"
        status = "Running" if tool_status == "success" else "Error"
        detail = tool_name if tool_status == "success" else f"{tool_name}: error"
        return step, status, detail
    if event_type == "final_answer":
        return step, "Completing", "Final answer"
    if event_type == "timeout":
        return step, "Timeout", "Step limit reached"
    if event_type == "stagnation_detected":
        return step, "Stalled", payload.get("reason") or "Stagnation detected"
    if event_type == "llm_error":
        return step, "Error", payload.get("error") or "LLM error"
    if event_type == "agent_error":
        return step, "Error", payload.get("message") or "Agent error"
    if event_type == "background_update":
        return step, "Running", "Background update"
    if event_type == "compaction_notice":
        return step, "Running", "compacting context"
    return step, None, None


def progress_updates_to_events(progress_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": datetime.now().isoformat(),
            "event_type": str(update.get("event_type") or ""),
            "payload": _json_safe_full(
                update.get("payload_full") if update.get("payload_full") is not None else update.get("payload") or {}
            ),
        }
        for update in progress_updates
    ]


def collect_terminal_outputs(workspace: Optional[Path]) -> List[Dict[str, Any]]:
    if workspace is None:
        return []
    terminals_dir = workspace / "memory" / "terminals"
    if not terminals_dir.exists():
        return []
    outputs: List[Dict[str, Any]] = []
    for file_path in sorted(terminals_dir.glob("*")):
        if file_path.is_file():
            outputs.append(
                {
                    "path": str(file_path.relative_to(workspace)),
                    "content": _read_text_if_exists(file_path, limit=None) or "",
                    "truncated": False,
                }
            )
    return outputs


def collect_workspace_artifacts(workspace: Optional[Path], artifact_paths: Optional[List[str]]) -> Dict[str, str]:
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
        content = _read_text_if_exists(target, limit=None)
        if content is not None:
            artifacts[str(rel_path)] = content
    return artifacts


def extract_full_output_paths(value: Any) -> List[str]:
    paths: List[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "full_output_path" and isinstance(child, str) and child.strip():
                    paths.append(child.strip())
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return list(dict.fromkeys(paths))


def load_saved_tool_output(full_output_path: str) -> Optional[str]:
    path = Path(full_output_path)
    if not path.exists() or not path.is_file():
        return None
    raw_text = _read_text_if_exists(path, limit=None)
    if raw_text is None:
        return None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    output = payload.get("output")
    return output if isinstance(output, str) else raw_text


def collect_saved_tool_outputs(history: List[Dict[str, Any]]) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for full_output_path in extract_full_output_paths(history):
        content = load_saved_tool_output(full_output_path)
        if content is not None:
            outputs[full_output_path] = content
    return outputs


def trajectory_dir_for_task(trajectory_root: Path, benchmark_name: str, task_id: str) -> Path:
    task_dir = trajectory_root / benchmark_name / _safe_name(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def collect_agent_snapshot(agent: Optional[Any]) -> Dict[str, Any]:
    read_cache: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    if agent is not None:
        get_history = getattr(agent, "get_history", None)
        if callable(get_history):
            history = [
                _json_safe_full(message.to_dict() if hasattr(message, "to_dict") else message)
                for message in get_history()
            ]
        events = [_json_safe_full(event) for event in list(getattr(agent, "benchmark_events", []))]
        if getattr(agent, "tool_registry", None) is not None:
            read_cache = dict(getattr(agent.tool_registry, "read_metadata_cache", {}))

    return {
        "history": history,
        "events": events,
        "read_cache": _json_safe_full(read_cache),
        "saved_tool_outputs": collect_saved_tool_outputs(history),
    }


def build_trajectory_payload(
    *,
    benchmark_name: str,
    task_id: str,
    task: Dict[str, Any],
    workspace: Optional[Path],
    agent: Optional[Any],
    prompt_texts: Optional[List[str]],
    result: Optional[Dict[str, Any]],
    artifact_paths: Optional[List[str]],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "trajectory_format": "whale-code-benchmark-1.0",
        "benchmark": benchmark_name,
        "task_id": task_id,
        "saved_at": datetime.now().isoformat(),
        "task": _json_safe_full(task),
        "prompts": [str(text) for text in (prompt_texts or []) if text],
        "result": _json_safe_full(result or {}),
        "agent": collect_agent_snapshot(agent),
        "workspace": {
            "root": str(workspace) if workspace else None,
            "artifacts": collect_workspace_artifacts(workspace, artifact_paths),
            "terminals": collect_terminal_outputs(workspace),
        },
        "extra": _json_safe_full(extra or {}),
    }


def build_trajectory_readme(payload: Dict[str, Any]) -> str:
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
        if event_type in {"tool_call", "control_tool", "builtin_tool", "tool_result"}:
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
            lines.extend(["", f"### Prompt {idx}", "```text", _clip_text(prompt_text, 16000), "```"])

    if events:
        lines.extend(["", "## Event Summary"])
        lines.extend(format_event(event) for event in events[-40:])

    if artifacts:
        lines.extend(["", "## Workspace Artifacts"])
        for rel_path, content in artifacts.items():
            lines.extend(["", f"### {rel_path}", "```text", content, "```"])

    if terminals:
        lines.extend(["", "## Terminal Outputs"])
        for terminal in terminals[:10]:
            lines.extend(["", f"### {terminal.get('path')}", "```text", terminal.get("content", ""), "```"])

    return "\n".join(lines).rstrip() + "\n"


def load_completed_ids(resume_file: Path) -> set[str]:
    completed: set[str] = set()
    if not resume_file.exists():
        return completed
    for record in latest_result_records(load_result_records(resume_file)):
        if record.get("passed") is True and record.get("task_id") is not None:
            completed.add(str(record["task_id"]))
    return completed


def load_result_records(results_file: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not results_file.exists():
        return records
    with open(results_file, encoding="utf-8") as handle:
        for line in handle:
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


def latest_result_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest_by_task: Dict[str, Dict[str, Any]] = {}
    anonymous_records: List[Dict[str, Any]] = []
    for record in records:
        task_id = record.get("task_id")
        if task_id is None:
            anonymous_records.append(record)
            continue
        task_key = str(task_id)
        latest_by_task.pop(task_key, None)
        latest_by_task[task_key] = record
    return list(latest_by_task.values()) + anonymous_records


def write_result_records(results_file: Path, records: List[Dict[str, Any]]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=results_file.parent,
        prefix=f".{results_file.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        for record in records:
            tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    tmp_path.replace(results_file)


def upsert_result_record(records: List[Dict[str, Any]], record_index: Dict[str, int], result: Dict[str, Any]) -> None:
    task_id = result.get("task_id")
    if task_id is None:
        records.append(result)
        return
    task_key = str(task_id)
    existing_idx = record_index.get(task_key)
    if existing_idx is None:
        record_index[task_key] = len(records)
        records.append(result)
        return
    records[existing_idx] = result


def summarize_result_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_records = latest_result_records(records)
    passed = sum(1 for record in final_records if record.get("passed") is True)
    failed = sum(1 for record in final_records if record.get("passed") is False)
    unfinished = sum(1 for record in final_records if record.get("passed") is None)
    total_time = round(sum(float(record.get("elapsed_s", 0.0) or 0.0) for record in final_records), 2)
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
