# Benchmark Progress Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix benchmark CLI progress rendering so panel borders and progress lines remain aligned in narrow terminals without changing the current UI structure.

**Architecture:** Keep the existing `BenchmarkProgressManager` and its `rich`/fallback renderers, but tighten width budgeting in one place. Add regression tests first to lock in width invariants, then make a minimal implementation change so terminal width, panel width, and clipped line content all use the same conservative assumptions.

**Tech Stack:** Python 3.12, pytest, rich, conda (`agent` environment)

---

## File structure

- Modify: `tests/test_benchmark_base.py` — add regression tests for width budgeting and fallback panel line lengths.
- Modify: `code/benchmark/base.py` — make width helpers more conservative and ensure fallback panels use the same width assumptions as body rows.
- Reference only: `docs/superpowers/specs/2026-04-03-benchmark-progress-alignment-design.md` — approved design and scope.

### Task 1: Add regression tests for width invariants

**Files:**
- Modify: `tests/test_benchmark_base.py`
- Reference: `code/benchmark/base.py:194-566`

- [ ] **Step 1: Write the failing tests**

Add these imports near the top of `tests/test_benchmark_base.py` after the existing benchmark imports:

```python
from benchmark.base import BenchmarkProgressManager
from benchmark.base import _display_width
```

Append these tests at the end of `tests/test_benchmark_base.py`:

```python
def test_progress_manager_uses_safe_panel_width_in_narrow_terminals(monkeypatch):
    manager = BenchmarkProgressManager("mini", 1055)
    monkeypatch.setattr(manager, "_terminal_width", lambda: 100)

    assert manager._target_width() == 94
    assert manager._content_width() == 90


def test_progress_manager_fallback_lines_fit_within_terminal(monkeypatch):
    manager = BenchmarkProgressManager("mini", 1055)
    monkeypatch.setattr(manager, "_terminal_width", lambda: 100)
    manager.completed = 759
    manager.current_index = 759
    manager.current_task_id = "LCB6/abc380_d"
    manager.current_step = 2
    manager.current_detail = "Agent init"
    manager.current_started_at = manager.started_at - 21

    lines = manager._fallback_panels()

    assert lines
    assert all(_display_width(line) <= 94 for line in lines)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate agent && pytest /home/kemove/LLM_Projects/Whale_Code/tests/test_benchmark_base.py -k "safe_panel_width or fallback_lines_fit" -v
```

Expected: FAIL because the current implementation returns a wider `_target_width()` result for a 100-column terminal and produces fallback lines wider than the new safe budget.

- [ ] **Step 3: Record the failing-test checkpoint**

Use `git diff -- tests/test_benchmark_base.py` to confirm only the new regression tests were added. Do not commit yet unless the user explicitly asks for a commit.

### Task 2: Implement conservative width budgeting in the progress manager

**Files:**
- Modify: `code/benchmark/base.py:532-550`
- Test: `tests/test_benchmark_base.py`

- [ ] **Step 1: Update width helpers with a stable safety margin**

Replace the existing `_target_width`, `_content_width`, and `_recent_text_width` methods with:

```python
    def _target_width(self, terminal_width: Optional[int] = None) -> int:
        columns = max(60, int(terminal_width or self._terminal_width()))
        safe_columns = max(58, columns - 6)
        if safe_columns <= 90:
            return safe_columns
        return max(72, min(safe_columns, int(columns * 0.8)))

    def _content_width(self) -> int:
        return max(24, self._target_width() - 4)

    def _recent_text_width(self) -> int:
        return max(16, self._content_width() - 26)
```

- [ ] **Step 2: Make fallback panel title construction obey the same width budget**

Inside `BenchmarkProgressManager._fallback_panels`, replace the nested `panel()` helper with:

```python
        def panel(title: str, body_lines: List[str]) -> List[str]:
            inner_width = max(20, width - 4)
            title_text = f" {title} "
            available = max(0, width - 2)
            clipped_title = _clip_display(title_text, available)
            top_fill = max(0, available - _display_width(clipped_title))
            title_segment = "┌" + clipped_title + ("─" * top_fill) + "┐"
            lines = [left_pad + title_segment]
            for line in body_lines:
                clipped = _clip_display(line, inner_width)
                padded = _ljust_display(clipped, inner_width)
                lines.append(left_pad + f"│ {padded} │")
            lines.append(left_pad + "└" + ("─" * available) + "┘")
            return lines
```

- [ ] **Step 3: Run the focused tests to verify they pass**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate agent && pytest /home/kemove/LLM_Projects/Whale_Code/tests/test_benchmark_base.py -k "safe_panel_width or fallback_lines_fit" -v
```

Expected: PASS for both new regression tests.

- [ ] **Step 4: Record the implementation checkpoint**

Use `git diff -- code/benchmark/base.py tests/test_benchmark_base.py` to confirm the change set contains only the width-budget fix and the regression tests. Do not commit unless the user explicitly asks for it.

### Task 3: Verify the fix against the broader benchmark base tests

**Files:**
- Test: `tests/test_benchmark_base.py`
- Reference: `code/benchmark/base.py`

- [ ] **Step 1: Run the full benchmark base test file**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate agent && pytest /home/kemove/LLM_Projects/Whale_Code/tests/test_benchmark_base.py -v
```

Expected: PASS for the existing tests and the new width regression tests.

- [ ] **Step 2: Run a rendering-oriented check in the agent environment**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate agent && python - <<'PY'
from benchmark.base import BenchmarkProgressManager
from benchmark.base import _display_width

manager = BenchmarkProgressManager("LCB6", 1055)
manager.completed = 759
manager.current_index = 759
manager.current_task_id = "LCB6/abc380_d"
manager.current_step = 2
manager.current_detail = "Agent init"
manager.current_started_at = manager.started_at - 21
width = manager._target_width(100)
lines = manager._fallback_panels()
print("target_width", width)
for line in lines:
    print(_display_width(line), line)
PY
```

Expected: printed widths should all be less than or equal to `target_width 94`, with no line exceeding 94 display columns.

- [ ] **Step 3: Record the verification checkpoint**

Confirm the terminal has output `target_width 94` and all listed widths are ≤ 94. No further commit needed unless the user asks.
