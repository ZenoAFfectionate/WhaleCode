# 2026-04-03 benchmark progress alignment design

## Context
The benchmark CLI progress renderer in `code/benchmark/base.py` uses `rich` panels when available and a manual text fallback otherwise. In narrow terminals, the current width calculations can let rendered lines land too close to the terminal boundary, causing automatic wrapping. The result is visible border misalignment: the right border can spill onto the next line, and the top-left/title border can appear offset.

## Audience
Primary audience: repository maintainer debugging benchmark CLI rendering issues.

## Problem statement
We need to keep the current progress UI style while fixing width-related alignment bugs shown in the screenshot. The benchmark logic and displayed information should remain unchanged.

## Root cause hypothesis
The renderer mixes several width calculations that are individually reasonable but not consistently conservative enough:
- `_target_width()` can choose a panel width that is too close to the terminal width.
- `_progress_line()`, `_status_line()`, and fallback panel/title rendering each budget content separately.
- The fallback renderer computes visible line count based on terminal wrapping, which exposes any off-by-one or near-boundary width mismatch immediately.

The likely failure mode is not the `rich` package itself, but inconsistent effective width budgeting between panel width, panel content width, and terminal-visible width.

## Recommended approach
Apply a minimal, localized fix in `code/benchmark/base.py`:
1. Make panel width selection more conservative by reserving a stable safety margin from the terminal edge.
2. Centralize the relationship between terminal width, panel outer width, and panel inner width.
3. Ensure all rendered single-line strings are clipped against the same effective content width.
4. Make fallback title/border construction use the same width assumptions as body rows.

## Alternatives considered
### A. Minimal width-strategy fix (recommended)
- Pros: smallest behavior change, preserves current UI, directly targets root cause.
- Cons: still depends on terminal width heuristics.

### B. Split progress metadata onto more lines
- Pros: very robust against wrapping.
- Cons: changes the current output format and visual density.

### C. Default to append/static mode
- Pros: avoids live rendering edge cases.
- Cons: degrades the intended UX and does not fix the underlying width logic.

## Test strategy
Follow TDD for this bug fix:
1. Add focused tests to `tests/test_benchmark_base.py` that reproduce the width invariants.
2. Verify the new tests fail against the current implementation.
3. Implement the minimal width-budget fix in `code/benchmark/base.py`.
4. Re-run the focused tests in the `agent` conda environment.
5. Run a small rendering-oriented check to confirm no produced fallback line exceeds the chosen panel width assumptions.

## Expected code changes
- `tests/test_benchmark_base.py`: add regression tests for progress/fallback width handling.
- `code/benchmark/base.py`: adjust width helpers and fallback panel rendering to keep output within safe bounds.

## Non-goals
- No redesign of the progress UI.
- No benchmark execution logic changes.
- No new dependency or replacement of `rich`.
