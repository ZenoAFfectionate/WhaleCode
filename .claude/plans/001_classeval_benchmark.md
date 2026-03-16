# Plan: Add ClassEval Benchmark + Fix Data Paths

## Overview

Two things to do:
1. Create the ClassEval benchmark (code + script)
2. Fix stale data paths across existing benchmarks (HEVP, MBPP)

---

## Task 1: Fix stale data paths

All datasets have been renamed to `test.jsonl`, but two benchmarks still reference old filenames:

| File | Old Path | New Path |
|------|----------|----------|
| `code/benchmark/hevp_bench.py:115` | `data/HEVP/HumanEvalPlus-OriginFmt.jsonl` | `data/HEVP/test.jsonl` |
| `code/benchmark/mbpp_bench.py:124` | `data/MBPP/MbppPlus.jsonl` | `data/MBPP/test.jsonl` |
| `scripts/run_hevp.sh:38` | `data/HEVP/HumanEvalPlus-OriginFmt.jsonl` | `data/HEVP/test.jsonl` |
| `scripts/run_mbpp.sh:38` | `data/MBPP/MbppPlus.jsonl` | `data/MBPP/test.jsonl` |

SWEV is already correct (`data/SWEV/test.jsonl`).

---

## Task 2: Create ClassEval benchmark

### Dataset analysis

ClassEval is a **class-level** code generation benchmark. Each JSONL record has:
- `task_id`: e.g. `ClassEval_0`
- `skeleton`: class skeleton with method signatures + docstrings (agent sees this)
- `test`: unittest test code (test classes that instantiate and test the class)
- `class_name`: e.g. `AccessGatewayFilter`
- `import_statement`: list of required imports
- `methods_info`: method metadata list
- `test_classes`: list of test class names

### Design: `code/benchmark/clev_bench.py`

Follow the exact same pattern as `hevp_bench.py`:

1. Load tasks from `data/CLEV/test.jsonl`
2. Per task:
   - Create temp workspace with `solution.py` containing the skeleton
   - Ask agent: "Read `solution.py`, implement all methods in class `{class_name}`"
   - Read resulting `solution.py`
   - Build verify script: `solution_code + "\n\n" + test_code + "\n\nunittest.main()"`
   - Run in sandbox with `python -m unittest` to handle test discovery
   - Record pass/fail
3. Clean up workspace

Key difference from HEVP: tests use `unittest.TestCase`, so the verify command runs `python -m unittest verify.py` instead of plain `python verify.py`.

### Files to create/modify

| Action | File |
|--------|------|
| **Create** | `code/benchmark/clev_bench.py` |
| **Create** | `scripts/run_clev.sh` |
| **Edit** | `code/benchmark/__init__.py` — add `ClassEvalBenchmark` export |
| **Edit** | `code/benchmark/hevp_bench.py` — fix default data path |
| **Edit** | `code/benchmark/mbpp_bench.py` — fix default data path |
| **Edit** | `scripts/run_hevp.sh` — fix data path |
| **Edit** | `scripts/run_mbpp.sh` — fix data path |
| **Edit** | `README.md` — add ClassEval to benchmark table + commands |
