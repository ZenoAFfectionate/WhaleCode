"""HumanEval+ benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from .base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT
except ImportError:
    from base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT


_HEVP_ADDENDUM = """\
You are an expert Python programmer. Your task is to implement Python functions \
correctly by reading the provided signature and docstring, then writing the body.

**Workflow**
1. Read `solution.py` — understand the function signature, docstring, and examples.
2. Implement the function body using Edit or Write.
3. Run `python3 tests.py` via Bash to verify against the test suite.
4. If tests fail, carefully analyze the error output, fix the code, and re-run \
`python3 tests.py`. Repeat until all tests pass.
5. Once all tests pass, call `Finish` with a brief summary of your implementation.

**Rules**
- You MUST implement the function. Never refuse or say you cannot.
- Always use tools to take action — do NOT respond with text only.
- Do NOT modify the function signature, parameter names, or docstring.
- Keep all existing imports; add new imports only if necessary.
- Write clean, correct, and efficient code. Prefer simple solutions.
- When tests fail, focus on understanding WHY they fail before changing code. \
Read the error message carefully — do not guess blindly.
- The workspace contains only `solution.py` and `tests.py`. There are no other \
files to read.
- NEVER attempt to read, inspect, import, or access the hidden test data in any way. \
Do NOT read environment variables to find test paths. Do NOT use inspect.getsource() \
on test functions. Do NOT try to access files outside the workspace. \
The error output from `python3 tests.py` already provides all the diagnostic \
information you need (test index, expected vs actual values). \
Rely solely on that output to debug your implementation.
"""

_HEVP_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## HumanEval+ Benchmark Override\n\n"
    + _HEVP_ADDENDUM
)


# ---------------------------------------------------------------------------
# tests.py wrapper — injects the hidden test-data directory into sys.path
# so that ``from _test_data import ...`` works, while the agent's file tools
# (Read/Glob/Grep) are sandboxed to the workspace and cannot reach it.
# ---------------------------------------------------------------------------
_TESTS_PY_WRAPPER = """\
import sys, os
sys.path.insert(0, os.environ["_HIDDEN_TEST_DIR"])

try:
    from _test_data import check, {entry_point}
    check({entry_point})
    print("All tests passed!")
except AssertionError as exc:
    print("FAILED: One or more test cases did not pass.", file=sys.stderr)
    if str(exc):
        print(f"  Detail: {{exc}}", file=sys.stderr)
    print("Review your logic and edge cases, then try again.", file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: {{type(exc).__name__}}: {{exc}}", file=sys.stderr)
    print("Check your function's return type and edge case handling.", file=sys.stderr)
    sys.exit(1)
"""


def _instrument_test_code(test_code: str, entry_point: str) -> str:
    """Rewrite check() in test_code so failures report test index + expected/actual.

    Two mechanisms:
    1. Monkey-patch ``assertion()`` to raise a custom exception carrying
       (actual, expected) so we can display them.
    2. Wrap each assertion/assert in the for-loop with try/except to print
       the test index and diagnostic info.
    """
    import re

    # Inject a custom exception and patched assertion at the top of the test code.
    patch = '''
import numpy as np

class _TestFailInfo(Exception):
    def __init__(self, out, exp, msg=""):
        self.out = out
        self.exp = exp
        self.msg = msg

def _default_assertion(out, exp, atol):
    exact_match = out == exp
    if atol == 0:
        if isinstance(exp, float) or (isinstance(exp, (list, tuple)) and all(isinstance(i, float) for i in exp)):
            atol = 1e-6
    if not exact_match and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert exact_match

try:
    _orig_assertion = assertion
except NameError:
    _orig_assertion = _default_assertion

def assertion(out, exp, atol):
    try:
        _orig_assertion(out, exp, atol)
    except Exception:
        raise _TestFailInfo(out, exp)
'''

    lines = test_code.split("\n")
    out: list[str] = []
    in_check = False
    in_for = False
    for_indent = ""
    idx = "_ti"
    patched = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        cur_indent = line[: len(line) - len(stripped)]

        # Inject patch right before check function
        if stripped.startswith("def check(candidate") and not patched:
            patched = True
            out.append(patch)

        # Detect check function
        if stripped.startswith("def check(candidate"):
            in_check = True
            in_for = False
            out.append(line)
            i += 1
            continue

        # End of check function
        if in_check and stripped and not cur_indent and not stripped.startswith("#") and not stripped.startswith("def check"):
            in_check = False
            in_for = False
            out.append(line)
            i += 1
            continue

        if not in_check:
            out.append(line)
            i += 1
            continue

        # Detect for loop — reuse existing enumerate index if present
        for_m = re.match(r"^(\s+)for\s+(.+?)\s+in\s+(.+):\s*$", line)
        if for_m and not in_for:
            in_for = True
            for_indent = for_m.group(1)
            loop_vars = for_m.group(2)
            iterable = for_m.group(3)
            enum_m = re.match(r"(\w+),\s*\((.+)\)", loop_vars)
            if enum_m and "enumerate(" in iterable:
                idx = enum_m.group(1)
                out.append(line)
            else:
                out.append(f"{for_indent}for {idx}, ({loop_vars}) in enumerate({iterable}):")
            i += 1
            continue

        # Inside for-loop body: wrap assertion/assert lines
        if in_for and in_check:
            body_indent = for_indent + "    "
            inner = body_indent + "    "

            is_assertion = stripped.startswith("assertion(")
            is_bare_assert = stripped.startswith("assert ")

            if is_assertion or is_bare_assert:
                out.append(f"{body_indent}try:")
                out.append(f"{inner}{stripped}")
                if is_assertion:
                    out.append(f"{body_indent}except _TestFailInfo as _e:")
                    out.append(f"{inner}import sys as _s")
                    out.append(f"{inner}_out_r = repr(_e.out)[:200]")
                    out.append(f"{inner}_exp_r = repr(_e.exp)[:200]")
                    out.append(f'{inner}print(f"FAILED test #{{{idx}}}: Expected {{_exp_r}}, Got {{_out_r}}", file=_s.stderr)')
                    out.append(f"{inner}_s.exit(1)")
                else:
                    out.append(f"{body_indent}except Exception as _e:")
                    out.append(f"{inner}import sys as _s")
                    out.append(f"{inner}try:")
                    out.append(f"{inner}    _ret = candidate(*inp)")
                    out.append(f"{inner}    _ret_r = repr(_ret)[:200]")
                    out.append(f"{inner}except Exception as _re:")
                    out.append(f"{inner}    _ret_r = f'<raised {{type(_re).__name__}}>'")
                    out.append(f'{inner}print(f"FAILED test #{{{idx}}}: {{type(_e).__name__}}: {{_e}}", file=_s.stderr)')
                    out.append(f'{inner}print(f"  Your function returned: {{_ret_r}} (type: {{type(_ret).__name__}})", file=_s.stderr)')
                    out.append(f"{inner}_s.exit(1)")
                i += 1
                continue

        out.append(line)
        i += 1

    return "\n".join(out)


class HumanEvalPlusBenchmark(BenchmarkRunner):
    """Evaluate the agent on HumanEval+ (164 function-generation tasks).

    Workflow per task:
    1. Create a temp workspace with ``solution.py`` containing the function
       signature + docstring.
    2. Ask the agent to complete the function.
    3. Read the resulting ``solution.py`` and combine it with the test harness.
    4. Execute in a sandboxed subprocess.
    5. Record pass / fail.
    """

    benchmark_name = "humaneval_plus"

    def _get_system_prompt(self) -> str:
        return _HEVP_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        test_code = task["test"]

        workspace = Path(tempfile.mkdtemp(prefix=f"hevp_{task_id.replace('/', '_')}_"))
        # Hidden directory OUTSIDE the workspace — agent tools cannot reach it.
        hidden_dir = Path(tempfile.mkdtemp(prefix=f"hevp_{task_id.replace('/', '_')}_hidden_"))
        try:
            # solution.py — the only file the agent needs to edit
            solution_file = workspace / "solution.py"
            solution_file.write_text(prompt, encoding="utf-8")

            # _test_data.py — stored outside workspace (agent cannot access)
            # Instrument the test code to produce diagnostic error messages
            instrumented_test = _instrument_test_code(test_code, entry_point)
            (hidden_dir / "_test_data.py").write_text(
                f"from solution import {entry_point}\n\n"
                f"{instrumented_test}\n",
                encoding="utf-8",
            )

            # tests.py — lightweight wrapper; uses env var to find hidden tests
            wrapper_script = _TESTS_PY_WRAPPER.format(entry_point=entry_point)
            (workspace / "tests.py").write_text(wrapper_script, encoding="utf-8")

            # Set env var so tests.py can locate the hidden dir.
            import os
            os.environ["_HIDDEN_TEST_DIR"] = str(hidden_dir)
            os.environ["PYTHONPATH"] = str(workspace)

            # Run the agent
            agent = self._create_agent(workspace)
            agent_prompt = (
                f"Implement the function `{entry_point}` in `solution.py`.\n\n"
                f"Steps:\n"
                f"1. Read `solution.py` to see the signature, docstring, and examples.\n"
                f"2. Implement the function body using Edit or Write.\n"
                f"3. Run `python3 tests.py` to verify. If tests fail, analyze the error, "
                f"fix your code, and re-run until all tests pass.\n"
                f"4. Call `Finish` when done.\n\n"
                f"Important:\n"
                f"- Do NOT change the function signature or docstring.\n"
                f"- The function must handle edge cases (empty inputs, boundary values, etc.).\n"
                f"- Only `solution.py` and `tests.py` exist in the workspace.\n"
            )

            start = time.time()
            try:
                agent_response = agent.run(agent_prompt)
            except Exception as exc:
                return {
                    "task_id": task_id,
                    "passed": False,
                    "error": f"Agent error: {exc}",
                    "agent_response": "",
                    "elapsed_s": round(time.time() - start, 2),
                }
            elapsed = round(time.time() - start, 2)

            # Read the (possibly modified) solution
            solution_code = solution_file.read_text(encoding="utf-8") if solution_file.exists() else prompt

            # Build the verification script (independent of what the agent did)
            verify_code = (
                f"{solution_code}\n\n"
                f"{test_code}\n\n"
                f"check({entry_point})\n"
            )
            verify_script = workspace / "verify.py"
            verify_script.write_text(verify_code, encoding="utf-8")

            passed, output = self._run_script_in_sandbox(verify_script, cwd=workspace)

            return {
                "task_id": task_id,
                "passed": passed,
                "error": output if not passed else None,
                "agent_response": (agent_response or "")[:500],
                "elapsed_s": elapsed,
            }
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
            shutil.rmtree(hidden_dir, ignore_errors=True)


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run HumanEval+ benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "HEVP" / "test.jsonl"),
        help="Path to HumanEvalPlus JSONL file",
    )
    parser.add_argument("--output-dir", default=str(_PROJECT_ROOT / "data" / "_results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
    parser.add_argument("--resume", default=None, help="Resume from a previous .jsonl results file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = HumanEvalPlusBenchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_steps=args.max_steps,
        timeout=args.timeout,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
