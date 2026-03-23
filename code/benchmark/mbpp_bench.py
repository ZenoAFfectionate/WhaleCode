"""MBPP+ benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .base import BenchmarkRunner, _PROJECT_ROOT


# ---------------------------------------------------------------------------
# tests.py wrapper — transforms plain assert into informative checks
# ---------------------------------------------------------------------------
_MBPP_TESTS_PY_TEMPLATE = """\
import sys
sys.path.insert(0, ".")
from solution import *

_failed = 0
_total = 0

{checks}

print(f"{{_total - _failed}}/{{_total}} passed")
if _failed:
    sys.exit(1)
else:
    print("All tests passed!")
"""

_MBPP_CHECK_TEMPLATE = """\
_total += 1
try:
    _actual = {actual_expr}
    _expected = {expected_expr}
    assert _actual == _expected, ""
except AssertionError:
    _failed += 1
    print(f"[FAIL] {actual_expr_escaped}")
    print(f"  actual:   {{_actual!r}}")
    print(f"  expected: {{_expected!r}}")
except Exception as _e:
    _failed += 1
    print(f"[ERROR] {actual_expr_escaped}")
    print(f"  {{type(_e).__name__}}: {{_e}}")
"""


def _build_tests_py(assertion_code: str) -> str:
    """Convert plain assert statements into a tests.py with detailed output."""
    checks = []
    for line in assertion_code.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^assert\s+(.+?)\s*==\s*(.+)$", line)
        if m:
            actual_expr = m.group(1).strip()
            expected_expr = m.group(2).strip()
            escaped = actual_expr.replace("{", "{{").replace("}", "}}")
            checks.append(
                _MBPP_CHECK_TEMPLATE.format(
                    actual_expr=actual_expr,
                    expected_expr=expected_expr,
                    actual_expr_escaped=escaped,
                )
            )
        else:
            # Non-standard assert — wrap with try/except
            escaped = line.replace("{", "{{").replace("}", "}}")
            checks.append(
                f'_total += 1\n'
                f'try:\n'
                f'    {line}\n'
                f'except Exception as _e:\n'
                f'    _failed += 1\n'
                f'    print(f"[ERROR] {escaped}")\n'
                f'    print(f"  {{type(_e).__name__}}: {{_e}}")\n'
            )
    return _MBPP_TESTS_PY_TEMPLATE.format(checks="\n".join(checks))


class MBPPPlusBenchmark(BenchmarkRunner):
    """Evaluate the agent on MBPP+ (378 function-generation tasks).

    Workflow per task:
    1. Create a temp workspace with an empty ``solution.py``.
    2. Present the task prompt (includes docstring + example assertions).
    3. Ask the agent to implement the function in ``solution.py``.
    4. Combine the solution with assertion tests and execute.
    5. Record pass / fail.
    """

    benchmark_name = "mbpp_plus"

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    def _build_test_code(self, task: Dict[str, Any], solution_code: str) -> str:
        """Build a verification script from solution + assertions.

        MBPP+ provides both ``base_input`` + ``plus_input`` for functional
        testing and ``assertion`` for simple assert-based checks.  We use
        the assertion string which is the most straightforward.
        """
        assertion_code = task.get("assertion", "")
        return f"{solution_code}\n\n{assertion_code}\n"

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        prompt_text = task["prompt"]
        entry_point = task["entry_point"]
        assertion_code = task.get("assertion", "")

        workspace = Path(tempfile.mkdtemp(prefix=f"mbpp_{task_id.replace('/', '_')}_"))
        try:
            # Write an empty solution file with a hint comment
            solution_file = workspace / "solution.py"
            solution_file.write_text(
                f"# Implement the function: {entry_point}\n",
                encoding="utf-8",
            )

            # tests.py — assertion wrapper with detailed error output
            (workspace / "tests.py").write_text(
                _build_tests_py(assertion_code), encoding="utf-8"
            )

            # Run the agent
            agent = self._create_agent(workspace)
            agent_prompt = (
                f"Your task is to implement the Python function `{entry_point}` "
                f"in `solution.py`.\n\n"
                f"**Task description:**\n{prompt_text}\n\n"
                f"Follow these steps:\n"
                f"1. Carefully analyze the task description and identify edge cases.\n"
                f"2. Implement the function in `solution.py` using the Edit or Write tool.\n"
                f"3. Run `python3 tests.py` to verify. If tests fail, analyze the error, "
                f"fix your code, and re-run until all tests pass.\n"
                f"4. Once all tests pass, call `Finish` with a brief summary.\n\n"
                f"Rules:\n"
                f"- The function must satisfy all the assertions in the task description.\n"
                f"- Include any necessary imports at the top of the file.\n"
                f"- Handle edge cases gracefully (empty inputs, special values).\n"
                f"- Prefer simple, readable implementations over clever one-liners.\n"
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

            # Read the solution
            if solution_file.exists():
                solution_code = solution_file.read_text(encoding="utf-8")
            else:
                return {
                    "task_id": task_id,
                    "passed": False,
                    "error": "solution.py not found after agent run",
                    "agent_response": (agent_response or "")[:500],
                    "elapsed_s": elapsed,
                }

            # Build and run the test
            test_code = self._build_test_code(task, solution_code)
            verify_script = workspace / "verify.py"
            verify_script.write_text(test_code, encoding="utf-8")

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


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run MBPP+ benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "MBPP" / "test.jsonl"),
        help="Path to MbppPlus JSONL file",
    )
    parser.add_argument("--output-dir", default=str(_PROJECT_ROOT / "data" / "_results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--resume", default=None, help="Resume from a previous .jsonl results file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = MBPPPlusBenchmark(
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
