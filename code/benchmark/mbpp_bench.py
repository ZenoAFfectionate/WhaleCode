"""MBPP+ benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import re
import shutil
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from .base import BenchmarkRunner, _PROJECT_ROOT
except ImportError:
    from base import BenchmarkRunner, _PROJECT_ROOT


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
        return self._load_jsonl_tasks()

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

        workspace = self._make_workspace(f"mbpp_{task_id.replace('/', '_')}_")
        agent = None
        agent_response = ""
        agent_prompt = ""
        result: Optional[Dict[str, Any]] = None
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
            agent_response, error_result = self._run_agent_prompt(
                agent=agent,
                task_id=task_id,
                prompt_text=agent_prompt,
                start_time=start,
            )
            if error_result is not None:
                result = error_result
                return result
            elapsed = round(time.time() - start, 2)

            # Read the solution
            if solution_file.exists():
                solution_code = solution_file.read_text(encoding="utf-8")
            else:
                result = self._missing_output_result(
                    task_id,
                    path_label="solution.py",
                    elapsed_s=elapsed,
                    agent_response=agent_response,
                )
                return result

            # Build and run the test
            test_code = self._build_test_code(task, solution_code)
            verify_script = workspace / "verify.py"
            verify_script.write_text(test_code, encoding="utf-8")

            passed, output = self._run_script_in_sandbox(verify_script, cwd=workspace)

            result = self._build_result(
                task_id,
                passed=passed,
                error=output if not passed else None,
                agent_response=agent_response,
                elapsed_s=elapsed,
            )
            return result
        finally:
            self._save_task_trajectory(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=[agent_prompt] if agent_prompt else [],
                result=result,
                artifact_paths=["solution.py", "tests.py", "verify.py"],
                extra={"entry_point": entry_point},
            )
            shutil.rmtree(workspace, ignore_errors=True)


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run MBPP+ benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "MBPP" / "test.jsonl"),
        help="Path to MbppPlus JSONL file",
    )
    BenchmarkRunner.add_shared_run_args(
        parser,
        default_temperature=1.0,
        default_max_steps=64,
        default_timeout=60,
    )
    args = parser.parse_args()

    bench = MBPPPlusBenchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_steps=args.max_steps,
        timeout=args.timeout,
        trajectory_dir=args.trajectory_dir,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
