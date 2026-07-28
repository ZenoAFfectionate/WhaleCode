"""MBPP+ benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from .base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
    )
    from .runtime.python_adapters import PythonAssertionAdapter
except ImportError:
    from base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
    )
    from runtime.python_adapters import PythonAssertionAdapter

_MBPP_ADDENDUM = """\
You are implementing MBPP+ Python programming tasks.

**Workflow**
1. Read `solution.py` — understand the function signature and the task description.
2. Implement the function body using Edit or Write.
3. Submit with `Finish`; the runner evaluates externally and returns bounded feedback.
4. If feedback shows failures: analyze the error carefully, fix the root cause,
   then resubmit with `Finish`.

**Rules**
- Benchmark test files are not available in the workspace. There are no local tests.
- Do NOT create your own uncontrolled benchmark test loop.
- Do NOT try to reconstruct hidden benchmark files or inspect anything outside the workspace.
- Keep the required function signature exactly as provided. Do not rename parameters.
- Keep all existing imports; add new ones only when strictly necessary.
- Prefer simple, readable, correct code. Do not over-engineer.

**On revision after failure**
- Before editing, understand WHY the test failed — read the error message carefully.
- Fix the root cause, not the symptom. One correct fix is better than multiple guesses.
- If feedback shows the same failure after 2+ rounds, reconsider your approach completely.
"""

_MBPP_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## MBPP+ Benchmark Override\n\n"
    + _MBPP_ADDENDUM
)


_MBPP_VERIFY_TEMPLATE = """\
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


def _build_verify_script(assertion_code: str) -> str:
    """Convert plain assert statements into an internal verifier with detailed output."""
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
    return _MBPP_VERIFY_TEMPLATE.format(checks="\n".join(checks))


def _evaluate_solution(
    workspace: Path,
    solution_file: Path,
    assertion_code: str,
    timeout: int,
) -> tuple[bool, str]:
    return PythonAssertionAdapter(_build_verify_script).evaluate(
        workspace=workspace,
        solution_file=solution_file,
        assertion_code=assertion_code,
        timeout=timeout,
    )


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

    def __init__(self, *args, max_submission_rounds: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_submission_rounds = max(1, int(max_submission_rounds))

    def _get_system_prompt(self) -> str:
        return _MBPP_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        return self._load_jsonl_tasks()

    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        prompt_text = task["prompt"]
        entry_point = task["entry_point"]
        assertion_code = task.get("assertion", "")

        workspace = self._make_workspace(f"mbpp_{task_id.replace('/', '_')}_")
        agent = None
        agent_response = ""
        prompt_history: List[str] = []
        result: Optional[Dict[str, Any]] = None
        try:
            solution_file = workspace / "solution.py"
            solution_file.write_text(
                f"# Implement the function: {entry_point}\n",
                encoding="utf-8",
            )

            agent = self._create_agent(workspace)
            initial_prompt = (
                f"Implement Python function `{entry_point}` in `solution.py`.\n\n"
                f"Task description:\n{prompt_text}\n\n"
                f"Constraints:\n"
                f"- Controlled submissions only: call `Finish` alone when ready for evaluation.\n"
                f"- Benchmark test files are not present in the workspace.\n"
                f"- Do not run your own benchmark test loop.\n"
                f"- Keep the required signature intact.\n"
                f"- Handle edge cases and include required imports.\n"
                f"- Keep the implementation simple and readable.\n"
            )

            start = time.time()
            def _retry_prompt(round_idx: int, feedback: str) -> str:
                return (
                    f"Controlled evaluation feedback for submission round {round_idx - 1}:\n\n"
                    f"{feedback}\n\n"
                    f"Revise `solution.py` based on this feedback.\n"
                    f"- The failing check summaries are reliable.\n"
                    f"- Actual/expected values are intentionally bounded.\n"
                    f"- Use the feedback above plus the task description, not hidden benchmark files.\n"
                    f"When ready for the next controlled submission, call `Finish` alone with a brief summary of the revision."
                )

            def _evaluate_submission(round_idx: int, latest_response: str) -> Dict[str, Any]:
                if not solution_file.exists():
                    return {
                        "result": self._build_result(
                            task_id,
                            passed=False,
                            error="solution.py not found after agent run",
                            start_time=start,
                            agent_response=latest_response,
                            extra={"submission_rounds": round_idx},
                        )
                    }
                passed, output = _evaluate_solution(
                    workspace=workspace,
                    solution_file=solution_file,
                    assertion_code=assertion_code,
                    timeout=self.timeout,
                )
                return {"passed": passed, "output": output}

            loop = self._run_controlled_submission_rounds(
                task_id=task_id,
                agent=agent,
                start_time=start,
                initial_prompt=initial_prompt,
                max_rounds=self.max_submission_rounds,
                prompt_history=prompt_history,
                evaluate_submission=_evaluate_submission,
                retry_prompt_builder=_retry_prompt,
            )
            agent_response = loop["agent_response"]
            if loop["early_result"] is not None:
                result = loop["early_result"]
                return result

            elapsed = round(time.time() - start, 2)

            result = self._build_result(
                task_id,
                passed=bool(loop["passed"]),
                error=loop["output"] if not loop["passed"] else None,
                agent_response=agent_response,
                elapsed_s=elapsed,
                extra={"submission_rounds": int(loop["rounds_used"])},
            )
            return result
        finally:
            self._finalize_workspace_task(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=prompt_history,
                result=result,
                artifact_paths=["solution.py"],
                extra={"entry_point": entry_point},
            )


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
        default_timeout=120,
        include_task_timeout=True,
        default_task_timeout=600,
    )
    parser.add_argument("--max-submission-rounds", type=int, default=5)
    args = parser.parse_args()

    bench = MBPPPlusBenchmark(
        data_path=args.data_path,
        max_submission_rounds=args.max_submission_rounds,
        **BenchmarkRunner.runner_kwargs_from_args(args, include_task_timeout=True),
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume, fresh=args.fresh)


if __name__ == "__main__":
    main()
