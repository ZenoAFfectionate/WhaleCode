"""MBPP+ benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from .base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
        build_minimal_child_env,
        truncate_feedback,
    )
except ImportError:
    from base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
        build_minimal_child_env,
        truncate_feedback,
    )

_MBPP_ADDENDUM = """\
You are implementing MBPP+ Python programming tasks.

Workflow:
1. Read the task description carefully and identify the required function behavior.
2. Read `solution.py`, then implement the function there. Prefer `Edit` for focused changes and `Write` only if a full rewrite is simpler.
3. Use only your own lightweight checks if you want to validate ideas locally.
4. When ready for a controlled submission, call `Finish` alone with a short summary of the implementation.
5. The benchmark runner will execute the benchmark tests outside the workspace and return bounded feedback if another revision is needed.

Rules:
- Benchmark test files are not available in the workspace.
- Do not create your own uncontrolled benchmark test loop.
- Do not try to reconstruct hidden benchmark files or inspect anything outside the workspace.
- Keep the required function signature intact.
- Prefer simple, readable, correct code.
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
    if not solution_file.exists():
        return False, "solution.py not found"

    verify_script = workspace / "._mbpp_verify.py"
    verify_script.write_text(_build_verify_script(assertion_code), encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(verify_script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace),
            env=build_minimal_child_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: benchmark evaluation exceeded {timeout}s."
    except Exception as exc:
        return False, f"ERROR: benchmark evaluation failed: {exc}"
    finally:
        try:
            verify_script.unlink(missing_ok=True)
        except Exception:
            pass

    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return True, output or "All tests passed!"
    return False, output or "Benchmark evaluation failed."


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

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
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
                f"Your task is to implement the Python function `{entry_point}` "
                f"in `solution.py`.\n\n"
                f"**Task description:**\n{prompt_text}\n\n"
                f"Submission policy:\n"
                f"- This benchmark uses controlled submissions.\n"
                f"- Benchmark test files are not present in the workspace.\n"
                f"- Do not run your own benchmark test loop.\n"
                f"- After each completed submission, typically when you call `Finish`, the runner will execute benchmark tests and send bounded feedback if needed.\n\n"
                f"Follow these steps:\n"
                f"1. Carefully analyze the task description and identify edge cases.\n"
                f"2. Read `solution.py`, then implement the function there. Prefer `Edit` for focused changes and `Write` only if a full rewrite is simpler.\n"
                f"3. You may run lightweight self-checks of your own design, but do not rely on benchmark tests.\n"
                f"4. When ready for a controlled submission, call `Finish` alone with a brief summary of what you implemented.\n\n"
                f"Important:\n"
                f"- The function must satisfy the task requirements and examples.\n"
                f"- Include any necessary imports at the top of the file.\n"
                f"- Handle edge cases gracefully (empty inputs, special values).\n"
                f"- Prefer simple, readable implementations over clever one-liners.\n"
                f"- Only `solution.py` exists in the workspace.\n"
            )

            start = time.time()
            feedback = None
            passed = False
            output = ""
            rounds_used = 0

            for round_idx in range(1, self.max_submission_rounds + 1):
                rounds_used = round_idx
                prompt = initial_prompt if round_idx == 1 else (
                    f"Controlled evaluation feedback for submission round {round_idx - 1}:\n\n"
                    f"{feedback}\n\n"
                    f"Revise `solution.py` based on this feedback.\n"
                    f"- The failing check summaries are reliable.\n"
                    f"- Actual/expected values are intentionally bounded.\n"
                    f"- Use the feedback above plus the task description, not hidden benchmark files.\n"
                    f"When ready for the next controlled submission, call `Finish` alone with a brief summary of the revision."
                )
                prompt_history.append(prompt)

                agent_response, error_result = self._run_agent_prompt(
                    agent=agent,
                    task_id=task_id,
                    prompt_text=prompt,
                    start_time=start,
                    error_extra={"submission_rounds": round_idx},
                )
                if error_result is not None:
                    result = error_result
                    return result

                if not solution_file.exists():
                    result = self._missing_output_result(
                        task_id,
                        path_label="solution.py",
                        start_time=start,
                        agent_response=agent_response,
                        extra={"submission_rounds": round_idx},
                    )
                    return result

                passed, output = _evaluate_solution(
                    workspace=workspace,
                    solution_file=solution_file,
                    assertion_code=assertion_code,
                    timeout=self.timeout,
                )
                if passed:
                    break
                feedback = truncate_feedback(output, max_lines=80, max_chars=12000)

            elapsed = round(time.time() - start, 2)

            result = self._build_result(
                task_id,
                passed=passed,
                error=output if not passed else None,
                agent_response=agent_response,
                elapsed_s=elapsed,
                extra={"submission_rounds": rounds_used},
            )
            return result
        finally:
            self._save_task_trajectory(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=prompt_history,
                result=result,
                artifact_paths=["solution.py"],
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
    parser.add_argument("--max-submission-rounds", type=int, default=5)
    args = parser.parse_args()

    bench = MBPPPlusBenchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_submission_rounds=args.max_submission_rounds,
        timeout=args.timeout,
        trajectory_dir=args.trajectory_dir,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
