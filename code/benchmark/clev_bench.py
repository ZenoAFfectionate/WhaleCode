"""ClassEval benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
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


_CLEV_ADDENDUM = """\
You are an expert Python programmer. Your task is to implement all methods in a \
Python class by reading the skeleton (signatures + docstrings) and writing \
complete, correct method bodies.

**Workflow**
1. Read `solution.py` — understand the class skeleton: every method signature, \
docstring, `__init__`, and existing imports.
2. Implement every method according to its docstring using Edit or Write.
3. When ready, respond with a short plain-text summary of the current implementation.
4. The benchmark runner will execute hidden tests outside the workspace and send \
back controlled feedback if another revision is needed.
5. Revise `solution.py` based on that feedback and respond again.

**Rules**
- You MUST implement every method. Never refuse or say you cannot.
- Use tools to inspect and modify the workspace, then give a normal text response once you are ready for evaluation.
- Do NOT modify the class name, method signatures, or docstrings.
- Keep all existing imports; add new imports only if necessary.
- Update `__init__` if your implementations require additional instance attributes.
- Write clean, correct, and efficient code. Prefer simple solutions.
- When tests fail, focus on understanding WHY they fail before changing code. \
Read the test name and error message carefully — do not guess blindly.
- If you have tried the same fix multiple times without progress, reconsider \
your approach from scratch.
- The workspace contains only `solution.py`. There are no local benchmark tests to run.
"""

_CLEV_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## ClassEval Benchmark Override\n\n"
    + _CLEV_ADDENDUM
)


_CLEV_HOST_EVAL_SUFFIX = """\
import inspect
import sys
import unittest


def _extract_context(test):
    \"\"\"Extract non-assertion source lines from a failing test method.\"\"\"
    try:
        method = getattr(test, test._testMethodName)
        lines = inspect.getsource(method).splitlines()
        context = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith("def ") or s.startswith("self.assert") or s.startswith("#"):
                continue
            context.append(f"  > {s}")
        return "\\n".join(context[:8])
    except Exception:
        return ""


def _trim_traceback(tb, max_lines=28):
    lines = tb.strip().splitlines()
    if len(lines) <= max_lines:
        return "\\n".join(lines)
    return "\\n".join(lines[-max_lines:])


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TestResult()
    suite.run(result)

    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    report_budget = 3
    emitted = 0

    for label, bucket in (("FAIL", result.failures), ("ERROR", result.errors)):
        for test, tb in bucket:
            if emitted >= report_budget:
                break
            ctx = _extract_context(test)
            print(f"[{label}] {test}")
            if ctx:
                print(ctx)
            print(_trim_traceback(tb))
            emitted += 1

    omitted = failed + errors - emitted
    if omitted > 0:
        print(f"[{omitted} additional failing tests omitted]")

    print(f"{total - failed - errors}/{total} passed")
    if not result.failures and not result.errors:
        print("All hidden tests passed!")

    sys.exit(0 if not result.failures and not result.errors else 1)
"""

def _evaluate_solution(
    workspace: Path,
    solution_file: Path,
    fallback_solution: str,
    test_code: str,
    timeout: int,
) -> tuple[bool, str]:
    solution_code = solution_file.read_text(encoding="utf-8") if solution_file.exists() else fallback_solution
    verify_code = (
        f"{solution_code}\n\n"
        f"{test_code}\n\n"
        f"{_CLEV_HOST_EVAL_SUFFIX}\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", verify_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace),
            env=build_minimal_child_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: hidden evaluation exceeded {timeout}s."
    except Exception as exc:
        return False, f"ERROR: host-side evaluation failed: {exc}"

    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output or ("All hidden tests passed!" if result.returncode == 0 else "Hidden evaluation failed.")


class ClassEvalBenchmark(BenchmarkRunner):
    """Evaluate the agent on ClassEval (100 class-level generation tasks).

    Workflow per task:
    1. Create a temp workspace with ``solution.py`` containing the class
       skeleton (method signatures + docstrings).
    2. Ask the agent to implement all methods in the class.
    3. Read the resulting ``solution.py`` and combine it with the unittest
       test harness.
    4. Execute in a sandboxed subprocess.
    5. Record pass / fail.
    """

    benchmark_name = "classeval"

    def __init__(self, *args, max_submission_rounds: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_submission_rounds = max(1, int(max_submission_rounds))

    def _get_system_prompt(self) -> str:
        return _CLEV_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        return self._load_jsonl_tasks()

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        skeleton = task["skeleton"]
        test_code = task["test"]
        class_name = task["class_name"]

        workspace = self._make_workspace(f"clev_{task_id}_")
        agent = None
        agent_response = ""
        prompt_history: List[str] = []
        result: Optional[Dict[str, Any]] = None
        try:
            solution_file = workspace / "solution.py"
            solution_file.write_text(skeleton, encoding="utf-8")

            agent = self._create_agent(workspace)
            initial_prompt = (
                f"Implement all methods in the class `{class_name}` in `solution.py`.\n\n"
                f"Submission policy:\n"
                f"- Hidden tests are evaluated only by the benchmark runner.\n"
                f"- Do not create your own uncontrolled benchmark loop.\n"
                f"- Each time you finish with a normal text response, the runner will execute hidden tests and send bounded feedback if needed.\n\n"
                f"Steps:\n"
                f"1. Read `solution.py` to understand the class skeleton — method signatures, "
                f"docstrings, and `__init__`.\n"
                f"2. Implement every method according to its docstring. Update `__init__` "
                f"if you need additional instance attributes.\n"
                f"3. Perform lightweight self-checks if useful, but do not rely on local benchmark tests.\n"
                f"4. When you want a controlled submission, stop and provide a brief plain-text summary.\n\n"
                f"Important:\n"
                f"- Do NOT change the class name, method signatures, or docstrings.\n"
                f"- Pay attention to the docstring examples — they reveal expected behavior.\n"
                f"- The workspace contains only `solution.py`.\n"
            )

            start = time.time()
            agent_response = ""
            feedback = None
            passed = False
            output = ""
            rounds_used = 0

            for round_idx in range(1, self.max_submission_rounds + 1):
                rounds_used = round_idx
                prompt_text = initial_prompt if round_idx == 1 else (
                    f"Controlled hidden-test feedback for submission round {round_idx - 1}:\n\n"
                    f"{feedback}\n\n"
                    f"Revise `solution.py` based on this feedback.\n"
                    f"- Failing test names are reliable.\n"
                    f"- The returned context lines are selected from the hidden test body but omit direct assertions.\n"
                    f"- Tracebacks are truncated to the most relevant portion.\n"
                    f"When ready, respond again with a brief plain-text summary.\n"
                )
                prompt_history.append(prompt_text)

                agent_response, error_result = self._run_agent_prompt(
                    agent=agent,
                    task_id=task_id,
                    prompt_text=prompt_text,
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
                    fallback_solution=skeleton,
                    test_code=test_code,
                    timeout=self.timeout,
                )
                if passed:
                    break
                feedback = truncate_feedback(output, max_lines=100, max_chars=14000)

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
                extra={"class_name": class_name},
            )
            shutil.rmtree(workspace, ignore_errors=True)


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run ClassEval benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "CLEV" / "test.jsonl"),
        help="Path to ClassEval JSONL file",
    )
    BenchmarkRunner.add_shared_run_args(
        parser,
        default_temperature=1.0,
        default_max_steps=64,
        default_timeout=120,
    )
    parser.add_argument("--max-submission-rounds", type=int, default=5)
    args = parser.parse_args()

    bench = ClassEvalBenchmark(
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
