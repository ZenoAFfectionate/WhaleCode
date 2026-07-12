"""ClassEval benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
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
    )
except ImportError:
    from base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
        build_minimal_child_env,
    )


_CLEV_ADDENDUM = """\
You are an expert Python programmer. Your task is to implement all methods in a \
Python class by reading the skeleton (signatures + docstrings) and writing \
complete, correct method bodies.

**Workflow**
1. Read `solution.py` — understand the class skeleton: every method signature, \
docstring, `__init__`, and existing imports.
2. Implement every method according to its docstring using Edit or Write.
3. Submit with `Finish`; hidden tests run outside the workspace and return bounded feedback.
4. Revise `solution.py` and resubmit with `Finish` until done.

**Rules**
- You MUST implement every method. Never refuse or say you cannot.
- Use tools to inspect and modify the workspace, then call `Finish` once you are ready for evaluation.
- Do NOT modify the class name, method signatures, or docstrings.
- Keep all existing imports; add new imports only if necessary.
- Update `__init__` if your implementations require additional instance attributes.
- Write clean, correct, and efficient code. Prefer simple solutions.
- When tests fail, focus on understanding WHY they fail before changing code. \
Read the test name and error message carefully — do not guess blindly.
- If you have tried the same fix multiple times without progress, reconsider \
your approach from scratch.
- The workspace contains only `solution.py`. There are no local benchmark tests to run.

**Critical: Do NOT over-engineer**
- Implement EXACTLY what the docstring specifies. Do NOT add extra validation, \
edge-case handling, or business logic beyond what the docstring requires.
- If the docstring says "returns True if X exists", do NOT also check "but return False \
if the value didn't actually change". Just check if X exists.
- Do NOT add special-case branches for specific input values (e.g., `if n == 1: ...`).
- Do NOT make assumptions about hidden test behavior that contradict the docstring.
- When in doubt, follow the docstring literally — even if it seems unconventional.

**On revision after failure**
- Before editing, mentally trace through the failing test's expected vs actual output.
- Identify the ROOT CAUSE, not just the symptom. A single fix addressing the root cause \
is better than multiple band-aids.
- If 2+ rounds produce the same failure, your diagnosis is likely wrong — step back and \
re-examine your assumptions about what the method should do.
"""

_CLEV_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## ClassEval Benchmark Override\n\n"
    + _CLEV_ADDENDUM
)


_CLEV_HOST_EVAL_SUFFIX = """\
import ast
import inspect
import sys
import unittest


def _extract_exception_info(tb):
    lines = [line.strip() for line in tb.strip().splitlines() if line.strip()]
    if not lines:
        return "UnknownError", ""
    last = lines[-1]
    if ":" in last:
        exc_type, message = last.split(":", 1)
        return exc_type.strip(), message.strip()
    return last, ""


def _build_context_map_from_test_source(source_text):
    \"\"\"Build class/method context lines from the injected hidden test source.\"\"\"
    context_map = {}
    if not source_text:
        return context_map
    try:
        module = ast.parse(source_text)
    except Exception:
        return context_map

    source_lines = source_text.splitlines()
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_map = {}
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            method_name = str(getattr(item, "name", "") or "")
            if not method_name.startswith("test"):
                continue
            start = max(1, int(getattr(item, "lineno", 1)))
            end = max(start, int(getattr(item, "end_lineno", start)))
            context_lines = []
            for lineno in range(start, min(end, len(source_lines)) + 1):
                raw = source_lines[lineno - 1]
                stripped = raw.strip()
                if (
                    not stripped
                    or stripped.startswith("def ")
                    or stripped.startswith("self.assert")
                    or stripped.startswith("#")
                ):
                    continue
                context_lines.append((lineno, stripped))
            if len(context_lines) > 12:
                context_lines = context_lines[:12]
            if context_lines:
                class_map[method_name] = context_lines
        if class_map:
            context_map[node.name] = class_map
    return context_map


_TEST_CONTEXT_MAP = _build_context_map_from_test_source(globals().get("__CLEV_TEST_CODE__", ""))


def _extract_context(test):
    \"\"\"Extract context lines for a failing test in a stable way.\"\"\"
    class_name = test.__class__.__name__
    method_name = str(getattr(test, "_testMethodName", "") or "")
    lines = _TEST_CONTEXT_MAP.get(class_name, {}).get(method_name, [])
    if lines:
        return lines

    # Fallback for environments where injected test source is unavailable.
    try:
        method = getattr(test, method_name)
        raw_lines = inspect.getsource(method).splitlines()
    except Exception:
        return []
    fallback_lines = []
    for raw in raw_lines:
        stripped = raw.strip()
        if (
            not stripped
            or stripped.startswith("def ")
            or stripped.startswith("self.assert")
            or stripped.startswith("#")
        ):
            continue
        fallback_lines.append((0, stripped))
    return fallback_lines


def _print_context_block(context_lines):
    print("context:")
    if not context_lines:
        print("  (no context lines available)")
        return
    for lineno, text in context_lines:
        if lineno > 0:
            print(f"  L{lineno}: {text}")
        else:
            print(f"  {text}")


def _trim_traceback(tb, max_lines=10):
    lines = [line for line in tb.strip().splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\\n".join(lines)
    return "\\n".join(lines[-max_lines:])


def _print_case(case_idx, label, test, tb):
    test_id = test.id() if hasattr(test, "id") else str(test)
    exc_type, exc_msg = _extract_exception_info(tb)

    print(f"=== FAILURE_CASE {case_idx} ===")
    print(f"result_type: {label}")
    print(f"test_id: {test_id}")
    print(f"error: {exc_type}: {exc_msg}")
    _print_context_block(_extract_context(test))
    print("traceback_tail:")
    print(_trim_traceback(tb))


def _print_summary(total, failures, errors):
    \"\"\"Print a high-level summary before detailed failure cases.\"\"\"
    passed = total - len(failures) - len(errors)
    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed, {len(failures)} failed, {len(errors)} errors")
    if failures or errors:
        print("Failing/broken tests:")
        for label, bucket in (("FAIL", failures), ("ERROR", errors)):
            for test, _ in bucket:
                test_id = test.id() if hasattr(test, "id") else str(test)
                print(f"  [{label}] {test_id}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TestResult()
    suite.run(result)

    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)

    # Print summary first, then details
    _print_summary(total, list(result.failures), list(result.errors))

    case_idx = 0
    for label, bucket in (("FAIL", result.failures), ("ERROR", result.errors)):
        for test, tb in bucket:
            case_idx += 1
            _print_case(case_idx, label, test, tb)

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
    test_code_literal = repr(test_code)
    verify_code = (
        f"{solution_code}\n\n"
        f"{test_code}\n\n"
        f"__CLEV_TEST_CODE__ = {test_code_literal}\n\n"
        f"{_CLEV_HOST_EVAL_SUFFIX}\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-"],
            input=verify_code,
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

    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
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
                f"Requirements:\n"
                f"- Do NOT change the class name, method signatures, or docstrings.\n"
                f"- Implement every method according to docstrings/examples.\n"
                f"- Update `__init__` if additional instance state is required.\n"
                f"- Pay attention to the docstring examples — they reveal expected behavior.\n"
                f"- Hidden tests run only after `Finish`; do not create uncontrolled benchmark loops.\n"
                f"- The workspace contains only `solution.py`.\n"
            )

            start = time.time()
            previous_failures: List[str] = []

            def _retry_prompt(round_idx: int, feedback: str) -> str:
                # Detect persistent failures: extract test_ids from current feedback
                # and compare with previous rounds to identify stuck patterns.
                current_test_ids = []
                for line in feedback.splitlines():
                    if line.startswith("  test_id:"):
                        current_test_ids.append(line.split(":", 1)[1].strip())

                repeated = [tid for tid in current_test_ids if tid in previous_failures]
                new_ids = [tid for tid in current_test_ids if tid not in previous_failures]

                # Build persistence warning if same tests keep failing
                persistence_note = ""
                if repeated:
                    persistence_note = (
                        f"** WARNING: The following {len(repeated)} test(s) have failed "
                        f"in multiple previous rounds — your previous fixes did not resolve them. **\n"
                        f"Failed tests: {', '.join(repeated)}\n\n"
                        f"This means your diagnosis is likely wrong. DO NOT apply a similar fix again.\n"
                        f"Step back and reconsider: what assumption are you making that might be incorrect?\n"
                        f"Re-read the docstring for the failing method carefully.\n\n"
                    )

                # Track failures for next round
                previous_failures.extend(current_test_ids)

                return (
                    f"Hidden test feedback after submission round {round_idx - 1}:\n\n"
                    f"{persistence_note}"
                    f"{'='*60}\n"
                    f"FEEDBACK:\n"
                    f"{'='*60}\n\n"
                    f"{feedback}\n\n"
                    f"{'='*60}\n"
                    f"ACTION REQUIRED:\n"
                    f"{'='*60}\n\n"
                    f"1. **Analyze FIRST**: Before editing, explain to yourself WHY each test fails.\n"
                    f"   - What does the test expect vs what your code produces?\n"
                    f"   - What assumption in your implementation is wrong?\n"
                    f"2. **Fix the root cause**: One correct fix is better than multiple guesses.\n"
                    f"3. **Re-read the docstring**: The docstring examples are ground truth.\n"
                    f"   Follow them literally — do not add logic they don't imply.\n\n"
                    f"{'New failing tests' if new_ids else 'All failing tests'}: "
                    f"{', '.join(new_ids) if new_ids else ', '.join(current_test_ids)}\n\n"
                    f"When ready, revise `solution.py` and call `Finish` with a brief summary.\n"
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
                    fallback_solution=skeleton,
                    test_code=test_code,
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
                error_extra_builder=lambda round_idx: {"submission_rounds": round_idx},
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
                extra={"class_name": class_name},
            )


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
        default_max_steps=128,
        default_timeout=120,
        include_task_timeout=True,
        default_task_timeout=1200,
    )
    parser.add_argument("--max-submission-rounds", type=int, default=5)
    args = parser.parse_args()

    bench = ClassEvalBenchmark(
        data_path=args.data_path,
        max_submission_rounds=args.max_submission_rounds,
        **BenchmarkRunner.runner_kwargs_from_args(args, include_task_timeout=True),
    )

    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
