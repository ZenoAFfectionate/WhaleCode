"""ClassEval benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
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
    from .runtime.python_adapters import PythonVerifierAdapter
except ImportError:
    from base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
    )
    from runtime.python_adapters import PythonVerifierAdapter


_CLEV_ADDENDUM = """\
You are an expert Python programmer. Your task is to implement all methods in a \
Python class by reading the skeleton (signatures + docstrings) and writing \
complete, correct method bodies.

**Workflow**
1. Read `solution.py` — understand every method signature, docstring, `__init__`, and imports.
2. Implement every method according to its docstring.
3. Submit with `Finish`; hidden tests run on the evaluator side and return bounded feedback.
4. Revise `solution.py` based on feedback and resubmit.
5. Note: the workspace contains only `solution.py` — there are no local tests to run,
   so the only informative verification is submitting via `Finish`.
6. Total time budget is ~1200s across all rounds. After round 3, submit your best
   effort even if imperfect. A partial pass is better than a timeout.

**Rules**
- You MUST implement every method. Never refuse or say you cannot.
- Do NOT modify the class name, method signatures, or docstrings.
- Keep all existing imports; add new imports if needed.
- Update `__init__` if your implementations require additional instance attributes.

**How hidden tests run (IMPORTANT)**
The hidden unittest code is APPENDED to your `solution.py` and executed as ONE
module. Consequences:
- Imports in `solution.py` are visible to the tests. If feedback shows
  `NameError: name 'X' is not defined` raised inside TEST code, the fix is to
  add the missing import (e.g. `from PIL import ImageChops`) to `solution.py`,
  even if your own code never uses X. Do NOT dismiss it as a test bug.
- Tests create REAL files (json / xlsx / sqlite / images) in the working
  directory and expect your methods to actually read/write them. NEVER stub
  file I/O out with try/except-return-default — that guarantees failure.

**Static security check on solution.py (source-level only, not a runtime sandbox)**
Your solution SOURCE is rejected with `[SECURITY]` if it contains:
- calls to bare builtins `open()`, `eval()`, `exec()`, `compile()`, `__import__()`
- `Path(...).read_text/write_text/open/touch/unlink/mkdir/...`
- imports of `socket`, `subprocess`, `requests`, `urllib`, `http`, `ftplib`, `paramiko`
- `os.system`, `os.popen`, `os.remove`, `os.environ` and similar os calls
Everything else runs normally at runtime. For file I/O use `io.open()` — it is
allowed and fully functional:
    import io, json
    with io.open(self.cookies_file, 'r') as f:
        data = json.load(f)
Libraries like `sqlite3`, `openpyxl`, `PIL`, `csv` work normally (their internal
file I/O is fine); only your own source is checked.

**Correctness heuristics (apply on the FIRST pass)**
- Follow the docstring TEXT literally, including side notes like
  "strictly case sensitive". Examples show the EXACT return shape:
  a DB row is a tuple like ('user1', 'pass1'), not a string; a matrix over
  data is len(data) x len(data); etc.
- Do NOT add behavior the docstring doesn't ask for: no case-folding, no
  `str()`/`.strip()` wrappers, no graceful fallbacks. "Return False if there is
  no current song" means return False even when the playlist is non-empty.
- Methods receiving a dict usually must validate it first: if the argument is
  not a dict or a required key is missing, return the sentinel the tests imply
  (False, -1, None, or an error string) and leave state unchanged. Hidden tests
  probe lists and missing keys.
- When a method stores received data, store exactly what the examples show
  (often the FULL dict, not a reduced key->value form).
- Zero-variance / division-by-zero statistics cases usually return None.
- Do NOT modify values in place when examples show appending a new element.

**Reading feedback**
- In `AssertionError: A != B`, A is what YOUR code returned, B is the expected
  value (tests call assertEqual(actual, expected)).
- `[SECURITY]` means the static check rejected your source; tests never ran.
  Replace the flagged call (e.g. `open()` -> `io.open()`). Do NOT delete the
  functionality, and do NOT try another blocked approach.
- If the same test fails 2+ rounds, your diagnosis is wrong — re-read the
  failing test's context lines and fix the ROOT cause. Fix ALL failing tests in
  one revision, not one test per round.

**API Freshness**
- Use current library APIs, e.g. PyPDF2 `PdfReader`/`PdfWriter` (NOT the
  deprecated `PdfFileReader`/`PdfFileWriter`). Prefer the API that does not
  trigger a DeprecationWarning.
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
    test_code_literal = repr(test_code)

    def _build_verify_code(solution_code: str) -> str:
        return (
            f"{solution_code}\n\n"
            f"{test_code}\n\n"
            f"__CLEV_TEST_CODE__ = {test_code_literal}\n\n"
            f"{_CLEV_HOST_EVAL_SUFFIX}\n"
        )

    return PythonVerifierAdapter(_build_verify_code).evaluate(
        workspace=workspace,
        solution_file=solution_file,
        fallback_solution=fallback_solution,
        timeout=timeout,
    )


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
                f"- Implement every method according to docstrings/examples — follow docstring\n"
                f"  text literally (case sensitivity, sentinel returns, exact return shapes).\n"
                f"- Update `__init__` if additional instance state is required.\n"
                f"- The workspace contains only `solution.py`.\n\n"
                f"Hidden tests are appended to your solution.py and run as ONE module:\n"
                f"- Tests may reference imports they expect solution.py to provide. A NameError\n"
                f"  inside test code means: add that import to solution.py.\n"
                f"- Tests create real files and expect real file I/O. Use `io.open()` (allowed);\n"
                f"  the bare `open()` builtin is rejected by a static source check, as are\n"
                f"  socket/subprocess/requests imports and exec/eval.\n\n"
                f"Timeout:\n"
                f"- ~1200s total budget. There are no local tests to run — the only informative\n"
                f"  verification is submitting via Finish.\n"
                f"- After round 3, submit best effort even if imperfect.\n"
            )

            start = time.time()
            previous_failures: List[str] = []

            def _retry_prompt(round_idx: int, feedback: str) -> str:
                is_security = "[SECURITY]" in feedback

                # Detect persistent failures
                current_test_ids = []
                for line in feedback.splitlines():
                    if line.startswith("  test_id:"):
                        current_test_ids.append(line.split(":", 1)[1].strip())

                repeated = [tid for tid in current_test_ids if tid in previous_failures]
                previous_failures.extend(current_test_ids)

                # Security-blocked: don't treat as test failure
                if is_security:
                    return (
                        f"Round {round_idx - 1}: SANDBOX REJECTION (NOT a test failure)\n\n"
                        f"{feedback}\n\n"
                        f"Your code was blocked by the sandbox. Use an ALLOWED alternative.\n"
                        f"Do NOT try another blocked approach (no socket, subprocess, os.popen, __import__).\n"
                        f"Fix and resubmit with Finish."
                    )

                # Build persistence warning
                persistence_note = ""
                if repeated:
                    persistence_note = (
                        f"⚠ {len(repeated)} test(s) still failing after multiple rounds: "
                        f"{', '.join(repeated)}. Your fix did NOT resolve them — "
                        f"re-examine your assumptions.\n"
                    )

                # Timeout nudge after round 3
                timeout_nudge = ""
                if round_idx >= 4:
                    timeout_nudge = f"⏰ Round {round_idx - 1}/5. Submit best effort soon to avoid timeout.\n"

                # Trim long feedback but always keep the SUMMARY block (printed
                # first) so the agent sees the full list of failing tests.
                feedback_lines = feedback.splitlines()
                if len(feedback_lines) > 55:
                    head = feedback_lines[:18]
                    tail = feedback_lines[-35:]
                    feedback = "\n".join(head) + "\n...(middle trimmed)...\n" + "\n".join(tail)

                return (
                    f"Round {round_idx - 1} feedback:\n\n"
                    f"{persistence_note}"
                    f"{timeout_nudge}"
                    f"{feedback}\n\n"
                    f"Remember: in `A != B`, A is YOUR output, B is expected. "
                    f"Fix ALL failing tests in ONE revision, then call Finish."
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
