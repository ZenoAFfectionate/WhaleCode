"""HumanEval+ benchmark runner for Whale Code agent."""

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


_HEVP_ADDENDUM = """\
You are an expert Python programmer. Your task is to implement Python functions \
correctly by reading the provided signature and docstring, then writing the body.

**Workflow**
1. Read `solution.py` — understand the function signature, docstring, and examples.
2. Implement the function body using Edit or Write.
3. Submit with `Finish`; hidden tests run outside the workspace and return bounded feedback.
4. Revise `solution.py` and resubmit with `Finish` until done.

**Rules**
- You MUST implement the function. Never refuse or say you cannot.
- Use tools to inspect and modify the workspace, then call `Finish` once you are ready for evaluation.
- Do NOT modify the function signature, parameter names, or docstring.
- Keep all existing imports; add new imports only if necessary.
- Write clean, correct, and efficient code. Prefer simple solutions.
- When feedback arrives, focus on understanding WHY it failed before changing code. \
Read the feedback carefully — do not guess blindly.
- The workspace contains only `solution.py`. There are no local benchmark tests to run.
- NEVER attempt to read, inspect, import, or access hidden test data in any way. \
Do NOT read environment variables to find test paths. Do NOT try to access files \
outside the workspace. Hidden evaluation happens only in the runner.
"""

_HEVP_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## HumanEval+ Benchmark Override\n\n"
    + _HEVP_ADDENDUM
)


# ---------------------------------------------------------------------------
# Host-side test instrumentation for richer but bounded hidden-test feedback.
# ---------------------------------------------------------------------------
def _instrument_test_code(test_code: str, entry_point: str) -> str:
    """Rewrite check() so hidden-test failures are fully accumulated and reported.

    Mechanisms:
    1. Monkey-patch ``assertion()`` to raise a custom exception carrying
       (actual, expected) for consistent diagnostics.
    2. Wrap each assertion/assert in check() loops with try/except and
       collect failure details instead of exiting on the first mismatch.
    """
    import re

    # Inject a custom exception and patched assertion at the top of the test code.
    patch = '''
import numpy as np
import sys

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

_HEVP_FAILURES = []

def _record_failure(test_idx, inp, *, expected=None, actual=None, error=None):
    _HEVP_FAILURES.append(
        {
            "test_idx": test_idx,
            "input": repr(inp),
            "expected": repr(expected) if expected is not None else None,
            "actual": repr(actual) if actual is not None else None,
            "error": str(error) if error is not None else None,
        }
    )

def _report_failures():
    if not _HEVP_FAILURES:
        return 0
    for item in _HEVP_FAILURES:
        print(f"FAILED test #{item['test_idx']}", file=sys.stderr)
        print(f"  Input: {item['input']}", file=sys.stderr)
        if item["error"] is not None:
            print(f"  Error: {item['error']}", file=sys.stderr)
            if item["actual"] is not None:
                print(f"  Your function returned: {item['actual']}", file=sys.stderr)
        else:
            print(f"  Expected: {item['expected']}", file=sys.stderr)
            print(f"  Actual:   {item['actual']}", file=sys.stderr)
    print(f"{len(_HEVP_FAILURES)} failing test(s)", file=sys.stderr)
    return len(_HEVP_FAILURES)
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
                    out.append(f"{inner}_record_failure({idx}, inp, expected=_e.exp, actual=_e.out)")
                else:
                    out.append(f"{body_indent}except Exception as _e:")
                    out.append(f"{inner}try:")
                    out.append(f"{inner}    _ret = candidate(*inp)")
                    out.append(f"{inner}    _ret_r = repr(_ret)")
                    out.append(f"{inner}except Exception as _re:")
                    out.append(f"{inner}    _ret_r = f'<raised {{type(_re).__name__}}>'")
                    out.append(f"{inner}_record_failure({idx}, inp, error=f\"{{type(_e).__name__}}: {{_e}}\", actual=_ret_r)")
                i += 1
                continue

        out.append(line)
        i += 1

    return "\n".join(out)

def _evaluate_solution(
    workspace: Path,
    solution_file: Path,
    fallback_solution: str,
    entry_point: str,
    test_code: str,
    timeout: int,
) -> tuple[bool, str]:
    instrumented_test = _instrument_test_code(test_code, entry_point)

    def _build_verify_code(solution_code: str) -> str:
        return (
            f"{solution_code}\n\n"
            f"{instrumented_test}\n\n"
            f"try:\n"
            f"    check({entry_point})\n"
            f"except Exception:\n"
            f"    _report_failures()\n"
            f"    import traceback as _tb\n"
            f"    _tb.print_exc()\n"
            f"    raise\n"
            f"_fail_count = _report_failures()\n"
            f"if _fail_count:\n"
            f"    raise SystemExit(1)\n"
            f"print('All hidden tests passed!')\n"
        )

    return PythonVerifierAdapter(_build_verify_code).evaluate(
        workspace=workspace,
        solution_file=solution_file,
        fallback_solution=fallback_solution,
        timeout=timeout,
    )


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

    def __init__(self, *args, max_submission_rounds: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_submission_rounds = max(1, int(max_submission_rounds))

    def _get_system_prompt(self) -> str:
        return _HEVP_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        return self._load_jsonl_tasks()

    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        test_code = task["test"]

        workspace = self._make_workspace(f"hevp_{task_id.replace('/', '_')}_")
        agent = None
        agent_response = ""
        prompt_history: List[str] = []
        result: Optional[Dict[str, Any]] = None
        try:
            solution_file = workspace / "solution.py"
            solution_file.write_text(prompt, encoding="utf-8")

            agent = self._create_agent(workspace)
            initial_prompt = (
                f"Implement the function `{entry_point}` in `solution.py`.\n\n"
                f"Requirements:\n"
                f"- Do NOT change the function signature or docstring.\n"
                f"- Handle edge cases (empty inputs, boundary values, etc.).\n"
                f"- Hidden tests are evaluated only by the runner after each `Finish`.\n"
                f"- Do not create your own uncontrolled benchmark test loop.\n"
                f"- Only `solution.py` exists in the workspace.\n"
            )

            start = time.time()
            def _retry_prompt(round_idx: int, feedback: str) -> str:
                return (
                    f"Controlled hidden-test feedback for submission round {round_idx - 1}:\n\n"
                    f"{feedback}\n\n"
                    f"Revise `solution.py` based on this feedback.\n"
                    f"- The failing hidden test index is reliable.\n"
                    f"- The hidden-test diagnostics include all discovered failures.\n"
                    f"- Use this feedback to reason about edge cases and logic errors, then call `Finish` alone again with a brief summary of the revision.\n"
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
                    fallback_solution=prompt,
                    entry_point=entry_point,
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
                extra={"entry_point": entry_point},
            )


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run HumanEval+ benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "HEVP" / "test.jsonl"),
        help="Path to HumanEvalPlus JSONL file",
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

    bench = HumanEvalPlusBenchmark(
        data_path=args.data_path,
        max_submission_rounds=args.max_submission_rounds,
        **BenchmarkRunner.runner_kwargs_from_args(args, include_task_timeout=True),
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
