"""HumanEval+ benchmark runner for Whale Code agent."""

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


_HEVP_ADDENDUM = """\
You are an expert Python programmer. Your task is to implement Python functions \
correctly by reading the provided signature and docstring, then writing the body.

**Workflow**
1. Read `solution.py` — understand the function signature, docstring, and examples.
2. Implement the function body using Edit or Write.
3. When ready, respond with a short plain-text summary of the current implementation.
4. The benchmark runner will execute hidden tests outside the workspace and send \
back controlled feedback if another revision is needed.
5. Revise `solution.py` based on that feedback and respond again.

**Rules**
- You MUST implement the function. Never refuse or say you cannot.
- Use tools to inspect and modify the workspace, then give a normal text response once you are ready for evaluation.
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
                    out.append(f"{inner}_inp_r = repr(inp)[:200]")
                    out.append(f"{inner}_out_r = repr(_e.out)[:200]")
                    out.append(f"{inner}_exp_r = repr(_e.exp)[:200]")
                    out.append(f'{inner}print(f"FAILED test #{{{idx}}}", file=_s.stderr)')
                    out.append(f'{inner}print(f"  Input: {{_inp_r}}", file=_s.stderr)')
                    out.append(f'{inner}print(f"  Expected: {{_exp_r}}", file=_s.stderr)')
                    out.append(f'{inner}print(f"  Actual:   {{_out_r}}", file=_s.stderr)')
                    out.append(f"{inner}_s.exit(1)")
                else:
                    out.append(f"{body_indent}except Exception as _e:")
                    out.append(f"{inner}import sys as _s")
                    out.append(f"{inner}_inp_r = repr(inp)[:200]")
                    out.append(f"{inner}try:")
                    out.append(f"{inner}    _ret = candidate(*inp)")
                    out.append(f"{inner}    _ret_r = repr(_ret)[:200]")
                    out.append(f"{inner}except Exception as _re:")
                    out.append(f"{inner}    _ret_r = f'<raised {{type(_re).__name__}}>'")
                    out.append(f'{inner}print(f"FAILED test #{{{idx}}}", file=_s.stderr)')
                    out.append(f'{inner}print(f"  Input: {{_inp_r}}", file=_s.stderr)')
                    out.append(f'{inner}print(f"  Error: {{type(_e).__name__}}: {{_e}}", file=_s.stderr)')
                    out.append(f'{inner}print(f"  Your function returned: {{_ret_r}}", file=_s.stderr)')
                    out.append(f"{inner}_s.exit(1)")
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
    solution_code = solution_file.read_text(encoding="utf-8") if solution_file.exists() else fallback_solution
    instrumented_test = _instrument_test_code(test_code, entry_point)
    verify_code = (
        f"{solution_code}\n\n"
        f"{instrumented_test}\n\n"
        f"check({entry_point})\n"
        f"print('All hidden tests passed!')\n"
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

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
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
                f"Submission policy:\n"
                f"- Hidden tests are evaluated only by the benchmark runner.\n"
                f"- Do not create your own uncontrolled benchmark loop.\n"
                f"- Each time you finish with a normal text response, the runner will execute hidden tests and send bounded feedback if needed.\n\n"
                f"Steps:\n"
                f"1. Read `solution.py` to see the signature, docstring, and examples.\n"
                f"2. Implement the function body using Edit or Write.\n"
                f"3. Perform lightweight self-checks if useful, but do not rely on local benchmark tests.\n"
                f"4. When you want a controlled submission, stop and provide a brief plain-text summary.\n\n"
                f"Important:\n"
                f"- Do NOT change the function signature or docstring.\n"
                f"- The function must handle edge cases (empty inputs, boundary values, etc.).\n"
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
                    f"- The failing hidden test index is reliable.\n"
                    f"- The input/expected/actual previews are intentionally truncated.\n"
                    f"- Use this feedback to reason about edge cases and logic errors, then respond again with a brief plain-text summary.\n"
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
                    fallback_solution=prompt,
                    entry_point=entry_point,
                    test_code=test_code,
                    timeout=self.timeout,
                )
                if passed:
                    break
                feedback = truncate_feedback(output, max_lines=60, max_chars=10000)

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

    parser = argparse.ArgumentParser(description="Run HumanEval+ benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "HEVP" / "test.jsonl"),
        help="Path to HumanEvalPlus JSONL file",
    )
    BenchmarkRunner.add_shared_run_args(
        parser,
        default_temperature=1.0,
        default_max_steps=32,
        default_timeout=60,
    )
    parser.add_argument("--max-submission-rounds", type=int, default=5)
    args = parser.parse_args()

    bench = HumanEvalPlusBenchmark(
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
