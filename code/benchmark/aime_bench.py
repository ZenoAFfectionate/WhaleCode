"""AIME benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from .base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT
except ImportError:
    from base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT


_VALID_AIME_YEARS = {"24", "25", "26"}


def _normalize_year(year: Optional[str]) -> Optional[str]:
    if year is None:
        return None
    value = str(year).strip()
    if len(value) == 4 and value.startswith("20"):
        value = value[2:]
    if value not in _VALID_AIME_YEARS:
        raise ValueError(f"Unsupported AIME year: {year!r}. Expected one of 24, 25, 26.")
    return value


def _infer_year_from_path(data_path: Path) -> Optional[str]:
    match = re.search(r"test_(\d{2})\.jsonl$", str(data_path))
    if not match:
        return None
    year = match.group(1)
    return year if year in _VALID_AIME_YEARS else None


def _resolve_data_path(year: Optional[str], data_path: Optional[str]) -> tuple[Path, Optional[str]]:
    normalized_year = _normalize_year(year)

    if data_path:
        path = Path(data_path)
        inferred_year = _infer_year_from_path(path)
        if normalized_year and inferred_year and normalized_year != inferred_year:
            raise ValueError(
                f"Year mismatch: --year {normalized_year} does not match data path {path.name}."
            )
        return path, normalized_year or inferred_year

    effective_year = normalized_year or "24"
    path = _PROJECT_ROOT / "data" / "AIME" / f"test_{effective_year}.jsonl"
    return path, effective_year


class AIMEBenchmark(BenchmarkRunner):
    """Evaluate the agent on AIME (math competition problems).

    Workflow per task:
    1. Create a temp workspace.
    2. Present the math problem and ask the agent to write a Python
       program in ``solution.py`` that prints the final integer answer.
    3. Execute ``solution.py`` in a sandbox.
    4. Extract the last integer from stdout and compare with the
       expected answer.
    5. Record pass / fail.
    """

    benchmark_name = "aime"

    _AIME_ADDENDUM = """\
You are a mathematical problem-solving agent. You solve competition-level math \
problems (AIME) by combining careful mathematical reasoning with Python computation.

AIME answers are always integers from 000 to 999.

# Core Principle: Think First, Code Second

Before writing any code, you MUST reason about the problem mathematically using \
the Thought tool. Identify the domain, key constraints, and a clear solution \
strategy. Only then write code to execute your plan.

# Workflow

1. **Analyze** — Use Thought to identify the mathematical domain (number theory, \
combinatorics, geometry, algebra, probability) and outline your approach. This is \
the most important step.
2. **Explore** (optional) — Use `Bash` with `python3 -c "..."` for quick \
computations to test conjectures or verify small cases. Keep explorations focused \
(at most 2-3 attempts).
3. **Solve** — Write your solution in `solution.py`. The script must print exactly \
one integer (the answer) as its only output. No debug prints.
4. **Verify** — Run `python3 solution.py` via Bash. The output must be a single \
integer in 0-999. If it is not, your approach is wrong — go back to step 1.
5. **Finish** — Call `Finish` with your answer. You MUST run solution.py before \
calling Finish.

# Strategy Guidelines

- **Think before brute-forcing.** If values explode or computation is slow, switch \
to a smarter approach (modular arithmetic, closed-form, generating functions).
- **Verify on small cases first.** Before scaling up, check your formula works on \
cases you can compute by hand.
- **Sequences & Recurrences**: Look for periodicity or closed-form solutions. Use \
modular arithmetic if values grow large.
- **Counting**: Use inclusion-exclusion or generating functions for large cases. \
Verify formula on small n first.
- **Geometry**: Use coordinate geometry or sympy for exact symbolic computation.
- **Number Theory**: Work modulo the target. Use CRT, Fermat's little theorem, or \
sympy.ntheory.
- **Sanity check**: AIME answers are 0-999. If your answer is outside this range, \
your approach is fundamentally wrong. Do not force it — rethink.

# Available Libraries

sympy, math, fractions, itertools, functools, collections, numpy, decimal, cmath.
"""

    _MATH_SYSTEM_PROMPT = (
        BENCHMARK_BASE_SYSTEM_PROMPT
        + "\n\n---\n\n## AIME Benchmark Override\n\n"
        + _AIME_ADDENDUM
    )

    _SOLUTION_TEMPLATE = """\
# AIME Solution — answer must be an integer from 0 to 999.
# Write your solution below. Assign the final answer to `answer`.
# The sanity check at the bottom will print it.

answer = None  # Replace with your computed answer


# --- Sanity check (do not remove) ---
if answer is None:
    raise ValueError("No answer computed — set the `answer` variable.")
answer = int(answer)
if not (0 <= answer <= 999):
    raise ValueError(f"Answer {answer} is outside AIME range 0-999. Rethink your approach.")
print(answer)
"""

    def __init__(self, *args, year: Optional[str] = None, **kwargs):
        self.year = _normalize_year(year)
        super().__init__(*args, **kwargs)
        if self.year is None:
            self.year = _infer_year_from_path(self.data_path)
        if self.year is not None:
            self.benchmark_name = f"aime_{self.year}"

    def _get_system_prompt(self):
        return self._MATH_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        prefix = f"AIME_{self.year}" if self.year else "AIME"
        return self._load_jsonl_tasks(
            task_transform=lambda task: {
                **task,
                "task_id": task.get("task_id") or f"{prefix}_{task.get('id', 0)}",
            }
        )

    @staticmethod
    def _extract_answer(output: str) -> Optional[int]:
        """Extract the last integer printed by the solution."""
        integers = re.findall(r"-?\d+", output)
        if integers:
            try:
                return int(integers[-1])
            except ValueError:
                return None
        return None

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        problem = task["problem"]
        expected_answer = int(task["answer"])

        workspace = self._make_workspace(f"aime_{task_id}_")
        agent = None
        agent_response = ""
        agent_prompt = ""
        result: Optional[Dict[str, Any]] = None
        try:
            solution_file = workspace / "solution.py"
            solution_file.write_text(self._SOLUTION_TEMPLATE, encoding="utf-8")

            agent = self._create_agent(workspace)
            agent_prompt = (
                f"Solve this AIME problem. The answer is an integer from 0 to 999.\n\n"
                f"**Problem:**\n{problem}\n\n"
                f"**Instructions:**\n"
                f"1. First, use Thought to analyze the problem mathematically. "
                f"Identify the domain and plan your approach before writing any code.\n"
                f"2. Optionally use `python3 -c \"...\"` via Bash to explore ideas "
                f"(sympy, itertools, numpy are available). Keep this to 2-3 attempts.\n"
                f"3. Edit `solution.py` — set the `answer` variable to your computed "
                f"result. The template already handles printing and sanity checking.\n"
                f"4. Run `python3 solution.py` via Bash. If it raises a ValueError, "
                f"your answer is wrong — rethink your approach.\n"
                f"5. Call `Finish` with your answer.\n"
            )

            start = time.time()
            agent_response, error_result = self._run_agent_prompt(
                agent=agent,
                task_id=task_id,
                prompt_text=agent_prompt,
                start_time=start,
                error_extra={"expected": expected_answer, "actual": None},
            )
            if error_result is not None:
                result = error_result
                return result
            elapsed = round(time.time() - start, 2)

            if not solution_file.exists():
                result = self._missing_output_result(
                    task_id,
                    path_label="solution.py",
                    elapsed_s=elapsed,
                    agent_response=agent_response,
                    extra={"expected": expected_answer, "actual": None},
                )
                return result

            success, output = self._run_script_in_sandbox(
                solution_file, cwd=workspace, timeout=self.timeout
            )

            actual_answer = self._extract_answer(output) if success else None
            passed = actual_answer == expected_answer

            result = self._build_result(
                task_id,
                passed=passed,
                error=output if not passed else None,
                agent_response=agent_response,
                elapsed_s=elapsed,
                extra={
                    "expected": expected_answer,
                    "actual": actual_answer,
                },
            )
            return result
        finally:
            self._save_task_trajectory(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=[agent_prompt] if agent_prompt else [],
                result=result,
                artifact_paths=["solution.py"],
                extra={"expected_answer": expected_answer},
            )
            shutil.rmtree(workspace, ignore_errors=True)


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run AIME benchmark")
    parser.add_argument(
        "--year",
        default=None,
        help="AIME year to run: 24, 25, or 26. If omitted, defaults to 24 unless --data-path is set.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to AIME JSONL file. If omitted, resolves to data/AIME/test_<year>.jsonl.",
    )
    BenchmarkRunner.add_shared_run_args(
        parser,
        default_temperature=1.0,
        default_max_steps=128,
        default_timeout=120,
        timeout_help="Longer timeout for math computations",
    )
    args = parser.parse_args()

    try:
        data_path, effective_year = _resolve_data_path(args.year, args.data_path)
    except ValueError as exc:
        parser.error(str(exc))

    if not data_path.exists():
        parser.error(f"AIME data file not found: {data_path}")

    bench = AIMEBenchmark(
        year=effective_year,
        data_path=str(data_path),
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_steps=args.max_steps,
        timeout=args.timeout,
        trajectory_dir=args.trajectory_dir,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
