"""AIME benchmark runner for Whale Code agent."""

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

try:
    from .base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT
except ImportError:
    from base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT


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

    # Template for solution.py — includes a sanity-check wrapper
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

    def _get_system_prompt(self):
        return self._MATH_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    task = json.loads(line)
                    task.setdefault("task_id", f"AIME_{task.get('id', 0)}")
                    tasks.append(task)
        return tasks

    @staticmethod
    def _extract_answer(output: str) -> Optional[int]:
        """Extract the last integer printed by the solution."""
        # Find all integers in the output (including negative)
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

        workspace = Path(tempfile.mkdtemp(prefix=f"aime_{task_id}_"))
        try:
            # Pre-populate solution.py with sanity-check template
            solution_file = workspace / "solution.py"
            solution_file.write_text(self._SOLUTION_TEMPLATE, encoding="utf-8")

            # Run the agent
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
            try:
                agent_response = agent.run(agent_prompt)
            except Exception as exc:
                return {
                    "task_id": task_id,
                    "passed": False,
                    "expected": expected_answer,
                    "actual": None,
                    "error": f"Agent error: {exc}",
                    "agent_response": "",
                    "elapsed_s": round(time.time() - start, 2),
                }
            elapsed = round(time.time() - start, 2)

            # Run the solution
            if not solution_file.exists():
                return {
                    "task_id": task_id,
                    "passed": False,
                    "expected": expected_answer,
                    "actual": None,
                    "error": "solution.py not found after agent run",
                    "agent_response": (agent_response or "")[:500],
                    "elapsed_s": elapsed,
                }

            success, output = self._run_script_in_sandbox(
                solution_file, cwd=workspace, timeout=self.timeout
            )

            actual_answer = self._extract_answer(output) if success else None
            passed = actual_answer == expected_answer

            return {
                "task_id": task_id,
                "passed": passed,
                "expected": expected_answer,
                "actual": actual_answer,
                "error": output if not passed else None,
                "agent_response": (agent_response or "")[:500],
                "elapsed_s": elapsed,
            }
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run AIME benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "AIME" / "test.jsonl"),
        help="Path to AIME JSONL file",
    )
    parser.add_argument("--output-dir", default=str(_PROJECT_ROOT / "data" / "_results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120, help="Longer timeout for math computations")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
    parser.add_argument("--resume", default=None, help="Resume from a previous .jsonl results file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = AIMEBenchmark(
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
