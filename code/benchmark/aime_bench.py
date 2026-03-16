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

from .base import BenchmarkRunner, _PROJECT_ROOT


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

    _MATH_SYSTEM_PROMPT = (
        "You are a mathematical problem-solving agent. You solve competition-level "
        "math problems by combining mathematical reasoning with Python computation.\n\n"
        "**Output Format (STRICT)**\n"
        "- Use OpenAI function calling for tools. Do NOT emit tool calls in plain text.\n"
        "- If you need a tool, call it via tool_calls only.\n\n"
        "# Workflow\n"
        "1. Analyze the problem — identify the mathematical domain (number theory, "
        "combinatorics, geometry, algebra, probability) and the key techniques.\n"
        "2. Use `Bash` to run `python3 -c \"...\"` for exploratory computations, "
        "testing conjectures, and verifying intermediate results.\n"
        "3. Write the final solution in `solution.py` using `Write` or `Edit`.\n"
        "4. Run `python3 solution.py` via `Bash` to verify the output.\n"
        "5. When done, call `Finish` with a concise summary of your answer.\n\n"
        "# Available Python Libraries\n"
        "You may freely use: `sympy`, `math`, `fractions`, `itertools`, "
        "`functools`, `collections`, `numpy`, `decimal`, `cmath`.\n\n"
        "# Tool Calling Rules\n"
        "1. Use `tool_calls` only; do not output Action/ToolName text.\n"
        "2. Arguments must be valid JSON.\n"
        "3. Make independent tool calls in parallel when possible.\n\n"
        "## Available Tools\n\n{tools}"
    )

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
            # Create an empty solution file
            solution_file = workspace / "solution.py"
            solution_file.write_text(
                "# Write your solution here\n",
                encoding="utf-8",
            )

            # Run the agent
            agent = self._create_agent(workspace)
            agent_prompt = (
                f"Solve this AIME (American Invitational Mathematics Examination) problem. "
                f"The answer is always an integer from 000 to 999.\n\n"
                f"**Problem:**\n{problem}\n\n"
                f"**Steps:**\n"
                f"1. Reason about the problem mathematically. Identify the domain "
                f"(number theory, combinatorics, geometry, algebra, probability) and "
                f"plan your approach.\n"
                f"2. Use `Bash` with `python3 -c \"...\"` to explore and test ideas. "
                f"You have: sympy, itertools, math, fractions, numpy, collections.\n"
                f"3. Write the final solution in `solution.py` — it must print exactly "
                f"one integer (the answer, nothing else).\n"
                f"4. Run `python3 solution.py` via Bash to verify the output is correct "
                f"and in range 0–999.\n"
                f"5. If the output looks wrong, revise and re-test.\n"
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
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=60, help="Longer timeout for math computations")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
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
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
