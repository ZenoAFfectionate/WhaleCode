"""ClassEval benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .base import BenchmarkRunner, _PROJECT_ROOT


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

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        skeleton = task["skeleton"]
        test_code = task["test"]
        class_name = task["class_name"]

        workspace = Path(tempfile.mkdtemp(prefix=f"clev_{task_id}_"))
        try:
            # Write the class skeleton for the agent to complete
            solution_file = workspace / "solution.py"
            solution_file.write_text(skeleton, encoding="utf-8")

            # Write the test harness (agent can optionally read it)
            test_file = workspace / "tests.py"
            test_file.write_text(test_code, encoding="utf-8")

            # Run the agent
            agent = self._create_agent(workspace)
            agent_prompt = (
                f"Your task is to implement all methods in the Python class `{class_name}` "
                f"defined in `solution.py`.\n\n"
                f"Follow these steps:\n"
                f"1. Read `solution.py` to understand the class skeleton — method signatures, "
                f"docstrings, and any existing `__init__` logic.\n"
                f"2. Implement every method according to its docstring. Write complete, "
                f"correct method bodies using the Edit or Write tool.\n"
                f"3. Run `python tests.py` via Bash to verify your implementation.\n"
                f"4. If any tests fail, read the error output, fix the code, and re-run "
                f"until all tests pass.\n\n"
                f"Rules:\n"
                f"- Do NOT change the class name, method signatures, or docstrings.\n"
                f"- Keep existing imports; add more if needed.\n"
                f"- Update `__init__` if your methods require additional instance attributes.\n"
                f"- You may import any standard library module or commonly-available "
                f"third-party package (e.g. `docx`, `openpyxl`, `nltk`, `PyPDF2`).\n"
            )

            start = time.time()
            try:
                agent_response = agent.run(agent_prompt)
            except Exception as exc:
                return {
                    "task_id": task_id,
                    "passed": False,
                    "error": f"Agent error: {exc}",
                    "agent_response": "",
                    "elapsed_s": round(time.time() - start, 2),
                }
            elapsed = round(time.time() - start, 2)

            # Read the (possibly modified) solution
            solution_code = solution_file.read_text(encoding="utf-8") if solution_file.exists() else skeleton

            # Build the verification script:
            #   solution code  (defines the class)
            #   + test code    (unittest.TestCase subclasses that reference the class)
            #   + unittest.main()
            verify_code = (
                f"{solution_code}\n\n"
                f"{test_code}\n\n"
                f"if __name__ == '__main__':\n"
                f"    unittest.main()\n"
            )

            verify_script = workspace / "verify.py"
            verify_script.write_text(verify_code, encoding="utf-8")

            passed, output = self._run_script_in_sandbox(verify_script, cwd=workspace)

            return {
                "task_id": task_id,
                "passed": passed,
                "error": output if not passed else None,
                "agent_response": (agent_response or "")[:500],
                "elapsed_s": elapsed,
            }
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run ClassEval benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "CLEV" / "test.jsonl"),
        help="Path to ClassEval JSONL file",
    )
    parser.add_argument("--output-dir", default=str(_PROJECT_ROOT / "data" / "_results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = ClassEvalBenchmark(
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
