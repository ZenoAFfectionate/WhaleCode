"""ClassEval benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from .base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT
except ImportError:
    from base import BenchmarkRunner, BENCHMARK_BASE_SYSTEM_PROMPT, _PROJECT_ROOT


_CLEV_ADDENDUM = """\
You are an expert Python programmer. Your task is to implement all methods in a \
Python class by reading the skeleton (signatures + docstrings) and writing \
complete, correct method bodies.

**Workflow**
1. Read `solution.py` — understand the class skeleton: every method signature, \
docstring, `__init__`, and existing imports.
2. Implement every method according to its docstring using Edit or Write.
3. Run `python3 tests.py` via Bash to verify against the test suite.
4. If tests fail, carefully analyze the error output, fix the code, and re-run \
`python3 tests.py`. Repeat until all tests pass.
5. Once all tests pass, call `Finish` with a brief summary.

**Rules**
- You MUST implement every method. Never refuse or say you cannot.
- Always use tools to take action — do NOT respond with text only.
- Do NOT modify the class name, method signatures, or docstrings.
- Keep all existing imports; add new imports only if necessary.
- Update `__init__` if your implementations require additional instance attributes.
- Write clean, correct, and efficient code. Prefer simple solutions.
- When tests fail, focus on understanding WHY they fail before changing code. \
Read the test name and error message carefully — do not guess blindly.
- If you have tried the same fix multiple times without progress, reconsider \
your approach from scratch.
- The workspace contains only `solution.py` and `tests.py`. There are no other \
files to read.
"""

_CLEV_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## ClassEval Benchmark Override\n\n"
    + _CLEV_ADDENDUM
)


_CLEV_TESTS_PY_WRAPPER = """\
import sys, os, unittest, inspect, re
sys.path.insert(0, os.environ["_HIDDEN_TEST_DIR"])

from solution import *
from _test_data import *


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
        return "\\n".join(context)
    except Exception:
        return ""


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TestResult()
    suite.run(result)

    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)

    if result.failures:
        for test, tb in result.failures:
            ctx = _extract_context(test)
            print(f"[FAIL] {test}")
            if ctx:
                print(ctx)
            print(tb)

    if result.errors:
        for test, tb in result.errors:
            ctx = _extract_context(test)
            print(f"[ERROR] {test}")
            if ctx:
                print(ctx)
            print(tb)

    print(f"{total - failed - errors}/{total} passed")
    if not result.failures and not result.errors:
        print("All tests passed!")

    sys.exit(0 if not result.failures and not result.errors else 1)
"""


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

    def _get_system_prompt(self) -> str:
        return _CLEV_SYSTEM_PROMPT

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
        hidden_dir = Path(tempfile.mkdtemp(prefix=f"clev_{task_id}_hidden_"))
        try:
            # solution.py — the only file the agent needs to edit
            solution_file = workspace / "solution.py"
            solution_file.write_text(skeleton, encoding="utf-8")

            # _test_data.py — stored outside workspace (agent cannot access)
            (hidden_dir / "_test_data.py").write_text(
                f"from solution import *\n\n"
                f"{test_code}\n",
                encoding="utf-8",
            )

            # tests.py — lightweight wrapper
            (workspace / "tests.py").write_text(_CLEV_TESTS_PY_WRAPPER, encoding="utf-8")

            # Set env vars for the hidden dir and PYTHONPATH
            import os
            os.environ["_HIDDEN_TEST_DIR"] = str(hidden_dir)
            os.environ["PYTHONPATH"] = str(workspace)

            # Run the agent
            agent = self._create_agent(workspace)
            agent_prompt = (
                f"Implement all methods in the class `{class_name}` in `solution.py`.\n\n"
                f"Steps:\n"
                f"1. Read `solution.py` to understand the class skeleton — method signatures, "
                f"docstrings, and `__init__`.\n"
                f"2. Implement every method according to its docstring. Update `__init__` "
                f"if you need additional instance attributes.\n"
                f"3. Run `python3 tests.py` to verify. If tests fail, analyze the error, "
                f"fix your code, and re-run until all tests pass.\n"
                f"4. Call `Finish` when done.\n\n"
                f"Important:\n"
                f"- Do NOT change the class name, method signatures, or docstrings.\n"
                f"- Pay attention to the docstring examples — they reveal expected behavior.\n"
                f"- Only `solution.py` and `tests.py` exist in the workspace.\n"
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

            # Build the verification script (independent of agent)
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
            shutil.rmtree(hidden_dir, ignore_errors=True)


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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
    parser.add_argument("--resume", default=None, help="Resume from a previous .jsonl results file")
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

    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
