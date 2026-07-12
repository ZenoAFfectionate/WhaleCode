"""AIME benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from .base import BenchmarkRunner, _PROJECT_ROOT
except ImportError:
    from base import BenchmarkRunner, _PROJECT_ROOT


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
    2. Present the math problem and let the agent reason directly.
    3. Optionally allow scratch exploration in the workspace.
    4. Extract the final integer answer from model output text.
    5. Compare with the expected answer and record pass / fail.
    """

    benchmark_name = "aime"
    _VALIDATION_SOURCE = "finish_answer"
    _RETRY_MAX_TOKENS = 128
    _BOXED_ANSWER_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}", flags=re.IGNORECASE)
    _FINAL_ANSWER_RE = re.compile(r"Final answer is\s+(-?\d+)\.", flags=re.IGNORECASE)

    _MATH_SYSTEM_PROMPT = """\
You are an expert AIME competition math solver.

Objective:
- Maximize answer correctness on olympiad-style problems.
- Produce a final integer answer in the exact required format.

Tool protocol (mandatory):
1. Start with a Thought tool call to state your plan.
2. Use Bash only for short arithmetic checks when needed.
3. Before concluding, use Thought again for a concise verification note.
4. Conclude by calling Finish exactly once.
5. Do not provide the final answer in plain assistant text.

Reasoning protocol:
1. Identify what is being asked and define symbols/constraints clearly.
2. Build a mathematically sound solution path before computing.
3. Keep steps concise but explicit: equations, transformations, and key logic.
4. Use Bash only for short arithmetic checks, not as a replacement for reasoning.
5. Before finalizing, run a quick verification pass:
   - Check algebra/sign mistakes.
   - Check counting/combinatorics edge cases.
   - Check that the result satisfies problem constraints.
   - If needed, confirm by a second method or substitution.

Answer policy:
- Prefer a short, clean derivation over verbose narration.
- The Finish tool's `answer` field must be exactly one boxed integer: `\\boxed{N}`.
- N must be an integer in [0, 999].
- Do not output multiple boxed answers.
- Do not add any trailing text.
"""

    def __init__(self, *args, year: Optional[str] = None, max_submission_rounds: int = 3, **kwargs):
        self.year = _normalize_year(year)
        self.max_submission_rounds = max(1, int(max_submission_rounds))
        super().__init__(*args, **kwargs)
        if self.year is None:
            self.year = _infer_year_from_path(self.data_path)
        if self.year is not None:
            self.benchmark_name = f"aime_{self.year}"

    def _get_system_prompt(self):
        return self._MATH_SYSTEM_PROMPT

    def _configure_agent_config(self, config: Any) -> Any:
        config = super()._configure_agent_config(config)
        config.skills_enabled = False
        config.skills_auto_register = False
        config.todowrite_enabled = False
        return config

    def _register_agent_tools(self, *, registry: Any, workspace: Path, agent: Any) -> None:
        """AIME keeps a minimal tool surface while enforcing Thought/Finish control tools."""
        from hello_agents.agents.react_agent import (
            FINISH_TOOL_DESCRIPTION,
            FINISH_TOOL_NAME,
            THOUGHT_TOOL_DESCRIPTION,
            THOUGHT_TOOL_NAME,
            _FinishTool,
            _ThoughtTool,
        )
        from hello_agents.tools.builtin.bash import BashTool

        ws = str(workspace)
        if registry.get_tool(THOUGHT_TOOL_NAME) is None:
            registry.register_tool(_ThoughtTool(THOUGHT_TOOL_DESCRIPTION))
        if registry.get_tool(FINISH_TOOL_NAME) is None:
            registry.register_tool(_FinishTool(FINISH_TOOL_DESCRIPTION))
        registry.register_tool(BashTool(project_root=ws, working_dir=ws))

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
        """Extract AIME answer from Finish output."""
        text = str(output or "").strip()
        if not text:
            return None

        last_boxed = None
        for match in AIMEBenchmark._BOXED_ANSWER_RE.finditer(text):
            last_boxed = match
        if last_boxed is not None:
            return int(last_boxed.group(1))

        last_final = None
        for match in AIMEBenchmark._FINAL_ANSWER_RE.finditer(text):
            last_final = match
        if last_final is not None:
            return int(last_final.group(1))

        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return None

    @staticmethod
    def _retry_prompt(problem: str, round_idx: int) -> str:
        return (
            f"Previous attempt #{round_idx - 1} did not follow the required output format.\n"
            f"Use tools only (tool_choice is required).\n"
            f"Call Thought briefly, then call Finish once.\n"
            f"In Finish, set answer to exactly: \\\\boxed{{N}}\n"
            f"where N is an integer in [0, 999].\n"
            f"No extra text.\n\n"
            f"Problem:\n{problem}\n"
        )

    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        problem = task["problem"]
        expected_answer = int(task["answer"])
        validation_source = self._VALIDATION_SOURCE

        workspace = self._make_workspace(f"aime_{task_id}_")
        agent = None
        agent_response = ""
        prompt_history: List[str] = []
        result: Optional[Dict[str, Any]] = None
        try:
            agent = self._create_agent(workspace)
            initial_prompt = (
                f"Solve this AIME problem with competition-grade rigor.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Execution checklist:\n"
                f"1. Call Thought first with your plan.\n"
                f"2. Restate the target quantity and key constraints.\n"
                f"3. Derive the solution step by step with correct math.\n"
                f"4. Use Bash only for quick arithmetic validation if needed.\n"
                f"5. Call Thought for a short self-check.\n"
                f"6. Call Finish exactly once with the final answer.\n\n"
                f"Tool/output requirements:\n"
                f"- You must use tools; do not answer directly in assistant text.\n"
                f"- When you are ready, call `Finish` with the final answer only.\n"
                f"- The Finish `answer` must be exactly one boxed integer: `\\boxed{{N}}`.\n"
                f"- N must be in [0, 999].\n"
                f"- Do not output multiple boxed values.\n"
                f"- Do not include trailing text.\n"
            )
            start = time.time()
            actual_answer: Optional[int] = None

            def evaluate_submission(round_idx: int, response: str) -> Dict[str, Any]:
                nonlocal actual_answer
                actual_answer = self._extract_answer(response)
                if actual_answer is not None:
                    return {"passed": actual_answer == expected_answer, "output": str(actual_answer), "force_stop": True}
                return {
                    "passed": False,
                    "output": "",
                    "feedback": (
                        "Format invalid. End with exactly one boxed integer like "
                        "`\\boxed{113}` and no trailing explanation."
                    ),
                }

            def retry_prompt_builder(round_idx: int, _feedback: str) -> str:
                return self._retry_prompt(problem, round_idx)

            def run_kwargs_builder(round_idx: int) -> Optional[Dict[str, Any]]:
                if round_idx <= 1:
                    return None
                return {"max_tokens": self._RETRY_MAX_TOKENS}

            def error_extra_builder(round_idx: int) -> Dict[str, Any]:
                return {
                    "expected": expected_answer,
                    "actual": None,
                    "validation_source": validation_source,
                    "round": round_idx,
                }

            submission = self._run_controlled_submission_rounds(
                task_id=task_id,
                agent=agent,
                start_time=start,
                initial_prompt=initial_prompt,
                max_rounds=self.max_submission_rounds,
                prompt_history=prompt_history,
                evaluate_submission=evaluate_submission,
                retry_prompt_builder=retry_prompt_builder,
                run_kwargs_builder=run_kwargs_builder,
                error_extra_builder=error_extra_builder,
            )
            agent_response = submission.get("agent_response", "")
            if submission.get("early_result") is not None:
                result = submission["early_result"]
                return result

            elapsed = round(time.time() - start, 2)
            rounds_used = int(submission.get("rounds_used", 0) or 0)
            if actual_answer is None:
                result = self._build_result(
                    task_id,
                    passed=False,
                    error="Could not extract a final AIME integer from the Finish answer",
                    agent_response=agent_response,
                    elapsed_s=elapsed,
                    extra={
                        "expected": expected_answer,
                        "actual": None,
                        "validation_source": validation_source,
                        "rounds_used": rounds_used,
                    },
                )
                return result

            result = self._build_result(
                task_id,
                passed=actual_answer == expected_answer,
                error=(
                    None
                    if actual_answer == expected_answer
                    else f"Wrong answer: expected {expected_answer}, got {actual_answer}"
                ),
                agent_response=agent_response,
                elapsed_s=elapsed,
                extra={
                    "expected": expected_answer,
                    "actual": actual_answer,
                    "validation_source": validation_source,
                    "rounds_used": rounds_used,
                },
            )
            return result
        finally:
            self._finalize_workspace_task(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=prompt_history,
                result=result,
                artifact_paths=None,
                extra={
                    "expected_answer": expected_answer,
                    "submission_mode": "boxed_answer_preferred",
                },
            )


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
        include_task_timeout=True,
        default_task_timeout=1200,
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
        **BenchmarkRunner.runner_kwargs_from_args(args, include_task_timeout=True),
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
