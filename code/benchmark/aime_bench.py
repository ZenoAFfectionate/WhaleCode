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
    2. Present the math problem and let the agent reason directly.
    3. Optionally allow scratch exploration in the workspace.
    4. Extract the final integer answer from the returned ``Finish`` answer.
    5. Compare with the expected answer and record pass / fail.
    """

    benchmark_name = "aime"

    _AIME_ADDENDUM = """\
You are solving AIME competition math problems.

AIME-specific rules:
- The final answer is an integer from 000 to 999.
- Solve by mathematical reasoning first.
- For easier problems, it is fine to reason directly and call `Finish` without using any other tools.
- If computation helps, prefer short `python3 -c "..."` commands through `Bash`.
- You may use the workspace for scratch files if useful, but you do not need to create or edit `solution.py` or any other submission file.
- When you are ready to submit, call `Finish` with the final answer only.
- The `answer` value passed to `Finish` must contain just the final integer, with no explanation or extra text.
"""

    _MATH_SYSTEM_PROMPT = (
        BENCHMARK_BASE_SYSTEM_PROMPT
        + "\n\n---\n\n## AIME Benchmark Override\n\n"
        + _AIME_ADDENDUM
    )

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

    @staticmethod
    def _benchmark_agent_run_kwargs(run_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """AIME benefits from allowing direct answers without forcing tool calls every step."""
        effective_kwargs = dict(run_kwargs or {})
        effective_kwargs.setdefault("tool_choice", "auto")
        return effective_kwargs

    def _configure_agent_config(self, config: Any) -> Any:
        config = super()._configure_agent_config(config)
        config.skills_enabled = False
        config.skills_auto_register = False
        config.todowrite_enabled = False
        return config

    def _register_agent_tools(self, *, registry: Any, workspace: Path, agent: Any) -> None:
        """AIME keeps the tool surface minimal to reduce workflow noise."""
        from hello_agents.tools.builtin.bash import BashTool

        ws = str(workspace)
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
        """Extract the final AIME integer from the returned Finish answer."""
        text = str(output or "").strip()
        if not text:
            return None

        boxed_match = re.search(r"\\boxed\{(\d{1,3})\}", text)
        if boxed_match:
            return int(boxed_match.group(1))

        strict_patterns = (
            r"(?:final\s+answer\s*:\s*)?(\d{1,3})",
            r"(?:the\s+answer\s+is\s+)?(\d{1,3})\.?",
            r"answer\s*=\s*(\d{1,3})",
        )
        for pattern in strict_patterns:
            match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))

        integers = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", text)
        unique_integers = list(dict.fromkeys(integers))
        if len(unique_integers) == 1:
            return int(unique_integers[0])
        return None

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        problem = task["problem"]
        expected_answer = int(task["answer"])

        workspace = self._make_workspace(f"aime_{task_id}_")
        agent = None
        agent_response = ""
        prompt_history: List[str] = []
        result: Optional[Dict[str, Any]] = None
        try:
            agent = self._create_agent(workspace)
            initial_prompt = (
                f"Solve this AIME problem. The answer is an integer from 0 to 999.\n\n"
                f"**Problem:**\n{problem}\n\n"
                f"**Instructions:**\n"
                f"1. First solve it mathematically. For easier problems, you may answer directly.\n"
                f"2. If computation helps, use short `python3 -c \"...\"` commands via Bash. "
                f"You may use scratch files, but you do not need to create `solution.py`.\n"
                f"3. When you are confident, call `Finish` with the final answer only.\n"
                f"4. The `answer` passed to `Finish` must be a single integer from 0 to 999, with no explanation.\n"
            )
            prompt_history.append(initial_prompt)

            start = time.time()
            agent_response, error_result = self._run_agent_prompt(
                agent=agent,
                task_id=task_id,
                prompt_text=initial_prompt,
                start_time=start,
                error_extra={
                    "expected": expected_answer,
                    "actual": None,
                    "validation_source": "finish_answer",
                },
            )
            if error_result is not None:
                result = error_result
                return result

            actual_answer = self._extract_answer(agent_response)
            elapsed = round(time.time() - start, 2)

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
                        "validation_source": "finish_answer",
                    },
                )
                return result

            result = self._build_result(
                task_id,
                passed=actual_answer == expected_answer,
                error=None if actual_answer == expected_answer else f"Wrong answer: expected {expected_answer}, got {actual_answer}",
                agent_response=agent_response,
                elapsed_s=elapsed,
                extra={
                    "expected": expected_answer,
                    "actual": actual_answer,
                    "validation_source": "finish_answer",
                },
            )
            return result
        finally:
            self._save_task_trajectory(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=prompt_history,
                result=result,
                artifact_paths=None,
                extra={
                    "expected_answer": expected_answer,
                    "submission_mode": "finish_answer_only",
                },
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
        include_task_timeout=True,
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
        task_timeout=args.task_timeout,
        trajectory_dir=args.trajectory_dir,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
