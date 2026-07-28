"""AIME benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from .base import BENCHMARK_BASE_SYSTEM_PROMPT, BenchmarkRunner, _PROJECT_ROOT
except ImportError:
    from base import BENCHMARK_BASE_SYSTEM_PROMPT, BenchmarkRunner, _PROJECT_ROOT


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
    2. Present the math problem; agent writes Python code to solve it.
    3. Agent runs the code, verifies correctness via constraint checking,
       and submits via Finish.
    4. Extract the final integer answer from the Finish output.
    5. Compare with the expected answer and record pass / fail.
    """

    benchmark_name = "aime"
    _VALIDATION_SOURCE = "finish_answer"
    _RETRY_MAX_TOKENS = 2048  # enough for a direct Finish call or a brief strategy pivot
    _BOXED_ANSWER_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}", flags=re.IGNORECASE)
    _FINAL_ANSWER_RE = re.compile(r"Final answer is\s+(-?\d+)\.", flags=re.IGNORECASE)

    _MATH_SYSTEM_PROMPT = """\
## AIME Math Benchmark — Code-First Problem Solving

You are solving AIME competition math problems. Code execution is your primary
tool — write programs to compute answers. Mathematical reasoning is essential
for correct modeling, but the final answer must come from running code.

### Strategy Guide

- Algebra / word problems: set up equations, solve with fractions.Fraction.
- Geometry: use coordinates, law of cosines, power of a point. Prefer
  fractions.Fraction over floating point.
- Combinatorics: use itertools to enumerate all configurations, filter, count.
- Number theory: use modular arithmetic, search, gcd, factorization.

### Workflow

1. **Analyze** — understand the problem mathematically. Identify the correct
   strategy. A wrong mathematical model wastes far more time than careful
   analysis. Take the time you need to get the math right.

2. **Write → Bash** — create a self-contained `solve.py`, then IMMEDIATELY
   run it with `python solve.py`. NEVER write code without running it.
   If the output is clearly unreasonable (e.g. not in [0, 999], far from
   expected magnitude), your mathematical model is likely wrong — rethink
   the approach rather than tweaking code details.

3. **Verify** — before submitting, do a proper check of your answer:
   - Is the mathematical model correct for the problem?
   - Does the code correctly implement that model?
   - Does the answer satisfy obvious constraints from the problem?
   A single thorough check is enough. Skipping verification risks
   submitting a wrong answer; endless re-verification wastes time.

4. **Finish** — call Finish with `\\boxed{N}` where N is your integer answer.
   No extra text, no multiple boxed values.

### Critical Rules

- ALWAYS write and run Python code to compute answers. Never submit a guess
  without code execution.
- Verify your answer with one solid pass — check both the math and the
  code. Then commit. Neither blind submission nor endless re-checking
  produces good results.
- If your approach isn't working after 2-3 code attempts, step back and
  try a completely different strategy rather than tweaking code details.
- If you cannot get a valid integer after genuine effort, submit your
  best answer anyway rather than running out of time.
"""

    def __init__(self, *args, year: Optional[str] = None, max_submission_rounds: int = 3, **kwargs):
        self.year = _normalize_year(year)
        self.max_submission_rounds = max(1, int(max_submission_rounds))
        super().__init__(*args, **kwargs)
        if self.year is None:
            self.year = _infer_year_from_path(self.data_path)
        if self.year is not None:
            self.benchmark_name = f"aime_{self.year}"
        # Cap at 64: AIME solutions average ~33 steps; 64 is generous
        # enough for complex problems while preventing runaway loops.
        if self.max_steps > 64:
            self.max_steps = 64

    def _get_system_prompt(self):
        return BENCHMARK_BASE_SYSTEM_PROMPT + "\n\n---\n\n" + self._MATH_SYSTEM_PROMPT

    def _configure_agent_config(self, config: Any) -> Any:
        config = super()._configure_agent_config(config)
        config.skills_enabled = False
        config.skills_auto_register = False
        config.todowrite_enabled = False
        return config

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
    def _retry_prompt(round_idx: int) -> str:
        """Rounds 2-3 retry prompts: first nudge to submit or rethink, then final
        push to submit whatever answer is available."""
        if round_idx == 2:
            return (
                "Your previous response did not include a Finish tool call.\n\n"
                "If your code produced a reasonable integer answer (0-999), call\n"
                "Finish now with: \\\\boxed{N} where N is that integer.\n\n"
                "If your code did NOT produce a reasonable answer, your mathematical\n"
                "model may be incorrect. Try a different approach — then call Finish\n"
                "with your best integer answer."
            )
        # Final round — just submit
        return (
            "This is your FINAL attempt. You MUST call Finish now with your best\n"
            "integer answer as: \\\\boxed{N} where N is in [0, 999].\n"
            "Submit whatever you have — an approximate answer is better than nothing."
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
                f"Solve this AIME problem by writing and running Python code.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Workflow:\n"
                f"1. Analyze — understand the mathematics and choose the right\n"
                f"   strategy. Getting the math right is more important than speed.\n"
                f"2. Write → Bash — create solve.py, then IMMEDIATELY run it.\n"
                f"   If output is unreasonable (not in [0, 999] or clearly wrong\n"
                f"   magnitude), first reconsider your mathematical approach, then\n"
                f"   fix the code and re-run.\n"
                f"3. Verify — do a proper check: is the math correct? Does the\n"
                f"   code faithfully implement it? Does the answer make sense?\n"
                f"   One thorough check is enough.\n"
                f"4. Finish — submit the integer answer as \\boxed{{N}}.\n\n"
                f"Submission:\n"
                f"- Verify your answer before submitting: check both the mathematical\n"
                f"  model and the code. One solid pass is sufficient — skipping\n"
                f"  verification risks wrong answers, but endless re-checking\n"
                f"  wastes time.\n"
                f"- If the answer seems unreasonable, your math model is likely\n"
                f"  wrong — try a different strategy.\n"
                f"- Call Finish with answer = `\\boxed{{N}}`. No extra text.\n"
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
                return self._retry_prompt(round_idx)

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
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume, fresh=args.fresh)


if __name__ == "__main__":
    main()
