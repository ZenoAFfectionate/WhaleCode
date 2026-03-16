"""SWE-bench Verified benchmark runner for Whale Code agent.

Two-phase evaluation:
  Phase 1 (this file): Run the agent on each instance, collect diffs,
      and output a predictions JSONL compatible with the official harness.
  Phase 2 (scripts/run_swev_eval.sh): Feed predictions to
      ``swebench.harness.run_evaluation`` for Docker-based grading.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .base import BenchmarkRunner, _PROJECT_ROOT


class SWEBenchVerifiedBenchmark(BenchmarkRunner):
    """Evaluate the agent on SWE-bench Verified (500 real GitHub issue instances).

    Workflow per instance:
    1. Clone (or use cached) the repository at ``base_commit``.
    2. Set the cloned repo as the agent workspace.
    3. Present the ``problem_statement`` to the agent.
    4. After the agent finishes, capture ``git diff`` of all changes.
    5. Record the diff for offline Docker evaluation.

    Use ``--repo-cache-dir`` to persist cloned repos across runs.
    """

    benchmark_name = "swebench_verified"

    def __init__(
        self,
        *args,
        repo_cache_dir: Optional[str] = None,
        model_name: str = "whale-code",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.repo_cache_dir = Path(repo_cache_dir) if repo_cache_dir else None
        if self.repo_cache_dir:
            self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    task = json.loads(line)
                    task["task_id"] = task.get("instance_id", task.get("task_id"))
                    tasks.append(task)
        return tasks

    # ------------------------------------------------------------------
    # Repository management
    # ------------------------------------------------------------------

    def _clone_repo(self, repo: str, base_commit: str) -> Optional[Path]:
        """Clone a GitHub repo at a specific commit.

        Uses ``--filter=blob:none`` for a blobless clone (much faster for
        large repos like astropy/django) — blobs are fetched on demand.
        Returns the workspace path, or None on failure.
        """
        repo_slug = repo.replace("/", "__")

        # Check cache first
        if self.repo_cache_dir:
            cached = self.repo_cache_dir / repo_slug
            if cached.exists():
                try:
                    subprocess.run(
                        ["git", "checkout", "-f", base_commit],
                        cwd=str(cached),
                        capture_output=True,
                        timeout=120,
                    )
                    subprocess.run(
                        ["git", "clean", "-fdx"],
                        cwd=str(cached),
                        capture_output=True,
                        timeout=60,
                    )
                    return cached
                except Exception:
                    shutil.rmtree(cached, ignore_errors=True)

        # Clone fresh
        target = (self.repo_cache_dir / repo_slug) if self.repo_cache_dir else Path(
            tempfile.mkdtemp(prefix=f"swev_{repo_slug}_")
        )
        url = f"https://github.com/{repo}.git"

        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--filter=blob:none", url, str(target)],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            return target
        except subprocess.TimeoutExpired:
            print(f"\n  [WARN] Clone timed out for {repo}")
            if not self.repo_cache_dir:
                shutil.rmtree(target, ignore_errors=True)
            return None
        except subprocess.CalledProcessError as exc:
            print(f"\n  [WARN] Clone failed for {repo}: {(exc.stderr or '')[:200]}")
            if not self.repo_cache_dir:
                shutil.rmtree(target, ignore_errors=True)
            return None
        except Exception as exc:
            print(f"\n  [WARN] Clone error for {repo}: {exc}")
            if not self.repo_cache_dir:
                shutil.rmtree(target, ignore_errors=True)
            return None

    def _get_agent_diff(self, workspace: Path) -> str:
        """Capture the diff of all changes the agent made."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        repo = task["repo"]
        base_commit = task["base_commit"]
        problem_statement = task["problem_statement"]

        start = time.time()
        is_temp = not bool(self.repo_cache_dir)

        # Step 1: Clone repo
        workspace = self._clone_repo(repo, base_commit)
        if workspace is None:
            return {
                "task_id": task_id,
                "repo": repo,
                "passed": None,
                "error": f"Failed to clone {repo}@{base_commit}",
                "agent_diff": "",
                "elapsed_s": round(time.time() - start, 2),
            }

        try:
            # Step 2: Run agent
            agent = self._create_agent(workspace)
            hints = task.get("hints_text", "").strip()
            hints_block = f"\n\nHints:\n{hints}" if hints else ""

            agent_prompt = (
                f"You are a software engineer fixing a bug in the `{repo}` repository.\n\n"
                f"## Issue\n\n{problem_statement}\n"
                f"{hints_block}\n\n"
                f"## Instructions\n\n"
                f"1. Use `Grep` and `Glob` to locate the relevant source files. Start by "
                f"searching for key class names, function names, or error messages "
                f"mentioned in the issue.\n"
                f"2. Use `Read` to understand the code around the identified location. "
                f"Read enough context (surrounding functions, class definitions) to "
                f"fully understand the logic.\n"
                f"3. Reason about the root cause based on the issue description and the "
                f"code you read. Explain your understanding before making changes.\n"
                f"4. Use `Edit` to implement a minimal, targeted fix. Change only what "
                f"is necessary to resolve the issue.\n"
                f"5. Use `Read` to verify your edit was applied correctly.\n\n"
                f"## Rules\n\n"
                f"- Do NOT run `pip install`, `python setup.py`, or any installation "
                f"commands. The environment is pre-configured.\n"
                f"- Do NOT run test commands (`pytest`, `python -m pytest`, etc.). "
                f"Testing is handled separately after your changes.\n"
                f"- Do NOT modify any test files (files under `tests/` or `test_*.py`).\n"
                f"- Make the smallest possible change that fixes the issue.\n"
                f"- Preserve existing code style, indentation, and conventions.\n"
                f"- If the fix requires changes in multiple files, edit each one.\n"
            )

            try:
                agent_response = agent.run(agent_prompt)
            except Exception as exc:
                return {
                    "task_id": task_id,
                    "repo": repo,
                    "passed": None,
                    "error": f"Agent error: {exc}",
                    "agent_diff": "",
                    "agent_response": "",
                    "elapsed_s": round(time.time() - start, 2),
                }

            # Step 3: Capture diff (full, not truncated)
            agent_diff = self._get_agent_diff(workspace)

            return {
                "task_id": task_id,
                "repo": repo,
                "passed": None,  # Determined by Docker eval
                "has_diff": bool(agent_diff),
                "error": "Agent produced no changes" if not agent_diff else None,
                "agent_diff": agent_diff,
                "agent_response": (agent_response or "")[:500],
                "elapsed_s": round(time.time() - start, 2),
            }
        finally:
            if is_temp and workspace and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    # ------------------------------------------------------------------
    # Override run() to add predictions export
    # ------------------------------------------------------------------

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run the benchmark and export predictions for Docker evaluation."""
        summary = super().run(limit=limit, task_ids=task_ids, dry_run=dry_run)

        if dry_run:
            return summary

        # Read back the results JSONL written by the base class and produce
        # a predictions file in the official SWE-bench format.
        results_file = Path(summary.get("results_file", ""))
        if not results_file.exists():
            return summary

        timestamp = summary.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        predictions_file = self.output_dir / f"swev_predictions_{timestamp}.jsonl"

        diff_count = 0
        with open(results_file, encoding="utf-8") as fin, \
             open(predictions_file, "w", encoding="utf-8") as fout:
            for line in fin:
                result = json.loads(line.strip())
                instance_id = result.get("task_id", "")
                agent_diff = result.get("agent_diff", "")
                if agent_diff:
                    diff_count += 1
                prediction = {
                    "instance_id": instance_id,
                    "model_name_or_path": self.model_name,
                    "model_patch": agent_diff,
                }
                fout.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        total = summary.get("total", 0)
        print(f"\n{'=' * 60}")
        print(f"  Predictions: {predictions_file}")
        print(f"  Diffs produced: {diff_count}/{total}")
        print(f"\n  To evaluate with Docker:")
        print(f"  bash scripts/run_swev_eval.sh {predictions_file}")
        print(f"{'=' * 60}\n")

        summary["predictions_file"] = str(predictions_file)
        summary["diff_count"] = diff_count
        return summary


def main():
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run SWE-bench Verified benchmark (Phase 1: agent inference)")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "SWEV" / "test.jsonl"),
        help="Path to SWE-bench Verified JSONL file",
    )
    parser.add_argument("--output-dir", default=str(_PROJECT_ROOT / "data" / "_results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=50, help="More steps for complex repo tasks")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument(
        "--repo-cache-dir",
        default=None,
        help="Directory to cache cloned repos between runs",
    )
    parser.add_argument(
        "--model-name",
        default="whale-code",
        help="Model name for predictions file (default: whale-code)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench = SWEBenchVerifiedBenchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_steps=args.max_steps,
        timeout=args.timeout,
        repo_cache_dir=args.repo_cache_dir,
        model_name=args.model_name,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
