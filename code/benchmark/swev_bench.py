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
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

from .base import BenchmarkRunner, _PROJECT_ROOT

# ---------------------------------------------------------------------------
# Build the SWE-bench system prompt: base system prompt + benchmark overrides
# ---------------------------------------------------------------------------
_BASE_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "system_prompt.md"
_BASE_SYSTEM_PROMPT: str = _BASE_PROMPT_FILE.read_text(encoding="utf-8")

_SWEV_ADDENDUM = """\

---

## SWE-bench Override: Autonomous Issue Resolution

The general workflow above is overridden for this benchmark session. \
You are an autonomous software engineer. Your sole job is to resolve \
a GitHub issue by editing source code in the local repository. \
You work alone — there is no human in the loop.

### Workflow (follow strictly)

```
1. Locate — Use Grep and Glob to find the relevant code. Start from
   class names, function names, or error strings mentioned in the issue.
2. Understand — Use Read to study the surrounding code (callers, tests,
   related functions). Read ENOUGH context to be confident about the
   root cause.
3. Diagnose — Use Thought to reason about the root cause BEFORE editing.
   State clearly: what is wrong, why it happens, and what the fix should be.
4. Fix — Use Edit to make the minimal change. Only change
   what is necessary.
5. Verify — Read the edited region to confirm the fix looks correct.
6. Finish — Call Finish with a brief summary. Do NOT keep searching
   after a successful edit.
```

### Critical Rules

```
- You MUST produce a code change. Saying "I cannot fix this" is NOT
  acceptable. Always attempt a fix even if you are uncertain.
- Make the SMALLEST possible change. Prefer a 1-3 line fix over a
  large refactor.
- Do NOT run pip install, python setup.py, pytest, or any shell commands
  that install packages or run tests. Testing is handled externally.
- Do NOT modify test files (tests/, test_*.py, *_test.py).
- Do NOT add new dependencies or create new files unless absolutely
  necessary.
- Preserve existing code style, indentation, and naming conventions.
```

### Efficiency Rules (save tokens and steps)

```
- Do NOT read the whole repository. Search targeted: grep for the key
  symbol, read only the relevant file region.
- When you find the fix location, edit IMMEDIATELY. Do not read 5 more
  files to "make sure" — that wastes steps.
- Use Glob with specific patterns (e.g. **/models/sql/*.py) rather than
  broad **/*.py.
- If a Grep returns too many results, narrow the query or add file filters.
- Avoid Bash for file searching — use Grep/Glob/Read instead.
- If you are going in circles (reading the same files repeatedly), STOP,
  reason in a Thought, then commit to a fix.
```
"""

_SWEV_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + _SWEV_ADDENDUM


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
        task_timeout: int = 1800,
        resume_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.repo_cache_dir = Path(repo_cache_dir) if repo_cache_dir else None
        if self.repo_cache_dir:
            self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.task_timeout = task_timeout  # per-task wall-clock timeout (seconds)
        self.resume_file = resume_file

    def _get_system_prompt(self) -> Optional[str]:
        """Use the SWE-bench-specific system prompt."""
        return _SWEV_SYSTEM_PROMPT

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

    def _load_completed_task_ids(self) -> Set[str]:
        """Load task IDs already completed from a previous results file."""
        if not self.resume_file:
            return set()
        resume_path = Path(self.resume_file)
        if not resume_path.exists():
            return set()
        completed = set()
        with open(resume_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    # Only skip tasks that actually ran (not clone failures)
                    error = result.get("error") or ""
                    if "Failed to clone" not in error:
                        completed.add(result.get("task_id", ""))
                except json.JSONDecodeError:
                    continue
        completed.discard("")
        print(f"  [RESUME] Found {len(completed)} completed tasks to skip")
        return completed

    # ------------------------------------------------------------------
    # Repository management
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_git_lock_files(repo_path: Path) -> None:
        """Remove stale git lock files that prevent checkout/reset."""
        for lock_file in ["index.lock", "HEAD.lock", "refs/heads/*.lock"]:
            for p in repo_path.glob(f".git/{lock_file}"):
                try:
                    p.unlink()
                except OSError:
                    pass

    def _reset_cached_repo(self, cached: Path, base_commit: str) -> bool:
        """Reset a cached repo to a specific commit. Returns True on success."""
        ws = str(cached)
        try:
            # Clean up any stale lock files from previous crashed runs
            self._remove_git_lock_files(cached)

            # Hard reset to discard any staged/unstaged changes from previous task
            subprocess.run(
                ["git", "reset", "--hard"],
                cwd=ws, capture_output=True, timeout=120,
            )
            # Remove all untracked files and directories
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=ws, capture_output=True, timeout=120,
            )
            # Checkout the target commit
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=ws, capture_output=True, timeout=300, check=True,
            )
            # Clean again after checkout (in case checkout brought changes)
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=ws, capture_output=True, timeout=120,
            )
            return True
        except Exception:
            return False

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
                # Try direct checkout
                if self._reset_cached_repo(cached, base_commit):
                    return cached

                # First attempt failed — try fetching latest refs then retry
                print(f"\n  [WARN] Cache checkout failed for {repo}@{base_commit[:10]}, fetching...")
                try:
                    subprocess.run(
                        ["git", "fetch", "--all"],
                        cwd=str(cached), capture_output=True, timeout=300,
                    )
                except Exception:
                    pass

                if self._reset_cached_repo(cached, base_commit):
                    return cached

                print(f"\n  [WARN] Retry checkout also failed for {repo}@{base_commit[:10]}")
                return None

        # Clone fresh (no cache or cache dir doesn't have this repo yet)
        target = (self.repo_cache_dir / repo_slug) if self.repo_cache_dir else Path(
            tempfile.mkdtemp(prefix=f"swev_{repo_slug}_")
        )
        url = f"https://github.com/{repo}.git"

        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--filter=blob:none", url, str(target)],
                capture_output=True, text=True, timeout=600, check=True,
            )
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=str(target), capture_output=True, text=True, timeout=120, check=True,
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
        """Capture the diff of all changes the agent made, including new files."""
        ws = str(workspace)
        try:
            # Stage all new (untracked) files so they appear in the diff
            subprocess.run(
                ["git", "add", "-A"],
                cwd=ws, capture_output=True, timeout=30,
            )
            # Diff between base_commit (HEAD) and current staged state
            result = subprocess.run(
                ["git", "diff", "--no-color", "HEAD"],
                cwd=ws, capture_output=True, text=True, timeout=60,
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
            hints_block = f"\n\n## Hints\n\n{hints}" if hints else ""

            agent_prompt = (
                f"Fix the following issue in the `{repo}` repository.\n\n"
                f"## Issue\n\n{problem_statement}\n"
                f"{hints_block}\n\n"
                f"The repository is already checked out at the correct commit. "
                f"Your working directory is the repo root.\n\n"
                f"## Strategy\n\n"
                f"1. **Identify the bug location**: Grep for key symbols from the "
                f"issue (class names, method names, error messages). Narrow down "
                f"to 1-2 files.\n"
                f"2. **Read the relevant code**: Read the specific function or "
                f"method (not the whole file). Understand the logic.\n"
                f"3. **Diagnose**: Use `Thought` to state: (a) what the current "
                f"code does wrong, (b) what it should do instead.\n"
                f"4. **Edit**: Make the minimal fix with `Edit`.\n"
                f"5. **Verify & Finish**: `Read` the edited lines, then call "
                f"`Finish` with a summary of what you changed and why.\n"
            )

            try:
                agent_response = self._run_agent_with_timeout(agent, agent_prompt)
            except _TaskTimeout:
                return {
                    "task_id": task_id,
                    "repo": repo,
                    "passed": None,
                    "error": f"Task timed out after {self.task_timeout}s",
                    "agent_diff": self._get_agent_diff(workspace),
                    "agent_response": "",
                    "elapsed_s": round(time.time() - start, 2),
                }
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

    def _run_agent_with_timeout(self, agent, prompt: str) -> str:
        """Run agent.run() with a wall-clock timeout using SIGALRM."""
        if self.task_timeout <= 0:
            return agent.run(prompt)

        # Use a threading-based timeout on all platforms
        import threading

        result_holder: Dict[str, Any] = {}
        exception_holder: List[BaseException] = []

        def target():
            try:
                result_holder["value"] = agent.run(prompt)
            except Exception as exc:
                exception_holder.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.task_timeout)

        if thread.is_alive():
            # Thread is still running — we can't forcefully kill it,
            # but we signal timeout and move on. The daemon thread will
            # be cleaned up when the process exits or on next task.
            raise _TaskTimeout(f"Agent did not finish within {self.task_timeout}s")

        if exception_holder:
            raise exception_holder[0]

        return result_holder.get("value", "")

    # ------------------------------------------------------------------
    # Override run() to add resume support and predictions export
    # ------------------------------------------------------------------

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark and export predictions for Docker evaluation."""
        # Use the constructor's resume_file (swev has custom clone-failure logic)
        effective_resume = resume or self.resume_file

        summary = super().run(
            limit=limit, task_ids=task_ids, dry_run=dry_run,
            resume=effective_resume,
        )

        if dry_run:
            return summary

        # Read back the results JSONL written by the base class and produce
        # a predictions file in the official SWE-bench format.
        results_file = Path(summary.get("results_file", ""))
        if not results_file.exists():
            return summary

        # If resuming, also include results from the resume file
        all_results_files = []
        if self.resume_file and Path(self.resume_file).exists():
            all_results_files.append(Path(self.resume_file))
        all_results_files.append(results_file)

        timestamp = summary.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        predictions_file = self.output_dir / f"swev_predictions_{timestamp}.jsonl"

        diff_count = 0
        seen_ids: Set[str] = set()
        with open(predictions_file, "w", encoding="utf-8") as fout:
            for rf in all_results_files:
                with open(rf, encoding="utf-8") as fin:
                    for line in fin:
                        result = json.loads(line.strip())
                        instance_id = result.get("task_id", "")
                        if instance_id in seen_ids:
                            continue
                        seen_ids.add(instance_id)
                        agent_diff = result.get("agent_diff", "")
                        if agent_diff:
                            diff_count += 1
                        prediction = {
                            "instance_id": instance_id,
                            "model_name_or_path": self.model_name,
                            "model_patch": agent_diff,
                        }
                        fout.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        total = len(seen_ids)
        print(f"\n{'=' * 60}")
        print(f"  Predictions: {predictions_file}")
        print(f"  Diffs produced: {diff_count}/{total}")
        print(f"\n  To evaluate with Docker:")
        print(f"  bash scripts/run_swev_eval.sh {predictions_file}")
        print(f"{'=' * 60}\n")

        summary["predictions_file"] = str(predictions_file)
        summary["diff_count"] = diff_count
        return summary


class _TaskTimeout(Exception):
    """Raised when a single task exceeds its wall-clock timeout."""


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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=128)
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
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=1800,
        help="Per-task wall-clock timeout in seconds (default: 1800 = 30min)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RESULTS_FILE",
        help="Resume from a previous results JSONL file, skipping completed tasks",
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
        task_timeout=args.task_timeout,
        resume_file=args.resume,
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
