"""Base benchmark runner for Whale Code agent evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _bootstrap_package() -> None:
    """Expose the local ``code/`` directory as the ``hello_agents`` package."""
    import types

    if "hello_agents" in sys.modules:
        return
    code_dir = _PROJECT_ROOT / "code"
    package = types.ModuleType("hello_agents")
    package.__path__ = [str(code_dir)]
    package.__file__ = str(code_dir / "__init__.py")
    sys.modules["hello_agents"] = package


_bootstrap_package()

from hello_agents.agents.code_agent import CodeAgent
from hello_agents.core.config import Config
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.registry import ToolRegistry


class BenchmarkRunner(ABC):
    """Base class for all benchmark runners.

    Subclasses must implement :meth:`_load_tasks` and :meth:`_evaluate_task`.
    """

    benchmark_name: str = "base"

    def __init__(
        self,
        data_path: str,
        output_dir: str = "benchmark_results",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_steps: int = 30,
        timeout: int = 30,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_steps = max_steps
        self.timeout = timeout  # seconds for sandboxed code execution

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> Optional[str]:
        """Return a custom system prompt for this benchmark, or None for default."""
        return None

    def _create_agent(self, workspace: Path) -> CodeAgent:
        """Create a fresh CodeAgent with coding tools only (no web tools)."""
        from hello_agents.tools.builtin.bash import BashTool
        from hello_agents.tools.builtin.file_tools import (
            DeleteTool,
            EditTool,
            ListFilesTool,
            ReadTool,
            WriteTool,
        )
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool

        llm_kwargs: Dict[str, Any] = {"temperature": self.temperature}
        if self.model:
            llm_kwargs["model"] = self.model
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        if self.api_key:
            llm_kwargs["api_key"] = self.api_key

        llm = HelloAgentsLLM(**llm_kwargs)
        registry = ToolRegistry(verbose=False)

        config = Config.from_env()
        config.trace_enabled = False

        agent = CodeAgent(
            name="bench-agent",
            llm=llm,
            tool_registry=registry,
            project_root=str(workspace),
            working_dir=str(workspace),
            config=config,
            max_steps=self.max_steps,
            register_default_tools=False,  # We register manually below
            enable_task_tool=False,
            interactive=False,
            system_prompt=self._get_system_prompt(),
        )

        ws = str(workspace)
        registry.register_tool(ListFilesTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(GlobTool(project_root=ws, working_dir=ws))
        registry.register_tool(GrepTool(project_root=ws, working_dir=ws))
        registry.register_tool(ReadTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(WriteTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(DeleteTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(EditTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(BashTool(project_root=ws, working_dir=ws))

        return agent

    # ------------------------------------------------------------------
    # Sandboxed code execution
    # ------------------------------------------------------------------

    def _run_code_in_sandbox(
        self,
        code: str,
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Execute *code* in a subprocess and return ``(passed, output)``."""
        timeout = timeout or self.timeout
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
            )
            passed = result.returncode == 0
            output = (result.stdout + result.stderr).strip()
            return passed, output
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as exc:
            return False, f"Execution error: {exc}"

    def _run_script_in_sandbox(
        self,
        script_path: Path,
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Execute a Python script file in a subprocess."""
        timeout = timeout or self.timeout
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
            )
            passed = result.returncode == 0
            output = (result.stdout + result.stderr).strip()
            return passed, output
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as exc:
            return False, f"Execution error: {exc}"

    # ------------------------------------------------------------------
    # Data loading & evaluation (subclass hooks)
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load and return the task list from ``self.data_path``."""

    @abstractmethod
    def _evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on *task* and return a result dict.

        The returned dict must include at least:
        ``task_id``, ``passed`` (bool), ``error`` (str or None),
        ``elapsed_s`` (float).
        """

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    @staticmethod
    def _load_completed_ids(resume_file: Path) -> set:
        """Read a previous results JSONL and return the set of completed task_ids."""
        completed: set = set()
        if not resume_file.exists():
            return completed
        with open(resume_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    tid = record.get("task_id")
                    if tid is not None:
                        completed.add(tid)
                except json.JSONDecodeError:
                    continue
        return completed

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark and return a summary dict.

        Args:
            limit: Only run the first *limit* tasks.
            task_ids: Only run tasks whose ``task_id`` is in this list.
            dry_run: Print tasks without executing.
            resume: Path to a previous results ``.jsonl`` file.  Already-
                completed task IDs are skipped and new results are **appended**
                to the same file.
        """
        tasks = self._load_tasks()
        if task_ids:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.get("task_id") in id_set]
        if limit and limit > 0:
            tasks = tasks[:limit]

        # --- Resume handling ---
        completed_ids: set = set()
        if resume:
            resume_path = Path(resume)
            if not resume_path.exists():
                print(f"  ⚠ Resume file not found: {resume}")
                print(f"    Starting a fresh run instead.\n")
            else:
                completed_ids = self._load_completed_ids(resume_path)
                print(f"  ▶ Resuming from: {resume}")
                print(f"    Already completed: {len(completed_ids)} tasks")

        # Decide which file to write results to
        if resume and Path(resume).exists():
            results_file = Path(resume)
            # Derive timestamp from the resume filename for the summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"{self.benchmark_name}_{timestamp}.jsonl"

        print(f"\n{'=' * 60}")
        print(f"  Benchmark: {self.benchmark_name}")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Model: {self.model or '(from env)'}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Timeout: {self.timeout}s")
        if completed_ids:
            remaining = sum(1 for t in tasks if t.get("task_id", "") not in completed_ids)
            print(f"  Resume: {len(completed_ids)} done, {remaining} remaining")
        print(f"{'=' * 60}\n")

        if dry_run:
            for t in tasks:
                tid = t.get("task_id")
                tag = " [SKIP]" if tid in completed_ids else ""
                print(f"  [dry-run] {tid}{tag}")
            return {"benchmark": self.benchmark_name, "total": len(tasks), "dry_run": True}

        results: List[Dict[str, Any]] = []
        passed_count = 0
        total_time = 0.0
        skipped = 0

        for i, task in enumerate(tasks):
            task_id = task.get("task_id", f"task_{i}")

            if task_id in completed_ids:
                skipped += 1
                print(f"\n[{i + 1}/{len(tasks)}] {task_id}  [SKIP - already completed]")
                continue

            print(f"\n[{i + 1}/{len(tasks)}] {task_id}")

            try:
                result = self._evaluate_task(task)
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "passed": False,
                    "error": f"Runner exception: {exc}",
                    "elapsed_s": 0.0,
                }

            results.append(result)
            passed = result.get("passed")
            if passed is True:
                passed_count += 1
                print(f"  => PASS ({result['elapsed_s']:.1f}s)")
            elif passed is None:
                has_diff = result.get("has_diff", False)
                if has_diff:
                    print(f"  => DIFF ({result['elapsed_s']:.1f}s) awaiting Docker eval")
                else:
                    err_msg = (result.get("error") or "")[:80]
                    print(f"  => NO_DIFF ({result['elapsed_s']:.1f}s) {err_msg}")
            else:
                err_msg = (result.get("error") or "")[:80]
                print(f"  => FAIL ({result['elapsed_s']:.1f}s) {err_msg}")

            total_time += result.get("elapsed_s", 0)

            # Append results to file after each task (works for both new and resumed runs)
            with open(results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Summary (includes both previous and new results)
        evaluated = len(results)
        total_with_resumed = evaluated + skipped
        pass_rate = (passed_count / evaluated * 100) if evaluated > 0 else 0
        summary = {
            "benchmark": self.benchmark_name,
            "model": self.model or "(from env)",
            "total": len(tasks),
            "evaluated": evaluated,
            "skipped": skipped,
            "passed": passed_count,
            "failed": evaluated - passed_count,
            "pass_rate": round(pass_rate, 2),
            "total_time_s": round(total_time, 2),
            "avg_time_s": round(total_time / evaluated, 2) if evaluated > 0 else 0,
            "timestamp": timestamp,
            "results_file": str(results_file),
            "resumed_from": resume if resume else None,
        }

        summary_file = self.output_dir / f"{self.benchmark_name}_{timestamp}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        if skipped:
            print(f"  Resumed: {skipped} previously completed")
        print(f"  New results: {passed_count}/{evaluated} passed ({pass_rate:.1f}%)")
        if skipped:
            # Also show overall stats from the combined file
            all_ids = self._load_completed_ids(results_file)
            print(f"  Total in file: {len(all_ids)} tasks")
        print(f"  Time: {total_time:.1f}s total, {summary['avg_time_s']:.1f}s avg")
        print(f"  Output: {results_file}")
        print(f"  Summary: {summary_file}")
        print(f"{'=' * 60}\n")

        return summary
