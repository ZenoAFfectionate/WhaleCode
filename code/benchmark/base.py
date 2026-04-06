"""Base benchmark runner for Whale Code agent evaluation."""

from __future__ import annotations

import importlib
import inspect
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import tempfile
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional


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
CodeAgent = importlib.import_module("hello_agents.agents.code_agent").CodeAgent
Message = importlib.import_module("hello_agents.core.message").Message

try:
    from ._utils import (
        BenchmarkProgressManager,
        _clip_text,
        _display_width,
        _json_safe,
        _json_safe_full,
        build_benchmark_system_prompt as _build_benchmark_system_prompt,
        build_minimal_child_env,
        build_progress_update as _build_progress_update,
        build_trajectory_payload as _build_trajectory_payload,
        build_trajectory_readme as _build_trajectory_readme,
        describe_progress_update as _describe_progress_update,
        load_completed_ids as _load_completed_ids,
        load_result_records as _load_result_records,
        latest_result_records as _latest_result_records,
        progress_updates_to_events as _progress_updates_to_events,
        summarize_result_records as _summarize_result_records,
        trajectory_dir_for_task as _trajectory_dir_for_task,
        truncate_feedback,
        upsert_result_record as _upsert_result_record,
        write_result_records as _write_result_records,
    )
except ImportError:
    from _utils import (
        BenchmarkProgressManager,
        _clip_text,
        _display_width,
        _json_safe,
        _json_safe_full,
        build_benchmark_system_prompt as _build_benchmark_system_prompt,
        build_minimal_child_env,
        build_progress_update as _build_progress_update,
        build_trajectory_payload as _build_trajectory_payload,
        build_trajectory_readme as _build_trajectory_readme,
        describe_progress_update as _describe_progress_update,
        load_completed_ids as _load_completed_ids,
        load_result_records as _load_result_records,
        latest_result_records as _latest_result_records,
        progress_updates_to_events as _progress_updates_to_events,
        summarize_result_records as _summarize_result_records,
        trajectory_dir_for_task as _trajectory_dir_for_task,
        truncate_feedback,
        upsert_result_record as _upsert_result_record,
        write_result_records as _write_result_records,
    )


_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "data" / "_results"
_DEFAULT_TRAJECTORY_DIR = _PROJECT_ROOT / "data" / "_trajectory"
__all__ = [
    "BENCHMARK_BASE_SYSTEM_PROMPT",
    "BenchmarkCodeAgent",
    "BenchmarkProgressManager",
    "BenchmarkRunner",
    "_display_width",
    "build_minimal_child_env",
    "truncate_feedback",
]


class BenchmarkCodeAgent(CodeAgent):
    """Benchmark-specific agent that records events and suppresses console spam."""

    def __init__(
        self,
        *args,
        task_id: str,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        self.task_id = task_id
        self._event_sink = event_sink
        self.benchmark_events: List[Dict[str, Any]] = []
        self._benchmark_required_tool_choice = False
        self._benchmark_protocol_errors = 0
        super().__init__(*args, **kwargs)

    @staticmethod
    def _contains_embedded_tool_markup(text: Optional[str]) -> bool:
        if not text:
            return False
        return bool(re.search(r"<tool_call>|<function=[^>]+>|<parameter=", text))

    def _console(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        return

    def _render_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        full_payload = _json_safe_full(payload)
        safe_payload = _json_safe(payload, max_string=2000)
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "payload": full_payload,
        }
        self.benchmark_events.append(event_record)
        if self._event_sink is not None:
            try:
                self._event_sink(event_type, safe_payload)
            except Exception:
                pass

    def run(self, input_text: str, **kwargs) -> str:
        tool_choice = kwargs.get("tool_choice")
        self._benchmark_required_tool_choice = tool_choice == "required"
        self._benchmark_protocol_errors = 0
        try:
            return super().run(input_text, **kwargs)
        finally:
            self._benchmark_required_tool_choice = False

    def _resolve_no_tool_call_response(
        self,
        response_message: Any,
        text_content: str,
        *,
        structured_output: Optional[Any] = None,
        fallback_text: str = "",
        reasoning_content: Optional[str] = None,
        reasoning_source: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        if self._benchmark_required_tool_choice and structured_output is None:
            self._benchmark_protocol_errors += 1
            reasoning_has_tool_markup = self._contains_embedded_tool_markup(reasoning_content)
            response_unfinished = self._response_unfinished_flag(response_message)

            self._render_event(
                "protocol_error",
                {
                    "task_id": self.task_id,
                    "retry_count": self._benchmark_protocol_errors,
                    "error": "tool_choice='required' but assistant returned no structured tool_calls",
                    "text_content_length": len(text_content or ""),
                    "reasoning_content_length": len(reasoning_content or ""),
                    "reasoning_contains_tool_markup": reasoning_has_tool_markup,
                    "response_unfinished": response_unfinished,
                    "reasoning_source": reasoning_source,
                    "text_content": _clip_text(text_content, 400),
                    "reasoning_excerpt": _clip_text(reasoning_content, 600),
                },
            )

            feedback_lines = [
                "Protocol error: the previous assistant response did not contain any structured tool_calls.",
                "The benchmark is running with tool_choice='required'.",
            ]
            if reasoning_has_tool_markup:
                feedback_lines.append(
                    "Do not place tool-call markup inside the reasoning field. Emit a real tool_calls response instead."
                )
            elif text_content.strip():
                feedback_lines.append(
                    "Do not answer in plain text at this step. Emit a real tool_calls response instead."
                )
            feedback_lines.append(
                "Retry now and return exactly one valid structured tool call."
            )

            self._append_history_message(
                Message("\n".join(feedback_lines), "user"),
                allow_compact=False,
            )
            return True, None, "protocol_error"

        return super()._resolve_no_tool_call_response(
            response_message,
            text_content,
            structured_output=structured_output,
            fallback_text=fallback_text,
            reasoning_content=reasoning_content,
            reasoning_source=reasoning_source,
        )


def _evaluate_task_in_subprocess(
    runner: "BenchmarkRunner",
    task: Dict[str, Any],
    result_queue: Any,
    progress_queue: Any,
    task_id: str,
) -> None:
    """Execute ``runner._evaluate_task`` and send a serializable payload back."""
    try:
        runner._bind_progress_queue(task_id, progress_queue)
        result_queue.put({"ok": True, "result": runner._evaluate_task(task)})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        runner._bind_progress_queue(None, None)


BENCHMARK_BASE_SYSTEM_PROMPT: str = _build_benchmark_system_prompt(_PROJECT_ROOT)


class BenchmarkRunner(ABC):
    """Base class for all benchmark runners.

    Subclasses must implement :meth:`_load_tasks` and :meth:`_evaluate_task`.
    """

    benchmark_name: str = "base"

    @staticmethod
    def add_shared_run_args(
        parser,
        *,
        default_temperature: float = 1.0,
        default_max_steps: int = 64,
        default_timeout: int = 60,
        timeout_help: Optional[str] = None,
        include_task_timeout: bool = False,
        default_task_timeout: int = 1200,
    ) -> None:
        parser.add_argument("--output-dir", default=str(_DEFAULT_RESULTS_DIR))
        parser.add_argument("--temperature", type=float, default=default_temperature)
        parser.add_argument("--max-steps", type=int, default=default_max_steps)
        if timeout_help is None:
            parser.add_argument("--timeout", type=int, default=default_timeout)
        else:
            parser.add_argument("--timeout", type=int, default=default_timeout, help=timeout_help)
        if include_task_timeout:
            parser.add_argument("--task-timeout", type=int, default=default_task_timeout)
        parser.add_argument("--trajectory-dir", default=str(_DEFAULT_TRAJECTORY_DIR))
        parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
        parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
        parser.add_argument("--resume", default=None, help="Resume from a previous .jsonl results file")
        parser.add_argument("--dry-run", action="store_true")

    def __init__(
        self,
        data_path: str,
        output_dir: str = str(_DEFAULT_RESULTS_DIR),
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_steps: int = 30,
        timeout: int = 30,
        task_timeout: int = 1200,
        trajectory_dir: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else _DEFAULT_TRAJECTORY_DIR
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_steps = max_steps
        self.timeout = timeout  # seconds for sandboxed code execution
        self.task_timeout = task_timeout  # seconds for one end-to-end benchmark task
        self._progress_manager: Optional[BenchmarkProgressManager] = None
        self._progress_queue = None
        self._current_task_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> Optional[str]:
        """Return a custom system prompt for this benchmark, or None for default."""
        return None

    def _load_jsonl_tasks(
        self,
        *,
        task_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                if task_transform is not None:
                    task = task_transform(task)
                tasks.append(task)
        return tasks

    def _make_workspace(self, prefix: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=prefix))

    def _build_result(
        self,
        task_id: str,
        *,
        passed: Optional[bool],
        error: Optional[str],
        agent_response: str = "",
        start_time: Optional[float] = None,
        elapsed_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if elapsed_s is None:
            elapsed_s = 0.0 if start_time is None else round(time.time() - start_time, 2)
        result: Dict[str, Any] = {
            "task_id": task_id,
            "passed": passed,
            "error": error,
            "agent_response": (agent_response or "")[:500],
            "elapsed_s": round(float(elapsed_s), 2),
        }
        if extra:
            result.update(extra)
        return result

    @staticmethod
    def _benchmark_agent_run_kwargs(run_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Default benchmark agent settings.

        Benchmarks now rely on explicit tool calls plus ``Finish`` to terminate.
        Using ``tool_choice="required"`` prevents the model from "thinking about"
        a tool call in plain text or reasoning metadata and then being treated as
        a completed no-tool response by the ReAct loop.
        """
        effective_kwargs = dict(run_kwargs or {})
        effective_kwargs.setdefault("tool_choice", "required")
        return effective_kwargs

    def _run_agent_prompt(
        self,
        *,
        agent: CodeAgent,
        task_id: str,
        prompt_text: str,
        start_time: float,
        run_kwargs: Optional[Dict[str, Any]] = None,
        error_extra: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        try:
            effective_kwargs = self._benchmark_agent_run_kwargs(run_kwargs)
            if effective_kwargs:
                run_signature = inspect.signature(agent.run)
                accepts_var_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in run_signature.parameters.values()
                )
                if not accepts_var_kwargs:
                    effective_kwargs = {
                        key: value
                        for key, value in effective_kwargs.items()
                        if key in run_signature.parameters
                    }
            agent_response = agent.run(prompt_text, **effective_kwargs)
            if agent_response is None:
                normalized_response = ""
            else:
                normalized_response = str(agent_response).strip()
            return normalized_response, None
        except Exception as exc:
            return "", self._build_result(
                task_id,
                passed=False,
                error=f"Agent error: {exc}",
                start_time=start_time,
                extra=error_extra,
            )

    def _missing_output_result(
        self,
        task_id: str,
        *,
        path_label: str,
        start_time: Optional[float] = None,
        elapsed_s: Optional[float] = None,
        agent_response: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._build_result(
            task_id,
            passed=False,
            error=f"{path_label} not found after agent run",
            agent_response=agent_response,
            start_time=start_time,
            elapsed_s=elapsed_s,
            extra=extra,
        )

    def _build_llm_kwargs(self) -> Dict[str, Any]:
        llm_kwargs: Dict[str, Any] = {"temperature": self.temperature}
        if self.model:
            llm_kwargs["model"] = self.model
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        if self.api_key:
            llm_kwargs["api_key"] = self.api_key
        return llm_kwargs

    def _configure_agent_config(self, config: Any) -> Any:
        config.trace_enabled = False
        return config

    def _build_agent(self, *, workspace: Path, registry: Any, config: Any, llm: Any) -> BenchmarkCodeAgent:
        task_id = self._current_task_id or workspace.name
        return BenchmarkCodeAgent(
            name="bench-agent",
            llm=llm,
            tool_registry=registry,
            project_root=str(workspace),
            working_dir=str(workspace),
            config=config,
            max_steps=self.max_steps,
            register_default_tools=False,
            enable_task_tool=False,
            interactive=False,
            system_prompt=self._get_system_prompt() or BENCHMARK_BASE_SYSTEM_PROMPT,
            task_id=task_id,
            event_sink=lambda event_type, payload: self._emit_progress_event(task_id, event_type, payload),
        )

    def _register_agent_tools(self, *, registry: Any, workspace: Path, agent: BenchmarkCodeAgent) -> None:
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
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool

        ws = str(workspace)
        registry.register_tool(ListFilesTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(GlobTool(project_root=ws, working_dir=ws))
        registry.register_tool(GrepTool(project_root=ws, working_dir=ws))
        registry.register_tool(ReadTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(WriteTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(DeleteTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(EditTool(project_root=ws, working_dir=ws, registry=registry))
        registry.register_tool(BashTool(project_root=ws, working_dir=ws))

        # Persist session todo state outside the repo to avoid diff pollution.
        todo_dir = Path(tempfile.gettempdir()) / "whale_bench_tasks" / uuid.uuid4().hex[:8]
        todo_dir.mkdir(parents=True, exist_ok=True)
        registry.register_tool(
            TodoWriteTool(
                project_root=ws,
                persistence_dir=str(todo_dir),
                session_id=agent.session_id,
            )
        )

    def _create_agent(self, workspace: Path) -> CodeAgent:
        """Create a fresh CodeAgent with coding tools only (no web tools)."""
        from hello_agents.core.config import Config
        from hello_agents.core.llm import HelloAgentsLLM
        from hello_agents.tools.registry import ToolRegistry

        config = self._configure_agent_config(Config.from_env())
        llm = HelloAgentsLLM(**self._build_llm_kwargs())
        registry = ToolRegistry(config=config, verbose=False)
        agent = self._build_agent(workspace=workspace, registry=registry, config=config, llm=llm)
        self._register_agent_tools(registry=registry, workspace=workspace, agent=agent)
        return agent

    def _bind_progress_queue(self, task_id: Optional[str], progress_queue: Any) -> None:
        self._current_task_id = task_id
        self._progress_queue = progress_queue

    def _emit_progress_event(self, task_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        update = _build_progress_update(task_id, event_type, payload)
        if self._progress_queue is not None:
            try:
                self._progress_queue.put_nowait(update)
            except Exception:
                pass
            return
        self._handle_progress_update(update)

    def _handle_progress_update(self, update: Dict[str, Any]) -> None:
        if self._progress_manager is None:
            return
        step, status, detail = _describe_progress_update(update)
        if detail is not None or status is not None or step is not None:
            self._progress_manager.update(step=step, status=status, detail=detail)

    def _drain_progress_queue(self, progress_queue: Any) -> None:
        while progress_queue is not None:
            try:
                update = progress_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            self._handle_progress_update(update)

    _progress_updates_to_events = staticmethod(_progress_updates_to_events)

    def _save_timeout_stub_trajectory(
        self,
        *,
        task: Dict[str, Any],
        result: Dict[str, Any],
        progress_updates: List[Dict[str, Any]],
    ) -> None:
        agent_stub = SimpleNamespace(
            benchmark_events=self._progress_updates_to_events(progress_updates),
            tool_registry=SimpleNamespace(read_metadata_cache={}),
            get_history=lambda: [],
        )
        try:
            self._save_task_trajectory(
                task=task,
                workspace=None,
                agent=agent_stub,
                prompt_texts=[],
                result=result,
                artifact_paths=None,
                extra={"timeout_stub": True},
            )
        except Exception:
            pass

    def _save_task_trajectory(
        self,
        *,
        task: Dict[str, Any],
        workspace: Optional[Path],
        agent: Optional[CodeAgent],
        prompt_texts: Optional[List[str]] = None,
        result: Optional[Dict[str, Any]] = None,
        artifact_paths: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = str(task.get("task_id") or uuid.uuid4().hex)
        task_dir = _trajectory_dir_for_task(self.trajectory_dir, self.benchmark_name, task_id)
        trajectory_path = task_dir / "trajectory.json"
        payload = _build_trajectory_payload(
            benchmark_name=self.benchmark_name,
            task_id=task_id,
            task=task,
            workspace=workspace,
            agent=agent,
            prompt_texts=prompt_texts,
            result=result,
            artifact_paths=artifact_paths,
            extra=extra,
        )
        trajectory_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (task_dir / "README.md").write_text(_build_trajectory_readme(payload), encoding="utf-8")
        return str(trajectory_path)

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

    def _evaluate_task_with_timeout(self, task: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Run ``_evaluate_task`` with an end-to-end wall-clock timeout."""
        if self.task_timeout <= 0:
            return self._evaluate_task(task)

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        result_queue = ctx.Queue(maxsize=1)
        progress_queue = ctx.Queue(maxsize=512)
        process = ctx.Process(
            target=_evaluate_task_in_subprocess,
            args=(self, task, result_queue, progress_queue, task_id),
            name=f"{self.benchmark_name}-{task_id}",
            daemon=True,
        )
        process.start()

        deadline = time.time() + self.task_timeout
        timed_out = False
        progress_updates: List[Dict[str, Any]] = []
        while process.is_alive():
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break
            try:
                update = progress_queue.get(timeout=min(0.2, max(0.01, remaining)))
            except queue.Empty:
                update = None
            except Exception:
                update = None
            if update is not None:
                progress_updates.append(update)
                self._handle_progress_update(update)

        self._drain_progress_queue(progress_queue)

        if timed_out and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5)
            timeout_result = {
                "task_id": task_id,
                "passed": False,
                "error": f"Timeout: problem solving exceeded {self.task_timeout}s",
                "elapsed_s": float(self.task_timeout),
                "timeout": True,
            }
            self._save_timeout_stub_trajectory(
                task=task,
                result=timeout_result,
                progress_updates=progress_updates,
            )
            return timeout_result

        process.join(timeout=1)
        try:
            payload = result_queue.get_nowait()
        except Exception:
            payload = None

        if not payload:
            return {
                "task_id": task_id,
                "passed": False,
                "error": "Runner process exited without returning a result",
                "elapsed_s": 0.0,
            }

        if not payload.get("ok"):
            raise RuntimeError(payload.get("traceback") or payload.get("error") or "Unknown runner error")

        return payload["result"]

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    _load_completed_ids = staticmethod(_load_completed_ids)
    _load_result_records = staticmethod(_load_result_records)
    _latest_result_records = staticmethod(_latest_result_records)
    _write_result_records = staticmethod(_write_result_records)
    _upsert_result_record = staticmethod(_upsert_result_record)
    _summarize_result_records = staticmethod(_summarize_result_records)

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
                completed task IDs are skipped and rerun task IDs replace their
                existing records in the same file.
        """
        tasks = self._load_tasks()
        if task_ids:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.get("task_id") in id_set]
        if limit and limit > 0:
            tasks = tasks[:limit]

        # --- Resume handling ---
        completed_ids: set = set()
        resume_path: Optional[Path] = Path(resume) if resume else None
        persisted_records: List[Dict[str, Any]] = []
        record_index: Dict[str, int] = {}
        if resume:
            if not resume_path.exists():
                print(f"  ▶ Resume target does not exist yet: {resume}")
                print("    A new results file will be created at this path.\n")
            else:
                raw_records = self._load_result_records(resume_path)
                persisted_records = self._latest_result_records(raw_records)
                if len(persisted_records) != len(raw_records):
                    duplicate_count = len(raw_records) - len(persisted_records)
                    self._write_result_records(resume_path, persisted_records)
                    print(f"  ▶ Cleaned {duplicate_count} duplicate result record(s) before resuming")
                completed_ids = self._load_completed_ids(resume_path)
                print(f"  ▶ Resuming from: {resume}")
                print(f"    Already completed: {len(completed_ids)} tasks")

        # Decide which file to write results to
        if resume_path is not None:
            results_file = resume_path
            results_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"{self.benchmark_name}_{timestamp}.jsonl"

        if not persisted_records and results_file.exists():
            persisted_records = self._latest_result_records(self._load_result_records(results_file))
        for idx, record in enumerate(persisted_records):
            task_id = record.get("task_id")
            if task_id is not None:
                record_index[str(task_id)] = idx

        print(f"\n{'=' * 60}")
        print(f"  Benchmark: {self.benchmark_name}")
        print(f"  Tasks: {len(tasks)}")
        model_label = self.model or os.getenv("LLM_MODEL_ID") or "(from env)"
        print(f"  Model: {model_label}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Task timeout: {self.task_timeout}s")
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
        progress = BenchmarkProgressManager(self.benchmark_name, len(tasks))
        self._progress_manager = progress
        progress.start()

        try:
            for i, task in enumerate(tasks):
                task_id = str(task.get("task_id", f"task_{i}"))
                self._current_task_id = task_id

                if task_id in completed_ids:
                    skipped += 1
                    progress.skip_task(i + 1, task_id)
                    continue

                progress.begin_task(i + 1, task_id)

                try:
                    result = self._evaluate_task_with_timeout(task, task_id)
                except Exception as exc:
                    result = {
                        "task_id": task_id,
                        "passed": False,
                        "error": f"Runner exception: {exc}",
                        "elapsed_s": 0.0,
                    }

                self._drain_progress_queue(self._progress_queue)
                results.append(result)
                if result.get("passed") is True:
                    passed_count += 1
                total_time += result.get("elapsed_s", 0)
                progress.finish_task(result)

                self._upsert_result_record(persisted_records, record_index, result)
                self._write_result_records(results_file, persisted_records)
        finally:
            progress.close()
            self._progress_manager = None
            self._progress_queue = None
            self._current_task_id = None

        # Summary
        evaluated = len(results)
        new_pass_rate = (passed_count / evaluated * 100) if evaluated > 0 else 0
        combined = self._summarize_result_records(persisted_records)
        summary = {
            "benchmark": self.benchmark_name,
            "model": self.model or "(from env)",
            "total": len(tasks),
            "evaluated": combined["tasks"],
            "new_evaluated": evaluated,
            "skipped": skipped,
            "passed": combined["passed"],
            "failed": combined["failed"],
            "unfinished": combined["unfinished"],
            "pass_rate": combined["pass_rate"],
            "total_time_s": combined["total_time_s"],
            "avg_time_s": combined["avg_time_s"],
            "records_in_file": combined["records_in_file"],
            "new_passed": passed_count,
            "new_failed": sum(1 for r in results if r.get("passed") is False),
            "new_unfinished": sum(1 for r in results if r.get("passed") is None),
            "new_pass_rate": round(new_pass_rate, 2),
            "new_total_time_s": round(total_time, 2),
            "new_avg_time_s": round(total_time / evaluated, 2) if evaluated > 0 else 0,
            "timestamp": timestamp,
            "results_file": str(results_file),
            "trajectory_dir": str(self.trajectory_dir),
            "resumed_from": resume if resume else None,
        }

        summary_file = self.output_dir / f"{self.benchmark_name}_{timestamp}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        if skipped:
            print(f"  Resumed: {skipped} previously completed")
        print(
            f"  Combined results: {combined['passed']}/{combined['tasks']} passed "
            f"({combined['pass_rate']:.1f}%)"
        )
        if combined["unfinished"]:
            print(f"  Combined unfinished: {combined['unfinished']}")
        print(
            f"  New results: {passed_count}/{evaluated} passed "
            f"({new_pass_rate:.1f}%)"
        )
        print(
            f"  Combined time: {combined['total_time_s']:.1f}s total, "
            f"{combined['avg_time_s']:.1f}s avg"
        )
        print(
            f"  New time: {total_time:.1f}s total, "
            f"{summary['new_avg_time_s']:.1f}s avg"
        )
        print(f"  Output: {results_file}")
        print(f"  Trajectory: {self.trajectory_dir}")
        print(f"  Summary: {summary_file}")
        print(f"{'=' * 60}\n")

        return summary
