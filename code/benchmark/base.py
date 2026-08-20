"""Base benchmark runner for Whale Code agent evaluation."""

from __future__ import annotations

import importlib
import inspect
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
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
        append_result_record as _append_result_record,
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
    from .runtime.config import BenchmarkRuntimeConfig
except ImportError:
    from _utils import (
        BenchmarkProgressManager,
        _clip_text,
        _display_width,
        _json_safe,
        _json_safe_full,
        append_result_record as _append_result_record,
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
    from runtime.config import BenchmarkRuntimeConfig


_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "result" / "_results"
_DEFAULT_TRAJECTORY_DIR = _PROJECT_ROOT / "result" / "_trajectory"
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
        self._benchmark_consecutive_code_writes = 0
        self._benchmark_verification_loop_detected = False
        super().__init__(*args, **kwargs)

    def _console(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        return

    def _apply_builtin_tool_prompt(self, messages: list) -> None:
        """Override: in benchmark mode, the agent MUST call Finish to submit.

        The default builtin-tool instruction says "you may answer in plain text",
        which is correct for interactive CLI use but wrong for benchmarks —
        the evaluation runner only processes ``Finish`` tool calls.
        """
        instruction = (
            "Builtin control tools are available.\n"
            "- Use Thought to record concise reasoning or planning when helpful.\n"
            "- Use normal tools to gather information and perform work.\n"
            "- IMPORTANT: You MUST call Finish to submit your final answer.\n"
            "  Do NOT answer in plain text — only Finish tool calls are\n"
            "  processed by the evaluation system. Plain text answers will be\n"
            "  treated as incomplete and the task will fail.\n"
            "- Call Finish alone after all other tool work is complete."
        )
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content", "")
            message["content"] = f"{content}\n\n{instruction}" if content else instruction
            return
        messages.insert(0, {"role": "system", "content": instruction})

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
        state: Optional[Any] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        if self._benchmark_required_tool_choice and structured_output is None:
            self._benchmark_protocol_errors += 1
            response_unfinished = self._response_unfinished_flag(response_message)

            self._render_event(
                "protocol_error",
                {
                    "task_id": self.task_id,
                    "retry_count": self._benchmark_protocol_errors,
                    "error": "tool_choice='required' but assistant returned no structured tool_calls",
                    "text_content_length": len(text_content or ""),
                    "reasoning_content_length": len(reasoning_content or ""),
                    "response_unfinished": response_unfinished,
                    "reasoning_source": reasoning_source,
                    "text_content": _clip_text(text_content, 400),
                    "reasoning_excerpt": _clip_text(reasoning_content, 600),
                },
            )

            feedback_lines = [
                "Protocol error: the previous assistant response did not contain any structured tool_calls.",
                "The benchmark is running with tool_choice='required'.",
                "Do not answer in plain text at this step. Emit a real tool_calls response instead.",
                "Retry now and return exactly one valid structured tool call.",
            ]

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
            state=state,
        )

    def _update_stagnation_state(
        self,
        tool_name: str,
        tool_call_id: str,
        result_content: str,
        current_step: int,
        state: Any,
        *,
        tool_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extend stagnation detection to catch benchmark-specific verification loops.

        The base stagnation detector only catches repeated failed edits and
        identical test results.  In benchmark mode the most common stuck pattern
        is the *verification loop*: the model keeps writing new .py files and
        running them, each producing slightly different output, without ever
        calling Finish.  This method adds a counter that triggers stagnation
        after 6 consecutive code writes without a finalising tool call.
        """
        super()._update_stagnation_state(
            tool_name, tool_call_id, result_content, current_step, state, tool_arguments=tool_arguments
        )

        # --- verification-loop detection -----------------------------------
        if tool_name == "Write":
            path = ""
            if isinstance(tool_arguments, dict):
                path = str(tool_arguments.get("path") or tool_arguments.get("file_path") or "")
            if path.endswith(".py"):
                self._benchmark_consecutive_code_writes += 1
            # Writing a non-Python file resets the counter (e.g. .txt, .md)
            elif path:
                self._benchmark_consecutive_code_writes = 0
        elif tool_name == "Finish":
            self._benchmark_consecutive_code_writes = 0
        elif tool_name not in ("Edit", "Read", "Bash", "TodoWrite", "Thought"):
            # An unrelated tool call resets the counter
            self._benchmark_consecutive_code_writes = 0

        if self._benchmark_consecutive_code_writes >= 6:
            self._benchmark_verification_loop_detected = True
            state.stagnation_detected = True
            self._render_event(
                "stagnation_detected",
                {
                    "reason": f"Verification loop: {self._benchmark_consecutive_code_writes} consecutive .py writes without Finish",
                    "step": current_step,
                },
            )


def _run_task_in_subprocess(
    runner: "BenchmarkRunner",
    task: Dict[str, Any],
    result_queue: Any,
    progress_queue: Any,
    task_id: str,
) -> None:
    """Execute ``runner._run_task`` and send a serializable payload back."""
    os.setpgid(0, 0)  # create independent process group for clean tree-kill on timeout
    try:
        runner._current_task_id = task_id
        runner._progress_queue = progress_queue
        result_queue.put({"ok": True, "result": runner._run_task(task)})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        runner._current_task_id = None
        runner._progress_queue = None


BENCHMARK_BASE_SYSTEM_PROMPT: str = _build_benchmark_system_prompt(_PROJECT_ROOT)


class BenchmarkRunner(ABC):
    """Base class for all benchmark runners.

    Subclasses must implement :meth:`_load_tasks` and :meth:`_run_task`.
    """

    benchmark_name: str = "base"

    # Runtime sandbox profile used when recording summary metadata.
    # Docker-based benchmarks should override this (e.g. ``"repo_docker"``).
    runtime_profile: str = "python_strict"

    # ========== 1. Configuration & Data Loading ==========

    @staticmethod
    def add_shared_run_args(
        parser,
        *,
        default_temperature: float = 1.0,
        default_max_steps: int = 64,
        default_timeout: int = 60,
        default_max_tokens: int = 16384,
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
        parser.add_argument("--max-tokens", type=int, default=default_max_tokens,
                            help="Max output tokens per LLM call (reasoning+content). 0 disables the cap.")
        if include_task_timeout:
            parser.add_argument("--task-timeout", type=int, default=default_task_timeout)
        parser.add_argument("--trajectory-dir", default=str(_DEFAULT_TRAJECTORY_DIR))
        parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
        parser.add_argument("--task-ids", nargs="*", default=None, help="Specific task IDs to run")
        parser.add_argument("--resume", default=None, help="Resume from a previous .jsonl results file")
        parser.add_argument("--fresh", action="store_true", help="Ignore existing results file and start a fresh run (overwrites {benchmark}.jsonl)")
        parser.add_argument("--dry-run", action="store_true")

    @staticmethod
    def runner_kwargs_from_args(args: Any, *, include_task_timeout: bool = False) -> Dict[str, Any]:
        kwargs = {
            "output_dir": args.output_dir,
            "model": getattr(args, "model", None),
            "base_url": getattr(args, "base_url", None),
            "api_key": getattr(args, "api_key", None),
            "temperature": args.temperature,
            "max_steps": args.max_steps,
            "timeout": args.timeout,
            "trajectory_dir": args.trajectory_dir,
            "max_tokens": getattr(args, "max_tokens", None),
        }
        if include_task_timeout:
            kwargs["task_timeout"] = args.task_timeout
        return kwargs

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
        max_tokens: Optional[int] = None,
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
        self.max_tokens = max_tokens  # None → use env or built-in default
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

    @abstractmethod
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load and return the task list from ``self.data_path``."""

    # ========== 2. Agent Factory ==========

    def _configure_agent_config(self, config: Any) -> Any:
        """Allow subclasses to tweak agent config after benchmark defaults are applied."""
        return config

    @staticmethod
    def _benchmark_agent_run_kwargs(run_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Default benchmark agent settings.

        Benchmarks now rely on explicit tool calls plus ``Finish`` to terminate.
        Using ``tool_choice="required"`` prevents the model from "thinking about"
        a tool call in plain text or reasoning metadata and then being treated as
        a completed no-tool response by the ReAct loop.
        """
        effective_kwargs = dict(run_kwargs or {})
        effective_kwargs.setdefault("tool_choice", "auto")
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
        registry.register_tool(BashTool(project_root=ws, working_dir=ws, config=agent.config, output_truncator=agent.truncator))

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

        # LSP tools — help the agent navigate large codebases (e.g. SWEV).
        # These degrade gracefully when the language server is not installed.
        from hello_agents.tools.lsp import (
            LSPDefinitionTool,
            LSPDiagnosticsTool,
            LSPHoverTool,
            LSPReferencesTool,
            LSPManager,
        )
        lsp_manager = LSPManager(workspace)
        registry.register_tool(LSPDefinitionTool(workspace_root=ws, manager=lsp_manager))
        registry.register_tool(LSPReferencesTool(workspace_root=ws, manager=lsp_manager))
        registry.register_tool(LSPHoverTool(workspace_root=ws, manager=lsp_manager))
        registry.register_tool(LSPDiagnosticsTool(workspace_root=ws, manager=lsp_manager))

    def _create_agent(self, workspace: Path) -> CodeAgent:
        """Create a fresh CodeAgent with coding tools only (no web tools)."""
        from hello_agents.core.config import Config
        from hello_agents.core.llm import HelloAgentsLLM
        from hello_agents.tools.registry import ToolRegistry

        config = Config.from_env()
        config.trace_enabled = False
        config = self._configure_agent_config(config)

        llm_kwargs: Dict[str, Any] = {"temperature": self.temperature}
        # Cap per-request output tokens (reasoning + content). Without this,
        # reasoning models can spend the whole task budget in one generation.
        # Priority: CLI --max-tokens > env BENCH_LLM_MAX_TOKENS > built-in default 16384.
        # Set to 0 to disable the cap.
        if self.max_tokens is not None:
            max_tokens = self.max_tokens
        else:
            max_tokens = int(os.getenv("BENCH_LLM_MAX_TOKENS", "16384"))
        if max_tokens > 0:
            llm_kwargs["max_tokens"] = max_tokens
        if self.model:
            llm_kwargs["model"] = self.model
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        if self.api_key:
            llm_kwargs["api_key"] = self.api_key

        llm = HelloAgentsLLM(**llm_kwargs)
        registry = ToolRegistry(config=config, verbose=False)
        task_id = self._current_task_id or workspace.name
        agent = BenchmarkCodeAgent(
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
        self._register_agent_tools(registry=registry, workspace=workspace, agent=agent)
        return agent

    # ========== 3. Task Execution & Retry Loop ==========

    def _use_subprocess_task_timeout(self) -> bool:
        """Return True when ``task_timeout`` should wrap task execution in a subprocess."""
        return True

    def _run_controlled_submission_rounds(
        self,
        *,
        task_id: str,
        agent: CodeAgent,
        start_time: float,
        initial_prompt: str,
        max_rounds: int,
        prompt_history: List[str],
        evaluate_submission: Callable[[int, str], Dict[str, Any]],
        retry_prompt_builder: Callable[[int, str], str],
        run_kwargs_builder: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
        error_extra_builder: Optional[Callable[[int], Optional[Dict[str, Any]]]] = (
            lambda round_idx: {"submission_rounds": round_idx}
        ),
        feedback_max_lines: int = 80,
        feedback_max_chars: int = 12000,
    ) -> Dict[str, Any]:
        """Run repeated controlled submissions with bounded evaluator feedback.

        ``feedback_max_lines`` / ``feedback_max_chars``: 评测反馈进入下一轮
        retry prompt 前的统一截断上限 (与 lcb6 的防线一致)。评测器输出
        (stdout+stderr) 无天然上限 — 大 repr / 长 traceback 直接进 prompt
        会撑爆 LLM 上下文并随轮次在对话历史中累积, 必须在此统一设界。
        """
        total_rounds = max(1, int(max_rounds))
        agent_response = ""
        evaluation: Dict[str, Any] = {"passed": False, "output": ""}
        feedback = ""

        for round_idx in range(1, total_rounds + 1):
            prompt_text = (
                initial_prompt
                if round_idx == 1
                else retry_prompt_builder(round_idx, feedback)
            )
            prompt_history.append(prompt_text)

            run_kwargs = run_kwargs_builder(round_idx) if run_kwargs_builder is not None else None
            error_extra = error_extra_builder(round_idx) if error_extra_builder is not None else None
            agent_response, error_result = self._run_agent_prompt(
                agent=agent,
                task_id=task_id,
                prompt_text=prompt_text,
                start_time=start_time,
                run_kwargs=run_kwargs,
                error_extra=error_extra,
            )
            if error_result is not None:
                return {
                    "agent_response": agent_response,
                    "early_result": error_result,
                    "rounds_used": round_idx,
                }

            evaluation = evaluate_submission(round_idx, agent_response) or {}
            early_result = evaluation.get("result")
            if early_result is not None:
                return {
                    "agent_response": agent_response,
                    "early_result": early_result,
                    "rounds_used": round_idx,
                }

            passed = bool(evaluation.get("passed"))
            output = str(evaluation.get("output") or "")
            if passed or evaluation.get("force_stop"):
                return {
                    "agent_response": agent_response,
                    "early_result": None,
                    "rounds_used": round_idx,
                    "passed": passed,
                    "output": output,
                }

            feedback = truncate_feedback(
                str(evaluation.get("feedback") or output),
                max_lines=feedback_max_lines,
                max_chars=feedback_max_chars,
            )

        return {
            "agent_response": agent_response,
            "early_result": None,
            "rounds_used": total_rounds,
            "passed": bool(evaluation.get("passed")),
            "output": str(evaluation.get("output") or ""),
        }

    @abstractmethod
    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on *task* and return a result dict.

        The returned dict must include at least:
        ``task_id``, ``passed`` (bool), ``error`` (str or None),
        ``elapsed_s`` (float).
        """

    def evaluate(self, task: Dict[str, Any], *, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute one benchmark task with optional end-to-end wall-clock timeout."""
        resolved_task_id = str(task_id or task.get("task_id") or uuid.uuid4().hex)

        if self.task_timeout <= 0 or not self._use_subprocess_task_timeout():
            return self._run_task(task)

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        result_queue = ctx.Queue(maxsize=1)
        progress_queue = ctx.Queue(maxsize=512)
        process = ctx.Process(
            target=_run_task_in_subprocess,
            args=(self, task, result_queue, progress_queue, resolved_task_id),
            name=f"{self.benchmark_name}-{resolved_task_id}",
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

        while progress_queue is not None:
            try:
                update = progress_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            progress_updates.append(update)
            self._handle_progress_update(update)

        if timed_out and process.is_alive():
            # Kill entire process group (child called os.setpgid(0,0) on entry).
            # Falls back gracefully when PG info is unavailable.
            child_pid = process.pid
            try:
                child_pgid = os.getpgid(child_pid) if child_pid else None
                if child_pgid is not None and child_pgid != os.getpgid(0):
                    os.killpg(child_pgid, signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, OSError):
                try:
                    process.kill()
                except Exception:
                    pass
            process.join(timeout=5)
            timeout_result = {
                "task_id": resolved_task_id,
                "passed": False,
                "error": f"Timeout: problem solving exceeded {self.task_timeout}s",
                "elapsed_s": float(self.task_timeout),
                "timeout": True,
            }
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
                    result=timeout_result,
                    artifact_paths=None,
                    extra={"timeout_stub": True},
                )
            except Exception:
                pass
            return timeout_result

        process.join(timeout=1)
        try:
            payload = result_queue.get_nowait()
        except Exception:
            payload = None

        if not payload:
            return {
                "task_id": resolved_task_id,
                "passed": False,
                "error": "Runner process exited without returning a result",
                "elapsed_s": 0.0,
            }

        if not payload.get("ok"):
            raise RuntimeError(payload.get("traceback") or payload.get("error") or "Unknown runner error")

        return payload["result"]

    # ========== 4. Progress & Trajectory ==========

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

    _progress_updates_to_events = staticmethod(_progress_updates_to_events)

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

    def _finalize_workspace_task(
        self,
        *,
        task: Dict[str, Any],
        workspace: Optional[Path],
        agent: Optional[CodeAgent],
        prompt_texts: Optional[List[str]] = None,
        result: Optional[Dict[str, Any]] = None,
        artifact_paths: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self._save_task_trajectory(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=prompt_texts,
                result=result,
                artifact_paths=artifact_paths,
                extra=extra,
            )
        except Exception:
            pass

        if workspace is None:
            return
        try:
            shutil.rmtree(workspace)
        except Exception:
            pass

    # ========== 5. Result Building ==========

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

    _load_completed_ids = staticmethod(_load_completed_ids)
    _load_result_records = staticmethod(_load_result_records)
    _latest_result_records = staticmethod(_latest_result_records)
    _write_result_records = staticmethod(_write_result_records)
    _append_result_record = staticmethod(_append_result_record)
    _upsert_result_record = staticmethod(_upsert_result_record)
    _summarize_result_records = staticmethod(_summarize_result_records)

    # ========== 6. Main Run Loop ==========

    def run(
        self,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: Optional[str] = None,
        fresh: bool = False,
    ) -> Dict[str, Any]:
        """Run the benchmark and return a summary dict.

        Args:
            limit: Only run the first *limit* tasks.
            task_ids: Only run tasks whose ``task_id`` is in this list.
            dry_run: Print tasks without executing.
            resume: Path to a previous results ``.jsonl`` file.  Already-
                completed task IDs are skipped and rerun task IDs replace their
                existing records in the same file.
            fresh: Ignore any existing ``{benchmark}.jsonl`` and start a
                fresh run (overwrites the canonical file).
        """
        tasks = self._load_tasks()
        if task_ids:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.get("task_id") in id_set]
        if limit and limit > 0:
            tasks = tasks[:limit]

        # --- Decide which file to write results to ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        completed_ids: set = set()
        resume_path: Optional[Path] = None
        persisted_records: List[Dict[str, Any]] = []
        record_index: Dict[str, int] = {}

        if resume:
            # Explicit --resume: always use the specified file.
            resume_path = Path(resume)
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
        else:
            # Canonical per-dataset file (e.g. ``aime_24.jsonl``).
            canonical = self.output_dir / f"{self.benchmark_name}.jsonl"

            if fresh and canonical.exists():
                # --fresh: discard all prior records for this benchmark.
                canonical.unlink()
                print(f"  ▶ Fresh run requested — removed previous results: {canonical}\n")

            if canonical.exists():
                # Auto-resume: the canonical file already exists.
                resume_path = canonical
                raw_records = self._load_result_records(resume_path)
                persisted_records = self._latest_result_records(raw_records)
                if len(persisted_records) != len(raw_records):
                    duplicate_count = len(raw_records) - len(persisted_records)
                    self._write_result_records(resume_path, persisted_records)
                    print(f"  ▶ Cleaned {duplicate_count} duplicate result record(s)")
                completed_ids = self._load_completed_ids(resume_path)
                print(f"  ▶ Auto-resuming from: {resume_path}")
                print(f"    Already completed: {len(completed_ids)} tasks\n")

        # Resolve the actual results file path.
        if resume_path is not None:
            results_file = resume_path
            results_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            results_file = self.output_dir / f"{self.benchmark_name}.jsonl"

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
                    result = self.evaluate(task, task_id=task_id)
                except Exception as exc:
                    result = {
                        "task_id": task_id,
                        "passed": False,
                        "error": f"Runner exception: {exc}",
                        "elapsed_s": 0.0,
                    }

                while self._progress_queue is not None:
                    try:
                        update = self._progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception:
                        break
                    self._handle_progress_update(update)
                results.append(result)
                if result.get("passed") is True:
                    passed_count += 1
                total_time += result.get("elapsed_s", 0)
                progress.finish_task(result)

                self._upsert_result_record(persisted_records, record_index, result)
                # Q3-1: append one JSONL line instead of rewriting the whole
                # results file per task (O(n²) → O(n) IO). Resume collapses
                # duplicate task_ids via latest_result_records.
                self._append_result_record(results_file, result)
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
            "benchmark_runtime": BenchmarkRuntimeConfig.from_env(
                profile=self.runtime_profile
            ).to_metadata(),
        }

        summary_file = self.output_dir / f"{self.benchmark_name}_summary.json"
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
