#!/usr/bin/env python3
"""WhaleCode web console server.

This module intentionally uses only the Python standard library plus the
project's existing WhaleCode modules. It serves the static console in
``web/static`` and exposes a small JSON/SSE API for:

- running the existing CodeAgent in a background job
- listing/loading/deleting persisted sessions
- starting/stopping/switching a vLLM process
- launching benchmark scripts as streaming jobs
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
CODE_DIR = PROJECT_ROOT / "code"
RESULTS_DIR = PROJECT_ROOT / "data" / "_results"
RESULT_DIRS = [
    PROJECT_ROOT / "result" / "_results",
    PROJECT_ROOT / "data" / "_results",
]
TRAJECTORY_ROOT = PROJECT_ROOT / "result" / "_trajectory"


def load_project_env(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Load project .env values without adding a hard dependency.

    The CLI uses python-dotenv, but this web server is intentionally standard
    library only. Existing process environment variables win over .env values.
    """
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


def bootstrap_package() -> None:
    """Expose the local ``code/`` directory as ``hello_agents``."""
    if "hello_agents" in sys.modules:
        return
    package = types.ModuleType("hello_agents")
    package.__path__ = [str(CODE_DIR)]
    package.__file__ = str(CODE_DIR / "__init__.py")
    sys.modules["hello_agents"] = package


bootstrap_package()

from hello_agents.agents.code_agent import CodeAgent  # noqa: E402
from hello_agents.core.config import Config  # noqa: E402
from hello_agents.core.llm import HelloAgentsLLM  # noqa: E402
from hello_agents.tools.registry import ToolRegistry  # noqa: E402
from hello_agents.tools.builtin.file_tools import ListFilesTool, ReadTool  # noqa: E402


MODEL_SCAN_ROOTS = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "modelscope" / "hub",
    Path.home() / "models",
    Path.home() / "Models",
    Path.home() / "LLM_Models",
    PROJECT_ROOT / "models",
]


DATASETS = [
    {
        "id": "hevp",
        "name": "HumanEval+",
        "cases": 164,
        "script": "scripts/run_hevp.sh",
        "description": "函数级代码生成评测，覆盖增强边界用例。",
    },
    {
        "id": "lcb6",
        "name": "LiveCodeBench v6",
        "cases": 400,
        "script": "scripts/run_lcb6.sh",
        "description": "LiveCodeBench v6 代码生成评测，覆盖 LeetCode / AtCoder 风格任务。",
    },
    {
        "id": "clev",
        "name": "ClassEval",
        "cases": 100,
        "script": "scripts/run_clev.sh",
        "description": "类与对象场景评测，覆盖多方法依赖。",
    },
    {
        "id": "aime",
        "name": "AIME",
        "cases": 90,
        "script": "scripts/run_aime.sh",
        "description": "数学推理评测，默认覆盖 24/25/26 三年。",
    },
    {
        "id": "swev",
        "name": "SWE-bench Verified",
        "cases": 500,
        "script": "scripts/run_swev.sh",
        "description": "真实仓库修复任务评测。",
    },
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return str(value)


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}

# AskUser 提问等待用户回答的最长秒数；超时后工具返回错误，模型自行继续
ASK_USER_TIMEOUT_SECONDS = float(os.getenv("ASK_USER_TIMEOUT_SECONDS", "300"))


class JobCancelled(Exception):
    """Raised inside a running job when the user requests cancellation."""


def compact_session_title(value: Any, limit: int = 8) -> str:
    """Return a short, UI-friendly session title from the first user prompt."""
    text = str(value or "").strip()
    text = " ".join(text.split())
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit]


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    return str(getattr(message, "role", "") or getattr(message, "type", "")).lower()


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        value = message.get("content") or message.get("text") or message.get("input_text") or ""
    else:
        value = getattr(message, "content", "") or getattr(message, "text", "") or getattr(message, "input_text", "")
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        value = "\n".join(part for part in parts if part)
    return str(value or "")


def first_user_prompt_from_session(data: dict[str, Any]) -> str:
    """Best-effort extraction for sessions saved by different history formats."""
    candidates = []
    for key in ("messages", "conversation", "history", "turns", "records"):
        value = data.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    state = data.get("state")
    if isinstance(state, dict):
        for key in ("messages", "conversation", "history", "turns"):
            value = state.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    for message in candidates:
        if _message_role(message) in {"user", "human"}:
            title = compact_session_title(_message_content(message))
            if title:
                return title
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    for key in ("first_prompt", "prompt", "title", "task", "input_text"):
        title = compact_session_title(metadata.get(key) or data.get(key))
        if title:
            return title
    return ""


def format_session_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("T", " ")[:16]


def openai_models_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    value = base_url.rstrip("/")
    if value.endswith("/models"):
        return value
    if value.endswith("/v1"):
        return f"{value}/models"
    return f"{value}/v1/models"


def detect_served_model(base_url: str | None, timeout: float = 2.5) -> str | None:
    """Return the first model id from an OpenAI-compatible /models endpoint."""
    url = openai_models_url(base_url)
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    return None


def configured_model_name() -> str | None:
    return os.getenv("LLM_MODEL_ID") or models.active_model or detect_served_model(os.getenv("LLM_BASE_URL"))


def slugify_model_name(model_name: str) -> str:
    safe = []
    for char in model_name:
        safe.append(char.lower() if char.isalnum() else "-")
    return "-".join(part for part in "".join(safe).split("-") if part)


def parse_parameter_hint(model_name: str) -> str:
    import re

    match = re.search(r"(?i)(\d+(?:\.\d+)?\s*[bB])", model_name)
    return match.group(1).upper().replace(" ", "") if match else "unknown"


def parse_quant_hint(model_name: str) -> str:
    upper = model_name.upper()
    for token in ("FP8", "FP16", "BF16", "INT8", "INT4", "GPTQ", "AWQ", "GGUF"):
        if token in upper:
            return token
    return "auto"


def latest_snapshot_dir(model_dir: Path) -> Path | None:
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [item for item in snapshots.iterdir() if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def hf_cache_model_id(model_dir: Path) -> str | None:
    name = model_dir.name
    if not name.startswith("models--"):
        return None
    parts = name.split("--")
    if len(parts) < 3:
        return None
    return f"{parts[1]}/{'--'.join(parts[2:])}"


def config_architecture(snapshot: Path | None) -> str | None:
    if not snapshot:
        return None
    config_path = snapshot / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    architectures = data.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    model_type = data.get("model_type")
    return str(model_type) if model_type else None


def scan_huggingface_models(root: Path) -> list[dict[str, Any]]:
    models_found = []
    if not root.exists():
        return models_found
    for model_dir in sorted(root.glob("models--*")):
        if not model_dir.is_dir():
            continue
        model_name = hf_cache_model_id(model_dir)
        if not model_name:
            continue
        snapshot = latest_snapshot_dir(model_dir)
        models_found.append(
            {
                "id": slugify_model_name(model_name),
                "name": model_name,
                "source": "huggingface-cache",
                "path": str(snapshot or model_dir),
                "parameters": parse_parameter_hint(model_name),
                "quantization": parse_quant_hint(model_name),
                "architecture": config_architecture(snapshot),
                "runnable": bool(snapshot),
                "default_command": build_vllm_command(model_name),
            }
        )
    return models_found


def scan_plain_model_dirs(root: Path) -> list[dict[str, Any]]:
    models_found = []
    if not root.exists():
        return models_found
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "config.json").exists() and not (model_dir / "model_index.json").exists():
            continue
        model_name = model_dir.name
        models_found.append(
            {
                "id": slugify_model_name(str(model_dir)),
                "name": model_name,
                "source": "local-dir",
                "path": str(model_dir),
                "parameters": parse_parameter_hint(model_name),
                "quantization": parse_quant_hint(model_name),
                "architecture": config_architecture(model_dir),
                "runnable": True,
                "default_command": build_vllm_command(str(model_dir)),
            }
        )
    return models_found


def build_vllm_command(model_ref: str) -> str:
    return (
        f"vllm serve {model_ref} --port 8000 "
        "--gpu-memory-utilization 0.92 --enable-auto-tool-choice"
    )


def discovered_models(active_model: str | None = None, served_model: str | None = None) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for root in MODEL_SCAN_ROOTS:
        scanners = [scan_huggingface_models] if root.name == "hub" else [scan_plain_model_dirs]
        for scanner in scanners:
            for item in scanner(root):
                by_name.setdefault(item["name"], item)

    for source, model_name in (("served", served_model), ("env", os.getenv("LLM_MODEL_ID")), ("active", active_model)):
        if model_name and model_name not in by_name:
            by_name[model_name] = {
                "id": slugify_model_name(model_name),
                "name": model_name,
                "source": source,
                "path": "",
                "parameters": parse_parameter_hint(model_name),
                "quantization": parse_quant_hint(model_name),
                "architecture": None,
                "runnable": True,
                "default_command": build_vllm_command(model_name),
            }

    result = []
    for item in by_name.values():
        state = "loaded" if active_model and item["name"] == active_model else "available"
        if served_model and item["name"] == served_model:
            state = "served"
        result.append({**item, "state": state})
    result.sort(key=lambda item: (item["state"] not in {"served", "loaded"}, item["source"], item["name"].lower()))
    return result


def model_matches_discovered(discovered_model: dict[str, Any], model_name: str | None) -> bool:
    if not model_name:
        return False
    aliases = {
        discovered_model.get("id"),
        discovered_model.get("name"),
    }
    return model_name in aliases


@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: str = "queued"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    progress: float = 0.0
    total: int | None = None
    completed: int = 0
    result: Any = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    cancel_requested: bool = False
    cancel_reason: str | None = None
    process: subprocess.Popen | None = None
    # AskUser 交互：question_id -> 等待该问题答案的队列（见 IMPROVEMENT.md 改进 1）
    answer_queues: dict[str, queue.Queue] = field(default_factory=dict)


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._counter = 0

    def create(self, kind: str, title: str) -> Job:
        with self._lock:
            self._counter += 1
            job = Job(id=f"{kind}-{int(time.time())}-{self._counter:04d}", kind=kind, title=title)
            self._jobs[job.id] = job
            self.emit(job.id, "job_created", {"title": title, "status": job.status})
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
            if kind:
                jobs = [job for job in jobs if job.kind == kind]
            jobs.sort(key=lambda item: item.created_at, reverse=True)
            return [self.snapshot(job) for job in jobs]

    def snapshot(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id,
            "kind": job.kind,
            "title": job.title,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "duration_seconds": job.duration_seconds,
            "progress": job.progress,
            "total": job.total,
            "completed": job.completed,
            "result": job.result,
            "error": job.error,
            "cancel_requested": job.cancel_requested,
            "cancel_reason": job.cancel_reason,
            "events": job.events[-200:],
        }

    @staticmethod
    def _terminate_process(process: subprocess.Popen | None) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception:
            pass

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)

    def attach_process(self, job_id: str, process: subprocess.Popen) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.process = process

    def clear_process(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.process = None

    def cancel(self, job_id: str, reason: str = "用户取消运行") -> dict[str, Any]:
        process: subprocess.Popen | None = None
        should_emit = False
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status not in TERMINAL_JOB_STATUSES:
                job.cancel_requested = True
                job.cancel_reason = reason
                job.status = "cancelled"
                job.completed_at = now_iso()
                job.error = reason
                job.updated_at = now_iso()
                process = job.process
                should_emit = True
            snapshot = self.snapshot(job)
        self._terminate_process(process)
        self.wake_all_waiters(job_id)
        if should_emit:
            self.emit(job_id, "cancelled", {"reason": reason})
        return snapshot

    def finish_cancelled(self, job_id: str, reason: str, duration_seconds: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.cancel_requested = True
            job.cancel_reason = reason
            job.status = "cancelled"
            job.completed_at = job.completed_at or now_iso()
            job.duration_seconds = duration_seconds
            job.error = reason
            job.updated_at = now_iso()

    def subscribe(self, job_id: str, last_event_id: int = 0) -> queue.Queue:
        with self._lock:
            job = self._jobs[job_id]
            subscriber: queue.Queue = queue.Queue()
            job.subscribers.append(subscriber)
            # Replay backlog, but only events the client has not seen yet
            # (EventSource sends Last-Event-ID on automatic reconnects —
            # without this filter every reconnect would duplicate the trace).
            for event in job.events[-200:]:
                if int(event.get("id", 0)) > last_event_id:
                    subscriber.put(event)
            return subscriber

    def unsubscribe(self, job_id: str, subscriber: queue.Queue) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if subscriber in job.subscribers:
                job.subscribers.remove(subscriber)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == "cancelled" and changes.get("status") not in (None, "cancelled"):
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = now_iso()

    def emit(self, job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == "cancelled" and event_type != "cancelled":
                return
            event = {
                # 1-based monotonic position within this job's event log; used
                # as the SSE `id:` frame for Last-Event-ID resume support.
                "id": len(job.events) + 1,
                "timestamp": now_iso(),
                "type": event_type,
                "payload": json_safe(payload or {}),
                "job": {
                    "id": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "completed": job.completed,
                    "total": job.total,
                },
            }
            job.events.append(event)
            for subscriber in list(job.subscribers):
                subscriber.put(event)

    # ── AskUser 交互通道（IMPROVEMENT.md 改进 1）──────────────────────

    def open_answer_queues(self, job_id: str, question_ids: list[str]) -> None:
        """为问题建立等待队列（job 线程调用，agent 线程等待消费）。"""
        with self._lock:
            job = self._jobs[job_id]
            for qid in question_ids:
                job.answer_queues[qid] = queue.Queue()

    def submit_answer(self, job_id: str, question_id: str, answer: Any) -> bool:
        """前端提交答案 → 放入对应队列。返回 False 表示队列不存在/已取消。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status == "cancelled":
                return False
            q = job.answer_queues.get(question_id)
        if q is None:
            return False
        q.put({"id": question_id, "answer": answer})
        return True

    def wait_for_answer(self, job_id: str, question_id: str, timeout: float) -> dict[str, Any]:
        """阻塞等待一个问题的答案（agent 线程调用）。

        取消/超时抛 AskUserTimeout 或 JobCancelled，由上层转为工具错误。
        """
        with self._lock:
            job = self._jobs.get(job_id)
            q = job.answer_queues.get(question_id) if job else None
        if q is None:
            raise KeyError(f"no answer queue for question {question_id!r} in job {job_id}")
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self.is_cancel_requested(job_id):
                raise JobCancelled(job.cancel_reason or "用户取消运行")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"waiting for answer to {question_id!r} timed out")
            try:
                item = q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if item is None:  # wake_all_waiters 投递的取消哨兵
                raise JobCancelled(job.cancel_reason or "用户取消运行")
            return item

    def wake_all_waiters(self, job_id: str) -> None:
        """取消时唤醒所有等待答案的队列（投递哨兵，避免 agent 线程卡死）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for q in job.answer_queues.values():
                try:
                    q.put(None)
                except Exception:
                    pass


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.active_model: str | None = os.getenv("LLM_MODEL_ID")
        self.status = "stopped"
        self.last_error: str | None = None
        self.started_at: str | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self.process and self.process.poll() is not None:
                self.status = "stopped"
                self.last_error = f"vLLM exited with code {self.process.returncode}"
                self.process = None
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
            served_model = detect_served_model(base_url)
            if served_model:
                self.active_model = served_model
                if self.status == "stopped":
                    self.status = "running"
                    self.last_error = None
            items = discovered_models(active_model=self.active_model, served_model=served_model)
            return {
                "status": self.status,
                "active_model": self.active_model,
                "served_model": served_model,
                "last_error": self.last_error,
                "started_at": self.started_at,
                "base_url": base_url,
                "models": items,
                "scan_roots": [str(root) for root in MODEL_SCAN_ROOTS],
            }

    def start(self, model_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            model = self._find_model(model_id)
            if self.process and self.process.poll() is None:
                if model and self.active_model != model["name"]:
                    self.stop()
                else:
                    self.status = "running"
                    return self.snapshot()

            command = self._command_for(model)
            if not command:
                self.status = "error"
                self.last_error = "No vLLM command configured. Set WHALE_WEB_VLLM_COMMAND or model-specific env."
                return self.snapshot()

            try:
                log_dir = WEB_ROOT / "runtime"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = open(log_dir / "vllm.log", "a", encoding="utf-8")
                self.process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.status = "running"
                self.active_model = model["name"] if model else os.getenv("LLM_MODEL_ID")
                self.started_at = now_iso()
                self.last_error = None
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self.process and self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    self.process.wait(timeout=15)
                except Exception:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except Exception:
                        pass
            self.process = None
            self.status = "stopped"
            self.started_at = None
            return self.snapshot()

    def unload(self) -> dict[str, Any]:
        snapshot = self.stop()
        self.active_model = None
        return snapshot | {"active_model": None}

    def _find_model(self, model_id: str | None) -> dict[str, Any] | None:
        candidates = discovered_models(active_model=self.active_model)
        if not model_id:
            current = configured_model_name()
            for model in candidates:
                if model_matches_discovered(model, current):
                    return model
            return candidates[0] if candidates else None
        for model in candidates:
            if model["id"] == model_id or model["name"] == model_id:
                return model
        return None

    def _command_for(self, model: dict[str, Any] | None) -> str | None:
        if model:
            env_key = f"WHALE_WEB_VLLM_COMMAND_{model['id'].upper().replace('-', '_')}"
            return os.getenv(env_key) or os.getenv("WHALE_WEB_VLLM_COMMAND") or model.get("default_command")
        return os.getenv("WHALE_WEB_VLLM_COMMAND")


jobs = JobManager()
models = ModelManager()


class WebCodeAgent(CodeAgent):
    def __init__(self, *args: Any, event_sink: Callable[[str, dict[str, Any]], None] | None = None, **kwargs: Any) -> None:
        self._web_event_sink = event_sink
        super().__init__(*args, **kwargs)

    def _console(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        if message and self._web_event_sink:
            self._web_event_sink("console", {"message": message})

    def _render_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._web_event_sink:
            self._web_event_sink(event_type, payload)


def _maybe_test_llm():
    """测试钩子 (B2): ``WHALE_WEB_TEST_LLM=stub`` 时返回确定性 stub LLM.

    仅用于 pytest 子进程中的 job 生命周期 / SSE / AskUser 契约测试
    (tests/test_web_jobs.py); 生产环境不设置该变量时返回 None,
    对真实行为零影响.

    行为契约:
        - 默认: 每轮返回 Finish 工具调用 (answer 固定为 stub 标记)
        - ``WHALE_WEB_TEST_LLM_ASKUSER=1``: 第一轮返回 AskUser 工具调用,
          之后把工具结果 (测试提交的答案) 原样带回 Finish.answer
    """
    if os.getenv("WHALE_WEB_TEST_LLM") != "stub":
        return None

    from types import SimpleNamespace

    ask_first = os.getenv("WHALE_WEB_TEST_LLM_ASKUSER") == "1"

    class _StubLLM:
        model = "stub-test-model"

        def __init__(self) -> None:
            self._asked = False

        @staticmethod
        def _usage(p: int = 10, c: int = 5) -> SimpleNamespace:
            return SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)

        def invoke_with_tools(self, messages, tools=None, tool_choice="auto", **kwargs):
            tool_result_text = ""
            for msg in reversed(messages):
                # messages 兼容 dict (ReAct 主循环) 与对象 (其他适配层) 两种形态
                if isinstance(msg, dict):
                    if msg.get("role") == "tool":
                        tool_result_text = str(msg.get("content") or "")
                        break
                elif getattr(msg, "role", "") == "tool":
                    tool_result_text = str(getattr(msg, "content", "") or "")
                    break

            def _call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
                # OpenAI 风格两层结构: tool_call.function.{name, arguments(JSON str)}
                function = SimpleNamespace(
                    name=name, arguments=json.dumps(arguments, ensure_ascii=False)
                )
                return SimpleNamespace(id=call_id, function=function)

            if ask_first and not self._asked and not tool_result_text:
                self._asked = True
                call = _call(
                    "stub_call_1",
                    "AskUser",
                    {"questions": [{"id": "q1", "text": "Stub question for tests", "type": "text"}]},
                )
                message = SimpleNamespace(content=None, tool_calls=[call])
            else:
                answer = "stub-finish"
                if tool_result_text:
                    answer = f"stub-finish: {tool_result_text[:200]}"
                call = _call("stub_call_fin", "Finish", {"answer": answer})
                message = SimpleNamespace(content=None, tool_calls=[call])

            choice = SimpleNamespace(message=message, finish_reason="tool_calls")
            return SimpleNamespace(choices=[choice], model=self.model, usage=self._usage())

        async def ainvoke_with_tools(self, messages, tools=None, tool_choice="auto", **kwargs):
            return self.invoke_with_tools(messages, tools, tool_choice, **kwargs)

    return _StubLLM()


def create_web_agent(
    *,
    workspace: Path,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    temperature: float | None,
    max_steps: int | None,
    event_sink: Callable[[str, dict[str, Any]], None] | None,
    answer_provider: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> WebCodeAgent:
    workspace.mkdir(parents=True, exist_ok=True)
    config = Config.from_env()
    config.trace_enabled = True
    config.trace_dir = str(WEB_ROOT / "runtime" / "traces")
    config.session_dir = str(WEB_ROOT / "runtime" / "sessions")
    config.todowrite_persistence_dir = str(WEB_ROOT / "runtime" / "todos")

    llm_kwargs: dict[str, Any] = {}
    effective_model = model or configured_model_name()
    effective_base_url = base_url or os.getenv("LLM_BASE_URL")
    effective_api_key = api_key or os.getenv("LLM_API_KEY")
    if effective_model:
        llm_kwargs["model"] = effective_model
    if effective_base_url:
        llm_kwargs["base_url"] = effective_base_url
    if effective_api_key:
        llm_kwargs["api_key"] = effective_api_key
    if temperature is not None:
        llm_kwargs["temperature"] = temperature

    # B2 测试钩子: 测试子进程注入确定性 stub, 生产路径零影响
    llm = _maybe_test_llm() or HelloAgentsLLM(**llm_kwargs)
    registry = ToolRegistry(config=config, verbose=False)
    agent = WebCodeAgent(
        name="web-code-agent",
        llm=llm,
        tool_registry=registry,
        project_root=str(workspace),
        working_dir=str(workspace),
        config=config,
        max_steps=max_steps or config.code_agent_max_steps,
        register_default_tools=True,
        enable_task_tool=True,        # TodoWrite (命名历史遗留, 与 Task 工具无关)
        enable_subagent_task=True,    # Task 工具: LLM 子代理动态派生 (A1)
        interactive=False,
        event_sink=event_sink,
    )
    # AskUser：默认 interactive=False 会直接报错；web 场景用 answer_provider
    # 走事件通道（见 IMPROVEMENT.md 改进 1），覆盖默认注册的同名工具。
    if answer_provider is not None:
        from hello_agents.tools.builtin.ask_user import AskUserTool  # noqa: E402

        agent.tool_registry.register_tool(
            AskUserTool(interactive=False, answer_provider=answer_provider),
            auto_expand=False,
        )
    return agent


def run_agent_job(job: Job, payload: dict[str, Any]) -> None:
    start = time.time()
    jobs.update(job.id, status="running", started_at=now_iso(), progress=5)
    jobs.emit(job.id, "status", {"message": "Agent run started"})
    try:
        if jobs.is_cancel_requested(job.id):
            raise JobCancelled(job.cancel_reason or "用户取消运行")
        workspace = Path(payload.get("workspace") or PROJECT_ROOT).expanduser().resolve()
        workspace.relative_to(PROJECT_ROOT.parent)
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")

        agent_holder: dict[str, Any] = {"agent": None}

        def sink(event_type: str, event_payload: dict[str, Any]) -> None:
            if jobs.is_cancel_requested(job.id):
                raise JobCancelled(job.cancel_reason or "用户取消运行")
            # C5: 每个事件附带当前累计 tokens，前端据此绘制 sparkline 与汇总。
            agent = agent_holder.get("agent")
            total = getattr(agent, "_total_tokens", 0)
            if isinstance(total, int) and total > 0:
                event_payload = {**event_payload, "tokens": total}
            jobs.emit(job.id, event_type, event_payload)

        def ask_user_provider(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """AskUser 事件通道：发 ask_user 事件 → 阻塞等待前端提交答案。"""
            q_ids = [str(q.get("id") or f"q{i + 1}") for i, q in enumerate(questions)]
            for i, q in enumerate(questions):
                q["id"] = q_ids[i]
            jobs.open_answer_queues(job.id, q_ids)
            jobs.emit(job.id, "ask_user", {"job_id": job.id, "questions": questions})
            answers: list[dict[str, Any]] = []
            for qid in q_ids:
                try:
                    item = jobs.wait_for_answer(
                        job.id, qid, timeout=ASK_USER_TIMEOUT_SECONDS
                    )
                except JobCancelled:
                    raise
                except Exception as exc:
                    raise RuntimeError(f"AskUser 等待回答超时或中断: {exc}") from exc
                if item is None:  # 取消哨兵
                    raise JobCancelled(job.cancel_reason or "用户取消运行")
                answers.append({"id": qid, "answer": item.get("answer", "")})
            return answers

        agent = create_web_agent(
            workspace=workspace,
            model=payload.get("model") or configured_model_name(),
            base_url=payload.get("base_url") or os.getenv("LLM_BASE_URL"),
            api_key=payload.get("api_key") or os.getenv("LLM_API_KEY"),
            temperature=payload.get("temperature"),
            max_steps=payload.get("max_steps"),
            event_sink=sink,
            answer_provider=ask_user_provider,
        )
        agent_holder["agent"] = agent

        resume_path = payload.get("resume_path")
        if resume_path:
            agent.load_session(str(Path(resume_path).expanduser()), check_consistency=False)
            jobs.emit(job.id, "session_loaded", {"path": resume_path})

        if jobs.is_cancel_requested(job.id):
            raise JobCancelled(job.cancel_reason or "用户取消运行")
        result = agent.run(prompt)
        if jobs.is_cancel_requested(job.id):
            raise JobCancelled(job.cancel_reason or "用户取消运行")
        session_name = payload.get("session_name") or f"web-{job.id}"
        session_path = agent.save_session(session_name)
        duration = time.time() - start
        model_name = payload.get("model") or configured_model_name()
        tokens = getattr(agent, "_total_tokens", None)
        steps = getattr(agent, "_current_step", None)
        jobs.update(
            job.id,
            status="completed",
            completed_at=now_iso(),
            duration_seconds=duration,
            progress=100,
            result={
                "answer": result,
                "session_path": session_path,
                "tokens": tokens,
                "steps": steps,
                "model": model_name,
            },
        )
        jobs.emit(
            job.id,
            "completed",
            {
                "answer": result,
                "session_path": session_path,
                "duration_seconds": duration,
                "tokens": tokens,
                "steps": steps,
                "model": model_name,
            },
        )
    except JobCancelled as exc:
        jobs.finish_cancelled(job.id, str(exc), time.time() - start)
    except Exception as exc:
        if jobs.is_cancel_requested(job.id):
            jobs.finish_cancelled(job.id, job.cancel_reason or "用户取消运行", time.time() - start)
            return
        jobs.update(
            job.id,
            status="failed",
            completed_at=now_iso(),
            duration_seconds=time.time() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        jobs.emit(job.id, "failed", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def benchmark_command(dataset_id: str, payload: dict[str, Any]) -> list[str]:
    dataset = next((item for item in DATASETS if item["id"] == dataset_id), None)
    if not dataset:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    script = PROJECT_ROOT / dataset["script"]
    if not script.exists():
        raise FileNotFoundError(script)
    command = ["bash", str(script)]
    if payload.get("limit"):
        command += ["--limit", str(int(payload["limit"]))]
    if payload.get("model"):
        command += ["--model", str(payload["model"])]
    if payload.get("base_url"):
        command += ["--base-url", str(payload["base_url"])]
    if payload.get("max_tokens"):
        command += ["--max-tokens", str(int(payload["max_tokens"]))]
    if payload.get("pass_k") and dataset_id in {"hevp", "lcb6", "clev"}:
        pass
    return command


def run_benchmark_job(job: Job, payload: dict[str, Any]) -> None:
    start = time.time()
    selected = payload.get("datasets") or []
    if isinstance(selected, str):
        selected = [selected]
    selected = [str(item) for item in selected if str(item)]
    total = sum((next((d["cases"] for d in DATASETS if d["id"] == item), 0) for item in selected))
    jobs.update(job.id, status="running", started_at=now_iso(), total=total or None, progress=2)
    jobs.emit(job.id, "status", {"message": "Benchmark job started", "datasets": selected})

    summaries: list[dict[str, Any]] = []
    completed_cases = 0
    try:
        for dataset_id in selected:
            if jobs.is_cancel_requested(job.id):
                raise JobCancelled(job.cancel_reason or "用户取消运行")
            dataset = next(item for item in DATASETS if item["id"] == dataset_id)
            command = benchmark_command(dataset_id, payload)
            jobs.emit(job.id, "benchmark_started", {"dataset": dataset, "command": command})
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            jobs.attach_process(job.id, proc)
            assert proc.stdout is not None
            recent_lines: list[str] = []
            for line in proc.stdout:
                if jobs.is_cancel_requested(job.id):
                    raise JobCancelled(job.cancel_reason or "用户取消运行")
                text = line.rstrip()
                recent_lines.append(text)
                recent_lines = recent_lines[-120:]
                if text:
                    jobs.emit(job.id, "benchmark_output", {"dataset_id": dataset_id, "line": text})
                if any(token in text.lower() for token in ["passed", "completed", "task", "progress"]):
                    completed_cases = min((total or dataset["cases"]), completed_cases + 1)
                    progress = min(98, (completed_cases / max(total or dataset["cases"], 1)) * 100)
                    jobs.update(job.id, completed=completed_cases, progress=progress)
            code = proc.wait()
            jobs.clear_process(job.id)
            summaries.append(
                {
                    "dataset": dataset["name"],
                    "dataset_id": dataset_id,
                    "returncode": code,
                    "recent_output": recent_lines,
                }
            )
            if code != 0:
                raise RuntimeError(f"{dataset['name']} benchmark failed with exit code {code}")
            completed_cases += dataset["cases"]
            jobs.update(job.id, completed=min(completed_cases, total or completed_cases), progress=min(98, job.progress + 10))

        result = {"summaries": summaries, "history": load_benchmark_history()}
        jobs.update(
            job.id,
            status="completed",
            completed_at=now_iso(),
            duration_seconds=time.time() - start,
            progress=100,
            completed=total or completed_cases,
            result=result,
        )
        jobs.emit(job.id, "completed", result)
    except JobCancelled as exc:
        jobs.finish_cancelled(job.id, str(exc), time.time() - start)
        jobs.clear_process(job.id)
    except Exception as exc:
        jobs.clear_process(job.id)
        if jobs.is_cancel_requested(job.id):
            jobs.finish_cancelled(job.id, job.cancel_reason or "用户取消运行", time.time() - start)
            return
        jobs.update(
            job.id,
            status="failed",
            completed_at=now_iso(),
            duration_seconds=time.time() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        jobs.emit(job.id, "failed", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _case_passed(record: Any) -> bool | None:
    """Best-effort verdict for a single benchmark case record."""
    if not isinstance(record, dict):
        return None
    for key in ("passed", "correct", "success", "is_correct", "ok"):
        if key in record:
            value = record[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"passed", "pass", "ok", "success", "true"}
    status = record.get("status")
    if isinstance(status, str):
        if status.strip().lower() in {"passed", "pass", "ok", "success"}:
            return True
        if status.strip().lower() in {"failed", "fail", "error"}:
            return False
    return None


def _summarize_records(records: list[Any]) -> dict[str, Any]:
    """Compute passed/total/pass_rate from a list of case records."""
    passed = 0
    total = 0
    for record in records:
        verdict = _case_passed(record)
        if verdict is None:
            continue
        total += 1
        if verdict:
            passed += 1
    if not total:
        return {}
    return {"passed": passed, "total": total, "failed": total - passed, "pass_rate": round(passed / total, 4)}


def _read_jsonl_records(path: Path, limit: int | None = None) -> list[Any]:
    records: list[Any] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if limit is not None and idx >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            records.append({"output": line})
    return records


def _summary_path_for(path: Path) -> Path:
    if path.suffix == ".jsonl":
        return path.with_name(f"{path.stem}_summary.json")
    return path


def _benchmark_from_name(name: str) -> str:
    stem = Path(name).stem.replace("_summary", "")
    if stem.startswith("lcb6"):
        return "lcb6"
    if stem.startswith("humaneval"):
        return "humaneval_plus"
    if stem.startswith("classeval"):
        return "classeval"
    if stem.startswith("aime_24"):
        return "aime_24"
    if stem.startswith("aime_25"):
        return "aime_25"
    if stem.startswith("aime_26"):
        return "aime_26"
    return stem.split("_2026", 1)[0]


def _trajectory_task_dir(task_id: str) -> str:
    return str(task_id or "").strip().replace("/", "_").replace("\\", "_")


def _find_trajectory_path(task_id: str, benchmark: str | None = None) -> Path | None:
    safe = _trajectory_task_dir(task_id)
    if not safe:
        return None
    roots = []
    if benchmark:
        roots.append(TRAJECTORY_ROOT / benchmark)
    roots.append(TRAJECTORY_ROOT)
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        direct = root / safe / "trajectory.json"
        if direct.exists():
            return direct
        matches = list(root.glob(f"**/{safe}/trajectory.json"))
        if matches:
            return matches[0]
    return None


def _annotate_record(record: Any, benchmark: str | None = None) -> Any:
    if not isinstance(record, dict):
        return record
    item = dict(record)
    task_id = item.get("task_id") or item.get("id") or item.get("name")
    if task_id and _find_trajectory_path(str(task_id), benchmark):
        item["trajectory_available"] = True
    return item


def _history_summary(path: Path) -> dict[str, Any]:
    """Extract a lightweight summary (incl. pass rate) for a result artifact.

    Works for both ``.json`` (dict with explicit fields or an embedded records
    list, or a bare list) and ``.jsonl`` (one case per line). The previous
    implementation only handled ``.json`` dicts, so real ``.jsonl`` runs never
    surfaced a pass rate.
    """
    try:
        summary_file = _summary_path_for(path)
        if summary_file.exists() and summary_file != path:
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
            if isinstance(summary_data, dict):
                return summary_data
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                summary = {k: data.get(k) for k in ("benchmark", "model", "pass_rate", "passed", "total", "failed") if k in data}
                if "passed" not in summary or "total" not in summary:
                    for key in ("cases", "items", "results", "details", "samples", "records", "examples"):
                        value = data.get(key)
                        if isinstance(value, list):
                            summary.update(_summarize_records(value))
                            break
                return summary
            if isinstance(data, list):
                return _summarize_records(data)
            return {}
        return _summarize_records([r for r in _read_jsonl_records(path, limit=5000) if isinstance(r, dict)])
    except Exception:
        return {}


def _iter_result_files() -> list[Path]:
    files: list[Path] = []
    for root in RESULT_DIRS:
        root.mkdir(parents=True, exist_ok=True)
        files.extend(
            path for path in root.glob("*")
            if path.is_file()
            and path.suffix in {".json", ".jsonl"}
            and not path.name.endswith("_summary.json")
        )
    return sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)


def load_benchmark_history() -> list[dict[str, Any]]:
    records = []
    for path in _iter_result_files()[:80]:
        benchmark = _benchmark_from_name(path.name)
        item = {
            "file": str(path),
            "name": path.name,
            "source_dir": str(path.parent),
            "benchmark": benchmark,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "size_bytes": path.stat().st_size,
        }
        summary = _history_summary(path)
        if summary:
            summary.setdefault("benchmark", benchmark)
            item["summary"] = summary
        records.append(item)
    return records


def load_benchmark_detail(file_value: str) -> dict[str, Any]:
    requested = Path(file_value)
    if not requested.is_absolute():
        candidates = [root / requested.name for root in RESULT_DIRS]
        requested = next((item for item in candidates if item.exists()), candidates[0])
    target = requested.resolve()
    if not any(_is_relative_to(target, root.resolve()) for root in RESULT_DIRS):
        raise PermissionError("benchmark result outside allowed directories")
    if not target.exists() or target.suffix not in {".json", ".jsonl"}:
        raise FileNotFoundError("benchmark result not found")

    stat = target.stat()
    records: list[Any] = []
    summary: dict[str, Any] = _history_summary(target)
    raw_preview = ""
    benchmark = str(summary.get("benchmark") or _benchmark_from_name(target.name))
    if target.suffix == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            summary = summary or data
            for key in ("cases", "items", "results", "details", "samples", "records", "examples"):
                value = data.get(key)
                if isinstance(value, list):
                    records = value
                    break
        elif isinstance(data, list):
            records = data
    else:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        raw_preview = "\n".join(lines[:80])
        records = _read_jsonl_records(target, limit=500)

    if not raw_preview:
        raw_preview = target.read_text(encoding="utf-8", errors="replace")[:12000]
    annotated = [_annotate_record(record, benchmark) for record in records[:200]]

    return {
        "file": str(target),
        "name": target.name,
        "source_dir": str(target.parent),
        "benchmark": benchmark,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "size_bytes": stat.st_size,
        "summary": summary,
        "records": annotated,
        "record_count": len(records),
        "raw_preview": raw_preview,
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_benchmark_trajectory(task_id: str, benchmark: str | None = None) -> dict[str, Any]:
    path = _find_trajectory_path(task_id, benchmark)
    if not path:
        raise FileNotFoundError("trajectory not found")
    target = path.resolve()
    if not _is_relative_to(target, TRAJECTORY_ROOT.resolve()):
        raise PermissionError("trajectory outside allowed directory")
    data = json.loads(target.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("trajectory payload is not an object")
    agent = data.get("agent") if isinstance(data.get("agent"), dict) else {}
    events = agent.get("events") if isinstance(agent, dict) else []
    history = agent.get("history") if isinstance(agent, dict) else []
    return {
        "file": str(target),
        "benchmark": data.get("benchmark") or benchmark,
        "task_id": data.get("task_id") or task_id,
        "saved_at": data.get("saved_at"),
        "task": data.get("task") if isinstance(data.get("task"), dict) else {},
        "result": data.get("result") if isinstance(data.get("result"), dict) else {},
        "events": events[:500] if isinstance(events, list) else [],
        "history": history[-80:] if isinstance(history, list) else [],
        "workspace": data.get("workspace") if isinstance(data.get("workspace"), dict) else {},
        "raw_preview": json.dumps(json_safe(data), ensure_ascii=False, indent=2)[:30000],
    }


# ── C1: 工作区浏览（复用 ListFilesTool/ReadTool 的路径安全与截断逻辑）─────


def _resolve_workspace_root(root_value: str | None) -> Path:
    """校验工作区根与 run_agent_job 的边界约束一致（PROJECT_ROOT.parent 之下）。"""
    root = Path(root_value or PROJECT_ROOT).expanduser()
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root not found: {root}")
    root.relative_to(PROJECT_ROOT.parent)  # 越界时抛 ValueError
    return root


def _run_workspace_tool(tool_cls, root: Path, params: dict[str, Any]) -> dict[str, Any]:
    """实例化文件工具并执行，把 ToolResponse 三态转换为 API 响应。

    工具自身的 ``_resolve_path`` 已做"项目根内"校验，这里 root 即项目根，
    双重校验防止越界访问。
    """
    response = tool_cls(project_root=str(root), working_dir=str(root)).run(params)
    payload = {"status": response.status.value if hasattr(response.status, "value") else str(response.status)}
    if response.status.value == "error":
        code = getattr(getattr(response, "error_info", None), "code", None)
        payload["error"] = f"{code}: {response.text}" if code else response.text
        return payload
    payload["text"] = response.text
    payload["data"] = response.data or {}
    return payload


def list_workspace_tree(root_value: str | None, path: str, offset: int, limit: int) -> dict[str, Any]:
    root = _resolve_workspace_root(root_value)
    return _run_workspace_tool(ListFilesTool, root, {"path": path or ".", "offset": offset, "limit": limit})


def read_workspace_file(root_value: str | None, path: str, offset: int, limit: int) -> dict[str, Any]:
    if not path:
        raise ValueError("path is required")
    root = _resolve_workspace_root(root_value)
    return _run_workspace_tool(ReadTool, root, {"path": path, "offset": offset, "limit": limit})


# ── C8: Trace 报告（TraceLogger 已产出 HTML，直接服务）──────────────────


TRACE_DIR = WEB_ROOT / "runtime" / "traces"


def list_trace_reports() -> list[dict[str, Any]]:
    """列出可用的 trace HTML 报告（按 mtime 倒序，最多 50 个）。"""
    if not TRACE_DIR.exists():
        return []
    reports = []
    for path in TRACE_DIR.glob("*.html"):
        stat = path.stat()
        reports.append({
            "name": path.name,
            "url": f"/traces/{path.name}",
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    reports.sort(key=lambda item: item["modified_at"], reverse=True)
    return reports[:50]


def read_trace_html(name: str) -> bytes | None:
    """读取 trace HTML 文件内容（路径穿越防护：basename 提取 + 目录内校验）。"""
    target = (TRACE_DIR / Path(name).name).resolve()
    try:
        target.relative_to(TRACE_DIR.resolve())
    except ValueError:
        return None
    if target.suffix != ".html" or not target.is_file():
        return None
    if target.stat().st_size > 20 * 1024 * 1024:
        return None
    return target.read_bytes()


def parse_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8") or "{}")


class WhaleWebHandler(SimpleHTTPRequestHandler):
    server_version = "WhaleCodeWeb/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def end_headers(self) -> None:
        # 静态文件无显式缓存策略时浏览器会做启发式缓存，导致改了代码看不到效果；
        # no-cache 让浏览器每次重新验证。API 响应已在 send_json/SSE 中设置 no-store。
        if not self.path.split("?", 1)[0].startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.path = "/index.html"
            return super().do_GET()
        if path == "/api/status":
            return self.send_json({
                "ok": True,
                "project_root": str(PROJECT_ROOT),
                "model": models.snapshot(),
            })
        if path == "/api/models":
            return self.send_json(models.snapshot())
        if path == "/api/datasets":
            return self.send_json({"datasets": DATASETS})
        if path == "/api/jobs":
            params = urllib.parse.parse_qs(parsed.query)
            return self.send_json({"jobs": jobs.list(kind=(params.get("kind") or [None])[0])})
        if path.startswith("/api/jobs/") and path.endswith("/events"):
            return self.stream_events(path.split("/")[3])
        if path.startswith("/api/jobs/"):
            job = jobs.get(path.split("/")[3])
            if not job:
                return self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            return self.send_json(jobs.snapshot(job))
        if path == "/api/sessions":
            return self.send_json({"sessions": self.list_sessions()})
        if path.startswith("/api/sessions/"):
            name = urllib.parse.unquote(path.split("/")[-1])
            try:
                detail = self.get_session_detail(name)
            except FileNotFoundError:
                return self.send_json({"error": "session not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"session": detail})
        if path == "/api/benchmarks/history":
            return self.send_json({"history": load_benchmark_history()})
        if path == "/api/workspace/tree":
            params = urllib.parse.parse_qs(parsed.query)
            try:
                result = list_workspace_tree(
                    (params.get("root") or [None])[0],
                    (params.get("path") or ["."])[0],
                    int((params.get("offset") or ["0"])[0] or 0),
                    min(int((params.get("limit") or ["400"])[0] or 400), 1000),
                )
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            status = HTTPStatus.OK if result.get("status") != "error" else HTTPStatus.BAD_REQUEST
            return self.send_json(result, status)
        if path == "/api/workspace/file":
            params = urllib.parse.parse_qs(parsed.query)
            try:
                result = read_workspace_file(
                    (params.get("root") or [None])[0],
                    (params.get("path") or [""])[0],
                    int((params.get("offset") or ["0"])[0] or 0),
                    min(int((params.get("limit") or ["2000"])[0] or 2000), 5000),
                )
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            status = HTTPStatus.OK if result.get("status") != "error" else HTTPStatus.BAD_REQUEST
            return self.send_json(result, status)
        if path == "/api/traces":
            return self.send_json({"traces": list_trace_reports()})
        if path.startswith("/traces/"):
            content = read_trace_html(path.split("/", 2)[-1])
            if content is None:
                return self.send_json({"error": "trace report not found"}, HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/api/benchmarks/trajectory":
            params = urllib.parse.parse_qs(parsed.query)
            task_id = (params.get("task_id") or [""])[0]
            benchmark = (params.get("benchmark") or [None])[0]
            try:
                return self.send_json({"trajectory": load_benchmark_trajectory(task_id, benchmark)})
            except Exception as exc:
                return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.NOT_FOUND)
        if path.startswith("/api/benchmarks/history/"):
            name = urllib.parse.unquote(path.split("/")[-1])
            try:
                return self.send_json({"detail": load_benchmark_detail(name)})
            except Exception as exc:
                return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            payload = parse_json_body(self)
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[3]
                try:
                    snapshot = jobs.cancel(job_id, str(payload.get("reason") or "用户取消运行"))
                    return self.send_json({"ok": True, "job": snapshot})
                except KeyError:
                    return self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            if path == "/api/agent/runs":
                title = str(payload.get("prompt") or "Agent run")[:80]
                job = jobs.create("agent", title)
                threading.Thread(target=run_agent_job, args=(job, payload), daemon=True).start()
                return self.send_json(jobs.snapshot(job), HTTPStatus.ACCEPTED)
            if path == "/api/agent/answers":
                # AskUser 回答提交（IMPROVEMENT.md 改进 1）
                job_id = str(payload.get("job_id") or "")
                answers = payload.get("answers") or []
                if not job_id or not isinstance(answers, list):
                    return self.send_json({"error": "job_id and answers list are required"}, HTTPStatus.BAD_REQUEST)
                accepted = 0
                for item in answers:
                    if not isinstance(item, dict):
                        continue
                    qid = str(item.get("id") or "")
                    if qid and jobs.submit_answer(job_id, qid, item.get("answer", "")):
                        accepted += 1
                if accepted == 0:
                    return self.send_json({"error": "no matching question found for this job"}, HTTPStatus.NOT_FOUND)
                return self.send_json({"ok": True, "accepted": accepted})
            if path == "/api/models/start":
                return self.send_json(models.start(payload.get("model_id")))
            if path == "/api/models/stop":
                return self.send_json(models.stop())
            if path == "/api/models/unload":
                return self.send_json(models.unload())
            if path == "/api/benchmarks/runs":
                title = "Benchmark: " + ", ".join(payload.get("datasets") or [])
                job = jobs.create("benchmark", title)
                threading.Thread(target=run_benchmark_job, args=(job, payload), daemon=True).start()
                return self.send_json(jobs.snapshot(job), HTTPStatus.ACCEPTED)
            return self.send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/sessions/"):
            name = urllib.parse.unquote(parsed.path.split("/")[-1])
            target = WEB_ROOT / "runtime" / "sessions" / name
            if target.suffix != ".json":
                target = target.with_suffix(".json")
            try:
                target.resolve().relative_to((WEB_ROOT / "runtime" / "sessions").resolve())
                if target.exists():
                    target.unlink()
                return self.send_json({"ok": True})
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return self.send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)

    def stream_events(self, job_id: str) -> None:
        if not jobs.get(job_id):
            return self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
        try:
            last_event_id = int(self.headers.get("Last-Event-ID") or 0)
        except (TypeError, ValueError):
            last_event_id = 0
        subscriber = jobs.subscribe(job_id, last_event_id=last_event_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    event = subscriber.get(timeout=15)
                    payload = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(f"id: {event.get('id', 0)}\n".encode("utf-8"))
                    self.wfile.write(f"event: {event['type']}\n".encode("utf-8"))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if event["type"] in {"completed", "failed", "cancelled"}:
                        break
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        finally:
            jobs.unsubscribe(job_id, subscriber)

    def list_sessions(self) -> list[dict[str, Any]]:
        session_dir = WEB_ROOT / "runtime" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        sessions = []
        for path in sorted(session_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metadata = data.get("metadata", {})
                first_prompt_title = first_user_prompt_from_session(data)
                saved_at = data.get("saved_at") or data.get("updated_at") or data.get("created_at")
                sessions.append(
                    {
                        "filename": path.name,
                        "filepath": str(path),
                        "session_id": data.get("session_id"),
                        "created_at": data.get("created_at"),
                        "saved_at": data.get("saved_at"),
                        "display_time": format_session_time(saved_at),
                        "title": first_prompt_title or compact_session_title(metadata.get("title") or path.stem),
                        "metadata": metadata,
                    }
                )
            except Exception:
                sessions.append({
                    "filename": path.name,
                    "filepath": str(path),
                    "title": compact_session_title(path.stem),
                    "display_time": format_session_time(datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")),
                })
        return sessions

    def get_session_detail(self, name: str) -> dict[str, Any]:
        """返回单个会话的详情（history 等），供前端渲染历史对话。

        路径穿越防护与 do_DELETE 一致：只允许 runtime/sessions 内的文件。
        """
        target = WEB_ROOT / "runtime" / "sessions" / name
        if target.suffix != ".json":
            target = target.with_suffix(".json")
        target.resolve().relative_to((WEB_ROOT / "runtime" / "sessions").resolve())
        if not target.exists():
            raise FileNotFoundError(name)
        data = json.loads(target.read_text(encoding="utf-8"))
        return {
            "filename": target.name,
            "session_id": data.get("session_id"),
            "created_at": data.get("created_at"),
            "saved_at": data.get("saved_at"),
            "title": first_user_prompt_from_session(data) or compact_session_title(data.get("metadata", {}).get("title") or target.stem),
            "history": data.get("history", []),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WhaleCode web console")
    parser.add_argument("--host", default=os.getenv("WHALE_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WHALE_WEB_PORT", "8765")))
    args = parser.parse_args()

    (WEB_ROOT / "runtime").mkdir(parents=True, exist_ok=True)
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), WhaleWebHandler)
    print(f"WhaleCode web console: http://{args.host}:{args.port}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web console...")
        models.stop()
        server.server_close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
