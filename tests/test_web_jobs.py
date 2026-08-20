"""Web job-execution tests (IMPROVEMENT.md B2 剩余部分).

与 ``test_web_api.py`` (只读端点) 互补, 本文件覆盖 **job 执行链路**:
    - job 完整生命周期: POST /api/agent/runs → running → completed/failed
    - SSE 事件流: 帧契约 + id 单调递增 + 终止事件
    - SSE Last-Event-ID 断线续传: 只回放未见过的 backlog
    - AskUser 交互: ask_user 事件 → POST /api/agent/answers → 结果回传

依赖测试钩子: ``web/server.py`` 的 ``_maybe_test_llm()``
(``WHALE_WEB_TEST_LLM=stub`` 注入确定性 stub LLM, 生产路径零影响).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "web" / "server.py"
# run_agent_job 校验 workspace 必须在 PROJECT_ROOT.parent 之下,
# 测试用 web/runtime/ 下的临时子目录满足该约束.
WORKSPACE_BASE = PROJECT_ROOT / "web" / "runtime" / "test-jobs-ws"

_TERMINAL_SSE = {"completed", "failed", "cancelled"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _launch_server(extra_env: dict[str, str]):
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_PATH), "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **extra_env},
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/api/status", timeout=2) as resp:
                if resp.status == 200:
                    break
        except Exception:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"web server died during startup:\n{output}")
            time.sleep(0.3)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        pytest.fail("web server did not become ready in 20s")

    def _stop():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    return base, _stop


@pytest.fixture(scope="module")
def stub_server():
    """默认 stub 模式: LLM 每轮直接 Finish."""
    base, stop = _launch_server({"WHALE_WEB_TEST_LLM": "stub"})
    yield base
    stop()


@pytest.fixture(scope="module")
def askuser_server():
    """AskUser stub 模式: 第一轮发 AskUser 工具调用, 之后 Finish."""
    base, stop = _launch_server({
        "WHALE_WEB_TEST_LLM": "stub",
        "WHALE_WEB_TEST_LLM_ASKUSER": "1",
    })
    yield base
    stop()


@pytest.fixture
def workspace():
    """PROJECT_ROOT 内的临时 workspace (满足 run_agent_job 的边界校验)."""
    ws = WORKSPACE_BASE / f"ws-{uuid.uuid4().hex[:8]}"
    ws.mkdir(parents=True, exist_ok=True)
    yield str(ws)
    shutil.rmtree(ws, ignore_errors=True)
    if WORKSPACE_BASE.exists() and not any(WORKSPACE_BASE.iterdir()):
        WORKSPACE_BASE.rmdir()


# ── HTTP helpers ───────────────────────────────────────────────────────────


def _post(base: str, path: str, payload: dict, timeout: float = 10):
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def _get(base: str, path: str, timeout: float = 10):
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def _run_job(base: str, prompt: str, ws: str):
    status, payload = _post(base, "/api/agent/runs", {"prompt": prompt, "workspace": ws})
    assert status == 202, payload
    return payload["id"]


def _wait_for_job(base: str, job_id: str, terminal: set[str], timeout: float = 30):
    deadline = time.time() + timeout
    payload: dict[str, Any] = {}
    while time.time() < deadline:
        status, payload = _get(base, f"/api/jobs/{job_id}")
        assert status == 200, payload
        if payload["status"] in terminal:
            return payload
        time.sleep(0.2)
    pytest.fail(f"job {job_id} did not reach {terminal} within {timeout}s (last: {payload})")


def _read_sse(base: str, job_id: str, last_event_id: int = 0, stop_types=None, timeout: float = 30):
    """读取 SSE 流 → [(id, type, payload)].

    遇到 stop_types 中的事件或 server 关闭连接 (终止事件后) 即停止;
    超时由 urlopen 的 socket timeout 兜底.
    """
    stop_types = stop_types or set()
    events: list[tuple[int, str, dict]] = []
    headers = {"Last-Event-ID": str(last_event_id)} if last_event_id else {}
    req = urllib.request.Request(f"{base}/api/jobs/{job_id}/events", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event_id, event_type, data_lines = 0, "", []
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\n")
            if line.startswith("id: "):
                event_id = int(line[4:])
            elif line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
            elif not line and event_type:  # 帧结束空行
                events.append((event_id, event_type, json.loads("".join(data_lines))))
                event_type, data_lines = "", []
                if events[-1][1] in stop_types or events[-1][1] in _TERMINAL_SSE:
                    break
    return events


# ── 1. Job 生命周期 ────────────────────────────────────────────────────────


class TestJobLifecycle:
    def test_completed_job_full_lifecycle(self, stub_server, workspace):
        job_id = _run_job(stub_server, "hello stub", workspace)
        final = _wait_for_job(stub_server, job_id, {"completed", "failed"})
        assert final["status"] == "completed"
        assert final["result"]["answer"] == "stub-finish"
        assert final["completed_at"]
        assert final["duration_seconds"] >= 0
        assert final["progress"] == 100

    def test_job_visible_in_list(self, stub_server, workspace):
        job_id = _run_job(stub_server, "list me", workspace)
        _wait_for_job(stub_server, job_id, {"completed", "failed"})
        status, payload = _get(stub_server, "/api/jobs?kind=agent")
        assert status == 200
        assert job_id in [job["id"] for job in payload["jobs"]]

    def test_failed_job_empty_prompt(self, stub_server, workspace):
        """空 prompt → job 线程内 ValueError → status failed + error 消息."""
        status, payload = _post(
            stub_server, "/api/agent/runs", {"prompt": "  ", "workspace": workspace}
        )
        assert status == 202
        final = _wait_for_job(stub_server, payload["id"], {"completed", "failed"})
        assert final["status"] == "failed"
        assert "prompt is required" in (final.get("error") or "")

    def test_unknown_job_404(self, stub_server):
        assert _get(stub_server, "/api/jobs/agent-0-0000")[0] == 404


# ── 2. SSE 事件流 ─────────────────────────────────────────────────────────


class TestSSEStream:
    def test_stream_frames_and_monotonic_ids(self, stub_server, workspace):
        job_id = _run_job(stub_server, "stream me", workspace)
        events = _read_sse(stub_server, job_id)
        assert events, "no SSE events received"
        # 帧契约: 每个事件有唯一 id + 类型 + JSON payload, id 单调递增
        ids = [event_id for event_id, _, _ in events]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
        # 生命周期: 以 status 开场, 以 completed 收尾 (server 随后关流)
        assert events[0][1] in {"job_created", "status"}
        assert events[-1][1] == "completed"
        assert events[-1][2]["payload"]["answer"] == "stub-finish"

    def test_resume_replays_only_unseen_events(self, stub_server, workspace):
        """Last-Event-ID 断线续传: 只回放 id > N 的 backlog, 不重复."""
        job_id = _run_job(stub_server, "resume me", workspace)
        _wait_for_job(stub_server, job_id, {"completed", "failed"})  # 全部事件入 backlog

        full = _read_sse(stub_server, job_id)
        assert len(full) >= 3
        cutoff = full[1][0]  # 第二个事件的 id 作为断线点
        resumed = _read_sse(stub_server, job_id, last_event_id=cutoff)
        assert resumed, "resume stream is empty"
        # 契约: 续传流不包含 id <= 断点的事件
        assert all(event_id > cutoff for event_id, _, _ in resumed)
        # 续传流恰好是完整流中 id > cutoff 的子序列 (顺序一致)
        expected = [(i, t) for i, t, _ in full if i > cutoff]
        assert [(i, t) for i, t, _ in resumed] == expected
        assert resumed[-1][1] == "completed"


# ── 3. AskUser 交互契约 ────────────────────────────────────────────────────


class TestAskUserFlow:
    def test_ask_user_event_answer_roundtrip(self, askuser_server, workspace):
        """ask_user 事件 → 提交答案 → job 完成 → 答案回传至最终 result."""
        job_id = _run_job(askuser_server, "ask me something", workspace)

        # 1. SSE 流上等到 ask_user 事件 (此时 agent 阻塞在 wait_for_answer)
        events = _read_sse(askuser_server, job_id, stop_types={"ask_user"})
        ask_events = [payload for _, etype, payload in events if etype == "ask_user"]
        assert ask_events, "ask_user event not observed on SSE stream"
        questions = ask_events[0]["payload"]["questions"]
        assert questions and questions[0]["id"] == "q1"

        # 2. 提交答案
        answer_text = "my-test-answer-42"
        status, payload = _post(
            askuser_server,
            "/api/agent/answers",
            {
                "job_id": job_id,
                "answers": [{"id": "q1", "answer": answer_text}],
            },
        )
        assert status == 200
        assert payload["accepted"] == 1

        # 3. job 完成, 答案经 stub LLM 回传至最终 result
        final = _wait_for_job(askuser_server, job_id, {"completed", "failed"})
        assert final["status"] == "completed", final.get("error")
        assert answer_text in final["result"]["answer"]

    def test_answers_endpoint_validation(self, askuser_server, workspace):
        """契约校验: 缺 job_id → 400; 无匹配问题 → 404."""
        status, _ = _post(
            askuser_server, "/api/agent/answers", {"answers": [{"id": "q1", "answer": "x"}]}
        )
        assert status == 400

        job_id = _run_job(askuser_server, "ask for validation", workspace)
        # 等待 ask_user 队列建立后, 提交不存在的问题 id
        _read_sse(askuser_server, job_id, stop_types={"ask_user"})
        status, _ = _post(
            askuser_server,
            "/api/agent/answers",
            {"job_id": job_id, "answers": [{"id": "no-such-q", "answer": "x"}]},
        )
        assert status == 404

        # 收尾: 回答真实问题让 job 正常终止, 避免线程悬挂到超时
        _post(
            askuser_server,
            "/api/agent/answers",
            {"job_id": job_id, "answers": [{"id": "q1", "answer": "cleanup"}]},
        )
        _wait_for_job(askuser_server, job_id, {"completed", "failed"})
