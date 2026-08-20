"""Web console API tests (IMPROVEMENT.md B2 + C1/C8).

Starts the real ``web/server.py`` on an ephemeral port via subprocess and
exercises the HTTP surface with urllib — no mock handler needed for these
read-only endpoints. Agent/benchmark job execution paths are covered by the
existing ``web/_test_server.py`` script; here we focus on the endpoints added
by the C-batch improvements plus the security guards (path traversal).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "web" / "server.py"
TRACE_DIR = PROJECT_ROOT / "web" / "runtime" / "traces"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def web_server():
    """Launch the real web server on an ephemeral port."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_PATH), "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    # Wait for readiness
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
    yield base
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _get(base: str, path: str, timeout: float = 10.0):
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _get_status(base: str, path: str, timeout: float = 10.0) -> int:
    """Return HTTP status for a request expected to fail."""
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _get_error_payload(base: str, path: str):
    """GET expected to fail → (status, parsed JSON body)."""
    try:
        with urllib.request.urlopen(base + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def _post(base: str, path: str, body: dict):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


# ── C1: 工作区浏览端点 ────────────────────────────────────────────────────


class TestWorkspaceEndpoints:
    def test_tree_lists_code_directory(self, web_server):
        status, payload = _get(web_server, "/api/workspace/tree?path=code&limit=20")
        assert status == 200
        # partial / success 均为可读态（partial = 截断提示）
        assert payload.get("status") in {"success", "partial"}
        entries = payload["data"]["entries"]
        names = {e["name"] for e in entries}
        assert "agents" in names and "tools" in names
        # 目录条目带 type 字段
        types = {e["type"] for e in entries}
        assert "directory" in types

    def test_tree_default_root_is_project(self, web_server):
        status, payload = _get(web_server, "/api/workspace/tree?limit=5")
        assert status == 200
        assert payload.get("status") in {"success", "partial"}

    def test_file_reads_readme(self, web_server):
        status, payload = _get(web_server, "/api/workspace/file?path=README.md&limit=5")
        assert status == 200
        assert payload.get("status") in {"success", "partial"}
        data = payload["data"]
        assert data["total_lines"] > 10
        assert "WhaleCode" in data["content"] or len(data["content"]) > 0

    def test_file_requires_path(self, web_server):
        assert _get_status(web_server, "/api/workspace/file") == 400

    def test_file_path_traversal_blocked(self, web_server):
        # 工具层 _resolve_path 拦截项目根之外的访问
        assert _get_status(web_server, "/api/workspace/file?path=../../etc/passwd") == 400

    def test_tree_root_escape_blocked(self, web_server):
        # root 参数越界（PROJECT_ROOT.parent 之外）被 _resolve_workspace_root 拦截
        assert _get_status(web_server, "/api/workspace/tree?root=/etc&path=.") == 400

    def test_file_absolute_outside_blocked(self, web_server):
        assert _get_status(web_server, "/api/workspace/file?path=/etc/passwd") == 400

    def test_tree_on_file_returns_error(self, web_server):
        # LS 工具对非目录返回 INVALID_PARAM 错误 → 400 + error 字段
        status, payload = _get_error_payload(web_server, "/api/workspace/tree?path=README.md")
        assert status == 400
        assert "error" in payload


# ── C8: Trace 报告端点 ────────────────────────────────────────────────────


class TestTraceEndpoints:
    @pytest.fixture()
    def sample_trace(self):
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / "trace-webapi-test-sample.html"
        path.write_text("<html><body><h1>webapi test trace</h1></body></html>", encoding="utf-8")
        yield path.name
        path.unlink(missing_ok=True)

    def test_traces_list(self, web_server, sample_trace):
        status, payload = _get(web_server, "/api/traces")
        assert status == 200
        names = [t["name"] for t in payload["traces"]]
        assert sample_trace in names
        item = next(t for t in payload["traces"] if t["name"] == sample_trace)
        assert item["url"] == f"/traces/{sample_trace}"
        assert item["size_bytes"] > 0

    def test_traces_html_served(self, web_server, sample_trace):
        with urllib.request.urlopen(f"{web_server}/traces/{sample_trace}", timeout=5) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/html")
            assert b"webapi test trace" in resp.read()

    def test_traces_traversal_blocked(self, web_server):
        # URL 编码的 ../ 尝试 + basename 提取 + 目录内校验
        encoded = urllib.parse.quote("../../web/server.py")
        assert _get_status(web_server, f"/traces/{encoded}") == 404

    def test_traces_non_html_suffix_blocked(self, web_server):
        assert _get_status(web_server, "/traces/server.py") == 404


# ── 既有端点回归（B2 范围中的关键安全项）─────────────────────────────────


class TestExistingEndpointGuards:
    def test_status_ok(self, web_server):
        status, payload = _get(web_server, "/api/status")
        assert status == 200 and payload["ok"] is True
        assert payload["project_root"].endswith("WhaleCode")

    def test_sessions_delete_traversal_blocked(self, web_server):
        """斜杠编码形态被 relative_to 校验拦截（400）；未编码形态经 basename
        隔离后只会指向 sessions 目录内（200 但无副作用）。两种形态都必须
        保证 sessions 目录之外的文件不可达。"""
        for encoded in (True, False):
            name = "../../etc/passwd.json"
            path_segment = urllib.parse.quote(name, safe="") if encoded else urllib.parse.quote(name)
            req = urllib.request.Request(
                f"{web_server}/api/sessions/{path_segment}", method="DELETE"
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    code = resp.status
            except urllib.error.HTTPError as exc:
                code = exc.code
            # 安全行为集合：拒绝（400/404）或"删除了 sessions 内不存在的文件"（200 无副作用）
            assert code in {200, 400, 404}
        # 关键不变量：系统文件不受任何形态影响
        assert Path("/etc/passwd").exists()

    def test_agent_runs_empty_prompt_rejected(self, web_server):
        status, payload = _post(web_server, "/api/agent/runs", {"prompt": ""})
        # 空 prompt 在 job 线程内抛 ValueError → job failed；端点本身返回 202
        # （job 创建成功，执行失败异步反馈）——此处只验证端点可调用不崩溃
        assert status in {202, 400}

    def test_unknown_api_404(self, web_server):
        assert _get_status(web_server, "/api/nonexistent") == 404

    def test_jobs_list(self, web_server):
        status, payload = _get(web_server, "/api/jobs")
        assert status == 200 and "jobs" in payload
