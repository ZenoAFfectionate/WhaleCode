#!/usr/bin/env python3
"""Mock backend for UI verification — pure stdlib, no model required.

Serves web/static and fakes every API the frontend calls, including a scripted
SSE agent run (think -> act -> observe, incl. a failing tool) and a benchmark
run, so the Dive Line and scoreboard can be screenshotted with real content.
"""
import json
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"

MODEL = {
    "status": "running",
    "active_model": "Qwen3.5-2B-Instruct",
    "gpu": {"available": True, "gpus": [
        {"index": 0, "name": "RTX 6000 Ada", "memory_used_mb": 41216, "memory_total_mb": 49140, "utilization": 63},
        {"index": 1, "name": "RTX 6000 Ada", "memory_used_mb": 1200, "memory_total_mb": 49140, "utilization": 2},
    ]},
}
SESSIONS = [
    {"filename": "web-agent-1.json", "filepath": "/x/web-agent-1.json", "title": "修复登录接口的空指针", "saved_at": "2026-07-12T22:14:05"},
    {"filename": "web-agent-2.json", "filepath": "/x/web-agent-2.json", "title": "为 CSV 导出补充测试", "saved_at": "2026-07-12T18:02:40"},
]
DATASETS = [
    {"id": "hevp", "name": "HumanEval+", "cases": 164, "script": "scripts/run_hevp.sh", "description": "函数级代码生成评测，覆盖增强边界用例。"},
    {"id": "mbpp", "name": "MBPP+", "cases": 378, "script": "scripts/run_mbpp.sh", "description": "Python 编程题增强集，适合日常代码能力回归。"},
    {"id": "clev", "name": "ClassEval", "cases": 100, "script": "scripts/run_clev.sh", "description": "类与对象场景评测，覆盖多方法依赖。"},
    {"id": "aime", "name": "AIME", "cases": 90, "script": "scripts/run_aime.sh", "description": "数学推理评测，默认覆盖 24/25/26 三年。"},
    {"id": "swev", "name": "SWE-bench Verified", "cases": 500, "script": "scripts/run_swev.sh", "description": "真实仓库修复任务评测。"},
]
HISTORY = [
    {"name": "aime_24_20260712.jsonl", "modified_at": "2026-07-12T23:21:49", "size_bytes": 48213,
     "summary": {"passed": 35, "total": 45, "failed": 10, "pass_rate": 0.7778, "model": "Qwen3.5-2B-Instruct"}},
    {"name": "hevp_20260712.jsonl", "modified_at": "2026-07-12T20:10:02", "size_bytes": 91002,
     "summary": {"passed": 132, "total": 164, "failed": 32, "pass_rate": 0.8049, "model": "Qwen3.5-2B-Instruct"}},
    {"name": "mbpp_smoke.jsonl", "modified_at": "2026-07-11T09:44:00", "size_bytes": 2140,
     "summary": {"passed": 0, "total": 5, "failed": 5, "pass_rate": 0.0, "model": "Qwen3.5-2B-Instruct"}},
]
DETAIL_RECORDS = [
    {"task_id": "aime24_60", "passed": True, "expected": 204, "actual": 204, "elapsed_s": 63.7, "agent_response": "\\boxed{204}"},
    {"task_id": "aime24_61", "passed": True, "expected": 113, "actual": 113, "elapsed_s": 62.5, "agent_response": "\\boxed{113}"},
    {"task_id": "aime24_71", "passed": False, "expected": 116, "actual": None, "elapsed_s": 58.2, "agent_response": "timeout"},
]

def sse(type_, payload, status="running", progress=0):
    return {"timestamp": time.strftime("%H:%M:%S"), "type": type_, "payload": payload,
            "job": {"id": "job-1", "status": status, "progress": progress, "completed": 0, "total": 0}}

AGENT_SCRIPT = [
    ("model_output", {"thought": "先阅读 config.py 了解现有配置，再定位失败的测试并修复。"}, 0.4),
    ("tool_call", {"name": "Read", "path": "config.py"}, 0.5),
    ("tool_result", {"status": "success", "output": "PORT = 8000\nDEBUG = False\nTIMEOUT = 30"}, 0.5),
    ("tool_call", {"name": "Edit", "path": "test_api.py",
                   "output": "  def test_login():\n-    assert login(None) == 200\n+    assert login(None) == 400"}, 0.6),
    ("tool_result", {"status": "success", "output": "已写入 test_api.py（+1 −1）"}, 0.5),
    ("console", {"message": "collecting tests..."}, 0.2),
    ("console", {"message": "test_api.py::test_login"}, 0.2),
    ("tool_call", {"name": "Bash", "command": "pytest -q test_api.py"}, 0.5),
    ("tool_result", {"status": "failed", "error": "1 failed, 2 passed in 1.2s\nAssertionError: 400 != 200"}, 0.6),
    ("model_output", {"thought": "断言方向反了，修正期望值后重跑。"}, 0.5),
    ("tool_call", {"name": "Bash", "command": "pytest -q test_api.py"}, 0.5),
    ("tool_result", {"status": "success", "output": "3 passed in 1.1s"}, 0.5),
]
BENCH_SCRIPT = [
    ("benchmark_started", {"dataset": {"name": "HumanEval+"}}, 0.3),
    ("benchmark_output", {"line": "loading 164 tasks..."}, 0.3),
    ("benchmark_output", {"line": "task 1/164 passed"}, 0.3),
    ("benchmark_output", {"line": "task 2/164 passed"}, 0.3),
]

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, script, done_payload, done_type="completed"):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for type_, payload, delay in script:
                time.sleep(delay)
                ev = sse(type_, payload)
                self.wfile.write(f"event: {type_}\n".encode())
                self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            time.sleep(0.3)
            ev = sse(done_type, done_payload, status=done_type, progress=100)
            self.wfile.write(f"event: {done_type}\n".encode())
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/": self.path = "/index.html"; return super().do_GET()
        if p == "/api/status": return self._json({"ok": True, "project_root": "/home/kemove/CodeingAgent/WhaleCode", "model": MODEL})
        if p == "/api/models": return self._json(MODEL)
        if p == "/api/sessions": return self._json({"sessions": SESSIONS})
        if p == "/api/datasets": return self._json({"datasets": DATASETS})
        if p == "/api/benchmarks/history": return self._json({"history": HISTORY})
        if p.startswith("/api/benchmarks/history/"):
            return self._json({"detail": {"name": p.split("/")[-1], "summary": {"model": "Qwen3.5-2B-Instruct", "temperature": 0.2},
                                          "records": DETAIL_RECORDS, "record_count": len(DETAIL_RECORDS),
                                          "raw_preview": "\n".join(json.dumps(r, ensure_ascii=False) for r in DETAIL_RECORDS)}})
        if p.endswith("/events"):
            if "benchmark" in p:
                return self._stream(BENCH_SCRIPT, {"summaries": [], "history": HISTORY})
            return self._stream(AGENT_SCRIPT, {"answer": "已修正 test_login 的断言方向并通过全部 3 个用例；根因是期望状态码写反（200 应为 400）。",
                                               "duration_seconds": 6.4, "tokens": 12480, "steps": 5})
        return super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length: self.rfile.read(length)
        if self.path == "/api/agent/runs": return self._json({"id": "job-1", "status": "queued"}, 202)
        if self.path == "/api/benchmarks/runs": return self._json({"id": "bench-1", "status": "queued"}, 202)
        return self._json({"error": "unknown"}, 404)

    def do_DELETE(self):
        return self._json({"ok": True})

    def log_message(self, *a): pass

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    print(f"mock backend on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
