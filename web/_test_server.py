#!/usr/bin/env python3
"""Unit test for the real server.py result-summary logic (no heavy deps).

We stub the ``hello_agents`` modules that server.py imports at top level so we
can import the real module and exercise ``load_benchmark_history`` /
``load_benchmark_detail`` against the real artifacts in ``data/_results``.
"""
import importlib.util
import sys
import types
from pathlib import Path

WEB = Path(__file__).resolve().parent

# --- stub hello_agents so `from hello_agents... import X` succeeds cheaply ---
for name in [
    "hello_agents",
    "hello_agents.agents", "hello_agents.agents.code_agent",
    "hello_agents.core", "hello_agents.core.config", "hello_agents.core.llm",
    "hello_agents.tools", "hello_agents.tools.registry",
]:
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as package
    sys.modules[name] = mod
sys.modules["hello_agents.agents.code_agent"].CodeAgent = type("CodeAgent", (), {})
sys.modules["hello_agents.core.config"].Config = type("Config", (), {})
sys.modules["hello_agents.core.llm"].HelloAgentsLLM = type("HelloAgentsLLM", (), {})
sys.modules["hello_agents.tools.registry"].ToolRegistry = type("ToolRegistry", (), {})

spec = importlib.util.spec_from_file_location("whale_server", WEB / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

failures = []
def check(label, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + extra) if extra else ''}")
    if not cond:
        failures.append(label)

print("=== _summarize_records ===")
recs = [{"passed": True}, {"passed": False}, {"passed": True}, {"note": "no verdict"}]
s = server._summarize_records(recs)
check("counts passed/total, ignores verdictless", s == {"passed": 2, "total": 3, "failed": 1, "pass_rate": round(2/3, 4)}, str(s))
check("empty list -> {}", server._summarize_records([]) == {})

print("\n=== _case_passed variants ===")
check("bool true", server._case_passed({"passed": True}) is True)
check("string 'pass'", server._case_passed({"status": "pass"}) is True)
check("string 'error'", server._case_passed({"status": "error"}) is False)
check("no verdict -> None", server._case_passed({"x": 1}) is None)

print("\n=== load_benchmark_history on real data/_results ===")
hist = server.load_benchmark_history()
print(f"  found {len(hist)} artifact(s)")
jsonl = [h for h in hist if h["name"].endswith(".jsonl")]
if jsonl:
    h = jsonl[0]
    sm = h.get("summary", {})
    print(f"  {h['name']} summary -> {sm}")
    check(".jsonl now has a summary", bool(sm), "was previously absent")
    check(".jsonl summary has pass_rate", "pass_rate" in sm)
    check(".jsonl summary has passed & total", "passed" in sm and "total" in sm)
    check("passed <= total and total > 0", sm.get("total", 0) > 0 and sm.get("passed", 0) <= sm.get("total", 1))
else:
    print("  (no .jsonl artifacts present — skipping live-file assertions)")

print("\n=== load_benchmark_detail on the same file ===")
if jsonl:
    detail = server.load_benchmark_detail(jsonl[0]["name"])
    check("detail returns records", detail.get("record_count", 0) > 0, f"{detail.get('record_count')} records")
    r0 = (detail.get("records") or [{}])[0]
    check("case record has task_id", "task_id" in r0, str(list(r0.keys())[:6]))

print("\n=== RESULT ===")
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL SERVER-SIDE CHECKS PASSED")
