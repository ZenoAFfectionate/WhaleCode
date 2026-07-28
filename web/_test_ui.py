#!/usr/bin/env python3
"""Drive the redesigned console in a real browser against the mock backend.

Screenshots every surface and asserts the behaviours IMPROVEMENT.md promises.
Exit code 0 = all assertions passed.
"""
import subprocess, sys, time, socket
from pathlib import Path

WEB = Path(__file__).resolve().parent
SHOTS = WEB / "_shots"; SHOTS.mkdir(exist_ok=True)

try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    print(f"SKIP: playwright not available ({e})"); sys.exit(2)

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

PORT = free_port()
proc = subprocess.Popen([sys.executable, str(WEB / "_mock_server.py"), str(PORT)])
time.sleep(1.2)

fails = []
def check(label, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + extra) if extra else ''}")
    if not cond: fails.append(label)

try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        base = f"http://127.0.0.1:{PORT}"
        page.goto(base, wait_until="networkidle")
        page.wait_for_timeout(600)

        # ---- Agent idle / empty state + reactor + vitals ----
        page.screenshot(path=str(SHOTS / "1_agent_idle.png"))
        check("vitals bar has 4 vital tiles", page.locator("#agentVitals .vital").count() == 4)
        check("empty-state invitation shows", page.locator(".empty-state").is_visible())
        check("3 example chips", page.locator(".example-chip").count() == 3)
        check("agent cancel button exists", page.locator("#cancelAgentButton").count() == 1)
        check("agent cancel hidden while idle", page.locator("#cancelAgentButton").is_hidden())
        check("reactor status is running", "运行中" in page.locator("#serviceStatus").inner_text())
        check("active model shown", "Qwen" in page.locator("#activeModel").inner_text())

        # ---- Run the scripted agent -> Dive Line ----
        page.fill("#promptInput", "修复 test_login 失败的用例")
        page.click("#runAgentButton")
        page.wait_for_selector(".answer", timeout=20000)
        page.wait_for_timeout(400)
        page.screenshot(path=str(SHOTS / "2_agent_trace.png"), full_page=True)

        steps = page.locator(".step").count()
        check("trace rendered multiple steps", steps >= 6, f"{steps} steps")
        check("has a THINK step", page.locator(".step.think").count() >= 1)
        check("has an ACT step", page.locator(".step.act").count() >= 1)
        check("has a FAILED observe step", page.locator(".step.fail").count() >= 1)
        check("answer card surfaced", page.locator(".answer").count() == 1)
        check("answer meta chips surfaced", page.locator(".answer-meta span").count() >= 3)
        check("no raw <pre> visible by default (collapsed)", page.locator(".step-body pre:visible").count() == 0)
        check("no step body visible by default", page.locator(".step-body:visible").count() == 0)
        tok = page.locator("#vitalTokens").inner_text()
        check("tokens vital populated after run", tok not in ("—", ""), tok)
        check("status vital = 已完成", "已完成" in page.locator("#vitalStatus").inner_text())
        stepv = page.locator("#vitalStep").inner_text()
        check("step vital > 0", stepv not in ("0", ""), stepv)

        # ---- Expand one step ----
        page.locator(".step.act .step-head").first.click()
        page.wait_for_timeout(300)
        check("expanding a step reveals its body", page.locator(".step-main.open .step-body:visible").count() >= 1)
        page.screenshot(path=str(SHOTS / "3_step_expanded.png"))

        # ---- Benchmarks view ----
        page.click('.nav-item[data-view="benchmarks"]')
        page.wait_for_timeout(500)
        page.screenshot(path=str(SHOTS / "4_benchmarks.png"), full_page=True)
        check("dataset cards render", page.locator(".dataset-card").count() == 5)
        check("LiveCodeBench v6 card renders", page.locator('.dataset-card:has-text("LiveCodeBench v6")').count() == 1)
        check("MBPP is not shown", page.locator('.dataset-card:has-text("MBPP")').count() == 0)
        check("selected datasets highlighted", page.locator(".dataset-card.selected").count() >= 1)
        rates = page.locator(".scoreboard .sb-rate")
        check("every result shows a pass-rate scoreboard", rates.count() == 3, f"{rates.count()} rows")
        texts = [rates.nth(i).inner_text() for i in range(rates.count())]
        check("scoreboards show % (not KB)", all("%" in t for t in texts), str(texts))

        # ---- Expand a result detail ----
        page.locator(".result-summary").first.click()
        page.wait_for_selector(".result-detail", timeout=8000)
        page.wait_for_timeout(300)
        check("detail stat-strip renders", page.locator(".result-detail .stat-strip div").count() >= 4)
        check("case rows render", page.locator(".case-row").count() == 3)
        check("passed & failed cases both present",
              page.locator(".case-row.passed").count() >= 1 and page.locator(".case-row.failed").count() >= 1)
        check("trajectory buttons render", page.locator(".trajectory-button").count() >= 1)
        page.locator(".trajectory-button").first.click()
        page.wait_for_selector(".trajectory-panel", timeout=8000)
        check("trajectory panel renders", page.locator(".trajectory-panel").count() >= 1)
        check("trajectory timeline renders", page.locator(".trajectory-event").count() >= 1)
        page.screenshot(path=str(SHOTS / "5_result_detail.png"), full_page=True)

        # ---- Mobile ----
        page.set_viewport_size({"width": 390, "height": 844})
        page.click('.nav-item[data-view="agent"]')
        page.wait_for_timeout(400)
        check("sessions still reachable on mobile", page.locator(".sessions").is_visible())
        page.screenshot(path=str(SHOTS / "6_mobile.png"), full_page=True)

        browser.close()
finally:
    proc.terminate()

print("\n=== RESULT ===")
if fails:
    print(f"FAILED ({len(fails)}): {fails}"); sys.exit(1)
print("ALL UI CHECKS PASSED"); print(f"screenshots in {SHOTS}")
