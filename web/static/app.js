const state = {
  currentView: "agent",
  projectRoot: "",
  selectedSession: null,
  selectedSessionTitle: "新会话",
  selectedDatasets: new Set(),
  datasets: [],
  modelSnapshot: null,
  benchmarkHistory: [],
  benchmarkDetails: new Map(),
  activeEventSource: null,
  run: { active: false, steps: 0, nodeIdx: 0, startedAt: 0, timer: null, pending: null, logStep: null },
};

const EXAMPLES = ["修复失败的测试", "解释这段代码的作用", "为接口新增一个端点"];

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

/* ----------------------------- helpers -------------------------------- */
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]
  ));
}
function clean(value) { return String(value ?? "").replace(/^\s+|\s+$/g, ""); }
function firstLine(value) { const t = clean(value); const i = t.indexOf("\n"); return i === -1 ? t : t.slice(0, i); }

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}
function formatTokens(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}
function statusLabel(status) {
  return { queued: "排队中", running: "运行中", completed: "已完成", failed: "失败", idle: "就绪",
    stopped: "已停止", loading: "加载中" }[status] || status || "就绪";
}

function payloadText(payload) {
  if (payload == null) return "";
  if (typeof payload === "string") return clean(payload);
  for (const key of ["thought", "thinking", "reasoning", "content", "text", "output", "message", "line", "answer", "error"]) {
    if (payload[key]) return clean(payload[key]);
  }
  return clean(JSON.stringify(payload, null, 2));
}
function toolName(payload) {
  if (!payload || typeof payload !== "object") return "工具";
  return payload.name || payload.tool || payload.tool_name || payload.command || payload.action || "工具";
}
function basename(value) {
  const t = clean(value);
  if (!t) return "";
  const n = t.replace(/\\/g, "/").split(/[?#]/)[0];
  return n.split("/").filter(Boolean).pop() || n;
}
function fileNameFromPayload(payload) {
  if (!payload || typeof payload !== "object") return "";
  for (const k of ["path", "file_path", "filepath", "file", "target_file", "target_path", "filename"]) {
    if (typeof payload[k] === "string" && payload[k].trim()) return basename(payload[k]);
  }
  const args = payload.arguments || payload.args || payload.params || payload.input || payload.parameters;
  if (typeof args === "string") { try { return fileNameFromPayload(JSON.parse(args)); } catch { return ""; } }
  if (args && typeof args === "object") return fileNameFromPayload(args);
  return "";
}
function isFileTool(name) { return ["read", "write", "edit", "delete"].includes(String(name || "").toLowerCase()); }
function toolTarget(payload) {
  const name = toolName(payload).toLowerCase();
  if (isFileTool(name)) return fileNameFromPayload(payload);
  const args = (payload && (payload.arguments || payload.args || payload.params || payload.input)) || payload || {};
  const cmd = (args && (args.command || args.cmd || args.script)) || payload?.command;
  if (cmd && typeof cmd === "string") return cmd;
  return fileNameFromPayload(payload);
}
function toolResultStatus(payload) {
  if (!payload || typeof payload !== "object") return "success";
  const v = payload.status ?? payload.state ?? payload.ok ?? payload.success ?? payload.error;
  if (v === false || v === "failed" || v === "error" || payload.error) return "failed";
  return "success";
}
function renderMaybeDiff(text) {
  const esc = escapeHtml(text);
  if (!/^[+-]/m.test(text)) return esc;
  return esc.split("\n").map((l) => {
    if (l.startsWith("+")) return `<span class="diff-add">${l}</span>`;
    if (l.startsWith("-")) return `<span class="diff-del">${l}</span>`;
    return l;
  }).join("\n");
}
function rawBlock(payload) {
  const raw = JSON.stringify(payload || {}, null, 2);
  if (!raw || raw === "{}") return "";
  return `<details class="step-raw"><summary>raw</summary><pre>${escapeHtml(raw)}</pre></details>`;
}

/* --------------------------- view switching --------------------------- */
function setView(view) {
  state.currentView = view;
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `${view}View`));
}
function setBusy(button, busy, label) {
  button.disabled = busy;
  if (!button.dataset.originalText) button.dataset.originalText = button.textContent.trim();
  button.textContent = busy ? label : button.dataset.originalText;
}

/* ----------------------------- vitals --------------------------------- */
function setVitalsStatus(status) {
  const bar = $("#agentVitals");
  bar.classList.toggle("is-running", status === "running" || status === "queued");
  bar.classList.toggle("is-done", status === "completed");
  bar.classList.toggle("is-failed", status === "failed");
  $("#vitalStatus").textContent = statusLabel(status);
}
function startRunTimer() {
  stopRunTimer();
  state.run.startedAt = performance.now();
  state.run.timer = setInterval(() => {
    $("#vitalElapsed").textContent = formatDuration((performance.now() - state.run.startedAt) / 1000);
  }, 200);
}
function stopRunTimer() { if (state.run.timer) { clearInterval(state.run.timer); state.run.timer = null; } }

/* --------------------------- trace stream ----------------------------- */
function ensureStreamReady() {
  const empty = $("#chatStream .empty-state");
  if (empty) empty.remove();
}
function scrollStream() { const s = $("#traceStream"); s.scrollTop = s.scrollHeight; }
function nextIdx() { state.run.nodeIdx += 1; return String(state.run.nodeIdx).padStart(2, "0"); }

function stepShell({ phase, idx, kicker, primary, meta, body }) {
  const art = document.createElement("article");
  art.className = `step ${phase}`;
  const isFile = /^\S+\.\w+$/.test(primary || "");
  art.innerHTML = `
    <div class="step-gutter">
      <span class="step-node"></span>
      ${idx ? `<span class="step-idx">${idx}</span>` : ""}
    </div>
    <div class="step-main">
      <button class="step-head" type="button">
        <span class="step-kicker">${escapeHtml(kicker)}</span>
        <span class="step-primary">${primary ? (isFile ? `<span class="path">${escapeHtml(primary)}</span>` : escapeHtml(primary)) : ""}</span>
        <span class="step-meta">${meta || ""}<svg class="step-caret" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg></span>
      </button>
      <div class="step-body">${body || ""}</div>
    </div>`;
  const main = art.querySelector(".step-main");
  art.querySelector(".step-head").addEventListener("click", () => main.classList.toggle("open"));
  return art;
}

function addThinkStep(event) {
  ensureStreamReady();
  state.run.logStep = null;
  const text = payloadText(event.payload);
  const art = stepShell({
    phase: "think", idx: nextIdx(), kicker: "思考",
    primary: firstLine(text),
    body: `<div class="body-text">${escapeHtml(text)}</div>`,
  });
  $("#chatStream").appendChild(art);
  scrollStream();
}

function addActStep(event) {
  ensureStreamReady();
  state.run.logStep = null;
  state.run.steps += 1;
  $("#vitalStep").textContent = String(state.run.steps);
  const p = event.payload || {};
  const target = toolTarget(p);
  const art = stepShell({
    phase: "act running", idx: nextIdx(), kicker: toolName(p).toUpperCase(),
    primary: target,
    meta: `<span class="step-stat">运行中</span>`,
    body: `<div class="body-text">${renderMaybeDiff(payloadText(p) || "已发起调用")}</div>${rawBlock(p)}`,
  });
  $("#chatStream").appendChild(art);
  state.run.pending = { el: art, start: performance.now() };
  scrollStream();
}

function addStandaloneObserve(event) {
  const p = event.payload || {};
  const ok = toolResultStatus(p) === "success";
  const art = stepShell({
    phase: ok ? "pass" : "fail", idx: nextIdx(), kicker: "OBSERVE",
    primary: toolTarget(p),
    meta: `<span class="step-stat">${ok ? "✓" : "✗"}</span>`,
    body: `<div class="body-text">${renderMaybeDiff(payloadText(p))}</div>${rawBlock(p)}`,
  });
  $("#chatStream").appendChild(art);
  scrollStream();
}

function resolveActStep(event) {
  const pending = state.run.pending;
  const p = event.payload || {};
  const ok = toolResultStatus(p) === "success";
  if (!pending) return addStandaloneObserve(event);
  state.run.pending = null;
  const el = pending.el;
  el.classList.remove("running");
  el.classList.add(ok ? "pass" : "fail");
  const dur = formatDuration((performance.now() - pending.start) / 1000);
  el.querySelector(".step-meta").innerHTML =
    `<span class="step-stat">${ok ? "✓" : "✗"}</span><span>${dur}</span>` +
    `<svg class="step-caret" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>`;
  const resultText = payloadText(p);
  if (resultText) {
    const body = el.querySelector(".step-body");
    body.innerHTML = `<div class="body-text">${renderMaybeDiff(resultText)}</div>${rawBlock(p)}`;
  }
  scrollStream();
}

function addLogLine(event) {
  ensureStreamReady();
  const line = payloadText(event.payload);
  if (!line) return;
  if (state.run.logStep && state.run.logStep.isConnected) {
    const list = state.run.logStep.querySelector(".log-list");
    const el = document.createElement("div");
    el.className = "log-line";
    el.textContent = line;
    list.appendChild(el);
    const count = list.children.length;
    state.run.logStep.querySelector(".log-count").textContent = `${count} 行`;
    return;
  }
  const art = stepShell({
    phase: "log", kicker: "日志", primary: "运行日志",
    meta: `<span class="log-count">1 行</span>`,
    body: `<div class="log-list" style="display:grid;gap:5px">` +
          `<div class="log-line">${escapeHtml(line)}</div></div>`,
  });
  $("#chatStream").appendChild(art);
  state.run.logStep = art;
  scrollStream();
}

function addSystemLine(text) {
  ensureStreamReady();
  state.run.logStep = null;
  const art = document.createElement("article");
  art.className = "step system";
  art.innerHTML = `<div class="step-gutter"><span class="step-node"></span></div>
    <div class="step-main"><div class="step-head" style="cursor:default">
      <span class="step-kicker">系统</span>
      <span class="step-primary">${escapeHtml(text)}</span><span></span></div></div>`;
  $("#chatStream").appendChild(art);
  scrollStream();
}

function addAnswer(text, duration, failed = false) {
  ensureStreamReady();
  state.run.logStep = null;
  const art = document.createElement("article");
  art.className = `answer${failed ? " failed" : ""}`;
  art.innerHTML = `
    <div class="answer-head">
      <span class="label">${failed ? "执行失败" : "最终答案"}</span>
      <time>${duration != null ? formatDuration(duration) : ""}</time>
    </div>
    <div class="answer-body">${escapeHtml(clean(text)).replace(/\n/g, "<br>")}</div>`;
  $("#chatStream").appendChild(art);
  scrollStream();
}

function handleEvent(event) {
  const t = event.type;
  if (event.job) setVitalsStatus(event.job.status);
  switch (t) {
    case "model_output": return addThinkStep(event);
    case "tool_call": return addActStep(event);
    case "tool_result": return resolveActStep(event);
    case "console": return addLogLine(event);
    case "session_loaded": return addSystemLine("会话上下文已接入本次运行");
    default: return; // job_created / status / benchmark_* / completed / failed handled elsewhere
  }
}

function renderEmptyState() {
  $("#chatStream").innerHTML = `
    <div class="empty-state">
      <h2>&gt; 开始一次下潜</h2>
      <p>描述一个编码任务，智能体会边思考边调用工具，轨迹在此逐步展开。</p>
      <div class="example-row">
        ${EXAMPLES.map((e) => `<button class="example-chip" type="button" data-example="${escapeHtml(e)}">${escapeHtml(e)}</button>`).join("")}
      </div>
    </div>`;
}

function resetAgentConversation() {
  if (state.activeEventSource) { state.activeEventSource.close(); state.activeEventSource = null; }
  stopRunTimer();
  state.selectedSession = null;
  state.selectedSessionTitle = "新会话";
  state.run = { active: false, steps: 0, nodeIdx: 0, startedAt: 0, timer: null, pending: null, logStep: null };
  setVitalsStatus("idle");
  $("#vitalStep").textContent = "0";
  $("#vitalElapsed").textContent = "—";
  $("#vitalTokens").textContent = "—";
  renderEmptyState();
}

/* ------------------------------ status -------------------------------- */
async function refreshStatus() {
  const data = await api("/api/status");
  state.projectRoot = data.project_root || "";
  renderModelStatus(data.model || {});
}
function renderModelStatus(snapshot) {
  state.modelSnapshot = snapshot;
  const status = snapshot.status || "stopped";
  const running = status === "running";
  $("#serviceDot").className = `dot ${running ? "running" : status === "loading" ? "loading" : "stopped"}`;
  $("#serviceStatus").textContent = statusLabel(status);
  $("#activeModel").textContent = snapshot.active_model || "未加载";
  renderGpu(snapshot.gpu);
}
function renderGpu(gpu) {
  const readout = $("#gpuReadout");
  const bar = $("#gpuBar");
  if (!gpu || !gpu.available || !Array.isArray(gpu.gpus) || !gpu.gpus.length) {
    readout.textContent = "不可用"; bar.style.width = "0%"; return;
  }
  const g = gpu.gpus.reduce((a, b) => (b.memory_used_mb > a.memory_used_mb ? b : a));
  const used = g.memory_used_mb / 1024, total = g.memory_total_mb / 1024;
  readout.textContent = `G${g.index} ${used.toFixed(1)}/${total.toFixed(0)} GB`;
  bar.style.width = `${Math.min(100, (g.memory_used_mb / Math.max(g.memory_total_mb, 1)) * 100)}%`;
}

/* ----------------------------- sessions ------------------------------- */
async function refreshSessions() {
  const data = await api("/api/sessions");
  const sessions = data.sessions || [];
  $("#sessionList").innerHTML = sessions.length ? sessions.map((s) => `
    <article class="session-item ${state.selectedSession === s.filepath ? "active" : ""}"
             data-filepath="${escapeHtml(s.filepath)}" data-filename="${escapeHtml(s.filename)}">
      <div>
        <strong>${escapeHtml(s.title || s.filename)}</strong>
        <small>${escapeHtml((s.saved_at || s.created_at || "").replace("T", " ").slice(0, 16) || "—")}</small>
      </div>
      <button class="icon-button delete-session" title="删除会话" aria-label="删除会话">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 7h12m-9 0V5h6v2m-7 3v8m4-8v8m4-8v8M8 7l1 13h6l1-13"/></svg>
      </button>
    </article>`).join("")
    : `<div class="session-empty"><strong>暂无会话</strong>运行智能体后自动生成。</div>`;
}

/* ----------------------------- datasets ------------------------------- */
async function refreshDatasets() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets || [];
  if (!state.selectedDatasets.size) ["hevp", "mbpp"].forEach((id) => state.selectedDatasets.add(id));
  renderDatasets();
}
function renderDatasets() {
  $("#datasetGrid").innerHTML = state.datasets.map((d) => `
    <article class="dataset-card ${state.selectedDatasets.has(d.id) ? "selected" : ""}" data-dataset-id="${escapeHtml(d.id)}">
      <div class="ds-top">
        <h3>${escapeHtml(d.name)}</h3>
        <svg class="ds-check" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 13 4 4L19 7"/></svg>
      </div>
      <span class="ds-cases">${d.cases} cases</span>
      <p>${escapeHtml(d.description)}</p>
    </article>`).join("");
}

/* --------------------------- benchmark history ------------------------ */
function passRateOf(item) {
  const s = item.summary || {};
  const passed = s.passed, total = s.total;
  if (typeof passed === "number" && typeof total === "number" && total > 0) {
    return { passed, total, rate: passed / total };
  }
  if (typeof s.pass_rate === "number") return { passed: s.passed, total: s.total, rate: s.pass_rate };
  return null;
}
function datasetLabel(item) {
  const s = item.summary || {};
  return s.benchmark || s.dataset || s.dataset_name || item.name.replace(/\.(jsonl|json)$/i, "");
}
function modelLabel(item) {
  const s = item.summary || {};
  return s.model || s.model_id || state.modelSnapshot?.active_model || "unknown";
}
function scoreboardHtml(pr) {
  if (!pr) return `<div class="scoreboard na"><div class="sb-top"><span class="sb-rate">— 无摘要</span></div></div>`;
  const pct = Math.round(pr.rate * 100);
  const cls = pr.rate === 0 ? "zero" : pr.rate < 0.5 ? "low" : "";
  return `<div class="scoreboard ${cls}">
      <div class="sb-top"><span class="sb-rate">${pct}%</span>
        <span class="sb-count">${pr.passed ?? "?"}/${pr.total ?? "?"} 通过</span></div>
      <div class="meter"><span style="width:${pct}%"></span></div>
    </div>`;
}

async function refreshBenchmarkHistory() {
  const data = await api("/api/benchmarks/history");
  state.benchmarkHistory = data.history || [];
  renderBenchmarkHistory();
}
function renderBenchmarkHistory() {
  const filter = $("#historyFilter").value.trim().toLowerCase();
  const sort = $("#historySort").value;
  let rows = [...state.benchmarkHistory];
  if (filter) {
    rows = rows.filter((i) => `${i.name} ${datasetLabel(i)} ${modelLabel(i)}`.toLowerCase().includes(filter));
  }
  if (sort === "name") rows.sort((a, b) => a.name.localeCompare(b.name));
  if (sort === "rate") rows.sort((a, b) => (passRateOf(b)?.rate ?? -1) - (passRateOf(a)?.rate ?? -1));

  $("#historyTable").innerHTML = rows.length ? rows.map((item) => `
    <article class="result-row" data-result-file="${escapeHtml(item.name)}">
      <button class="result-summary" type="button" data-history-file="${escapeHtml(item.name)}">
        <div class="rs-name">
          <strong>${escapeHtml(datasetLabel(item))}</strong>
          <small>${escapeHtml((item.modified_at || "").replace("T", " "))}</small>
        </div>
        <div class="rs-model">${escapeHtml(modelLabel(item))}</div>
        ${scoreboardHtml(passRateOf(item))}
        <svg class="caret" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>
      </button>
      <div class="result-detail-host"></div>
    </article>`).join("")
    : `<div class="empty-block"><strong>暂无测试结果</strong><span>运行评测后，这里会以通过率为头条展示每次结果。</span></div>`;
}

/* --------------------------- benchmark detail ------------------------- */
function caseStatus(r) {
  if (!r || typeof r !== "object") return "unknown";
  const v = r.passed ?? r.correct ?? r.success ?? r.is_correct ?? r.ok ?? r.status;
  if (v === true || v === "passed" || v === "pass" || v === "ok" || v === "success") return "passed";
  if (v === false || v === "failed" || v === "fail" || v === "error") return "failed";
  return "unknown";
}
function caseTitle(r, i) {
  if (!r || typeof r !== "object") return `Case ${i + 1}`;
  return r.task_id || r.id || r.name || r.title || r.prompt_id || `Case ${i + 1}`;
}
function caseMeta(r) {
  if (!r || typeof r !== "object") return "";
  const parts = [];
  if (r.expected != null) parts.push(`期望 ${clean(r.expected).slice(0, 24)}`);
  if (r.actual != null) parts.push(`实际 ${clean(r.actual).slice(0, 24) || "—"}`);
  if (typeof r.elapsed_s === "number") parts.push(`${r.elapsed_s.toFixed(1)}s`);
  return parts.join(" · ");
}
function caseBody(r) {
  if (!r || typeof r !== "object") return String(r ?? "");
  const resp = r.agent_response || r.response || r.prediction || r.output;
  const head = resp ? `${clean(resp)}\n\n` : "";
  return head + JSON.stringify(r, null, 2);
}
function computeStats(records) {
  let passed = 0, total = 0;
  const els = [];
  for (const r of records) {
    const s = caseStatus(r);
    if (s === "passed" || s === "failed") { total += 1; if (s === "passed") passed += 1; }
    if (typeof r?.elapsed_s === "number") els.push(r.elapsed_s);
  }
  return {
    passed, total,
    rate: total ? passed / total : null,
    avg: els.length ? els.reduce((a, b) => a + b, 0) / els.length : null,
  };
}
function renderBenchmarkDetail(detail) {
  const records = detail.records || [];
  const stats = computeStats(records);
  const summary = detail.summary || {};
  const extra = Object.entries(summary)
    .filter(([, v]) => v == null || typeof v !== "object")
    .filter(([k]) => !["passed", "total", "pass_rate", "failed"].includes(k))
    .slice(0, 12);

  const casesHtml = records.length ? records.map((r, i) => {
    const s = caseStatus(r);
    return `<details class="case-row ${s}">
        <summary>
          <span class="case-status">${s === "passed" ? "正确" : s === "failed" ? "错误" : "未知"}</span>
          <span class="case-title">${escapeHtml(caseTitle(r, i))}</span>
          <span class="case-meta">${escapeHtml(caseMeta(r))}</span>
        </summary>
        <pre>${escapeHtml(caseBody(r))}</pre>
      </details>`;
  }).join("") : `<div class="empty-block">无结构化用例，已在下方保留原始轨迹预览。</div>`;

  const rateTile = stats.rate != null
    ? `${Math.round(stats.rate * 100)}%` : (summary.pass_rate != null ? `${summary.pass_rate}` : "—");

  return `<div class="result-detail">
      <div class="stat-strip">
        <div><span class="label">通过率</span><strong>${rateTile}</strong></div>
        <div><span class="label">通过 / 总数</span><strong>${stats.total ? `${stats.passed} / ${stats.total}` : (detail.record_count || 0)}</strong></div>
        <div><span class="label">模型</span><strong>${escapeHtml(summary.model || summary.model_id || state.modelSnapshot?.active_model || "—")}</strong></div>
        <div><span class="label">平均耗时</span><strong>${stats.avg != null ? `${stats.avg.toFixed(1)}s` : "—"}</strong></div>
      </div>
      ${extra.length ? `<details class="more-summary"><summary>更多摘要字段</summary>
        <div class="stat-strip" style="margin-top:10px">
          ${extra.map(([k, v]) => `<div><span class="label">${escapeHtml(k)}</span><strong>${escapeHtml(v)}</strong></div>`).join("")}
        </div></details>` : ""}
      <section>
        <h3>用例执行情况</h3>
        <div class="case-list">${casesHtml}</div>
      </section>
      <details class="more-summary">
        <summary>运行轨迹预览</summary>
        <pre class="trace-preview" style="margin-top:10px">${escapeHtml(detail.raw_preview || "暂无运行轨迹")}</pre>
      </details>
    </div>`;
}

async function toggleBenchmarkDetail(button) {
  const file = button.dataset.historyFile;
  const row = button.closest(".result-row");
  const host = row.querySelector(".result-detail-host");
  if (row.classList.contains("open")) { row.classList.remove("open"); host.innerHTML = ""; return; }
  row.classList.add("open");
  host.innerHTML = `<div class="empty-block">正在加载明细…</div>`;
  try {
    let detail = state.benchmarkDetails.get(file);
    if (!detail) {
      const data = await api(`/api/benchmarks/history/${encodeURIComponent(file)}`);
      detail = data.detail;
      state.benchmarkDetails.set(file, detail);
    }
    host.innerHTML = renderBenchmarkDetail(detail);
  } catch (error) {
    host.innerHTML = `<div class="empty-block error-text">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

/* ------------------------------- SSE ---------------------------------- */
function subscribeJob(job, handlers = {}) {
  const source = new EventSource(`/api/jobs/${encodeURIComponent(job.id)}/events`);
  const onAny = (event) => {
    const data = JSON.parse(event.data);
    handlers.any?.(data);
    if (handlers[data.type]) handlers[data.type](data);
  };
  ["job_created", "status", "console", "model_output", "tool_call", "tool_result",
   "session_loaded", "benchmark_started", "benchmark_output", "completed", "failed"]
    .forEach((name) => source.addEventListener(name, onAny));
  source.onerror = () => { if (job.status === "completed" || job.status === "failed") source.close(); };
  return source;
}

/* ---------------------------- run agent ------------------------------- */
async function runAgent(event) {
  event.preventDefault();
  const prompt = $("#promptInput").value.trim();
  if (!prompt || state.run.active) return;
  if (state.activeEventSource) { state.activeEventSource.close(); state.activeEventSource = null; }

  ensureStreamReady();
  const turn = document.createElement("article");
  turn.className = "turn-user";
  turn.innerHTML = `<div class="label">User</div>${escapeHtml(prompt).replace(/\n/g, "<br>")}`;
  $("#chatStream").appendChild(turn);

  $("#promptInput").value = "";
  autoGrow();
  state.run.active = true;
  state.run.steps = 0;
  state.run.pending = null;
  state.run.logStep = null;
  $("#vitalStep").textContent = "0";
  $("#vitalTokens").textContent = "—";
  setVitalsStatus("queued");
  startRunTimer();
  setBusy($("#runAgentButton"), true, "运行中");

  const finish = (status) => {
    stopRunTimer();
    state.run.active = false;
    setVitalsStatus(status);
    setBusy($("#runAgentButton"), false);
    state.activeEventSource?.close();
    state.activeEventSource = null;
  };

  try {
    const job = await api("/api/agent/runs", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        workspace: state.projectRoot,
        resume_path: state.selectedSession,
        model: state.modelSnapshot?.active_model || undefined,
      }),
    });
    state.activeEventSource = subscribeJob(job, {
      any: handleEvent,
      completed(data) {
        const r = data.payload || {};
        addAnswer(r.answer || "任务已完成，但没有返回文本。", r.duration_seconds);
        $("#vitalElapsed").textContent = formatDuration(r.duration_seconds);
        if (r.tokens != null) $("#vitalTokens").textContent = formatTokens(r.tokens);
        if (r.steps != null) $("#vitalStep").textContent = String(r.steps);
        finish("completed");
        refreshSessions();
      },
      failed(data) {
        addAnswer(data.payload?.error || "unknown error", null, true);
        finish("failed");
      },
    });
  } catch (error) {
    addAnswer(`提交失败：${error.message}`, null, true);
    finish("failed");
  }
}

/* --------------------------- run benchmark ---------------------------- */
function appendBenchLog(line) {
  const empty = $("#benchLog .log-empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = "log-line";
  el.textContent = line;
  $("#benchLog").appendChild(el);
  $("#benchLog").scrollTop = $("#benchLog").scrollHeight;
}
async function runBenchmark(event) {
  event.preventDefault();
  const datasets = [...state.selectedDatasets];
  if (!datasets.length) return;
  $("#benchLog").innerHTML = "";
  setBusy($("#runBenchButton"), true, "运行中");
  try {
    const job = await api("/api/benchmarks/runs", {
      method: "POST",
      body: JSON.stringify({
        datasets,
        model: $("#benchModel").value.trim() || state.modelSnapshot?.active_model || undefined,
        limit: $("#benchLimit").value ? Number($("#benchLimit").value) : undefined,
        pass_k: $("#benchPassK").value,
      }),
    });
    subscribeJob(job, {
      benchmark_started(data) { appendBenchLog(`=== ${data.payload.dataset?.name || "Benchmark"} 开始 ===`); },
      benchmark_output(data) { appendBenchLog(data.payload.line || ""); },
      completed() { appendBenchLog("评测完成"); setBusy($("#runBenchButton"), false); refreshBenchmarkHistory(); },
      failed(data) { appendBenchLog(`错误：${data.payload?.error || "unknown"}`); setBusy($("#runBenchButton"), false); },
    });
  } catch (error) {
    appendBenchLog(`提交失败：${error.message}`);
    setBusy($("#runBenchButton"), false);
  }
}

/* ----------------------------- textarea ------------------------------- */
function autoGrow() {
  const ta = $("#promptInput");
  ta.style.height = "auto";
  ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
}

/* ------------------------------ events -------------------------------- */
function bindEvents() {
  $$(".nav-item").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#agentForm").addEventListener("submit", runAgent);
  $("#promptInput").addEventListener("input", autoGrow);
  $("#promptInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#agentForm").requestSubmit(); }
  });
  $("#benchForm").addEventListener("submit", runBenchmark);
  $("#refreshSessions").addEventListener("click", refreshSessions);
  $("#newSessionButton").addEventListener("click", async () => { resetAgentConversation(); await refreshSessions(); });

  $("#chatStream").addEventListener("click", (e) => {
    const chip = e.target.closest("[data-example]");
    if (chip) { $("#promptInput").value = chip.dataset.example; autoGrow(); $("#promptInput").focus(); }
  });

  $("#sessionList").addEventListener("click", async (e) => {
    const row = e.target.closest(".session-item");
    if (!row?.dataset.filepath) return;
    if (e.target.closest(".delete-session")) {
      await api(`/api/sessions/${encodeURIComponent(row.dataset.filename)}`, { method: "DELETE" });
      if (state.selectedSession === row.dataset.filepath) { state.selectedSession = null; state.selectedSessionTitle = "新会话"; }
      await refreshSessions();
      return;
    }
    state.selectedSession = row.dataset.filepath;
    state.selectedSessionTitle = row.querySelector("strong")?.textContent || row.dataset.filename;
    addSystemLine(`已选择历史会话：${state.selectedSessionTitle}，下一次运行将基于该上下文继续`);
    await refreshSessions();
  });

  $("#datasetGrid").addEventListener("click", (e) => {
    const card = e.target.closest(".dataset-card");
    if (!card) return;
    const id = card.dataset.datasetId;
    state.selectedDatasets.has(id) ? state.selectedDatasets.delete(id) : state.selectedDatasets.add(id);
    renderDatasets();
  });
  $("#historyFilter").addEventListener("input", renderBenchmarkHistory);
  $("#historySort").addEventListener("change", renderBenchmarkHistory);
  $("#historyTable").addEventListener("click", (e) => {
    const button = e.target.closest("[data-history-file]");
    if (button) toggleBenchmarkDetail(button);
  });
}

async function init() {
  bindEvents();
  renderEmptyState();
  await Promise.allSettled([refreshStatus(), refreshSessions(), refreshDatasets(), refreshBenchmarkHistory()]);
}

init().catch((error) => addSystemLine(`初始化失败：${error.message}`));
