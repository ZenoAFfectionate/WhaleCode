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
};

const viewMeta = {
  agent: ["", ""],
  benchmarks: ["数据集测试", "调用项目现有 benchmark scripts，实时展示进度、日志和历史结果。"],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

function formatDuration(seconds) {
  if (seconds == null) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

function cleanDisplayText(value) {
  return String(value ?? "").replace(/^\s+|\s+$/g, "");
}

function setView(view) {
  state.currentView = view;
  $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach((section) => section.classList.toggle("active", section.id === `${view}View`));
  const isAgent = view === "agent";
  $(".topbar").classList.toggle("agent-topbar", isAgent);
  $("#viewTitle").textContent = isAgent ? "" : viewMeta[view][0];
  $("#viewSubtitle").textContent = isAgent ? "" : viewMeta[view][1];
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (!button.dataset.originalText) button.dataset.originalText = button.textContent;
  button.textContent = busy ? label : button.dataset.originalText;
}

function addMessage(role, text, meta = "") {
  const displayText = cleanDisplayText(text);
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="message-label">${role === "user" ? "User" : "WhaleCode"} ${meta ? `· ${escapeHtml(meta)}` : ""}</div>
    <div class="bubble">${escapeHtml(displayText).replace(/\n/g, "<br>")}</div>
  `;
  $("#chatStream").appendChild(article);
  $("#chatStream").scrollTop = $("#chatStream").scrollHeight;
}

function payloadText(payload) {
  if (payload == null) return "";
  if (typeof payload === "string") return cleanDisplayText(payload);
  const priority = ["thought", "thinking", "reasoning", "content", "text", "output", "message", "line", "answer", "error"];
  for (const key of priority) {
    if (payload[key]) return cleanDisplayText(payload[key]);
  }
  return cleanDisplayText(JSON.stringify(payload, null, 2));
}

function payloadTitle(payload, fallback) {
  if (!payload || typeof payload !== "object") return fallback;
  return payload.name || payload.tool || payload.command || payload.action || payload.status || payload.model || fallback;
}

function toolName(payload) {
  if (!payload || typeof payload !== "object") return "工具调用";
  return payload.name || payload.tool || payload.tool_name || payload.command || payload.action || "工具调用";
}

function basename(value) {
  const text = cleanDisplayText(value);
  if (!text) return "";
  const normalized = text.replace(/\\/g, "/").split(/[?#]/)[0];
  return normalized.split("/").filter(Boolean).pop() || normalized;
}

function fileNameFromPayload(payload) {
  if (!payload || typeof payload !== "object") return "";
  const directKeys = ["path", "file_path", "filepath", "file", "target_file", "target_path", "filename"];
  for (const key of directKeys) {
    if (typeof payload[key] === "string" && payload[key].trim()) return basename(payload[key]);
  }
  const args = payload.arguments || payload.args || payload.params || payload.input || payload.parameters;
  if (typeof args === "string") {
    try {
      return fileNameFromPayload(JSON.parse(args));
    } catch {
      return "";
    }
  }
  if (args && typeof args === "object") return fileNameFromPayload(args);
  return "";
}

function isFileTool(name) {
  return ["read", "write", "edit", "delete"].includes(String(name || "").toLowerCase());
}

function toolDisplayName(payload) {
  const name = toolName(payload);
  const file = isFileTool(name) ? fileNameFromPayload(payload) : "";
  return file ? `${name} · ${file}` : name;
}

function toolResultStatus(payload) {
  if (!payload || typeof payload !== "object") return "success";
  const value = payload.status ?? payload.state ?? payload.ok ?? payload.success ?? payload.error;
  if (value === false || value === "failed" || value === "error" || value instanceof Error || payload.error) return "failed";
  return "success";
}

function eventDescriptor(event) {
  const payload = event.payload || {};
  const map = {
    console: ["console", "Console Log", payloadText(payload)],
    model_output: ["thinking", "Thinking", payloadText(payload)],
    tool_call: ["tool-call", `发起工具调用：${toolDisplayName(payload)}`, payloadText(payload)],
    tool_result: [`tool-result ${toolResultStatus(payload)}`, `工具执行${toolResultStatus(payload) === "success" ? "成功" : "失败"}：${toolDisplayName(payload)}`, payloadText(payload)],
    session_loaded: ["system", "会话已加载", payloadText(payload) || "历史上下文已接入本次运行。"],
    benchmark_started: ["system", "Benchmark Started", payloadText(payload)],
    benchmark_output: ["console", "Benchmark Output", payloadText(payload)],
    completed: ["final", "Final Answer", payloadText(payload)],
    failed: ["error", "执行失败", payloadText(payload)],
  };
  return map[event.type] || ["system", event.type, payloadText(payload)];
}

function appendRunEvent(event) {
  if (event.type === "job_created" || event.type === "status") return;
  if (event.type === "completed" || event.type === "failed") return;

  const [kind, title, rawText] = eventDescriptor(event);
  const text = cleanDisplayText(rawText);
  const raw = JSON.stringify(event.payload || {}, null, 2);
  const hasRaw = kind !== "thinking" && raw && raw !== "{}" && raw !== JSON.stringify(text);
  const item = document.createElement("article");
  item.className = `run-event ${kind}`;
  if (event.type === "tool_call" || event.type === "tool_result") {
    item.innerHTML = `
      <div class="event-rail"></div>
      <div class="tool-status-card">
        <div>
          <span class="event-kicker">${event.type === "tool_call" ? "tool call" : "tool result"}</span>
          <strong>${escapeHtml(title)}</strong>
        </div>
        ${text ? `<p>${escapeHtml(text).replace(/\n/g, "<br>")}</p>` : ""}
      </div>
    `;
    $("#chatStream").appendChild(item);
    $("#chatStream").scrollTop = $("#chatStream").scrollHeight;
    return;
  }
  const open = kind === "error" || kind === "tool-call";
  item.innerHTML = `
    <div class="event-rail"></div>
    <details ${open ? "open" : ""}>
      <summary>
        <span class="event-kicker">${escapeHtml(kind.replace("-", " "))}</span>
        <strong>${escapeHtml(title)}</strong>
        <time>${escapeHtml(event.timestamp || "")}</time>
      </summary>
      <div class="event-body">${escapeHtml(text || "无文本内容").replace(/\n/g, "<br>")}</div>
      ${hasRaw ? `<pre>${escapeHtml(raw)}</pre>` : ""}
    </details>
  `;
  $("#chatStream").appendChild(item);
  $("#chatStream").scrollTop = $("#chatStream").scrollHeight;
}

function resetAgentConversation() {
  if (state.activeEventSource) {
    state.activeEventSource.close();
    state.activeEventSource = null;
  }
  state.selectedSession = null;
  state.selectedSessionTitle = "新会话";
  $("#activeSessionName").textContent = "新会话";
  $("#detailStatus").textContent = "idle";
  $("#detailDuration").textContent = "-";
  $("#detailProgress").textContent = "0%";
  $("#agentRunStatus").textContent = "idle";
  $("#chatStream").innerHTML = "";
}

function appendBenchLog(line) {
  const el = document.createElement("div");
  el.className = "log-line";
  el.textContent = line;
  $("#benchLog").appendChild(el);
  $("#benchLog").scrollTop = $("#benchLog").scrollHeight;
}

function subscribeJob(job, handlers = {}) {
  const source = new EventSource(`/api/jobs/${encodeURIComponent(job.id)}/events`);
  const onAny = (event) => {
    const data = JSON.parse(event.data);
    handlers.any?.(data);
    if (handlers[data.type]) handlers[data.type](data);
  };
  [
    "job_created",
    "status",
    "console",
    "model_output",
    "tool_call",
    "tool_result",
    "session_loaded",
    "benchmark_started",
    "benchmark_output",
    "completed",
    "failed",
  ].forEach((name) => source.addEventListener(name, onAny));
  source.onerror = () => {
    if (job.status === "completed" || job.status === "failed") source.close();
  };
  return source;
}

async function refreshStatus() {
  const data = await api("/api/status");
  state.projectRoot = data.project_root;
  $("#projectRoot").textContent = data.project_root;
  $("#workspaceInput").value ||= data.project_root;
  renderModelStatus(data.model);
}

function renderModelStatus(snapshot) {
  state.modelSnapshot = snapshot;
  const isRunning = snapshot.status === "running";
  $("#serviceDot").className = `dot ${isRunning ? "" : "stopped"}`;
  $("#serviceStatus").textContent = snapshot.status;
  $("#activeModel").textContent = snapshot.active_model || "未加载";
}

async function refreshSessions() {
  const data = await api("/api/sessions");
  const sessions = data.sessions || [];
  const html = sessions.length ? sessions.map((session) => `
    <article class="session-item ${state.selectedSession === session.filepath ? "active" : ""}" data-filepath="${escapeHtml(session.filepath)}" data-filename="${escapeHtml(session.filename)}">
      <div>
        <strong>${escapeHtml(session.title || session.filename)}</strong>
        <small>${escapeHtml(session.saved_at || session.created_at || "unknown")}</small>
      </div>
      <button class="icon-button delete-session" title="删除会话">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 7h12m-9 0V5h6v2m-7 3v8m4-8v8m4-8v8M8 7l1 13h6l1-13"/></svg>
      </button>
    </article>
  `).join("") : `<div class="session-item"><div><strong>暂无 Web 会话</strong><small>运行智能体后自动生成</small></div></div>`;
  $("#sessionList").innerHTML = html;
  const drawerList = $("#drawerSessionList");
  if (drawerList) drawerList.innerHTML = html;
}

async function refreshDatasets() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets || [];
  if (!state.selectedDatasets.size) {
    ["hevp", "mbpp"].forEach((id) => state.selectedDatasets.add(id));
  }
  renderDatasets();
}

function renderDatasets() {
  $("#datasetGrid").innerHTML = state.datasets.map((dataset) => `
    <article class="dataset-card ${state.selectedDatasets.has(dataset.id) ? "selected" : ""}" data-dataset-id="${escapeHtml(dataset.id)}">
      <h3>${escapeHtml(dataset.name)}</h3>
      <p>${escapeHtml(dataset.description)}</p>
      <div class="card-meta">
        <span class="pill">${dataset.cases} cases</span>
        <span class="pill">${escapeHtml(dataset.script)}</span>
      </div>
    </article>
  `).join("");
}

async function refreshBenchmarkHistory() {
  const data = await api("/api/benchmarks/history");
  state.benchmarkHistory = data.history || [];
  renderBenchmarkHistory();
}

function benchmarkSummary(item) {
  return item.summary || {};
}

function benchmarkDatasetLabel(item) {
  const summary = benchmarkSummary(item);
  return summary.benchmark || summary.dataset || summary.dataset_name || item.name.replace(/\.(jsonl|json)$/i, "");
}

function benchmarkModelLabel(item) {
  const summary = benchmarkSummary(item);
  return summary.model || summary.model_id || state.modelSnapshot?.active_model || "unknown model";
}

function benchmarkResultSummary(item) {
  const summary = benchmarkSummary(item);
  const parts = [];
  if (summary.pass_rate != null) parts.push(`通过率 ${summary.pass_rate}`);
  if (summary.passed != null && summary.total != null) parts.push(`${summary.passed}/${summary.total} passed`);
  if (summary.failed != null) parts.push(`${summary.failed} failed`);
  if (!parts.length) parts.push(`${Math.round(item.size_bytes / 1024)} KB`);
  return parts.join(" · ");
}

function caseStatus(record) {
  if (!record || typeof record !== "object") return "unknown";
  const value = record.passed ?? record.correct ?? record.success ?? record.is_correct ?? record.ok ?? record.status;
  if (value === true || value === "passed" || value === "pass" || value === "ok" || value === "success") return "passed";
  if (value === false || value === "failed" || value === "fail" || value === "error") return "failed";
  return "unknown";
}

function caseTitle(record, index) {
  if (!record || typeof record !== "object") return `Case ${index + 1}`;
  return record.task_id || record.id || record.name || record.title || record.prompt_id || `Case ${index + 1}`;
}

function caseTrace(record) {
  if (!record || typeof record !== "object") return String(record ?? "");
  const trace = record.trace || record.trajectory || record.reasoning || record.logs || record.output || record.prediction || record.response;
  if (Array.isArray(trace)) return trace.map((item) => typeof item === "string" ? item : JSON.stringify(item, null, 2)).join("\n");
  if (trace && typeof trace === "object") return JSON.stringify(trace, null, 2);
  return trace ? String(trace) : JSON.stringify(record, null, 2);
}

function renderBenchmarkDetail(detail) {
  const summary = detail.summary || {};
  const summaryPairs = Object.entries(summary)
    .filter(([, value]) => value == null || typeof value !== "object")
    .slice(0, 12);
  const records = detail.records || [];
  const casesHtml = records.length ? records.map((record, index) => {
    const status = caseStatus(record);
    return `
      <details class="case-row ${escapeHtml(status)}">
        <summary>
          <span class="case-status">${status === "passed" ? "正确" : status === "failed" ? "错误" : "未知"}</span>
          <strong>${escapeHtml(caseTitle(record, index))}</strong>
        </summary>
        <pre>${escapeHtml(caseTrace(record))}</pre>
      </details>
    `;
  }).join("") : `<div class="benchmark-empty compact-empty">该结果文件没有结构化用例明细，已展示原始运行轨迹预览。</div>`;

  return `
    <div class="result-detail">
      <div class="result-detail-grid">
        ${summaryPairs.map(([key, value]) => `
          <div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>
        `).join("") || `<div><span>records</span><strong>${detail.record_count || 0}</strong></div>`}
      </div>
      <section>
        <h3>用例执行情况</h3>
        <div class="case-list">${casesHtml}</div>
      </section>
      <section>
        <h3>运行轨迹</h3>
        <pre class="trace-preview">${escapeHtml(detail.raw_preview || "暂无运行轨迹")}</pre>
      </section>
    </div>
  `;
}

function renderBenchmarkHistory() {
  const filter = $("#historyFilter").value.trim().toLowerCase();
  const sort = $("#historySort").value;
  let rows = [...state.benchmarkHistory];
  if (filter) {
    rows = rows.filter((item) => {
      const haystack = `${item.name} ${benchmarkDatasetLabel(item)} ${benchmarkModelLabel(item)} ${benchmarkResultSummary(item)}`.toLowerCase();
      return haystack.includes(filter);
    });
  }
  if (sort === "name") rows.sort((a, b) => a.name.localeCompare(b.name));
  if (sort === "size") rows.sort((a, b) => b.size_bytes - a.size_bytes);

  $("#historyTable").innerHTML = rows.length ? rows.map((item) => `
    <article class="history-row result-row" data-result-file="${escapeHtml(item.name)}">
      <button class="result-summary" type="button" data-history-file="${escapeHtml(item.name)}">
        <div>
          <strong>${escapeHtml(benchmarkDatasetLabel(item))}</strong>
          <small>${escapeHtml(item.modified_at)} · ${escapeHtml(item.name)}</small>
        </div>
        <div>
          <span>模型</span>
          <strong>${escapeHtml(benchmarkModelLabel(item))}</strong>
        </div>
        <div>
          <span>结果摘要</span>
          <strong>${escapeHtml(benchmarkResultSummary(item))}</strong>
        </div>
      </button>
      <div class="result-detail-host"></div>
    </article>
  `).join("") : `<article class="benchmark-empty"><strong>暂无测试结果</strong><span>运行完成后，这里会添加包含数据集类型、模型和结果摘要的条目。</span></article>`;
}

async function runAgent(event) {
  event.preventDefault();
  const prompt = $("#promptInput").value.trim();
  if (!prompt) return;

  if (state.activeEventSource) {
    state.activeEventSource.close();
    state.activeEventSource = null;
  }
  addMessage("user", prompt, "submitted");
  $("#promptInput").value = "";
  $("#detailStatus").textContent = "queued";
  $("#detailDuration").textContent = "-";
  $("#detailProgress").textContent = "0%";
  $("#agentRunStatus").textContent = "queued";
  setBusy($("#runAgentButton"), true, "运行中...");

  try {
    const job = await api("/api/agent/runs", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        workspace: $("#workspaceInput").value.trim() || state.projectRoot,
        resume_path: state.selectedSession,
        model: state.modelSnapshot?.active_model || undefined,
      }),
    });

    state.activeEventSource = subscribeJob(job, {
      any(data) {
        $("#detailStatus").textContent = data.job.status;
        $("#detailProgress").textContent = `${Math.round(data.job.progress || 0)}%`;
        $("#agentRunStatus").textContent = data.job.status;
        appendRunEvent(data);
      },
      completed(data) {
        const result = data.payload || {};
        addMessage("agent", result.answer || "任务已完成，但没有返回文本。", `final · 耗时 ${formatDuration(result.duration_seconds)}`);
        $("#detailDuration").textContent = formatDuration(result.duration_seconds);
        setBusy($("#runAgentButton"), false);
        state.activeEventSource?.close();
        state.activeEventSource = null;
        refreshSessions();
      },
      failed(data) {
        addMessage("agent", `执行失败：${data.payload?.error || "unknown error"}`, "failed");
        setBusy($("#runAgentButton"), false);
        state.activeEventSource?.close();
        state.activeEventSource = null;
      },
    });
  } catch (error) {
    addMessage("agent", `提交失败：${error.message}`, "error");
    setBusy($("#runAgentButton"), false);
  }
}

async function runBenchmark(event) {
  event.preventDefault();
  const datasets = [...state.selectedDatasets];
  if (!datasets.length) return;

  $("#benchLog").innerHTML = "";
  setBusy($("#runBenchButton"), true, "测试运行中...");

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
      any(data) {
        const completed = data.job.completed || 0;
        const total = data.job.total || 0;
        if (total) appendBenchLog(`progress: ${completed} / ${total} (${Math.round(data.job.progress || 0)}%)`);
      },
      benchmark_started(data) {
        appendBenchLog(`=== ${data.payload.dataset?.name || "Benchmark"} started ===`);
      },
      benchmark_output(data) {
        appendBenchLog(data.payload.line || "");
      },
      completed(data) {
        appendBenchLog("benchmark completed");
        setBusy($("#runBenchButton"), false);
        refreshBenchmarkHistory();
      },
      failed(data) {
        appendBenchLog(`ERROR: ${data.payload?.error || "unknown error"}`);
        setBusy($("#runBenchButton"), false);
      },
    });
  } catch (error) {
    appendBenchLog(`提交失败：${error.message}`);
    setBusy($("#runBenchButton"), false);
  }
}

async function toggleBenchmarkDetail(button) {
  const file = button.dataset.historyFile;
  const row = button.closest(".result-row");
  const host = row.querySelector(".result-detail-host");
  if (row.classList.contains("open")) {
    row.classList.remove("open");
    host.innerHTML = "";
    return;
  }
  row.classList.add("open");
  host.innerHTML = `<div class="benchmark-empty compact-empty">正在加载测试明细...</div>`;
  try {
    let detail = state.benchmarkDetails.get(file);
    if (!detail) {
      const data = await api(`/api/benchmarks/history/${encodeURIComponent(file)}`);
      detail = data.detail;
      state.benchmarkDetails.set(file, detail);
    }
    host.innerHTML = renderBenchmarkDetail(detail);
  } catch (error) {
    host.innerHTML = `<div class="benchmark-empty compact-empty error-text">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $$("[data-view-jump]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.viewJump)));
  $("#agentForm").addEventListener("submit", runAgent);
  $("#benchForm").addEventListener("submit", runBenchmark);
  $("#refreshSessions").addEventListener("click", refreshSessions);
  $("#newSessionButton").addEventListener("click", async () => {
    resetAgentConversation();
    await refreshSessions();
  });
  $("#openSessionDrawer").addEventListener("click", () => {
    $("#sessionDrawer").classList.add("open");
    $("#sessionDrawer").setAttribute("aria-hidden", "false");
  });
  ["closeSessionDrawer", "drawerCloseButton"].forEach((id) => {
    $(`#${id}`).addEventListener("click", () => {
      $("#sessionDrawer").classList.remove("open");
      $("#sessionDrawer").setAttribute("aria-hidden", "true");
    });
  });
  const handleSessionClick = async (event) => {
    const row = event.target.closest(".session-item");
    if (!row?.dataset.filepath) return;
    if (event.target.closest(".delete-session")) {
      await api(`/api/sessions/${encodeURIComponent(row.dataset.filename)}`, { method: "DELETE" });
      if (state.selectedSession === row.dataset.filepath) {
        state.selectedSession = null;
        state.selectedSessionTitle = "新会话";
        $("#activeSessionName").textContent = "新会话";
      }
      await refreshSessions();
      return;
    }
    state.selectedSession = row.dataset.filepath;
    state.selectedSessionTitle = row.querySelector("strong")?.textContent || row.dataset.filename;
    $("#activeSessionName").textContent = state.selectedSessionTitle;
    $("#sessionDrawer").classList.remove("open");
    $("#sessionDrawer").setAttribute("aria-hidden", "true");
    addMessage("agent", `已选择历史会话：${state.selectedSessionTitle}。下一次运行会基于该上下文继续。`, "session");
    await refreshSessions();
  };
  $("#sessionList").addEventListener("click", handleSessionClick);
  $("#drawerSessionList").addEventListener("click", handleSessionClick);
  $("#datasetGrid").addEventListener("click", (event) => {
    const card = event.target.closest(".dataset-card");
    if (!card) return;
    const id = card.dataset.datasetId;
    if (state.selectedDatasets.has(id)) state.selectedDatasets.delete(id);
    else state.selectedDatasets.add(id);
    renderDatasets();
  });
  $("#historyFilter").addEventListener("input", renderBenchmarkHistory);
  $("#historySort").addEventListener("change", renderBenchmarkHistory);
  $("#historyTable").addEventListener("click", (event) => {
    const button = event.target.closest("[data-history-file]");
    if (button) toggleBenchmarkDetail(button);
  });
}

async function init() {
  bindEvents();
  await Promise.all([
    refreshStatus(),
    refreshSessions(),
    refreshDatasets(),
    refreshBenchmarkHistory(),
  ]);
}

init().catch((error) => {
  addMessage("agent", `初始化失败：${error.message}`, "error");
});
