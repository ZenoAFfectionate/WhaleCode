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
  benchmarkTrajectories: new Map(),
  activeBenchmarkSource: null,
  activeBenchmarkJobId: null,
  blobs: new Map(),
  blobSeq: 0,
  /* C6: 多会话 Tab——state.run / state.activeEventSource 通过 getter 桥接到
     当前激活 tab，既有的 43 处 state.run.* 引用无需改动。 */
  get run() { return _tabs.active.run; },
  set run(value) { _tabs.active.run = value; },
  get activeEventSource() { return _tabs.active.eventSource; },
  set activeEventSource(value) { _tabs.active.eventSource = value; },
};

/* --------------------- C6: 多会话 Tab 管理 ----------------------------- */
const _tabs = { list: [], active: null, seq: 0 };

let _streamOverride = null;  // 后台 tab 事件渲染重定向（见 subscribeJob）
let _eventTab = null;        // 当前正在处理的事件所属 tab（后台 tab 事件期间非空）
function chatStreamEl() { return _streamOverride || _tabs.active.streamEl; }
function isActiveTab(tab) { return tab === _tabs.active; }
/* C6: 事件归属感知——渲染函数经 evtRun() 读写"事件所属 tab"的运行状态，
   经 evtIsForeground() 决定是否触碰全局 vitals DOM / 滚动激活视图。 */
function evtRun() { return _eventTab ? _eventTab.run : state.run; }
function evtIsForeground() { return !_eventTab || _eventTab === _tabs.active; }

function makeRunState() {
  return {
    active: false, jobId: null, steps: 0, nodeIdx: 0, startedAt: 0, timer: null,
    pendingTools: new Map(), pendingOrder: [], logStep: null,
  };
}
function createTab(title) {
  _tabs.seq += 1;
  // 首个 tab 复用 index.html 里的静态 #chatStream 节点，避免重复容器
  const existing = _tabs.list.length === 0 ? $("#traceStream > .trace-inner") : null;
  const streamEl = existing || document.createElement("div");
  streamEl.classList.add("trace-inner");
  const tab = {
    id: "tab-" + _tabs.seq,
    title: title || `会话 ${_tabs.seq}`,
    streamEl,
    vitals: { status: "idle", steps: "0", elapsedText: "—", tokensText: "—" },
    run: makeRunState(),
    eventSource: null,
  };
  _tabs.list.push(tab);
  switchTab(tab.id);
  renderEmptyState();
  renderTabs();
  return tab;
}
function switchTab(tabId) {
  const target = _tabs.list.find((t) => t.id === tabId);
  if (!target || target === _tabs.active) return;
  // vitals 快照保存
  snapshotVitals();
  // DOM 换挂
  if (_tabs.active) {
    const host = $("#traceStream");
    if (host && _tabs.active.streamEl.parentNode === host) host.removeChild(_tabs.active.streamEl);
  }
  _tabs.active = target;
  $("#traceStream").appendChild(target.streamEl);
  restoreVitals();
  // 计时器随 tab 换挂：新激活 tab 若在运行则用其自身 startedAt 续走
  if (_tabs.active.run.active && _tabs.active.run.startedAt) startRunTimer(target);
  else stopRunTimer(target);
  renderTabs();
}
function closeTab(tabId) {
  const idx = _tabs.list.findIndex((t) => t.id === tabId);
  if (idx === -1) return;
  const tab = _tabs.list[idx];
  if (tab.eventSource) tab.eventSource.close();
  tab.run.pendingTools?.clear?.();
  stopRunTimer(tab);
  _tabs.list.splice(idx, 1);
  if (_tabs.list.length === 0) {
    createTab();
    return;
  }
  if (_tabs.active === tab) {
    // 先摘除被关闭 tab 的 streamEl，再切换（switchTab 只摘除 active 的）
    const host = $("#traceStream");
    if (host && tab.streamEl.parentNode === host) host.removeChild(tab.streamEl);
    _tabs.active = null;
    switchTab(_tabs.list[Math.max(0, idx - 1)].id);
  } else {
    renderTabs();
  }
}
function snapshotVitals() {
  const tab = _tabs.active;
  if (!tab) return;
  tab.vitals = {
    status: $("#vitalStatus").textContent,
    steps: $("#vitalStep").textContent,
    elapsedText: $("#vitalElapsed").textContent,
    tokensText: $("#vitalTokens").textContent,
  };
  tab.title = tab.title || "会话";
}
function restoreVitals() {
  const tab = _tabs.active;
  if (!tab) return;
  $("#vitalStatus").textContent = tab.vitals.status;
  $("#vitalStep").textContent = tab.vitals.steps;
  $("#vitalElapsed").textContent = tab.vitals.elapsedText;
  $("#vitalTokens").textContent = tab.vitals.tokensText;
  const bar = $("#agentVitals");
  const statusMap = { 运行中: "running", 排队中: "queued", 已完成: "completed", 失败: "failed", 已取消: "cancelled" };
  const cls = statusMap[tab.vitals.status];
  bar.classList.toggle("is-running", cls === "running" || cls === "queued");
  bar.classList.toggle("is-done", cls === "completed");
  bar.classList.toggle("is-failed", cls === "failed");
  bar.classList.toggle("is-cancelled", cls === "cancelled");
}
function renderTabs() {
  const bar = $("#agentTabs");
  if (_tabs.list.length <= 1) { bar.hidden = true; return; }
  bar.hidden = false;
  bar.innerHTML = _tabs.list.map((t) => `
    <div class="tab-item ${t === _tabs.active ? "active" : ""}" data-tab-id="${t.id}">
      <button type="button" class="tab-label" data-switch-tab="${t.id}" title="${escapeHtml(t.title)}">
        ${t.run?.active ? '<span class="tab-dot"></span>' : ""}${escapeHtml(t.title)}
      </button>
      <button type="button" class="tab-close" data-close-tab="${t.id}" title="关闭标签页" aria-label="关闭标签页">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>
      </button>
    </div>`).join("");
}

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
    cancelled: "已取消", stopped: "已停止", loading: "加载中", starting: "启动中", error: "异常", unloaded: "未加载" }[status] || status || "就绪";
}

function payloadText(payload) {
  if (payload == null) return "";
  if (typeof payload === "string") return clean(payload);
  for (const key of [
    "thought", "thinking", "reasoning", "reasoning_content", "result_content",
    "content", "text", "output", "stdout", "stderr", "message", "line", "answer", "error",
  ]) {
    if (payload[key]) return clean(payload[key]);
  }
  if (payload.result && typeof payload.result === "object") return payloadText(payload.result);
  if (payload.arguments && typeof payload.arguments === "object") return clean(JSON.stringify(payload.arguments, null, 2));
  return clean(JSON.stringify(payload, null, 2));
}
function toolName(payload) {
  if (!payload || typeof payload !== "object") return "工具";
  return payload.name || payload.tool || payload.tool_name || payload.command || payload.action || "工具";
}
function toolCallId(payload) {
  if (!payload || typeof payload !== "object") return "";
  return String(payload.tool_call_id || payload.call_id || payload.id || payload.request_id || "");
}
function toolArgs(payload) {
  if (!payload || typeof payload !== "object") return {};
  const args = payload.arguments || payload.args || payload.params || payload.input || payload.parameters;
  if (typeof args === "string") { try { return JSON.parse(args); } catch { return { input: args }; } }
  return args && typeof args === "object" ? args : {};
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
/* C1: 提取完整相对路径（供工作区面板联动打开），无则返回空串 */
function filePathFromPayload(payload) {
  if (!payload || typeof payload !== "object") return "";
  for (const k of ["path", "file_path", "filepath", "target_file", "target_path"]) {
    if (typeof payload[k] === "string" && payload[k].trim()) return payload[k].trim();
  }
  const args = payload.arguments || payload.args || payload.params || payload.input || payload.parameters;
  if (typeof args === "string") { try { return filePathFromPayload(JSON.parse(args)); } catch { return ""; } }
  if (args && typeof args === "object") return filePathFromPayload(args);
  return "";
}
/* C1: 文件路径 → 可点击 chip（点击在工作区面板中打开该文件） */
function fileLinkChip(name, path) {
  if (!name) return "";
  if (!path) return `<strong>${escapeHtml(name)}</strong>`;
  return `<strong><a class="file-link" data-open-file="${escapeHtml(path)}" title="在工作区面板中打开 ${escapeHtml(path)}">${escapeHtml(name)}</a></strong>`;
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

/* ------------------- C2: unified diff 增强查看器 ----------------------- */
function parseUnifiedDiff(diffText) {
  /* 解析 unified diff → { hunks: [{ header, oldStart, newStart, rows: [...] }], adds, dels }
     rows: [{ type: 'ctx'|'add'|'del'|'hunk', oldNo, newNo, text }] */
  const lines = String(diffText || "").split("\n");
  const hunks = [];
  let current = null;
  let adds = 0, dels = 0;
  let oldNo = 0, newNo = 0;
  for (const line of lines) {
    const hunkMatch = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunkMatch) {
      current = { header: line, oldStart: Number(hunkMatch[1]), newStart: Number(hunkMatch[2]), rows: [] };
      hunks.push(current);
      oldNo = current.oldStart;
      newNo = current.newStart;
      continue;
    }
    if (!current) continue;  // hunk 头之前的文件名/索引行忽略
    if (line.startsWith("diff ") || line.startsWith("--- ") || line.startsWith("+++ ")) continue;
    if (line.startsWith("+")) {
      current.rows.push({ type: "add", newNo: newNo++, text: line.slice(1) });
      adds += 1;
    } else if (line.startsWith("-")) {
      current.rows.push({ type: "del", oldNo: oldNo++, text: line.slice(1) });
      dels += 1;
    } else if (line.startsWith(" ")) {
      current.rows.push({ type: "ctx", oldNo: oldNo++, newNo: newNo++, text: line.slice(1) });
    } else if (line === "") {
      current.rows.push({ type: "ctx", oldNo: oldNo, newNo: newNo, text: "" });
    }
  }
  return { hunks, adds, dels };
}
function renderDiffBlock(diffText, options = {}) {
  const parsed = parseUnifiedDiff(diffText);
  if (!parsed.hunks.length) return null;
  const ctxLimit = options.context ?? 3;
  const rowsHtml = [];
  for (const hunk of parsed.hunks) {
    // 上下文折叠：连续 ctx 行超过 2*ctxLimit+1 时只保留首尾
    const kept = new Set();
    let runStart = -1;
    const rows = hunk.rows;
    for (let i = 0; i <= rows.length; i += 1) {
      const isCtx = i < rows.length && rows[i].type === "ctx";
      if (isCtx && runStart === -1) runStart = i;
      if (!isCtx && runStart !== -1) {
        const runEnd = i - 1;
        const len = runEnd - runStart + 1;
        if (len > ctxLimit * 2 + 1) {
          for (let k = runStart; k < runStart + ctxLimit; k += 1) kept.add(k);
          for (let k = runEnd - ctxLimit + 1; k <= runEnd; k += 1) kept.add(k);
          kept.add(-runStart);  // 折叠占位标记
        } else {
          for (let k = runStart; k <= runEnd; k += 1) kept.add(k);
        }
        runStart = -1;
      }
    }
    rowsHtml.push(`<div class="diff-hunk-header">${escapeHtml(hunk.header)}</div>`);
    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      if (kept.has(-i)) {
        const prev = rows[i - 1];
        rowsHtml.push(`<div class="diff-row fold"><span class="dl">${prev ? prev.oldNo ?? "" : ""}</span><span class="dl">${prev ? prev.newNo ?? "" : ""}</span><span class="dc">⋯ 上下文已折叠 ⋯</span></div>`);
      }
      if (!kept.has(i)) continue;
      const cls = row.type === "add" ? "add" : row.type === "del" ? "del" : "ctx";
      const mark = row.type === "add" ? "+" : row.type === "del" ? "-" : " ";
      rowsHtml.push(`<div class="diff-row ${cls}"><span class="dl">${row.oldNo ?? ""}</span><span class="dl">${row.newNo ?? ""}</span><span class="dm">${mark}</span><span class="dc">${escapeHtml(row.text)}</span></div>`);
    }
  }
  const badge = `<span class="diff-badge"><i class="add">+${parsed.adds}</i><i class="del">−${parsed.dels}</i></span>`;
  return `<div class="diff-view"><div class="diff-toolbar"><span class="label">${escapeHtml(options.label || "diff")}</span>${badge}</div><div class="diff-body">${rowsHtml.join("")}</div></div>`;
}
function storeBlob(value) {
  const id = "blob-" + (++state.blobSeq);
  state.blobs.set(id, String(value ?? ""));
  return id;
}
function headTail(value, head = 40, tail = 24) {
  const lines = String(value ?? "").split("\n");
  if (lines.length <= head + tail) return { text: lines.join("\n"), truncated: false, omitted: 0 };
  return {
    text: lines.slice(0, head).concat(["… 已省略 " + (lines.length - head - tail) + " 行 …"], lines.slice(-tail)).join("\n"),
    truncated: true,
    omitted: lines.length - head - tail,
  };
}
function longTextBlock(value, options = {}) {
  const full = String(value ?? "");
  const preview = headTail(full, options.head ?? 40, options.tail ?? 24);
  const id = storeBlob(full);
  const label = escapeHtml(options.label || "输出") + (preview.truncated ? " · 已裁剪预览" : "");
  const lang = options.language ? " data-language=\"" + escapeHtml(options.language) + "\"" : "";
  return "<div class=\"long-output\" data-blob-id=\"" + id + "\">"
    + "<div class=\"output-toolbar\"><span>" + label + "</span>"
    + "<button type=\"button\" class=\"mini-button\" data-copy-blob=\"" + id + "\">复制</button>"
    + (preview.truncated ? "<button type=\"button\" class=\"mini-button\" data-expand-blob=\"" + id + "\">展开全部</button>" : "")
    + "</div><pre" + lang + ">" + escapeHtml(preview.text) + "</pre></div>";
}
function rawBlock(payload) {
  const raw = JSON.stringify(payload || {}, null, 2);
  if (!raw || raw === "{}") return "";
  return "<details class=\"step-raw\"><summary>raw</summary>" + longTextBlock(raw, { label: "raw JSON", language: "json" }) + "</details>";
}
function renderInline(text) {
  // 行内 markdown：行内代码 → 加粗 → 斜体 → 链接（按此顺序避免正则互相干扰）
  let out = escapeHtml(text);
  out = out.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  // C3: 链接协议白名单——javascript:/data: 等危险 scheme 降级为纯文本
  out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, label, href) => {
    const safe = /^(https?:\/\/|\/|\.\/|#)/i.test(href) ? href : "#";
    return `<a href="${safe}" target="_blank" rel="noopener">${label}</a>`;
  });
  return out;
}
function renderMarkdown(value) {
  const src = clean(value);
  if (!src) return "";
  const fence = String.fromCharCode(96, 96, 96);
  const chunks = src.split(fence);
  return chunks.map((chunk, idx) => {
    if (idx % 2 === 1) {
      const firstBreak = chunk.indexOf("\n");
      const lang = firstBreak >= 0 ? chunk.slice(0, firstBreak).trim() : "";
      const code = firstBreak >= 0 ? chunk.slice(firstBreak + 1) : chunk;
      const langAttr = lang ? " data-language=\"" + escapeHtml(lang) + "\"" : "";
      return "<pre class=\"md-code\"" + langAttr + "><code>" + escapeHtml(code) + "</code></pre>";
    }
    const lines = chunk.split("\n");
    const out = [];
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (!line.trim()) continue;
      // 标题 1-6 级
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        // Keep generated headings within valid HTML h2..h6 (h7/h8 are
        // custom unknown elements and lose browser semantics/styles).
        const level = Math.min(6, heading[1].length + 1);
        out.push("<h" + level + ">" + renderInline(heading[2]) + "</h" + level + ">");
        continue;
      }
      // C3: 块引用（支持连续多行合并）
      if (/^\s*>\s?/.test(line)) {
        const quoteLines = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^\s*>\s?/, ""));
          i += 1;
        }
        i -= 1;
        out.push("<blockquote>" + renderMarkdown(quoteLines.join("\n")) + "</blockquote>");
        continue;
      }
      // C3: 任务列表（- [ ] / - [x]，渲染为只读 checkbox）
      if (/^\s*[-*+]\s+\[[ xX]\]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*+]\s+\[[ xX]\]\s+/.test(lines[i])) {
          const checked = /\[[xX]\]/.test(lines[i]);
          items.push(`<li class="task-item"><span class="task-check${checked ? " done" : ""}">${checked ? "✓" : ""}</span>`
            + renderInline(lines[i].replace(/^\s*[-*+]\s+\[[ xX]\]\s+/, "")) + "</li>");
          i += 1;
        }
        i -= 1;
        out.push("<ul class=\"task-list\">" + items.join("") + "</ul>");
        continue;
      }
      // C3: 水平线
      if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) {
        out.push("<hr>");
        continue;
      }
      // 无序列表
      if (/^\s*[-*+]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          items.push("<li>" + renderInline(lines[i].replace(/^\s*[-*+]\s+/, "")) + "</li>");
          i += 1;
        }
        i -= 1;
        out.push("<ul>" + items.join("") + "</ul>");
        continue;
      }
      // 有序列表
      if (/^\s*\d+[.)]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          items.push("<li>" + renderInline(lines[i].replace(/^\s*\d+[.)]\s+/, "")) + "</li>");
          i += 1;
        }
        i -= 1;
        out.push("<ol>" + items.join("") + "</ol>");
        continue;
      }
      // 表格
      if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
        const headers = line.split("|").map((x) => clean(x)).filter(Boolean);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          rows.push(lines[i].split("|").map((x) => clean(x)).filter(Boolean));
          i += 1;
        }
        i -= 1;
        out.push("<table><thead><tr>" + headers.map((h) => "<th>" + renderInline(h) + "</th>").join("") + "</tr></thead><tbody>"
          + rows.map((row) => "<tr>" + row.map((c) => "<td>" + renderInline(c) + "</td>").join("") + "</tr>").join("")
          + "</tbody></table>");
        continue;
      }
      out.push("<p>" + renderInline(line) + "</p>");
    }
    return out.join("");
  }).join("");
}

/* --------------------------- view switching --------------------------- */
function setView(view) {
  state.currentView = view;
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `${view}View`));
}
function setBusy(button, busy, label) {
  button.disabled = busy;
  if (!button.dataset.originalHtml) button.dataset.originalHtml = button.innerHTML;
  button.innerHTML = busy ? '<span class="button-spinner" aria-hidden="true"></span>' + escapeHtml(label) : button.dataset.originalHtml;
}

/* ----------------------------- vitals --------------------------------- */
function setVitalsStatus(status) {
  const bar = $("#agentVitals");
  bar.classList.toggle("is-running", status === "running" || status === "queued");
  bar.classList.toggle("is-done", status === "completed");
  bar.classList.toggle("is-failed", status === "failed");
  bar.classList.toggle("is-cancelled", status === "cancelled");
  $("#vitalStatus").textContent = statusLabel(status);
  const label = statusLabel(status);
  document.title = (status === "running" || status === "queued")
    ? `● ${label} · ${BASE_TITLE}`
    : (status === "failed" ? `✕ ${label} · ${BASE_TITLE}` : BASE_TITLE);
}
/* C6: 计时器归属各 tab——后台 tab 结束只停自己的，不误停激活 tab。 */
function startRunTimer(tab = _tabs.active) {
  if (!tab) return;
  stopRunTimer(tab);
  tab.run.startedAt = performance.now();
  tab.run.timer = setInterval(() => {
    if (tab === _tabs.active) {
      $("#vitalElapsed").textContent = formatDuration((performance.now() - tab.run.startedAt) / 1000);
    }
  }, 200);
}
function stopRunTimer(tab = _tabs.active) {
  if (tab && tab.run.timer) { clearInterval(tab.run.timer); tab.run.timer = null; }
}

/* --------------------------- trace stream ----------------------------- */
function ensureStreamReady() {
  const empty = chatStreamEl().querySelector(".empty-state");
  if (empty) empty.remove();
}
function scrollStream() {
  // C6: 后台 tab 的事件渲染进其自身 DOM（detached），不滚动激活视图。
  if (!evtIsForeground()) return;
  const s = $("#traceStream");
  // Only auto-scroll while the user is already near the bottom; yanking the
  // viewport on every event would make it impossible to read earlier steps.
  const nearBottom = s.scrollHeight - s.scrollTop - s.clientHeight < 120;
  if (nearBottom) s.scrollTop = s.scrollHeight;
}
function nextIdx() { const run = evtRun(); run.nodeIdx += 1; return String(run.nodeIdx).padStart(2, "0"); }

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
      <button class="step-head" type="button" aria-expanded="false">
        <span class="step-kicker">${escapeHtml(kicker)}</span>
        <span class="step-primary">${primary ? (isFile ? `<span class="path">${escapeHtml(primary)}</span>` : escapeHtml(primary)) : ""}</span>
        <span class="step-meta">${meta || ""}<svg class="step-caret" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg></span>
      </button>
      <div class="step-body">${body || ""}</div>
    </div>`;
  const main = art.querySelector(".step-main");
  const head = art.querySelector(".step-head");
  head.addEventListener("click", () => {
    const open = main.classList.toggle("open");
    head.setAttribute("aria-expanded", String(open));
  });
  return art;
}

function addThinkStep(event) {
  ensureStreamReady();
  evtRun().logStep = null;
  const text = payloadText(event.payload);
  const art = stepShell({
    phase: "think", idx: nextIdx(), kicker: "思考",
    primary: firstLine(text) || "模型思考",
    body: "<div class=\"body-text\">" + renderMarkdown(text) + "</div>",
  });
  chatStreamEl().appendChild(art);
  scrollStream();
}

function toolKind(name) {
  const n = String(name || "").toLowerCase();
  if (n.includes("bash") || n.includes("shell")) return "bash";
  if (n === "read" || n.includes("read")) return "read";
  if (n === "edit" || n.includes("edit")) return "edit";
  if (n === "write" || n.includes("write")) return "write";
  if (n.includes("grep") || n.includes("search") || n.includes("glob") || n.includes("web")) return "search";
  return "tool";
}
function toolPrimary(payload) {
  const name = toolName(payload);
  const args = toolArgs(payload);
  const kind = toolKind(name);
  if (kind === "bash") return args.command || args.cmd || payload.command || "Bash command";
  if (kind === "read" || kind === "edit" || kind === "write") return fileNameFromPayload(args) || fileNameFromPayload(payload) || kind;
  if (kind === "search") return args.pattern || args.query || args.q || args.path || toolTarget(payload) || "Search";
  return toolTarget(payload) || name;
}
function renderToolCallBody(payload) {
  const name = toolName(payload);
  const args = toolArgs(payload);
  const kind = toolKind(name);
  const cards = [];
  const link = fileLinkChip(toolPrimary(payload), filePathFromPayload(payload));
  if (kind === "bash") {
    const cmd = args.command || args.cmd || payload.command || payloadText(payload);
    cards.push("<div class=\"tool-card bash\"><span class=\"label\">Command</span>" + longTextBlock(cmd, { label: "bash", language: "bash", head: 20, tail: 12 }) + "</div>");
  } else if (kind === "read") {
    cards.push("<div class=\"tool-card read\"><span class=\"label\">Read</span>" + link + "</div>");
  } else if (kind === "edit" || kind === "write") {
    // C2: Edit 调用参数自带 old/new，本地构造预览 diff（结果返回后由 diff_preview 接管）
    let previewDiff = "";
    if (kind === "edit" && typeof args.old_string === "string" && typeof args.new_string === "string") {
      const p = args.path || args.file_path || "";
      const oldLines = String(args.old_string).split("\n");
      const newLines = String(args.new_string).split("\n");
      const prefixDiffLines = (lines, marker) => lines.map((line) => marker + line).join("\n");
      previewDiff = renderDiffBlock(
        `--- a/${p}\n+++ b/${p}\n@@ -1,${oldLines.length} +1,${newLines.length} @@\n${prefixDiffLines(oldLines, "-")}\n${prefixDiffLines(newLines, "+")}`,
        { label: "pending edit" },
      ) || "";
    }
    const body = args.patch || args.content || payload.output || payloadText(payload);
    cards.push("<div class=\"tool-card " + kind + "\"><span class=\"label\">" + kind.toUpperCase() + "</span>" + link
      + (previewDiff || longTextBlock(body, { label: kind === "edit" ? "diff / patch" : "content", head: 28, tail: 18 })) + "</div>");
  } else if (kind === "search") {
    cards.push("<div class=\"tool-card search\"><span class=\"label\">Search</span><strong>" + escapeHtml(toolPrimary(payload)) + "</strong>" + longTextBlock(JSON.stringify(args, null, 2), { label: "query", language: "json", head: 16, tail: 8 }) + "</div>");
  } else {
    cards.push("<div class=\"tool-card\"><span class=\"label\">Input</span>" + longTextBlock(JSON.stringify(args && Object.keys(args).length ? args : payload, null, 2), { label: "tool input", language: "json", head: 24, tail: 12 }) + "</div>");
  }
  return cards.join("") + rawBlock(payload);
}
function errorAdvice(payload) {
  const name = toolName(payload);
  const kind = toolKind(name);
  if (kind === "bash") return "检查命令、退出码、stderr、依赖安装以及当前工作目录。";
  if (kind === "read") return "检查路径是否存在、权限是否足够，以及文件是否过大。";
  if (kind === "edit" || kind === "write") return "检查目标路径、补丁上下文、权限以及是否存在并发改动。";
  if (kind === "search") return "检查搜索词、目录范围、忽略规则以及结果是否被截断。";
  return "检查工具入参、错误摘要和 raw 详情。";
}
function renderToolResultBody(payload, ok) {
  const text = payloadText(payload) || (ok ? "工具执行完成。" : "工具执行失败。");
  const code = payload?.exit_code ?? payload?.returncode ?? payload?.code;
  const name = toolName(payload);
  const pieces = [];
  if (!ok) {
    pieces.push("<div class=\"error-card\"><span class=\"label\">错误摘要</span><strong>" + escapeHtml(firstLine(text) || "工具失败") + "</strong>"
      + (code != null ? "<small>Exit code: " + escapeHtml(code) + "</small>" : "")
      + "<p>" + escapeHtml(errorAdvice(payload)) + "</p></div>");
  }
  // C2: Edit/Write 结果优先渲染结构化 diff_preview（行号双列 + 统计徽标）
  const diffPreview = payload?.data?.diff_preview;
  const kind = toolKind(name);
  if (typeof diffPreview === "string" && diffPreview.trim()) {
    const diffHtml = renderDiffBlock(diffPreview, { label: kind === "edit" ? "edit diff" : "write diff" });
    if (diffHtml) {
      // 不能只显示 diff：Write/Edit 的 formatter、diagnostics、大小和
      // backup 等有效摘要位于 result_content 的 Unified diff 之前。
      const marker = "\nUnified diff preview:";
      const summary = text.includes(marker) ? text.split(marker)[0].trim() : text;
      if (summary) pieces.push(longTextBlock(summary, { label: "result summary", head: 24, tail: 12 }));
      pieces.push(diffHtml);
      return pieces.join("") + rawBlock(payload);
    }
  }
  pieces.push(longTextBlock(text, { label: ok ? "output" : "error output", head: 48, tail: 28 }));
  return pieces.join("") + rawBlock(payload);
}

function addActStep(event) {
  ensureStreamReady();
  const run = evtRun();
  run.logStep = null;
  run.steps += 1;
  if (evtIsForeground()) $("#vitalStep").textContent = String(run.steps);
  const p = event.payload || {};
  let callId = toolCallId(p);
  if (!callId) callId = "auto-" + run.steps + "-" + run.nodeIdx;
  const art = stepShell({
    phase: "act running tool-" + toolKind(toolName(p)), idx: nextIdx(), kicker: toolName(p).toUpperCase(),
    primary: toolPrimary(p),
    meta: "<span class=\"step-stat\">运行中</span>",
    body: renderToolCallBody(p),
  });
  chatStreamEl().appendChild(art);
  run.pendingTools.set(callId, { el: art, start: performance.now(), payload: p });
  run.pendingOrder.push(callId);
  scrollStream();
}

function addStandaloneObserve(event) {
  const p = event.payload || {};
  const ok = toolResultStatus(p) === "success";
  const art = stepShell({
    phase: ok ? "pass" : "fail", idx: nextIdx(), kicker: "OBSERVE",
    primary: toolPrimary(p),
    meta: "<span class=\"step-stat\">" + (ok ? "✓" : "✗") + "</span>",
    body: renderToolResultBody(p, ok),
  });
  chatStreamEl().appendChild(art);
  scrollStream();
}

function parseAskUserAnswers(text) {
  /* 解析工具 text 中的问答对：`User answered N question(s):` 后每行 `- qid: answer` */
  const lines = String(text || "").split("\n");
  const pairs = [];
  for (const line of lines) {
    const m = line.match(/^\s*-\s*([^:]+):\s*(.+)$/);
    if (m) pairs.push({ id: m[1].trim(), answer: m[2].trim() });
  }
  return pairs;
}

function renderAskUserResult(payload, ok, el) {
  /* AskUser 完成后的美观展示：绿色成功主题 + 问题/回答对照（见 IMPROVEMENT.md 改进 1） */
  if (!ok) return renderToolResultBody(payload, false);
  const text = payloadText(payload) || "";
  const pairs = parseAskUserAnswers(text);
  const questions = (el && el._askQuestions) || [];
  const qText = (qid) => {
    const q = questions.find((item) => String(item.id || "") === qid);
    return q ? String(q.text || "") : "";
  };

  const items = pairs.map((pair) => {
    const question = qText(pair.id);
    const qHtml = question
      ? `<span class="ask-result-q">${renderInline(question)}</span>`
      : `<span class="ask-result-q ask-result-q-muted">问题 ${escapeHtml(pair.id)}</span>`;
    return `<div class="ask-result-item">
      ${qHtml}
      <div class="ask-result-a">${escapeHtml(pair.answer).replace(/\n/g, "<br>")}</div>
    </div>`;
  }).join("");

  if (!items) return renderToolResultBody(payload, true);
  return `<div class="ask-result">
    <div class="ask-result-head"><span class="ask-result-badge">✓ 已收到你的回答</span></div>
    ${items}
  </div>`;
}

function resolveActStep(event) {
  const p = event.payload || {};
  const ok = toolResultStatus(p) === "success";
  const run = evtRun();
  let callId = toolCallId(p);
  if (!callId || !run.pendingTools.has(callId)) {
    callId = run.pendingOrder.find((id) => run.pendingTools.has(id)) || "";
  }
  const pending = callId ? run.pendingTools.get(callId) : null;
  if (!pending) return addStandaloneObserve(event);
  run.pendingTools.delete(callId);
  run.pendingOrder = run.pendingOrder.filter((id) => id !== callId);
  const el = pending.el;
  el.classList.remove("running");
  el.classList.add(ok ? "pass" : "fail");
  const dur = formatDuration((performance.now() - pending.start) / 1000);
  el.querySelector(".step-meta").innerHTML =
    "<span class=\"step-stat\">" + (ok ? "✓" : "✗") + "</span><span>" + dur + "</span>"
    + "<svg class=\"step-caret\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"m9 6 6 6-6 6\"/></svg>";
  const body = el.querySelector(".step-body");
  const name = String(toolName(p)).toLowerCase();
  body.innerHTML = name === "askuser"
    ? renderAskUserResult(p, ok, el)
    : renderToolResultBody(p, ok);
  scrollStream();
}

function addBuiltinToolStep(event) {
  // thought 工具承载模型的显式推理；finish 等其余内建工具由 completed 收尾，不重复展示。
  const p = event.payload || {};
  const name = String(toolName(p)).toLowerCase();
  if (name !== "thought" && name !== "thinking") return;
  const text = clean(p.result_content || payloadText(p)).replace(/^Reasoning:\s*/i, "");
  if (!text) return;
  ensureStreamReady();
  evtRun().logStep = null;
  const art = stepShell({
    phase: "think", idx: nextIdx(), kicker: "思考",
    primary: firstLine(text) || "模型思考",
    body: "<div class=\"body-text\">" + renderMarkdown(text) + "</div>",
  });
  chatStreamEl().appendChild(art);
  scrollStream();
}

function buildAskForm(questions, jobId, onSubmitted) {
  /* 构造提问表单 HTML，绑定选项选中/提交逻辑，返回 {html, wire}。
     wire() 在卡片挂载后调用以绑定事件。 */
  const qBlocks = questions.map((q, qi) => {
    const qid = String(q.id || `q${qi + 1}`);
    const text = String(q.text || "");
    const options = Array.isArray(q.options) && q.options.length ? q.options : [];
    let controlHtml;
    if (options.length) {
      const qidAttr = escapeHtml(qid);
      controlHtml = `<div class="ask-options" data-qid="${qidAttr}">` + options.map((opt, oi) => {
        const label = String((opt && (opt.label || opt.text || opt.value)) || "");
        const value = String((opt && opt.value) || label);
        return `<button type="button" class="ask-option" data-qid="${qidAttr}" data-value="${escapeHtml(value)}">
          <span class="ask-option-num">${oi + 1}</span>
          <span class="ask-option-label">${escapeHtml(label)}</span>
        </button>`;
      }).join("") + "</div>";
    } else {
      controlHtml = `<textarea class="ask-input" data-qid="${escapeHtml(qid)}" rows="2" placeholder="输入你的回答…"></textarea>`;
    }
    return `<div class="ask-question" data-qid="${escapeHtml(qid)}">
      <div class="ask-qtext">${renderInline(text)}</div>
      ${controlHtml}
    </div>`;
  }).join("");
  const html = `<div class="ask-box">${qBlocks}
      <div class="ask-actions">
        <button type="button" class="ask-submit" data-job="${escapeHtml(jobId)}">提交回答</button>
        <span class="ask-hint">回答提交后模型将继续执行</span>
      </div>
    </div>`;

  function wire(root) {
    const byQid = (selector, qid) => [...root.querySelectorAll(selector)]
      .filter((el) => el.dataset.qid === qid);
    root.querySelectorAll(".ask-option").forEach((btn) => {
      btn.addEventListener("click", () => {
        const qid = btn.dataset.qid;
        byQid(".ask-option", qid).forEach((b) => b.classList.remove("selected"));
        btn.classList.add("selected");
      });
    });
    const submit = root.querySelector(".ask-submit");
    submit.addEventListener("click", async () => {
      const answers = questions.map((q, qi) => {
        const qid = String(q.id || `q${qi + 1}`);
        let answer = "";
        const selected = byQid(".ask-option.selected", qid)[0];
        if (selected) answer = selected.dataset.value;
        else {
          const ta = byQid("textarea", qid)[0];
          if (ta) answer = ta.value.trim();
        }
        return { id: qid, answer };
      });
      submit.disabled = true;
      submit.textContent = "提交中…";
      try {
        await api("/api/agent/answers", {
          method: "POST",
          body: JSON.stringify({ job_id: jobId, answers }),
        });
        if (onSubmitted) onSubmitted(root, submit);
      } catch (err) {
        submit.disabled = false;
        submit.textContent = "提交回答";
        const hint = root.querySelector(".ask-hint");
        if (hint) hint.textContent = "提交失败：" + (err.message || err) + "（可能已超时）";
      }
    });
  }
  return { html, wire };
}

function addAskUserCard(event) {
  ensureStreamReady();
  const run = evtRun();
  run.logStep = null;
  const p = event.payload || {};
  const questions = Array.isArray(p.questions) ? p.questions : [];
  if (!questions.length) return;
  const jobId = p.job_id || run.jobId;
  const form = buildAskForm(questions, jobId, (root) => {
    const hint = root.querySelector(".ask-hint");
    if (hint) hint.textContent = "已提交回答，等待模型继续执行…";
    root.querySelectorAll(".ask-input, .ask-option").forEach((el) => { el.disabled = true; });
  });
  // 记住问题列表：tool_result 到达时用它渲染美观的问答结果（result_content 只有 qid+回答）
  form._questions = questions;

  // 优先注入到正在运行的 AskUser 工具调用卡片（复用其展示框，见 IMPROVEMENT.md 改进 1）
  const steps = document.querySelectorAll("#chatStream .step.running");
  let target = null;
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const kicker = steps[i].querySelector(".step-kicker");
    if (kicker && kicker.textContent.trim().toLowerCase() === "askuser") {
      target = steps[i];
      break;
    }
  }

  const storeQuestions = (el) => { el._askQuestions = questions; };

  if (target) {
    const body = target.querySelector(".step-body");
    const main = target.querySelector(".step-main");
    if (body) body.innerHTML = form.html;
    if (main) main.classList.add("open");
    if (form.wire) form.wire(target);
    storeQuestions(target);
    const meta = target.querySelector(".step-meta");
    if (meta) meta.innerHTML = "<span class=\"step-stat\">等待回答</span>";
    const head = target.querySelector(".step-head");
    if (head) head.setAttribute("aria-expanded", "true");
    scrollStream();
    return;
  }

  // 回退：独立提问卡片
  const art = stepShell({
    phase: "act running tool-ask", idx: nextIdx(), kicker: "AskUser",
    primary: "需要你的确认",
    meta: "<span class=\"step-stat\">等待回答</span>",
    body: form.html,
  });
  art.querySelector(".step-main").classList.add("open");
  art.querySelector(".step-head").setAttribute("aria-expanded", "true");
  if (form.wire) form.wire(art);
  storeQuestions(art);
  chatStreamEl().appendChild(art);
  scrollStream();
}

function addErrorLine(text) {
  ensureStreamReady();
  evtRun().logStep = null;
  const art = stepShell({
    phase: "fail", idx: nextIdx(), kicker: "错误",
    primary: firstLine(text) || "运行错误",
    meta: "<span class=\"step-stat\">✗</span>",
    body: "<div class=\"body-text\">" + escapeHtml(text) + "</div>",
  });
  art.querySelector(".step-main").classList.add("open");
  chatStreamEl().appendChild(art);
  scrollStream();
}

function addLogLine(event) {
  ensureStreamReady();
  const line = payloadText(event.payload);
  if (!line) return;
  const run = evtRun();
  if (run.logStep && run.logStep.isConnected) {
    const list = run.logStep.querySelector(".log-list");
    const el = document.createElement("div");
    el.className = "log-line";
    el.textContent = line;
    list.appendChild(el);
    const count = list.children.length;
    run.logStep.querySelector(".log-count").textContent = `${count} 行`;
    return;
  }
  const art = stepShell({
    phase: "log", kicker: "日志", primary: "运行日志",
    meta: `<span class="log-count">1 行</span>`,
    body: `<div class="log-list" style="display:grid;gap:5px">` +
          `<div class="log-line">${escapeHtml(line)}</div></div>`,
  });
  chatStreamEl().appendChild(art);
  run.logStep = art;
  scrollStream();
}

function addSystemLine(text) {
  ensureStreamReady();
  evtRun().logStep = null;
  const art = document.createElement("article");
  art.className = "step system";
  art.innerHTML = `<div class="step-gutter"><span class="step-node"></span></div>
    <div class="step-main"><div class="step-head" style="cursor:default">
      <span class="step-kicker">系统</span>
      <span class="step-primary">${escapeHtml(text)}</span><span></span></div></div>`;
  chatStreamEl().appendChild(art);
  scrollStream();
}

function addAnswer(text, meta = {}, failed = false) {
  ensureStreamReady();
  evtRun().logStep = null;
  const art = document.createElement("article");
  art.className = "answer" + (failed ? " failed" : "");
  art.innerHTML =
    "<div class=\"answer-body markdown-body\">" + renderMarkdown(text) + "</div>";
  chatStreamEl().appendChild(art);
  scrollStream();
}

function handleEvent(event, tab) {
  /* C6: tab 参数标记事件归属——后台 tab 的事件只渲染进它自己的 streamEl
     （_streamOverride 已重定向），vitals DOM 仅在事件属于激活 tab 时更新。 */
  const t = event.type;
  const active = !tab || isActiveTab(tab);
  if (event.job && active) setVitalsStatus(event.job.status);
  if (active && event.payload && event.payload.tokens != null) {
    $("#vitalTokens").textContent = formatTokens(event.payload.tokens);
    recordTokenSample(event.payload.tokens);  // C5: 运行中 token 曲线
  }
  switch (t) {
    case "model_output": return addThinkStep(event);
    case "tool_call": return addActStep(event);
    case "tool_result": return resolveActStep(event);
    case "builtin_tool": return addBuiltinToolStep(event);
    case "ask_user": return addAskUserCard(event);
    case "control_tool": {
      const p = event.payload || {};
      return addSystemLine(`${toolName(p)}：${firstLine(p.result_content || "") || "已完成"}`);
    }
    case "subagent_start": {
      // Task 工具派生: 子代理开始执行 (并行派生时多个 start 可能交错)
      const p = event.payload || {};
      return addSystemLine(`⎇ [${p.role || "subagent"}] ${firstLine(p.task || "")}`);
    }
    case "subagent_finish": {
      const p = event.payload || {};
      const ok = p.success !== false;
      const secs = typeof p.duration_seconds === "number" ? p.duration_seconds.toFixed(1) : "0.0";
      return addSystemLine(`${ok ? "✓" : "✗"} [${p.role || "subagent"}] ${secs}s — ${firstLine(p.summary || "")}`);
    }
    case "agent_error": return addErrorLine(String(event.payload?.message || "Agent 运行错误"));
    case "llm_error": return addErrorLine(`模型调用失败：${event.payload?.error || "未知错误"}`);
    case "console": return addLogLine(event);
    case "session_loaded": return addSystemLine("会话上下文已接入本次运行");
    default: return; // job_created / status / benchmark_* / completed / failed handled elsewhere
  }
}

function renderEmptyState() {
  chatStreamEl().innerHTML = `
    <div class="empty-state">
      <h2>开始一次编码协作</h2>
      <p>输入你想完成的开发任务，WhaleCode 会实时展示思考过程、工具调用和最终结果。</p>
      <div class="example-row">
        ${EXAMPLES.map((e) => `<button class="example-chip" type="button" data-example="${escapeHtml(e)}">${escapeHtml(e)}</button>`).join("")}
      </div>
    </div>`;
}

function resetAgentConversation() {
  const tab = _tabs.active;
  if (tab) {
    if (tab.eventSource) { tab.eventSource.close(); tab.eventSource = null; }
    if (tab.run.timer) clearInterval(tab.run.timer);
    tab.run = makeRunState();
    tab.vitals = { status: "idle", steps: "0", elapsedText: "—", tokensText: "—" };
  }
  state.selectedSession = null;
  state.selectedSessionTitle = "新会话";
  // The whole stream is being discarded, so the stored full-text blobs
  // backing its 复制/展开 buttons can be released too.
  state.blobs.clear();
  state.blobSeq = 0;
  tokenSpark.samples = [];
  tokenSpark.last = 0;
  drawTokenSpark();
  stopRunTimer();
  setVitalsStatus("idle");
  const cancel = $("#cancelAgentButton");
  if (cancel) { cancel.hidden = true; cancel.disabled = false; }
  $("#vitalStep").textContent = "0";
  $("#vitalElapsed").textContent = "—";
  $("#vitalTokens").textContent = "—";
  renderEmptyState();
  renderTabs();
}

/* ------------------------------ status -------------------------------- */
const BASE_TITLE = document.title;
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
}
/* ----------------------------- sessions ------------------------------- */
async function renderSessionHistory(history) {
  const stream = chatStreamEl();
  if (!stream) return;
  stream.innerHTML = "";
  for (const msg of history || []) {
    const role = msg && msg.role;
    const content = String(msg && msg.content != null ? msg.content : "");
    if (!content) continue;
    if (role === "user") {
      const turn = document.createElement("article");
      turn.className = "turn-user";
      turn.innerHTML = escapeHtml(content).replace(/\n/g, "<br>");
      stream.appendChild(turn);
    } else if (role === "assistant") {
      const art = document.createElement("article");
      art.className = "answer";
      art.innerHTML = `<div class="answer-body markdown-body">${renderMarkdown(content)}</div>`;
      stream.appendChild(art);
    }
    // tool 消息是过程噪音，历史回顾时跳过
  }
  evtRun().logStep = null;
  const s = $("#traceStream");
  if (s) s.scrollTop = s.scrollHeight;
}
async function refreshSessions() {
  const data = await api("/api/sessions");
  const sessions = data.sessions || [];
  $("#sessionList").innerHTML = sessions.length ? sessions.map((s) => `
    <article class="session-item ${state.selectedSession === s.filepath ? "active" : ""}"
             data-filepath="${escapeHtml(s.filepath)}" data-filename="${escapeHtml(s.filename)}"
             data-title="${escapeHtml(s.title || s.filename)}">
      <div>
        <strong>${escapeHtml(s.title || s.filename)}</strong>
        <small>${escapeHtml(s.display_time || (s.saved_at || s.created_at || "").replace("T", " ").slice(0, 16) || "—")}</small>
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
  if (!state.selectedDatasets.size) ["lcb6"].forEach((id) => state.selectedDatasets.add(id));
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
  if (typeof s.pass_rate === "number") {
    const rate = s.pass_rate > 1 ? s.pass_rate / 100 : s.pass_rate;
    return { passed: s.passed, total: s.total, rate };
  }
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
  const safeRate = Math.max(0, Math.min(1, Number(pr.rate) || 0));
  const pct = Math.round(safeRate * 100);
  const cls = pr.rate === 0 ? "zero" : pr.rate < 0.5 ? "low" : "";
  const count = Number.isFinite(pr.passed) && Number.isFinite(pr.total)
    ? `${pr.passed}/${pr.total} 通过`
    : "通过率";
  return `<div class="scoreboard ${cls}">
      <div class="sb-top"><span class="sb-rate">${pct}%</span>
        <span class="sb-count">${count}</span></div>
      <div class="meter"><span style="width:${pct}%"></span></div>
    </div>`;
}

async function refreshBenchmarkHistory() {
  const data = await api("/api/benchmarks/history");
  state.benchmarkHistory = data.history || [];
  renderBenchmarkHistory();
  renderBenchChart();  // C7: 通过率柱状图
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
  const normalized = typeof v === "string" ? v.trim().toLowerCase() : v;
  if (normalized === true || normalized === "passed" || normalized === "pass" || normalized === "ok" || normalized === "success" || normalized === "true") return "passed";
  if (normalized === false || normalized === "failed" || normalized === "fail" || normalized === "error" || normalized === "false") return "failed";
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
    .filter(([, v]) => v != null && typeof v !== "object")
    .filter(([k]) => !["passed", "total", "pass_rate", "failed"].includes(k))
    .slice(0, 12);

  const casesHtml = records.length ? records.map((r, i) => {
    const s = caseStatus(r);
    const taskId = caseTitle(r, i);
    const benchmark = r.benchmark || summary.benchmark || detail.benchmark || datasetLabel(detail);
    const trajectoryButton = r.trajectory_available ? "<button type=\"button\" class=\"trajectory-button\" data-trajectory-task=\""
      + escapeHtml(taskId) + "\" data-trajectory-benchmark=\"" + escapeHtml(benchmark || "") + "\">查看 trajectory</button>" : "";
    return `<details class="case-row ${s}"${s === "failed" ? " open" : ""}>
        <summary>
          <span class="case-status">${s === "passed" ? "正确" : s === "failed" ? "错误" : "未知"}</span>
          <span class="case-title">${escapeHtml(taskId)}</span>
          <span class="case-meta">${escapeHtml(caseMeta(r))}</span>
        </summary>
        <div class="case-actions">${trajectoryButton}</div>
        ${longTextBlock(caseBody(r), { label: "case JSON", language: "json", head: 36, tail: 20 })}
        <div class="trajectory-host"></div>
      </details>`;
  }).join("") : `<div class="empty-block">无结构化用例，已在下方保留原始轨迹预览。</div>`;

  const summaryRate = typeof summary.pass_rate === "number"
    ? (summary.pass_rate > 1 ? summary.pass_rate / 100 : summary.pass_rate)
    : null;
  const rateTile = stats.rate != null
    ? `${Math.round(stats.rate * 100)}%` : (summaryRate != null ? `${Math.round(summaryRate * 100)}%` : "—");

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

function boolLabel(value) {
  if (value === true) return "通过";
  if (value === false) return "未通过";
  if (value == null || value === "") return "—";
  return String(value);
}
function trajectoryEvents(trajectory) {
  const agent = trajectory.agent || {};
  if (Array.isArray(trajectory.events)) return trajectory.events;
  if (Array.isArray(agent.events)) return agent.events;
  return [];
}
function trajectoryHistory(trajectory) {
  const agent = trajectory.agent || {};
  if (Array.isArray(trajectory.history)) return trajectory.history;
  if (Array.isArray(agent.history)) return agent.history;
  return [];
}
function renderTrajectoryEvent(item, idx) {
  const type = item?.type || item?.event || item?.role || "event";
  const title = item?.name || item?.tool || item?.tool_name || item?.role || type;
  const payload = item?.payload ?? item?.content ?? item?.message ?? item;
  const text = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  const ok = toolResultStatus(item) === "success";
  return "<article class=\"trajectory-event " + (ok ? "" : "failed") + "\">"
    + "<div class=\"trajectory-dot\">" + String(idx + 1).padStart(2, "0") + "</div>"
    + "<div class=\"trajectory-card\"><div class=\"trajectory-event-head\">"
    + "<span class=\"label\">" + escapeHtml(type) + "</span><strong>" + escapeHtml(title) + "</strong></div>"
    + longTextBlock(text, { label: "event payload", language: "json", head: 26, tail: 14 })
    + "</div></article>";
}
function renderTrajectoryHistoryItem(item, idx) {
  const role = item?.role || item?.type || "message";
  const content = item?.content || item?.text || item?.message || JSON.stringify(item, null, 2);
  return "<article class=\"trajectory-event history\"><div class=\"trajectory-dot\">H" + (idx + 1) + "</div>"
    + "<div class=\"trajectory-card\"><div class=\"trajectory-event-head\"><span class=\"label\">"
    + escapeHtml(role) + "</span><strong>History</strong></div>"
    + longTextBlock(content, { label: "message", head: 24, tail: 12 }) + "</div></article>";
}
function renderBenchmarkTrajectory(data) {
  const trajectory = data.trajectory || data || {};
  const task = trajectory.task || {};
  const result = trajectory.result || {};
  const workspace = trajectory.workspace || {};
  const events = trajectoryEvents(trajectory);
  const history = trajectoryHistory(trajectory);
  const title = task.title || task.question_title || task.task_id || trajectory.task_id || "Trajectory";
  const prompt = task.prompt || task.question || task.question_content || task.description || "";
  const elapsed = result.elapsed_s ?? result.duration_seconds ?? result.elapsed;
  const tests = result.test_count ?? result.num_tests ?? result.tests;
  const error = result.error || result.stderr || result.exception;
  const timeline = events.length ? events.slice(0, 120).map(renderTrajectoryEvent).join("")
    : history.slice(0, 80).map(renderTrajectoryHistoryItem).join("");
  return "<div class=\"trajectory-panel\">"
    + "<div class=\"trajectory-title\"><span class=\"label\">LiveCodeBench Trajectory</span><strong>" + escapeHtml(title) + "</strong></div>"
    + "<div class=\"trajectory-grid\">"
    + "<div><span class=\"label\">Task</span><strong>" + escapeHtml(trajectory.task_id || task.task_id || title) + "</strong></div>"
    + "<div><span class=\"label\">Result</span><strong>" + escapeHtml(boolLabel(result.passed ?? result.success ?? result.ok)) + "</strong></div>"
    + "<div><span class=\"label\">Tests</span><strong>" + escapeHtml(tests ?? "—") + "</strong></div>"
    + "<div><span class=\"label\">Duration</span><strong>" + escapeHtml(typeof elapsed === "number" ? formatDuration(elapsed) : (elapsed || "—")) + "</strong></div>"
    + "</div>"
    + (prompt ? "<section class=\"trajectory-task\"><span class=\"label\">Problem</span>" + longTextBlock(prompt, { label: "prompt", head: 32, tail: 18 }) + "</section>" : "")
    + (error ? "<div class=\"error-card\"><span class=\"label\">错误摘要</span><strong>" + escapeHtml(firstLine(error)) + "</strong><p>建议检查生成代码、测试输入、超时限制与执行环境。</p></div>" : "")
    + "<section class=\"trajectory-timeline\"><div class=\"trajectory-title compact\"><span class=\"label\">Timeline</span><strong>"
    + escapeHtml(events.length ? events.length + " events" : history.length + " history messages") + "</strong></div>"
    + (timeline || "<div class=\"empty-block\">暂无可展示的 trajectory 事件。</div>") + "</section>"
    + "<details class=\"more-summary\"><summary>workspace / raw</summary>"
    + longTextBlock(JSON.stringify({ workspace, result, extra: trajectory.extra || null }, null, 2), { label: "trajectory raw", language: "json", head: 40, tail: 20 })
    + "</details></div>";
}
async function toggleBenchmarkTrajectory(button) {
  const row = button.closest(".case-row");
  const host = row?.querySelector(".trajectory-host");
  if (!host) return;
  if (host.dataset.open === "1") {
    host.dataset.open = "0";
    host.innerHTML = "";
    button.textContent = "查看 trajectory";
    return;
  }
  const taskId = button.dataset.trajectoryTask;
  const benchmark = button.dataset.trajectoryBenchmark || "";
  const key = benchmark + "::" + taskId;
  host.dataset.open = "1";
  button.textContent = "收起 trajectory";
  host.innerHTML = "<div class=\"empty-block\">正在加载 trajectory…</div>";
  try {
    let data = state.benchmarkTrajectories.get(key);
    if (!data) {
      data = await api("/api/benchmarks/trajectory?task_id=" + encodeURIComponent(taskId) + "&benchmark=" + encodeURIComponent(benchmark));
      state.benchmarkTrajectories.set(key, data);
    }
    host.innerHTML = renderBenchmarkTrajectory(data);
  } catch (error) {
    host.innerHTML = "<div class=\"empty-block error-text\">trajectory 加载失败：" + escapeHtml(error.message) + "</div>";
  }
}

/* ------------------------------- SSE ---------------------------------- */
function subscribeJob(job, handlers = {}, tab = null) {
  const source = new EventSource(`/api/jobs/${encodeURIComponent(job.id)}/events`);
  const onAny = (event) => {
    const data = JSON.parse(event.data);
    // C6: 事件归属重定向——后台 tab 的事件渲染进自己的 streamEl，
    // 且渲染函数读写的 run 状态 / vitals / 滚动行为都归属该 tab。
    const prevOverride = _streamOverride;
    const prevTab = _eventTab;
    _streamOverride = tab ? tab.streamEl : null;
    _eventTab = tab;
    try {
      handlers.any?.(data, tab);
      if (handlers[data.type]) handlers[data.type](data);
    } finally {
      _streamOverride = prevOverride;
      _eventTab = prevTab;
    }
    if (["completed", "failed", "cancelled"].includes(data.type)) source.close();
  };
  ["job_created", "status", "console", "model_output", "tool_call", "tool_result",
   "builtin_tool", "control_tool", "ask_user", "agent_error", "llm_error",
   "subagent_start", "subagent_finish", "session_loaded", "benchmark_started",
   "benchmark_output", "completed", "failed", "cancelled"]
    .forEach((name) => source.addEventListener(name, onAny));
  source.onerror = () => { if (["completed", "failed", "cancelled"].includes(job.status)) source.close(); };
  return source;
}

/* ---------------------------- run agent ------------------------------- */
async function runAgent(event) {
  event.preventDefault();
  const prompt = $("#promptInput").value.trim();
  if (!prompt) return;
  // C6: 当前 tab 正在运行 → 自动开新 tab 并行执行
  if (state.run.active) {
    createTab(prompt.slice(0, 12));
    addSystemLine("当前会话运行中，已在新标签页启动本任务。");
  }

  ensureStreamReady();
  const tab = _tabs.active;
  tab.title = prompt.slice(0, 12) || tab.title;
  const turn = document.createElement("article");
  turn.className = "turn-user";
  turn.innerHTML = escapeHtml(prompt).replace(/\n/g, "<br>");
  chatStreamEl().appendChild(turn);

  $("#promptInput").value = "";
  autoGrow();
  state.run.active = true;
  state.run.jobId = null;
  state.run.steps = 0;
  state.run.nodeIdx = 0;
  state.run.pendingTools = new Map();
  state.run.pendingOrder = [];
  state.run.logStep = null;
  tokenSpark.samples = [];
  tokenSpark.last = 0;
  drawTokenSpark();
  $("#vitalStep").textContent = "0";
  $("#vitalTokens").textContent = "—";
  setVitalsStatus("queued");
  startRunTimer();
  setBusy($("#runAgentButton"), true, "运行中");
  const cancelButton = $("#cancelAgentButton");
  // Disabled until the job id is known — a cancel click before that would no-op.
  if (cancelButton) { cancelButton.hidden = false; cancelButton.disabled = true; }
  renderTabs();

  const finish = (status) => {
    // C6: 计时器与 vitals 均归属各自 tab——后台结束只停自己的 timer。
    stopRunTimer(tab);
    tab.run.active = false;
    tab.run.jobId = null;
    tab.eventSource?.close();
    tab.eventSource = null;
    if (isActiveTab(tab)) {
      setVitalsStatus(status);
      setBusy($("#runAgentButton"), false);
      if (cancelButton) { cancelButton.hidden = true; cancelButton.disabled = false; }
    } else {
      tab.vitals.status = { completed: "已完成", failed: "失败", cancelled: "已取消" }[status] || tab.vitals.status;
    }
    renderTabs();
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
    tab.run.jobId = job.id;
    if (cancelButton && isActiveTab(tab)) cancelButton.disabled = false;
    // C6: subscribeJob 的 onAny 已把事件归属（DOM/状态/vitals 守卫）重定向到
    // 本 tab，completed/failed/cancelled 内无需再手动切换 override。
    tab.eventSource = subscribeJob(job, {
      any: (data, ownerTab) => handleEvent(data, ownerTab),
      completed(data) {
        const r = data.payload || {};
        addAnswer(r.answer || "任务已完成，但没有返回文本。", r);
        renderRunSummary(r);  // C5: 会话结束汇总卡片
        if (isActiveTab(tab)) {
          $("#vitalElapsed").textContent = formatDuration(r.duration_seconds);
          if (r.tokens != null) $("#vitalTokens").textContent = formatTokens(r.tokens);
          if (r.steps != null) $("#vitalStep").textContent = String(r.steps);
        } else {
          tab.vitals.elapsedText = formatDuration(r.duration_seconds);
          if (r.tokens != null) tab.vitals.tokensText = formatTokens(r.tokens);
          if (r.steps != null) tab.vitals.steps = String(r.steps);
        }
        finish("completed");
        refreshSessions();
      },
      failed(data) {
        addAnswer(data.payload?.error || "unknown error", data.payload || {}, true);
        finish("failed");
      },
      cancelled(data) {
        addSystemLine(data.payload?.reason || "已取消当前 Agent 运行");
        finish("cancelled");
      },
    }, tab);
  } catch (error) {
    addAnswer("提交失败：" + error.message, {}, true);
    finish("failed");
  }
}

async function cancelAgentRun() {
  if (!state.run.active || !state.run.jobId) return;
  const button = $("#cancelAgentButton");
  if (button) button.disabled = true;
  try {
    await api("/api/jobs/" + encodeURIComponent(state.run.jobId) + "/cancel", { method: "POST" });
    addSystemLine("已请求取消当前 Agent 运行，正在等待后端终止…");
  } catch (error) {
    addSystemLine("取消失败：" + error.message);
    if (button) button.disabled = false;
  }
}

/* --------------------------- run benchmark ---------------------------- */
function appendBenchLog(line) {
  const log = $("#benchLog");
  const empty = log.querySelector(".log-empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = "log-line";
  el.textContent = line;
  log.appendChild(el);
  // Cap DOM growth for long benchmark runs (thousands of output lines).
  while (log.children.length > 2000) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}
async function runBenchmark(event) {
  event.preventDefault();
  const datasets = [...state.selectedDatasets];
  if (!datasets.length || state.activeBenchmarkJobId) return;
  $("#benchLog").innerHTML = "";
  setBusy($("#runBenchButton"), true, "运行中");
  const cancelButton = $("#cancelBenchButton");
  if (cancelButton) { cancelButton.hidden = false; cancelButton.disabled = false; }
  const finish = () => {
    setBusy($("#runBenchButton"), false);
    if (cancelButton) { cancelButton.hidden = true; cancelButton.disabled = false; }
    state.activeBenchmarkSource?.close();
    state.activeBenchmarkSource = null;
    state.activeBenchmarkJobId = null;
  };
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
    state.activeBenchmarkJobId = job.id;
    state.activeBenchmarkSource = subscribeJob(job, {
      benchmark_started(data) { appendBenchLog(`=== ${data.payload.dataset?.name || "Benchmark"} 开始 ===`); },
      benchmark_output(data) { appendBenchLog(data.payload.line || ""); },
      completed() { appendBenchLog("评测完成"); finish(); refreshBenchmarkHistory(); },
      failed(data) { appendBenchLog(`错误：${data.payload?.error || "unknown"}`); finish(); },
      cancelled(data) { appendBenchLog(data.payload?.reason || "评测已取消"); finish(); refreshBenchmarkHistory(); },
    });
  } catch (error) {
    appendBenchLog(`提交失败：${error.message}`);
    finish();
  }
}

async function cancelBenchmarkRun() {
  if (!state.activeBenchmarkJobId) return;
  const button = $("#cancelBenchButton");
  if (button) button.disabled = true;
  try {
    await api("/api/jobs/" + encodeURIComponent(state.activeBenchmarkJobId) + "/cancel", { method: "POST" });
    appendBenchLog("已请求取消评测任务，正在等待后端终止…");
  } catch (error) {
    appendBenchLog("取消失败：" + error.message);
    if (button) button.disabled = false;
  }
}

/* ----------------------------- textarea ------------------------------- */
function autoGrow() {
  const ta = $("#promptInput");
  ta.style.height = "auto";
  ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
}

/* --------------------- C1: 工作区文件浏览器 ---------------------------- */
const workspace = {
  root: "",
  openDirs: new Set(),
  activeFile: null,
  loadedDirs: new Map(),  // path -> entries
};

function wpVisible() { return !$("#workspacePanel").hidden; }
function setWorkspacePanel(open) {
  $("#workspacePanel").hidden = !open;
  if (open) {
    if (!workspace.root && state.projectRoot) workspace.root = state.projectRoot;
    $("#wpRootPath").textContent = workspace.root ? workspace.root.split(/[\\/]/).pop() || workspace.root : "";
    if (!workspace.loadedDirs.size) loadWorkspaceDir(".");
  }
}
async function loadWorkspaceDir(path) {
  const host = document.querySelector(`.wp-tree [data-dir="${CSS.escape(path)}"] .wp-children`) || $("#wpTree");
  if (host) host.innerHTML = `<div class="wp-loading">加载中…</div>`;
  try {
    const result = await api(`/api/workspace/tree?path=${encodeURIComponent(path)}&limit=500&root=${encodeURIComponent(workspace.root || "")}`);
    if (result.status === "error") throw new Error(result.error || "加载失败");
    workspace.loadedDirs.set(path, result.data?.entries || []);
    renderWorkspaceTree();
  } catch (err) {
    if (host) host.innerHTML = `<div class="wp-error">${escapeHtml(err.message)}</div>`;
  }
}
function renderWorkspaceTree() {
  const rootEntries = workspace.loadedDirs.get(".") || [];
  const tree = $("#wpTree");
  const buildDir = (entry, depth) => {
    const path = entry.path;
    const isOpen = workspace.openDirs.has(path);
    const entries = workspace.loadedDirs.get(path) || [];
    return `<div class="wp-dir" data-dir="${escapeHtml(path)}" style="--depth:${depth}">
      <button type="button" class="wp-row dir ${isOpen ? "open" : ""}" data-toggle-dir="${escapeHtml(path)}">
        <svg class="wp-caret" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5z"/></svg>
        <span class="wp-name">${escapeHtml(entry.name)}</span>
      </button>
      <div class="wp-children">${isOpen ? entries.map((e) => e.type === "directory" ? buildDir(e, depth + 1) : buildFile(e, depth + 1)).join("") : ""}</div>
    </div>`;
  };
  const buildFile = (entry, depth) => {
    const active = workspace.activeFile === entry.path ? " active" : "";
    return `<button type="button" class="wp-row file${active}" data-open-file="${escapeHtml(entry.path)}" style="--depth:${depth}">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 12h6m-6 4h6M9 8h6M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/></svg>
      <span class="wp-name">${escapeHtml(entry.name)}</span>
      <span class="wp-size">${escapeHtml(entry.size)}</span>
    </button>`;
  };
  tree.innerHTML = rootEntries.map((e) => (e.type === "directory" ? buildDir(e, 0) : buildFile(e, 0))).join("")
    || `<div class="wp-loading">空目录</div>`;
}
async function openWorkspaceFile(path, opts = {}) {
  setWorkspacePanel(true);
  const viewer = $("#wpViewer");
  viewer.hidden = false;
  viewer.innerHTML = `<div class="wp-loading">读取 ${escapeHtml(path)} …</div>`;
  workspace.activeFile = path;
  renderWorkspaceTree();
  try {
    const result = await api(`/api/workspace/file?path=${encodeURIComponent(path)}&limit=2000&root=${encodeURIComponent(workspace.root || "")}`);
    if (result.status === "error") throw new Error(result.error || "读取失败");
    const data = result.data || {};
    const content = data.content ?? "";
    const total = data.total_lines ?? content.split("\n").length;
    const lines = content.split("\n");
    const numbered = lines.map((l, i) => `<div class="wp-line" data-line="${i + 1}"><span class="wp-ln">${i + 1}</span><span class="wp-code">${escapeHtml(l) || " "}</span></div>`).join("");
    viewer.innerHTML = `
      <div class="wp-viewer-head">
        <strong title="${escapeHtml(path)}">${escapeHtml(basename(path))}</strong>
        <span class="wp-viewer-meta">${total} 行${data.truncated ? " · 已截断" : ""}${data.encoding ? " · " + escapeHtml(data.encoding) : ""}</span>
        <button type="button" class="mini-button" data-copy-file="${escapeHtml(path)}">复制</button>
        <button type="button" class="mini-button" id="wpCloseViewer" title="关闭查看器">关闭</button>
      </div>
      <pre class="wp-content">${numbered}</pre>`;
    if (opts.line && opts.line > 0) {
      const target = viewer.querySelector(`.wp-line[data-line="${opts.line}"]`);
      if (target) {
        target.scrollIntoView({ block: "center" });
        target.classList.add("flash");
      }
    }
  } catch (err) {
    viewer.innerHTML = `<div class="wp-error">${escapeHtml(err.message)}</div>`;
  }
}
/* 工具卡片上的 data-open-file 委托入口：路径可能带 :line 后缀 */
function handleOpenFileRequest(pathWithLine) {
  const m = String(pathWithLine).match(/^(.+?)(?::(\d+))?$/);
  if (!m) return;
  openWorkspaceFile(m[1], { line: m[2] ? Number(m[2]) : undefined });
}

/* --------------------- C5: Token sparkline 与汇总 ---------------------- */
/* sparkline 为激活视图的全局简化实现：仅记录前台事件（后台 tab 的事件不采样，
   避免多会话曲线混叠）；切回历史 tab 时曲线不恢复（快照仅恢复数字 vitals）。 */
const tokenSpark = {
  samples: [],       // {t: elapsedMs, v: tokens}
  last: 0,
};
function recordTokenSample(tokens) {
  const v = Number(tokens);
  if (!Number.isFinite(v) || v <= 0 || v === tokenSpark.last) return;
  tokenSpark.last = v;
  tokenSpark.samples.push({ t: performance.now(), v });
  drawTokenSpark();
}
function drawTokenSpark() {
  const svg = $("#tokenSpark");
  if (!svg) return;
  const pts = tokenSpark.samples;
  if (pts.length < 2) { svg.innerHTML = ""; return; }
  const maxV = Math.max(...pts.map((p) => p.v), 1);
  const path = pts.map((p, i) => {
    const x = (i / (pts.length - 1)) * 100;
    const y = 22 - (p.v / maxV) * 20;
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  svg.innerHTML = `<path d="${path}" fill="none" stroke="currentColor" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`;
}
function renderRunSummary(data) {
  /* C5: 会话结束汇总卡片（tokens/步数/耗时/工具调用 Top）。
     工具统计来自本会话 stream 内的 act 步骤卡片（事件归属 tab 的 DOM）。 */
  const toolCounts = {};
  chatStreamEl().querySelectorAll(".step.act").forEach((el) => {
    const kicker = el.querySelector(".step-kicker");
    if (kicker) {
      const key = kicker.textContent.trim();
      toolCounts[key] = (toolCounts[key] || 0) + 1;
    }
  });
  const topTools = Object.entries(toolCounts).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const tokens = data.tokens != null ? formatTokens(data.tokens) : "—";
  const steps = data.steps ?? evtRun().steps;
  const dur = data.duration_seconds != null ? formatDuration(data.duration_seconds) : $("#vitalElapsed").textContent;
  const art = document.createElement("article");
  art.className = "run-summary";
  art.innerHTML = `
    <div class="rs-grid">
      <div><span class="label">Tokens</span><strong>${tokens}</strong></div>
      <div><span class="label">步数</span><strong>${steps}</strong></div>
      <div><span class="label">用时</span><strong>${dur}</strong></div>
      ${topTools.length ? `<div><span class="label">工具 Top</span><strong>${topTools.map(([n, c]) => `${escapeHtml(n)}×${c}`).join(" · ")}</strong></div>` : ""}
    </div>`;
  chatStreamEl().appendChild(art);
  scrollStream();
}

/* --------------------- C7: Benchmark 通过率图表 ------------------------ */
function renderBenchChart() {
  const host = $("#benchChart");
  const groups = new Map();  // dataset -> [{rate, time, model}]
  for (const item of state.benchmarkHistory) {
    const pr = passRateOf(item);
    if (!pr) continue;
    const name = datasetLabel(item);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push({ rate: pr.rate, time: item.modified_at || "", model: modelLabel(item) });
  }
  if (groups.size < 1) { host.hidden = true; return; }
  host.hidden = false;
  const maxPerGroup = Math.max(...[...groups.values()].map((v) => v.length));
  const cellW = 26, barW = 16, gapX = 10, rowH = 44, headH = 22, axisW = 108;
  const width = axisW + groups.size * (cellW * Math.max(maxPerGroup, 1) + gapX);
  const height = headH + groups.size * rowH + 8;
  let svg = "";
  let ci = 0;
  for (const [name, runs] of groups) {
    const y = headH + ci * rowH;
    svg += `<text x="${axisW - 10}" y="${y + 14}" text-anchor="end" class="chart-label">${escapeHtml(name.length > 16 ? name.slice(0, 15) + "…" : name)}</text>`;
    runs.forEach((run, ri) => {
      const pct = Math.round(Math.max(0, Math.min(1, run.rate)) * 100);
      const barH = Math.max(3, (run.rate * 28));
      const x = axisW + ri * cellW;
      const color = run.rate >= 0.8 ? "var(--pass)" : run.rate >= 0.5 ? "var(--aqua)" : run.rate >= 0.25 ? "var(--warn)" : "var(--fail)";
      svg += `<rect x="${x}" y="${y + 28 - barH}" width="${barW}" height="${barH}" rx="3" fill="${color}" opacity="0.85"><title>${escapeHtml(name)} · ${pct}% · ${escapeHtml(run.model || "")} · ${escapeHtml(run.time)}</title></rect>`;
      svg += `<text x="${x + barW / 2}" y="${y + 40}" text-anchor="middle" class="chart-value">${pct}</text>`;
    });
    ci += 1;
  }
  host.innerHTML = `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" width="${Math.min(width, 900)}" height="${height}" role="img" aria-label="各数据集通过率柱状图">${svg}</svg></div>`;
}

/* --------------------- C4: 主题切换 ------------------------------------ */
const THEMES = ["dark", "light"];
function applyTheme(theme) {
  document.body.dataset.theme = theme;
  try { localStorage.setItem("whalecode-theme", theme); } catch { /* 隐私模式 */ }
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("whalecode-theme"); } catch { /* ignore */ }
  if (!THEMES.includes(saved)) {
    saved = window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  applyTheme(saved);
}
function toggleTheme() {
  applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
}

/* --------------------- C9: 命令面板 ------------------------------------ */
const cmdk = {
  commands: [],
  visible: false,
};
function buildCommandRegistry() {
  cmdk.commands = [
    { id: "new-session", label: "新会话 / New session", hint: "Cmd+N", run: () => $("#newSessionButton").click() },
    { id: "view-agent", label: "切换到智能体视图 / Switch to Agent view", run: () => setView("agent") },
    { id: "view-bench", label: "切换到评测视图 / Switch to Benchmarks view", run: () => setView("benchmarks") },
    { id: "workspace", label: "打开工作区面板 / Toggle workspace panel", run: () => setWorkspacePanel(wpVisible() ? false : true) },
    { id: "theme", label: "切换浅色/深色主题 / Toggle theme", run: toggleTheme },
    { id: "trace", label: "打开最近 Trace 报告 / Open latest trace report", run: openLatestTrace },
    { id: "refresh-sessions", label: "刷新会话列表 / Refresh sessions", run: () => refreshSessions() },
    { id: "run-bench", label: "运行评测（跳转）/ Run benchmark", run: () => { setView("benchmarks"); $("#benchModel")?.focus(); } },
  ];
}
function renderCmdkList(filter) {
  const q = (filter || "").trim().toLowerCase();
  const items = cmdk.commands.filter((c) => !q || c.label.toLowerCase().includes(q));
  $("#cmdkList").innerHTML = items.length ? items.map((c) => `
    <button type="button" class="cmdk-item" id="cmdk-option-${escapeHtml(c.id)}" role="option"
            aria-selected="false" data-cmdk-id="${escapeHtml(c.id)}">
      <span>${escapeHtml(c.label)}</span>${c.hint ? `<kbd>${escapeHtml(c.hint)}</kbd>` : ""}
    </button>`).join("") : `<div class="cmdk-empty">无匹配命令</div>`;
  $("#cmdkInput")?.removeAttribute("aria-activedescendant");
  return items;
}
function openCmdk() {
  cmdk.visible = true;
  $("#cmdkOverlay").hidden = false;
  $("#cmdkInput").value = "";
  renderCmdkList("");
  $("#cmdkInput").focus();
}
function closeCmdk() {
  cmdk.visible = false;
  $("#cmdkOverlay").hidden = true;
}
function runCmdkCommand(id) {
  const command = cmdk.commands.find((c) => c.id === id);
  closeCmdk();
  if (command) command.run();
}
async function openLatestTrace() {
  try {
    const data = await api("/api/traces");
    const traces = data.traces || [];
    if (!traces.length) { addSystemLine("暂无 Trace 报告（运行一次任务后生成）。"); return; }
    window.open(traces[0].url, "_blank", "noopener");
  } catch (err) {
    addSystemLine(`Trace 列表加载失败：${err.message}`);
  }
}

/* ------------------------------ events -------------------------------- */
function bindEvents() {
  $$(".nav-item").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#agentForm").addEventListener("submit", runAgent);
  $("#cancelAgentButton")?.addEventListener("click", cancelAgentRun);
  $("#promptInput").addEventListener("input", autoGrow);
  $("#promptInput").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    // Enter 或 Cmd/Ctrl+Enter 发送；纯 Shift+Enter 换行
    const send = !e.shiftKey || e.metaKey || e.ctrlKey;
    if (send) { e.preventDefault(); $("#agentForm").requestSubmit(); }
  });
  $("#benchForm").addEventListener("submit", runBenchmark);
  $("#cancelBenchButton")?.addEventListener("click", cancelBenchmarkRun);
  $("#refreshSessions").addEventListener("click", refreshSessions);
  $("#newSessionButton").addEventListener("click", async () => {
    // C6: 当前 tab 有内容时开新 tab，否则原地重置
    const stream = chatStreamEl();
    if (_tabs.list.length > 1 || (stream && stream.querySelector(".step, .turn-user, .answer"))) {
      createTab();
    } else {
      resetAgentConversation();
    }
    await refreshSessions();
  });
  // C4/C1/C8: 新控件
  $("#themeToggle")?.addEventListener("click", toggleTheme);
  $("#traceReportButton")?.addEventListener("click", openLatestTrace);
  $("#workspaceToggle")?.addEventListener("click", () => setWorkspacePanel(wpVisible() ? false : true));
  $("#wpClose")?.addEventListener("click", () => setWorkspacePanel(false));
  $("#wpRefresh")?.addEventListener("click", () => {
    workspace.loadedDirs.clear();
    loadWorkspaceDir(".");
  });
  // C9: 命令面板（键盘导航高亮保持在 input 上，焦点不移入列表项）
  $("#cmdkInput")?.addEventListener("input", (e) => { renderCmdkList(e.target.value); });
  $("#cmdkInput")?.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); closeCmdk(); return; }
    const items = Array.from($("#cmdkList .cmdk-item"));
    if (!items.length && e.key !== "Enter") return;
    const currentIdx = Math.max(0, items.findIndex((el) => el.classList.contains("selected")));
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      items.forEach((el) => {
        el.classList.remove("selected");
        el.setAttribute("aria-selected", "false");
      });
      const nextIdx = e.key === "ArrowDown"
        ? (currentIdx + 1) % items.length
        : (currentIdx - 1 + items.length) % items.length;
      items[nextIdx].classList.add("selected");
      items[nextIdx].setAttribute("aria-selected", "true");
      $("#cmdkInput").setAttribute("aria-activedescendant", items[nextIdx].id);
      items[nextIdx].scrollIntoView({ block: "nearest" });
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      // Enter 执行高亮项；无高亮则执行第一项
      const target = items[currentIdx] || items[0];
      if (target) runCmdkCommand(target.dataset.cmdkId);
    }
  });
  document.addEventListener("keydown", (e) => {
    // C9: 全局快捷键
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      cmdk.visible ? closeCmdk() : openCmdk();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
      e.preventDefault();
      createTab();
      return;
    }
    if (e.key === "Escape") {
      if (cmdk.visible) { closeCmdk(); return; }
      const cancel = $("#cancelAgentButton");
      if (cancel && !cancel.hidden && !cancel.disabled) { cancelAgentRun(); return; }
      if (wpVisible()) { setWorkspacePanel(false); }
    }
  });

  document.addEventListener("click", async (e) => {
    const copy = e.target.closest("[data-copy-blob]");
    if (copy) {
      const text = state.blobs.get(copy.dataset.copyBlob) || "";
      try { await navigator.clipboard.writeText(text); copy.textContent = "已复制"; }
      catch { copy.textContent = "复制失败"; }
      setTimeout(() => { if (copy.isConnected) copy.textContent = "复制"; }, 1200);
      return;
    }
    // C1: 文件链接（工具卡片）→ 工作区面板打开
    const fileLink = e.target.closest("[data-open-file]");
    if (fileLink && !fileLink.dataset.toggleDir) {
      e.preventDefault();
      handleOpenFileRequest(fileLink.dataset.openFile);
      return;
    }
    // C1: 工作区面板文件树交互
    const dirToggle = e.target.closest("[data-toggle-dir]");
    if (dirToggle) {
      const dirPath = dirToggle.dataset.toggleDir;
      if (workspace.openDirs.has(dirPath)) workspace.openDirs.delete(dirPath);
      else {
        workspace.openDirs.add(dirPath);
        if (!workspace.loadedDirs.has(dirPath)) loadWorkspaceDir(dirPath);
      }
      renderWorkspaceTree();
      return;
    }
    const closeViewer = e.target.closest("#wpCloseViewer");
    if (closeViewer) {
      $("#wpViewer").hidden = true;
      $("#wpViewer").innerHTML = "";
      return;
    }
    const copyFile = e.target.closest("[data-copy-file]");
    if (copyFile) {
      // 查看器当前内容已在 DOM 中，从行元素拼接复制
      const lines = Array.from($("#wpViewer").querySelectorAll(".wp-code")).map((el) => el.textContent);
      try { await navigator.clipboard.writeText(lines.join("\n")); copyFile.textContent = "已复制"; }
      catch { copyFile.textContent = "复制失败"; }
      setTimeout(() => { if (copyFile.isConnected) copyFile.textContent = "复制"; }, 1200);
      return;
    }
    // C6: tab 切换/关闭
    const closeTabBtn = e.target.closest("[data-close-tab]");
    if (closeTabBtn) { closeTab(closeTabBtn.dataset.closeTab); return; }
    const switchTabBtn = e.target.closest("[data-switch-tab]");
    if (switchTabBtn) { switchTab(switchTabBtn.dataset.switchTab); return; }
    // C9: 命令面板项
    const cmdkItem = e.target.closest("[data-cmdk-id]");
    if (cmdkItem) { runCmdkCommand(cmdkItem.dataset.cmdkId); return; }
    // C9: 点击遮罩关闭命令面板
    if (cmdk.visible && e.target === $("#cmdkOverlay")) { closeCmdk(); return; }
    const expand = e.target.closest("[data-expand-blob]");
    if (expand) {
      const box = expand.closest(".long-output");
      const pre = box?.querySelector("pre");
      const full = state.blobs.get(expand.dataset.expandBlob) || "";
      if (pre) { pre.textContent = full; pre.classList.add("expanded"); }
      expand.remove();
    }
  });

  // 事件委托绑定在 traceStream 父容器上：tab 的 streamEl 会动态挂载/卸载
  $("#traceStream").addEventListener("click", (e) => {
    const chip = e.target.closest("[data-example]");
    if (chip) { $("#promptInput").value = chip.dataset.example; autoGrow(); $("#promptInput").focus(); }
  });

  $("#sessionList").addEventListener("click", async (e) => {
    const row = e.target.closest(".session-item");
    if (!row?.dataset.filepath) return;
    if (e.target.closest(".delete-session")) {
      if (!window.confirm(`删除会话「${row.dataset.title || row.dataset.filename}」？此操作不可恢复。`)) return;
      await api(`/api/sessions/${encodeURIComponent(row.dataset.filename)}`, { method: "DELETE" });
      if (state.selectedSession === row.dataset.filepath) { state.selectedSession = null; state.selectedSessionTitle = "新会话"; }
      await refreshSessions();
      return;
    }
    state.selectedSession = row.dataset.filepath;
    state.selectedSessionTitle = row.dataset.title || row.querySelector("strong")?.textContent || row.dataset.filename;
    addSystemLine(`已加载历史会话：${state.selectedSessionTitle}，可继续对话`);
    try {
      const detail = await api(`/api/sessions/${encodeURIComponent(row.dataset.filename)}`);
      if (detail && detail.session) {
        await renderSessionHistory(detail.session.history || []);
      } else {
        addSystemLine("该会话没有可展示的历史记录（可能只有工具调用过程）。");
      }
    } catch (err) {
      addSystemLine(`历史会话加载失败：${err.message || err}`);
    }
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
    const traj = e.target.closest("[data-trajectory-task]");
    if (traj) {
      e.preventDefault();
      e.stopPropagation();
      toggleBenchmarkTrajectory(traj);
      return;
    }
    const button = e.target.closest("[data-history-file]");
    if (button) toggleBenchmarkDetail(button);
  });
}

async function init() {
  initTheme();          // C4: 主题（系统偏好 + localStorage）
  buildCommandRegistry();  // C9: 命令面板
  createTab();          // C6: 首个会话 tab（含 streamEl 与空状态）
  bindEvents();
  await Promise.allSettled([refreshStatus(), refreshSessions(), refreshDatasets(), refreshBenchmarkHistory()]);
  // GPU 读数与服务状态随时间变化，低频轮询保持侧栏鲜活（15s，远轻于 SSE）。
  setInterval(() => { refreshStatus().catch(() => {}); }, 15000);
}

init().catch((error) => addSystemLine(`初始化失败：${error.message}`));
