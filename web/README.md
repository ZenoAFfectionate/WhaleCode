# WhaleCode Web Console / Web 控制台

A lightweight, zero-dependency web console for the WhaleCode agent framework — it drives the **real** agent infrastructure (not a static showcase), streaming live progress over SSE.

轻量级、零依赖的 WhaleCode 编程智能体 Web 控制台——驱动的 是**真实**智能体基础设施（非静态演示页），通过 SSE 实时流式展示执行进度。

---

## 🚀 Run / 启动

```bash
# From the cloned repository root / 在仓库根目录执行：
python3 web/server.py --host 127.0.0.1 --port 8765
```

Then open / 然后打开：

```text
http://127.0.0.1:8765
```

---

## 🖥️ UI Overview / 界面功能

The console ships two views / 控制台包含两个视图：

| Feature / 功能 | Description / 说明 |
| --- | --- |
| **Agent Console / Agent 控制台** | Start a coding-agent job against a workspace, watch Think/Act/Observe steps stream in via tool cards, answer the agent's interactive questions, and cancel a running job. 面向某个 workspace 启动编程智能体任务，实时查看 Think/Act/Observe 步骤（工具卡片渲染），回答智能体的交互式提问，取消运行中的任务。 |
| **Benchmark Panel / Benchmark 面板** | Browse datasets, launch benchmark scripts (`scripts/run_*.sh`), and inspect historical pass-rate records with per-task trajectory drill-down. 浏览数据集、启动评测脚本，查看历史通过率记录并下钻单个任务轨迹。 |
| **Sessions / 会话管理** | List / open / delete web-created persisted sessions. 列出、查看、删除 Web 创建的持久化会话。 |
| **vLLM Control / vLLM 管理** | Start / switch / stop / unload local vLLM processes from the model catalog. 启动、切换、停止、卸载本地 vLLM 进程。 |

**Interaction highlights / 交互要点**：

- SSE stream with `Last-Event-ID` resume — reload the page and the console reconnects without losing events. SSE 事件流支持 `Last-Event-ID` 断线续传，刷新页面不丢事件。
- **AskUser channel / 提问通道** — when the agent calls `AskUser`, a question card appears in the chat; answers are posted back through `POST /api/agent/answers` and the agent continues. 智能体调用 `AskUser` 时对话流中出现问题卡片，答案经 `POST /api/agent/answers` 回传，智能体继续执行。
- Long tool outputs are collapsed with copy buttons; diffs are colorized (+/- lines). 长工具输出自动折叠并可复制；diff 以 +/- 行着色。

---

## 🔧 Environment / 环境变量

The web server reuses the same environment variables as the CLI / Web 服务与 CLI 共用同一组环境变量：

- `LLM_MODEL_ID`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `CODE_AGENT_MAX_STEPS`

Optional vLLM process command overrides / 可选的 vLLM 启动命令覆盖：

- `WHALE_WEB_VLLM_COMMAND`
- `WHALE_WEB_VLLM_COMMAND_QWEN35_35B_FP8`
- `WHALE_WEB_VLLM_COMMAND_QWEN3_CODER_30B`
- `WHALE_WEB_VLLM_COMMAND_DEEPSEEK_CODER_LITE`

If no override is set, the server uses the default command embedded in `web/server.py` for the selected catalog model. 未设置覆盖时，服务端使用 `web/server.py` 内置的默认命令。

---

## 📡 API Reference / API 参考

### Status & Models / 状态与模型

| Method | Path | Description / 说明 |
| --- | --- | --- |
| `GET` | `/api/status` | Server status & project root. 服务状态与项目根路径。 |
| `GET` | `/api/models` | Model / vLLM / GPU status. 模型、vLLM 与 GPU 状态。 |
| `POST` | `/api/models/start` | Start or switch vLLM for a catalog model. 启动或切换 vLLM。 |
| `POST` | `/api/models/stop` | Stop vLLM. 停止 vLLM。 |
| `POST` | `/api/models/unload` | Stop vLLM and clear active model state. 停止 vLLM 并清除激活模型状态。 |

### Agent Jobs / 智能体任务

| Method | Path | Description / 说明 |
| --- | --- | --- |
| `POST` | `/api/agent/runs` | Create an agent job (workspace + task payload). 创建智能体任务。 |
| `GET` | `/api/jobs` | List jobs. 任务列表。 |
| `GET` | `/api/jobs/{job_id}` | Job status / detail. 任务状态与详情。 |
| `GET` | `/api/jobs/{job_id}/events` | **SSE** event stream for a job (supports `Last-Event-ID` resume). 任务的 SSE 事件流（支持 `Last-Event-ID` 续传）。 |
| `POST` | `/api/jobs/{job_id}/cancel` | Request graceful cancellation ( cooperative check inside the agent loop). 请求优雅取消（智能体循环内协作式检查）。 |
| `POST` | `/api/agent/answers` | Submit the user's answer to an `AskUser` prompt. 提交对 `AskUser` 提问的回答。 |

### Sessions / 会话

| Method | Path | Description / 说明 |
| --- | --- | --- |
| `GET` | `/api/sessions` | List web-created persisted sessions. Web 创建的持久化会话列表。 |
| `GET` | `/api/sessions/{filename}` | Session detail (history / metadata). 会话详情（历史与元数据）。 |
| `DELETE` | `/api/sessions/{filename}` | Delete a session file. 删除会话文件。 |

### Benchmarks / 评测

| Method | Path | Description / 说明 |
| --- | --- | --- |
| `GET` | `/api/datasets` | List benchmark datasets. 数据集列表。 |
| `POST` | `/api/benchmarks/runs` | Launch a benchmark script. 启动评测脚本。 |
| `GET` | `/api/benchmarks/history` | Aggregated pass-rate summary over `data/_results`. `data/_results` 的通过率汇总。 |
| `GET` | `/api/benchmarks/history/{name}` | Records of a specific benchmark run. 某次评测的明细记录。 |
| `GET` | `/api/benchmarks/trajectory` | Single-task trajectory drill-down. 单任务轨迹下钻。 |

### SSE Event Types / SSE 事件类型

Typical event stream for an agent job / 智能体任务的典型事件流：

```text
agent_started → step / llm_chunk → tool_call_start → tool_call_finish
→ (ask_user → answer) → agent_finish | cancelled | error
```

---

## 🏗️ Architecture / 架构说明

- **Backend / 后端**: pure Python standard library (`http.server` + `threading`) — no FastAPI/Flask dependency. The server instantiates the same `CodeAgent` stack as the CLI (tools, sandbox policy, context engine, trace). 纯标准库实现（`http.server` + 线程），与 CLI 共用同一套 `CodeAgent` 基础设施。
- **Frontend / 前端**: zero-dependency vanilla JS (`web/static/`) — hand-rolled markdown renderer, diff colorizer, tool cards. 零依赖原生 JS，自研 markdown 渲染、diff 着色与工具卡片。
- **Traces / 轨迹**: agent traces are written to `web/runtime/traces` (JSONL + HTML). 智能体轨迹写入 `web/runtime/traces`（JSONL + HTML 双格式）。
- **Testing / 测试**: `web/_mock_server.py` + `web/_test_server.py` + `web/_test_ui.py` (Playwright, optional). See `IMPROVEMENT.md` B2 for the plan to fold them into pytest. 测试脚本位于本目录，计划纳入 pytest 体系（见 `IMPROVEMENT.md` B2）。

---

**Last updated / 最后更新**: 2026-08-19
