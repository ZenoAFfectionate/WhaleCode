# Contributing to WhaleCode / 贡献指南

Thank you for contributing! This guide covers the conventions this repository actually follows. 欢迎贡献！本指南描述本仓库实际遵循的约定。

> **Naming / 命名**: project brand **WhaleCode** · PyPI distribution `hello-agents` · Python package `hello_agents` (source tree `code/`). 项目品牌 WhaleCode · PyPI 发行名 hello-agents · Python 包名 hello_agents（源码树 `code/`）。

---

## 1. Setup / 环境搭建

```bash
git clone https://github.com/ZenoAFfectionate/WhaleCode.git
cd WhaleCode

# Install runtime + dev deps (authoritative source: pyproject.toml)
# 安装运行时 + 开发依赖（权威来源：pyproject.toml）
pip install -e ".[dev]"

# Or the quick mirror / 或快速镜像方式：
pip install -r requirements.txt
```

- Python **3.10+** required / 需要 Python 3.10 及以上。
- Optional extras: `.[tokenizer]` (local token counting), `.[ui-tests]` (Playwright for `web/_test_ui.py`), `.[anthropic]` / `.[gemini]`. 可选能力组见 `pyproject.toml` 的 `[project.optional-dependencies]`。

---

## 2. Repository Layout / 仓库结构

| Path / 路径 | Contents / 内容 |
| --- | --- |
| `code/` | The `hello_agents` package: `agents/` (ReAct engine, roles), `tools/` (builtin, LSP, protocol), `core/` (LLM adapters, config), `context/` (compression engine), `benchmark/`, `observability/`, `skills/`. 框架源码包。 |
| `tests/` | Pytest suite (~1500 tests) + `conftest.py` mock fixtures. 测试套件与 mock 基建。 |
| `web/` | Zero-dependency web console (std-lib server + vanilla JS frontend). Web 控制台。 |
| `scripts/` | CLI entry (`cli.py`), benchmark runners (`run_*.sh`), demos. CLI 入口与评测脚本。 |
| `asserts/docs/` | Themed guides (per-feature deep dives). 专题指南文档。 |
| `main.py` | Unified entry: `python main.py cli | web | bench | test ...`. 统一入口。 |
| `IMPROVEMENT.md` | Improvement backlog with a per-item tracking table. 改进清单与逐项跟踪表。 |

---

## 3. Branching / 分支规范

- Branch from `main` for every change; keep branches short-lived. 每次改动从 `main` 拉出分支，短生命周期。
- Naming / 命名: `feat/<area>-<slug>`, `fix/<area>-<slug>`, `docs/<slug>`, `test/<slug>` (e.g. `feat/tools-task-tool`, `fix/web-sse-resume`).
- One logical change per branch/PR — avoid mixing features with refactors. 一个分支一个逻辑改动，功能与重构不混提。

---

## 4. Commit Messages / 提交信息格式

This repository tracks improvements with **tagged markers** (used consistently in code comments, tests, and `IMPROVEMENT.md`). Follow the existing convention: 本仓库用**带标记的编号**追踪改进（代码注释、测试与 `IMPROVEMENT.md` 保持一致）：

```text
<type>: <short summary in Chinese or English>

<body: what & why, referencing the marker>
```

- `type`: `feat` / `fix` / `docs` / `test` / `refactor` / `chore`
- For tracked improvements, include the marker, e.g. `feat: LLM 指数退避重试（重要-12）`. 已跟踪的改进项带上编号标记。
- Markers observed in history / 历史标记风格: `改进-1`、`严重-1`、`P1-1`、`重要-12` — register new markers in `IMPROVEMENT.md`'s tracking table (§八) when landing an item. 新标记需在 `IMPROVEMENT.md` 跟踪表登记。

---

## 5. Testing Requirements / 测试要求

**Write tests first or with the change — never after the fact.** 本项目的既定纪律：测试与改动同批落地，每个改进项配回归测试（如 `tests/test_improvements.py`、`tests/test_runtime_improvements.py` 的先例）。

```bash
# Run the full suite (config in pyproject.toml: pytest + coverage)
python main.py test
# or: pytest

# Run a focused subset / 运行子集
pytest tests/test_agents_unit.py -q
```

- New features need unit tests + (where applicable) regression tests named after the improvement marker. 新功能需要单测，改进项需配套以标记命名的回归测试。
- Sandbox / security changes MUST add adversarial cases (see `tests/test_safe_exec*.py`, `test_bash_sandbox*.py`). 沙箱与安全类改动必须补充对抗性用例。
- Web changes: extend `web/_test_server.py` / `web/_test_ui.py` (Playwright, skipped when missing). Web 改动同步更新 web 目录测试脚本。
- Keep coverage at or above the current baseline (`--cov` configured in `pyproject.toml`). 覆盖率不低于当前基线。

---

## 6. Code Style / 代码风格

- **Ruff** is the single linter/formatter (`ruff check` + `ruff format`, config in `pyproject.toml`; it replaced the former black + isort pair). ruff 是唯一 lint/format 工具（已替代 black + isort）。
- **mypy** runs progressively: strict on `code/core/` + `code/context/`, other trees whitelisted via `ignore_errors` — new code in strict trees must be fully typed. mypy 分层收紧中，严格区新代码必须完整标注类型。
- Docstrings & comments: `code/core` uses Chinese, `code/tools` uses English — for new code prefer English, and keep the language consistent within a file. 注释语言：core 中文 / tools 英文并存；新代码建议英文，单文件内保持一致。

---

## 7. Documentation / 文档要求

- Docs live in `asserts/docs/` (themed guides) — follow the existing template (emoji section headers, quick-start, tables, FAQ). 专题文档遵循现有模板。
- **Bilingual convention / 双语惯例**: titles are Chinese + English; new top-level docs (README-level, CONTRIBUTING-level) are written in both languages. 标题中英双语；顶层新文档采用中英对照。
- **Accuracy is a hard requirement / 准确性是硬要求**: docs must describe implemented APIs only — verified against `code/` (the 2026-08-19 audit removed several docs describing non-existent APIs such as `AgentLogger`/`DevLogTool`/`TaskTool` auto-registration). 文档只能描述已实现的 API；描述规划中功能时必须显式标注（参照 `subagent-guide.md` 顶部状态说明的先例）。
- Configuration/config-field changes must update the relevant guide (e.g. `observability-guide.md` for `trace_*` fields). 配置字段变更需同步对应指南。

---

## 8. Pull Request Checklist / PR 清单

- [ ] Tests added/updated & passing locally (`python main.py test`). 测试已补充且本地通过。
- [ ] `ruff check` and `ruff format` clean. lint 与格式化通过。
- [ ] mypy clean for `code/core/` + `code/context/` (strict zone). 严格区类型检查通过。
- [ ] Docs updated if behavior/API/config changed. 行为/API/配置变更已同步文档。
- [ ] Improvement items registered/checked off in `IMPROVEMENT.md` tracking table. 改进项已在 `IMPROVEMENT.md` 跟踪表登记或勾选。
- [ ] No hard-coded machine-specific paths (use `PROJECT_ROOT` / env vars — see D2 in `IMPROVEMENT.md`). 无机器特定硬编码路径。
- [ ] Commit messages follow the marker convention (§4). 提交信息遵循标记规范。

---

## 9. Getting Help / 获取帮助

- Open an issue: https://github.com/ZenoAFfectionate/WhaleCode/issues
- Improvement roadmap & history: `IMPROVEMENT.md` (per-round verification records in §七).

---

**Last updated / 最后更新**: 2026-08-19
