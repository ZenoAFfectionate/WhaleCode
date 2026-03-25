# Whale Code — A Production-Grade Coding Agent Framework

Whale Code is a from-scratch implementation of an autonomous coding agent that operates inside a local repository. It follows the **ReAct (Reasoning + Acting)** paradigm, powered by OpenAI-compatible function calling, and ships with a full suite of atomized programming tools, a multi-layer context management engine, and a persistent task scheduling system. The goal is to replicate — and deeply understand — the core architecture behind tools like Claude Code, Cursor Agent, and similar AI coding assistants.

![Whale Code Key Design](asserts/key_design.png)

## Table of Contents
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [CodeAgent Implementation Details](#codeagent-implementation-details)
- [Context Management](#context-management)
- [Atomized Tool System](#atomized-tool-system)
  - [Why Atomized Tools Instead of Bash?](#why-atomized-tools-instead-of-bash)
  - [Tool List](#tool-list)
  - [Tool Response Protocol](#tool-response-protocol)
  - [Circuit Breaker](#circuit-breaker)
- [Task System](#task-system)
  - [Unified Task Management (TodoWrite)](#unified-task-management-todowrite)
  - [Background Execution via Bash](#background-execution-via-bash)
- [Benchmarks](#benchmarks)
  - [Supported Benchmarks](#supported-benchmarks)
  - [Quick Start](#quick-start-1)
- [License](#license)

---

## Quick Start

### Installation

```bash
git clone https://github.com/ZenoAFfectionate/Coding_Agent.git
cd Whale_Code

# Create and activate a conda virtual environment
conda create -n WhaleCode python=3.12 -y
conda activate WhaleCode

# Install dependencies
pip install -r requirements.txt

# Configure `.env` (minimum required):
#   LLM_MODEL_ID=<your-model>
#   LLM_API_KEY=<your-key>
#   LLM_BASE_URL=<your-openai-compatible-endpoint>
```

### Running the Agent

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --language-model-only \
    --tool-call-parser qwen3_xml
```

```bash
python run_cli.py --workspace /working/space
```

The CLI provides an interactive loop where you can issue coding tasks:

```
> Read the main entry point and summarize its structure
> Find all TODO comments in the codebase
> Add error handling to the data processing pipeline
```

![Whale Code CLI Demo](asserts/cli-demo.png)

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show the help message with all available commands |
| `/info` `/model` | Display runtime info: workspace path, model, base URL, temperature, trace status, tool count |
| `/tools` | List all registered tools with their descriptions |
| `/pwd` | Print the current working directory |
| `/cd <path>` | Change the agent's working directory (must stay within workspace root) |
| `/history [n]` | Show conversation history; optional `n` limits to the last N entries |
| `/log` | Open all terminal output in a scrollable pager (`less`/`more`) |
| `/clear` | Clear the in-memory conversation history |
| `/save [name]` | Save the current session snapshot (default name: `session-latest`) |
| `/load [path\|name]` | Load a previously saved session by file path or name |
| `/resume [path\|name]` | Alias of `/load`; restores conversation + task snapshot |
| `/sessions` | List all saved session files with metadata (steps, tokens, timestamps) |
| `/compact [focus]` | Manually trigger context compaction; optional `focus` guides the summary |

---

## Architecture Overview

![Whale Code CLI Architecture](asserts/architecture.png)

The agent follows a strict **Think → Act → Observe → Re-think** loop implemented through the ReAct pattern. Every reasoning step, tool invocation, and observation is tracked, compressed, and persisted.

---

## CodeAgent Implementation Details

**Source**: `code/agents/code_agent.py`

`CodeAgent` is the top-level agent class, built by extending the inheritance chain:

```
Agent (base) → ReActAgent (function-calling ReAct loop) → CodeAgent (coding-specialized)
```

### Key Design Decisions

1. **Workspace Sandboxing**: The agent is initialized with a `project_root` and `working_dir`. Every file operation is validated to stay within the workspace root, preventing accidental access to system files.

2. **OpenAI Function Calling (not text-parsing)**: Unlike text-based ReAct implementations that rely on regex to parse `Action: ...` from model output, `CodeAgent` uses native function calling. The model's `tool_calls` are structured JSON, achieving a **99%+ parse success rate** with zero regex.

3. **Tool Auto-Registration**: On initialization, `register_default_tools()` instantiates and registers atomic tools, each bound to the workspace root:

   ```python
   def register_default_tools(self, enable_task_tool=True):
       self.tool_registry.register_tool(ReadTool(...))
       self.tool_registry.register_tool(WriteTool(...))
       self.tool_registry.register_tool(EditTool(...))
       self.tool_registry.register_tool(GlobTool(...))
       self.tool_registry.register_tool(GrepTool(...))
       self.tool_registry.register_tool(BashTool(...))
       # ... and more
   ```

4. **Sub-Agent Creation**: `_create_subagent()` spawns an isolated `CodeAgent` with its own `ToolRegistry`, separate history, and `interactive=False` (disabling `AskUser`). This enables context-isolated delegated work.

5. **Manual Context Compaction**: The `compact()` method exposes a public API for on-demand context compression, reconstructing messages from history, running the compactor, and replacing the stored history.

---

## Context Management

Context management is the core engineering challenge of a long-running coding agent. Whale Code implements a **multi-layer context engineering system** with five dedicated components.

### 1. HistoryManager (`code/context/history.py`)

An append-only message store with round-based compression:

- **Append-only writes** — cache-friendly, no in-place edits.
- **Round boundary detection** — identifies `user → assistant/tool*` round boundaries to determine compression granularity.
- **Compression** — replaces old rounds with a summary message while retaining the most recent `min_retain_rounds` complete rounds.
- **Serialization** — `to_dict()` / `load_from_dict()` for session persistence.

### 2. TokenCounter (`code/context/token_counter.py`)

Local token estimation without API calls:

- **tiktoken integration** — uses the `cl100k_base` encoding for accurate counts.
- **Caching** — `role:content` cache key avoids re-encoding identical messages.
- **Incremental counting** — `_history_token_count` is updated on every `add_message()` call, never re-scanning full history.
- **Graceful degradation** — falls back to `len(text) // 4` when tiktoken is unavailable.

### 3. ContextCompactor (`code/context/compactor.py`)

A three-layer compression engine adapted for OpenAI function-calling message format:

| Layer | Trigger | Strategy |
|-------|---------|----------|
| **Layer 1: micro_compact** | Every turn | Scans `tool` role messages from newest to oldest; keeps the N most recent tool results intact, replaces older ones with `[Previous tool result: {name} — truncated]` |
| **Layer 2: auto_compact** | Token threshold exceeded | Saves the full transcript to a JSONL file, calls the LLM to generate a structured summary, rebuilds messages as `[system] + [summary] + [ack]` |
| **Layer 3: manual_compact** | User-triggered | Same as Layer 2, but accepts an optional `focus` parameter to guide the summary (e.g. "focus on the authentication module") |

The compactor builds a `tool_call_id → tool_name` mapping to produce meaningful truncation labels, and serializes messages with per-message truncation (2000 chars max) before feeding them to the summary LLM.

### 4. ObservationTruncator (`code/context/truncator.py`)

Handles tool output that is too large to fit in context:

- **Multi-directional truncation** — `head` (keep first N lines), `tail` (keep last N lines), or `head_tail` (keep both ends with a gap marker).
- **Dual limits** — enforces both `max_lines` (default 2000) and `max_bytes` (default 50KB).
- **Full output persistence** — saves the complete untruncated output as a JSON file in `tool-output/`, so the agent can reference it later if needed.

### 5. ContextBuilder (`code/context/builder.py`)

A **GSSC (Gather-Select-Structure-Compress)** pipeline for structured context assembly:

1. **Gather** — collects context packets from multiple sources (system instructions, conversation history, additional data).
2. **Select** — scores packets using `0.7 × relevance + 0.3 × recency` with exponential time decay, filters by `min_relevance` threshold, and fills within the token budget.
3. **Structure** — organizes selected packets into a template: `[Role & Policies] → [Task] → [State] → [Evidence] → [Context] → [Output]`.
4. **Compress** — if the structured result exceeds the token budget, truncates by paragraph boundaries.

---

## Atomized Tool System

### Why Atomized Tools Instead of Bash?

A naive approach would give the agent a single `Bash` tool and let it run `cat`, `grep`, `sed`, etc. for all operations. Whale Code intentionally **splits file and search operations into dedicated, atomic tools**. Here's why:

#### 1. Workspace Sandboxing

Every atomic tool (Read, Write, Edit, Glob, Grep) uses `resolve_path()` to validate that the target path stays within the `project_root`. The Bash tool cannot enforce this for arbitrary shell commands.

```python
# Every tool validates paths against the workspace root
target = resolve_path(self.project_root, self.working_dir, raw_path)
# Raises ValueError if the path escapes the workspace
```

#### 2. Optimistic Locking for Concurrent Safety

`ReadTool` returns `expected_mtime_ms` and `expected_size_bytes` metadata with every file read. `WriteTool` and `EditTool` accept these values back and check them before writing — if the file has been modified externally since it was last read, the write is rejected. This prevents the agent from silently overwriting human edits. A raw `Bash` + `sed`/`echo` pipeline has no such protection.

#### 3. Structured Response Protocol

Every tool returns a `ToolResponse` object with a three-state status (`SUCCESS | PARTIAL | ERROR`), structured `data` payload, `error_info` with error codes, and `stats` with timing information. This gives the agent machine-readable feedback instead of parsing unstructured stdout/stderr.

```python
class ToolResponse:
    status: ToolStatus        # SUCCESS / PARTIAL / ERROR
    text: str                 # Human/LLM-readable text
    data: Dict[str, Any]      # Structured payload
    error_info: Dict          # Error code + message
    stats: Dict               # Timing, token counts
```

#### 4. Circuit Breaker Protection

The `ToolRegistry` wraps every tool call with a `CircuitBreaker`. If a tool fails 3 consecutive times, it is automatically disabled for 5 minutes. This prevents infinite retry loops where the agent keeps calling a broken tool. Bash-only architectures have no such guardrail.

#### 5. Context Efficiency

Atomic tools produce compact, well-formatted output. `GlobTool` returns sorted file paths. `GrepTool` returns `file:line: content` entries. `ReadTool` returns numbered lines with metadata headers. Raw shell output from `find`, `grep -r`, or `cat` is often verbose, includes color codes, and wastes context tokens.

#### 6. The Bash Tool Actively Redirects

The `BashTool` itself enforces this philosophy. It maintains a `PREFER_SPECIALIZED_TOOLS` blocklist:

```python
PREFER_SPECIALIZED_TOOLS = {"ls", "find", "grep", "rg", "sed", "awk"}
```

If the model tries to run `grep pattern .` or `sed -i ...` as a standalone command, Bash returns an error saying "Use the dedicated tools instead." However, piped usage like `git log | grep fix` is allowed, since `grep` is not the segment leader.

### Tool List

| Category | Tool | Description |
|----------|------|-------------|
| **File Discovery** | `Glob` | Find files by glob pattern (`**/*.py`, `src/**/*.ts`) with `fnmatch` + directory pruning |
| | `Grep` | Regex code search using ripgrep (with Python fallback) |
| | `LS` | List directory contents |
| | `Read` | Read file content with metadata for optimistic locking |
| **File Modification** | `Write` | Full-file rewrite with atomic write + optimistic locking + dry-run mode |
| | `Delete` | Safe file/directory deletion with guardrails and atomic trash move |
| | `Edit` | Single-snippet surgical replacement with conflict detection + backup |
| **Execution** | `Bash` | Shell commands with command policy validation and `block_until_ms` backgrounding |
| **Planning** | `TodoWrite` | Unified task manager (`action=create/update/list/get/bulk_create/delete`) with persisted dependency graph |
| **Web** | `WebSearch` | Search the web via DuckDuckGo |
| | `WebFetch` | Fetch and extract readable text from a URL |
| **Interaction** | `AskUser` | Ask the user a question (main agent only, disabled in sub-agents) |

### Tool Response Protocol

All tools return `ToolResponse` with a three-state status:

- **`SUCCESS`** — task completed as expected.
- **`PARTIAL`** — result is usable but degraded (output truncated, search limit reached, non-zero exit code).
- **`ERROR`** — no valid result; includes a structured error code (e.g. `NOT_FOUND`, `ACCESS_DENIED`, `TIMEOUT`, `CIRCUIT_OPEN`).

### Circuit Breaker

```
Closed (normal) ──[3 consecutive failures]──► Open (disabled)
      ▲                                           │
      └───────────[5 min timeout]─────────────────┘
```

The circuit breaker tracks per-tool failure counts. When a tool hits the threshold, it's automatically disabled for 5 minutes. This prevents the agent from burning tokens on a broken tool.

---

## Task System

Whale Code now uses a merged task architecture centered on one tool (`TodoWrite`) plus asynchronous execution through `Bash`.

### Unified Task Management (TodoWrite)

**Source**: `code/tools/builtin/todowrite_tool.py`

`TodoWrite` is a single action-based tool that replaces separate `task_*` tools.

- **Actions**: `create`, `update`, `list`, `get`, `bulk_create`, `delete`
- **Storage model**: one JSON file per task under `memory/tasks/task_{id}.json`
- **Status model**: `pending` → `in_progress` → `completed` or `cancelled`
- **Dependencies**: `blockedBy` / `blocks` are maintained bidirectionally
- **Auto-unblock behavior**: completing a task removes its ID from other tasks' `blockedBy`
- **Progress recap**: each operation returns a compact status recap for context efficiency

This merged design keeps task state persistent across context compaction and process restarts while reducing tool-surface complexity.

### Background Execution via Bash

**Source**: `code/tools/builtin/bash.py`

Long-running shell commands are handled directly by `Bash`:

1. **`block_until_ms`** — waits up to the configured window (default `30000`).
2. **Automatic backgrounding** — if the process exceeds the wait window, it continues in background.
3. **Immediate background mode** — set `block_until_ms: 0` to return immediately.
4. **Terminal artifacts** — background runs are persisted under `memory/terminals/` with status and output.

This removes the need for a dedicated background tool while preserving asynchronous workflow support.

---

## Benchmarks

Whale Code includes a built-in benchmark suite to evaluate the coding agent on five standard datasets. All benchmarks use `CodeAgent` with its full tool set (Read, Write, Edit, Bash, Glob, Grep, etc.) — web-related tools are disabled during evaluation to ensure fair, reproducible results.

**Source**: `code/benchmark/`

### Supported Benchmarks

| Benchmark | Dataset | Tasks | Metric | Description |
|-----------|---------|-------|--------|-------------|
| **MBPP+** | `data/MBPP/` | 378 | pass@1 | Crowd-sourced Python programming problems |
| **HumanEval+** | `data/HEVP/` | 164 | pass@1 | Function-generation tasks with 80× more tests than original HumanEval |
| **ClassEval** | `data/CLEV/` | 100 | pass@1 | Class-level code generation requiring multi-method implementation |
| **AIME** | `data/AIME/` | — | accuracy | Math competition problems solved via agent-written Python programs |

> **Note on SWE-bench**: SWE-bench uses a **two-phase evaluation**:
> 1. **Phase 1 — Agent inference** (`run_swev.sh`): The agent reads the issue, navigates the repo, and produces a patch (git diff). Results are saved as a predictions JSONL file.
> 2. **Phase 2 — Docker evaluation** (`run_swev_eval.sh`): The predictions are fed to the [official SWE-bench Docker harness](https://github.com/SWE-bench/SWE-bench) which applies each patch in an isolated container with the correct Python version and dependencies, then runs the test suite to grade the fix.

### Quick Start

> **Prerequisite**: The LLM backend must be running (e.g. vLLM, or set the API key in `.env`).

```bash
bash scripts/run_hevp.sh  # run HumanEval benchmark
bash scripts/run_clev.sh  # run ClassEval benchmark
bash scripts/run_mbpp.sh  # run MBPP benchmark
bash scripts/run_aime.sh  # run AIME benchmark

# run SWEV benchmark and evaluation
bash scripts/run_swev.sh  # (Phase 1: agent inference)

bash scripts/run_swev_eval.sh data/_results/swevbench_verified_<timestamp>.jsonl
```

### Result

> Model: **Qwen3.5-35B-A3B-FP8**

| Benchmark | Tasks | Passed | Pass Rate | Avg Time | Date |
|-----------|------:|-------:|----------:|---------:|------|
| **MBPP+**      | 378 | 375 | **99.2%** | 30.52s | 2026-03-24 |
| **HumanEval+** | 164 | 159 | **96.9%** | 32.71s | 2026-03-24 |
| **ClassEval**  | 100 | 94  | **94.0%** | 139.5s | 2026-03-24 |
| **AIME**       | 30  | 25  | **83.3%** | 171.1s | 2026-03-24 |

---

## License

This project is licensed under [CC-BY-NC-SA-4.0](LICENSE).
