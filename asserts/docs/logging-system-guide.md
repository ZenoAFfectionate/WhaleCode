# 日志系统指南（Logging System）

## 📖 概述

HelloAgents 框架提供**三种日志范式**，满足不同场景的日志需求 / The framework ships three logging paradigms:

1. **TraceLogger** - 执行轨迹审计（JSONL + HTML 双格式）
2. **agent_print / agent_eprint** - 框架统一输出门面
3. **标准 logging** - Python 标准日志（集成方自配）

> ⚠️ **API 变更（2026-08-20）**：`set_agent_print` / `set_agent_eprint` sink 注入函数已随死代码清理移除（生产与测试均零调用，注入机制从未接线——见 `IMPROVEMENT.md` Q1-A）。`agent_print` / `agent_eprint` 仍作为框架统一输出门面保留，但其输出固定走 `sys.stdout` / `sys.stderr`；本文后续章节中涉及 `set_agent_print` 的注入示例已失效，集成方如需重定向输出请拦截 `sys.stdout` 或配置标准 logging。

---

## 🚀 快速开始

### 1. TraceLogger（执行轨迹 / Execution Trace）

TraceLogger 通过 `Config` 启用（Agent 构造函数**不接收** `trace_logger` 参数）：

```python
from hello_agents import ReActAgent, HelloAgentsLLM, Config

# 启用 TraceLogger
config = Config(trace_enabled=True, trace_dir="logs")
agent = ReActAgent("assistant", HelloAgentsLLM(), config=config)

# 执行任务
agent.run("分析项目")

# 查看日志（文件名含会话 ID）
# - logs/trace-<session_id>.jsonl（机器可读）
# - logs/trace-<session_id>.html（人类可读）
```

### 2. agent_print / agent_eprint（框架输出门面 / Injectable Sink）

框架内部所有输出均经过 `agent_print` / `agent_eprint`（`core/logging.py`），默认转发到 stdout/stderr；集成方可注入自定义 sink（CLI 入口用它接入 Rich 渲染层，Web 端用它转发到 SSE）：

```python
from hello_agents.core.logging import (
    agent_print, agent_eprint,
    set_agent_print, set_agent_eprint,
)

# 重定向框架输出（无需 monkey-patch sys.stdout）
set_agent_print(lambda *a, **kw: my_logger.info(" ".join(map(str, a))))
set_agent_eprint(lambda *a, **kw: my_logger.error(" ".join(map(str, a))))
```

### 3. 标准 logging（通用日志 / Standard Logging）

```python
import logging
from hello_agents import ReActAgent, HelloAgentsLLM

# 配置标准 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

agent = ReActAgent("assistant", HelloAgentsLLM())
agent.run("分析项目")
```

---

## 💡 三种范式对比

| 范式 | 用途 | 格式 | 可读性 | 持久化 |
| --- | --- | --- | --- | --- |
| TraceLogger | 执行轨迹审计 | JSONL + HTML | 高 | ✅ |
| agent_print / agent_eprint | 框架运行输出（可重定向） | 文本 | 中 | 取决于 sink |
| 标准 logging | 集成方应用日志 | 文本 | 低 | ✅ |

---

## 📝 使用指南

### 1. TraceLogger 详细说明

**特点：**
- ✅ 记录会话、模型输出、工具调用等关键事件
- ✅ 双格式输出（JSONL + HTML，文件名含 session_id）
- ✅ 敏感信息脱敏（`trace_sanitize`）
- ✅ 支持审计与回放

**配置（全部为 `Config` 字段）：**
```python
config = Config(
    trace_enabled=True,                        # 总开关
    trace_dir="logs",                          # 输出目录
    trace_sanitize=True,                       # 敏感信息脱敏
    trace_html_include_raw_response=False,     # HTML 是否含原始响应
)
```

**JSONL 事件结构（字段为 `ts` / `event` / `payload`）：**
```json
{"ts": "2026-02-21T10:30:45.123Z", "event": "session_start", "payload": {"session_id": "s-20260221-103045-a1b2"}}
{"ts": "2026-02-21T10:30:46.456Z", "event": "tool_call", "payload": {"tool_name": "Read", "parameters": {"path": "config.py"}}}
{"ts": "2026-02-21T10:30:47.789Z", "event": "model_output", "payload": {"content": "...", "usage": {...}}}
```

**已记录的事件类型：** `session_start`、`session_end`、`tool_call`、`tool_result`、`message_written`、`model_output`、`hook_timeout`、`hook_error`

**查看 HTML 报告：**
```bash
open logs/trace-<session_id>.html
```

**直接使用 TraceLogger（不经 Agent）：**
```python
from hello_agents.observability import TraceLogger

logger = TraceLogger(
    output_dir="logs",                 # 输出目录
    sanitize=True,                     # 敏感信息脱敏
    html_include_raw_response=False,   # HTML 含原始响应
)
logger.log_event("tool_call", {"tool_name": "Read"}, step=1)
logger.finalize()  # 生成 HTML 汇总
```

### 2. agent_print 详细说明

**特点：**
- ✅ 框架内部输出的唯一通道（建议-7 改造，替代直接 print）
- ✅ sink 可注入：`set_agent_print` / `set_agent_eprint`
- ✅ 默认行为等价于 `print`（向后兼容）

**典型用法——把框架输出接入文件：**
```python
from hello_agents.core.logging import set_agent_print, set_agent_eprint

log_file = open("agent-run.log", "a", encoding="utf-8")
set_agent_print(lambda *a, **kw: print(*a, file=log_file, flush=True))
set_agent_eprint(lambda *a, **kw: print(*a, file=log_file, flush=True))
```

### 3. 标准 logging 详细说明

**特点：**
- ✅ Python 标准库，无需额外依赖
- ✅ 灵活配置（Handler、Formatter、轮转）
- ✅ 与其他库兼容

**配置：**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Agent 开始执行")
```

---

## 📊 实际案例

### 案例 1：生产环境监控

```python
import logging
from hello_agents import ReActAgent, HelloAgentsLLM, Config

# 应用级日志（标准 logging）
logging.basicConfig(level=logging.INFO)

# Agent 级轨迹（TraceLogger）
config = Config(trace_enabled=True, trace_dir="logs/prod")
agent = ReActAgent("assistant", HelloAgentsLLM(), config=config)

try:
    result = agent.run("处理用户请求")
except Exception as e:
    logging.error(f"Agent 执行失败: {e}")
```

### 案例 2：开发调试

```python
from hello_agents import ReActAgent, HelloAgentsLLM, Config

# 详细轨迹 + HTML 可视化 + 保留原始响应
config = Config(
    trace_enabled=True,
    trace_dir="debug_logs",
    trace_html_include_raw_response=True,  # 调试时查看原始 LLM 响应
)
agent = ReActAgent("assistant", llm, config=config)
agent.run("分析项目")

# 查看 debug_logs/trace-<session_id>.html 可视化轨迹
```

### 案例 3：捕获框架输出到自定义管道

```python
from hello_agents import ReActAgent, HelloAgentsLLM
from hello_agents.core.logging import set_agent_print

# 把框架输出送入消息队列/面板等自定义通道
events = []
set_agent_print(lambda *a, **kw: events.append(" ".join(map(str, a))))

agent = ReActAgent("assistant", HelloAgentsLLM())
agent.run("分析项目")
```

---

## 🎯 最佳实践

### 1. 根据场景选择范式

```python
# ✅ 生产审计：TraceLogger（脱敏开启）
config = Config(trace_enabled=True, trace_sanitize=True)

# ✅ 集成宿主（CLI/Web）：注入 agent_print sink
set_agent_print(my_render_fn)

# ✅ 应用日志：标准 logging
logging.basicConfig(level=logging.WARNING)
```

### 2. 日志分级（标准 logging）

```python
logger.debug(f"工具参数: {parameters}")      # 详细调试信息
logger.info("Agent 开始执行")                 # 普通信息
logger.warning("工具调用超时，重试中...")      # 警告
logger.error(f"Agent 执行失败: {error}")      # 错误
```

### 3. 日志轮转

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "agent.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5           # 保留 5 个备份
)

logging.basicConfig(handlers=[handler])
```

---

## 🔗 相关文档

- [可观测性](./observability-guide.md) - TraceLogger 详细说明

---

## ❓ 常见问题

**Q: 如何同时使用多种日志范式？**

A: 三者互不冲突，可同时开启：
```python
import logging
from hello_agents import ReActAgent, HelloAgentsLLM, Config
from hello_agents.core.logging import set_agent_print

logging.basicConfig(level=logging.INFO)                  # 应用日志
set_agent_print(lambda *a, **kw: print("[agent]", *a))  # 框架输出加前缀
config = Config(trace_enabled=True, trace_dir="logs")    # 执行轨迹
agent = ReActAgent("assistant", llm, config=config)
```

**Q: 日志文件太大怎么办？**

A: TraceLogger 每个会话独立文件（trace-<session_id>），按会话归档即可；应用日志用轮转：
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler("agent.log", maxBytes=10*1024*1024, backupCount=5)
```

**Q: 如何禁用 TraceLogger？**

A: `Config(trace_enabled=False)`（默认即为关闭状态）。

**Q: 框架输出能重定向到 SSE / WebSocket 吗？**

A: 可以，这正是 `set_agent_print` 的设计目的（建议-7）——Web 控制台通过注入 sink 把框架输出转发到 SSE 事件流。

---

**最后更新**: 2026-08-19
