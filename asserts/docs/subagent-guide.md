# 子代理机制指南（Subagent Mechanism）

## 📖 概述

**子代理机制**允许主 Agent 将复杂任务分解为子任务，委派给独立的子 Agent 执行，实现上下文隔离和工具权限控制。

### 核心特性

- ✅ **上下文隔离**：子代理是全新实例（独立 Config / 独立历史 / 独立 ToolRegistry），不污染主 Agent
- ✅ **角色化派生**：explorer / reviewer / tester 三种预置角色，按角色控制工具白名单与系统提示词
- ✅ **LLM 自动编排**：主循环内置 `Task` 工具，LLM 在 ReAct 循环中按需派生子代理
- ✅ **同轮并行**：独立的子任务可在同一轮响应中并行执行
- ✅ **成本优化**：子任务可用轻量模型（节省 ~70%）

> **✅ 状态说明 / Status**: 主 Agent 对话循环内置 **LLM 可自动调用的 `Task` 工具**——LLM 在 ReAct 循环中按需派生角色化子代理（explorer / reviewer / tester），独立任务可同轮多调用并行执行。
> 历史演进：原 `AgentOrchestra` 一次分解式编排已移除（由 Task 工具的动态派生取代）；编程式 `run_as_subagent()` API 与 `ToolFilter` 过滤器已于一并下线（2026-08-20，见 `IMPROVEMENT.md`）——其"同实例状态切换"模型有线程安全隐患且生产零调用，隔离能力由 `Role.create_subagent()` 的全新实例模型取代。
> The main-agent loop ships an **LLM-callable `Task` tool** — the LLM spawns role-specialized sub-agents on demand inside the ReAct loop; independent subtasks may be issued in the same response to run in parallel. The legacy `AgentOrchestra` pipeline and the programmatic `run_as_subagent()` / `ToolFilter` APIs have been removed (2026-08-20); isolation is now provided by `Role.create_subagent()` fresh-instance model.

---

## 🚀 快速开始

### 1. Task 工具（推荐 / Recommended）

无需编写任何编排代码——给主 Agent 下达任务，LLM 会在需要时自动调用 `Task` 工具：

```python
from hello_agents import ReActAgent, HelloAgentsLLM, Config

config = Config(subagent_timeout_seconds=300.0)  # 子代理超时预算
agent = ReActAgent("main", HelloAgentsLLM(), config=config)

# LLM 在推理过程中自行决定派生子代理，例如：
#   Task(role="explorer", task="探索项目结构，定位核心模块")
#   Task(role="reviewer", task="审查最近的改动")
result = agent.run("分析这个项目并给出架构改进建议")
```

**Task 工具参数**：

| 参数 | 说明 |
|------|------|
| `role` | 子代理角色：`explorer`（只读探索）/ `reviewer`（代码审查）/ `tester`（写测试与验证） |
| `task` | 子任务描述（自然语言） |

### 2. 编程式调用（Role 工厂）

需要显式控制子代理构造时，使用 `Role.create_subagent()`——这也是 Task 工具内部走的路径：

```python
from hello_agents.agents.roles import get_role

# 按角色创建隔离子代理（Config 深拷贝 / 独立 ToolRegistry / 禁交互 / 防递归）
# 注意：llm 与 working_dir 均为必传参数（与 Task 工具内部调用一致）
role = get_role("explorer")
subagent = role.create_subagent(
    llm=llm,
    parent_config=Config(),
    project_root="/path/to/project",
    working_dir="/path/to/project",
)

# 像普通 Agent 一样运行（独立历史，不影响主 Agent）
result = subagent.run("探索 hello_agents/core/ 目录，总结各模块职责")
```

---

## 💡 核心概念

### 1. 上下文隔离（全新实例模型）

子代理**不是**主 Agent 的状态切换，而是 `Role.create_subagent()` 构造的**全新实例**：

| 维度 | 主 Agent | 子代理 |
|------|----------|--------|
| 历史 | 原有历史继续累积 | 全新空历史 |
| Config | 原配置 | 深拷贝（修改不影响主 Agent） |
| ToolRegistry | 共享的注册表 | 独立注册表（只含角色白名单内工具） |
| 交互 | 可用 AskUser | 强制禁用（后台执行不能等待用户） |
| 递归 | 可派生子代理 | 强制 `subagent_task_enabled=False`（防递归） |

这保证了并发派生多个子代理时互不干扰——不存在共享可变状态的竞态。

### 2. 角色工具白名单

工具权限由各角色在 `code/agents/roles/` 中**硬编码白名单**控制：

| 角色 | 类别 | 可用工具 | 用途 |
|------|------|----------|------|
| 🔍 **explorer** | `readonly` | Glob, Grep, Read, LS, GitStatus, GitDiff, GitLog, GitBlame, Bash（受限） | 理解代码结构、追踪依赖、定位代码 |
| 📝 **reviewer** | `readonly` | Glob, Grep, Read, LS, GitDiff, GitLog, GitBlame, Bash（受限） | 多视角代码审查（正确性/安全/性能/风格） |
| 🧪 **tester** | `write` | Read, Write, Edit, Bash, GitDiff | 编写并运行测试、验证修复 |

> 为什么用角色白名单而不是通用过滤器？不同阶段有不同的安全画像：explorer 绝不应改文件，tester 需要写权限。角色化派生让边界显式且在代码层面可执行，而不依赖提示词约束。

### 3. Agent 工厂

**create_agent() - 统一创建接口**（用于构造自定义 Agent，非角色化子代理）：

```python
from hello_agents.agents.factory import create_agent

react_agent = create_agent("react", "explorer", llm, registry)
code_agent = create_agent("code", "coder", llm, registry)
```

> 注：reflection / plan / simple 三个类型已随教学演示型 agent 一并移除（2026-08-19）；工厂仅支持 react 与 code。

---

## 📝 使用指南

### 1. 并行与串行

**并行**：LLM 在同一轮响应中发起多个 Task 调用即可并行执行（主循环 `_execute_tools` 的 ThreadPoolExecutor）：

```python
# LLM 的单轮响应中同时发起（示意）：
# Task(role="explorer", task="分析模块 A 的依赖")
# Task(role="explorer", task="分析模块 B 的依赖")
# → 两个子代理并行执行
```

**串行**：依赖型子任务应等上游 Task 结果返回后再发起——ReAct "观察→思考→行动" 的分轮循环天然保证了这一点。单次 Task 调用是阻塞的（子代理跑完才返回结果给主循环）。

### 2. 行为配置（Config 字段）

```python
config = Config(
    subagent_task_enabled=True,      # 主循环是否注册 Task 工具（默认 True）
    subagent_timeout_seconds=300.0,  # 单个子代理超时（超时后放弃等待，结果丢弃）
)
```

> 注：**子代理步数上限不随 Config 配置**，由各角色内置的 `RoleConfig.max_steps` 控制：
> explorer = 20，reviewer = 30，tester = 25（定义于 `code/agents/roles/` 各角色）。

### 3. 成本优化（轻量模型）

角色工厂支持为子代理指定轻量模型：

```python
from hello_agents import HelloAgentsLLM

main_llm = HelloAgentsLLM(provider="openai", model="gpt-4")        # 复杂决策
light_llm = HelloAgentsLLM(provider="deepseek", model="deepseek-chat")  # 探索/简单处理

# Task 工具派生时经 get_role 按角色创建子代理，llm 参数传入轻量模型：
#   role.create_subagent(llm=light_llm, parent_config=..., ...)
# 主 Agent 保留强模型（main_llm）。
```

**成本节省示例**：
```
之前：100% GPT-4 = $30
之后：30% GPT-4 + 70% DeepSeek = $9 + $0.7 = $9.7
节省：68%
```

---

## 📊 实际案例

### 案例 1：复杂项目分析

**场景**：分析大型代码库，生成架构报告

直接向主 Agent 下达任务，LLM 自动分解并派生：

```
用户：分析这个项目并给出架构改进建议

LLM 的执行轨迹（自动）：
  → Task(role="explorer", task="探索项目结构，列出核心模块")     # 只读
  → Task(role="explorer", task="分析各模块的依赖关系")           # 只读
  → 综合两个子代理的结果，生成架构报告
```

**优势**：
- ✅ 每个子任务上下文隔离，不互相干扰
- ✅ 探索任务只能读取，不会误修改文件
- ✅ 独立子任务同轮并行，缩短总耗时

### 案例 2：多阶段代码审查

**场景**：代码审查 + 自动修复

```
用户：审查最近的改动，确认无误后写测试验证

LLM 的执行轨迹（自动）：
  → Task(role="reviewer", task="审查未提交的 diff，按严重度列出问题")  # 只读
  → 根据审查结论修复问题（主 Agent 自己做，有完整工具）
  → Task(role="tester", task="为修复的代码编写并运行测试")            # 写权限
```

---

## 🎯 最佳实践

### 1. 任务描述要具体

```text
❌ 不好：Task(role="explorer", task="看看代码")
✅ 好：Task(role="explorer", task="找到处理用户认证的模块，总结其调用链")
```

### 2. 让角色与任务性质匹配

探索/定位类任务用 `explorer`（只读最安全）；审查类用 `reviewer`（结构化输出）；需要写文件/跑测试用 `tester`。不要用 tester 跑纯探索任务——多余的写权限是风险。

### 3. 设置合理的步数与超时预算

```python
config = Config(subagent_timeout_seconds=300.0)
```

步数过高会放大失控循环的成本；超时过低会产生假失败。

---

## 🔧 高级用法

### 1. 子代理执行元数据

Task 工具返回给主循环的结果附带执行元数据（由 `Agent._get_subagent_metadata` 生成）：

```python
metadata = {
    "steps": 5,                  # 执行步数
    "tokens": 1500,              # Token 数（本地估算）
    "duration_seconds": 12.3,    # 执行时长
    "tools_used": ["Read", "Grep"],  # 使用的工具列表
    # "error": "..."             # 仅失败时出现
}
```

### 2. 子代理事件流（可观测性）

Task 工具在派生开始/结束时发出结构化事件，CLI 与 Web 前端均有渲染：

| 事件 | payload | 时机 |
|------|---------|------|
| `subagent_start` | `{role, task}` | 子代理开始执行 |
| `subagent_finish` | `{role, task, success, duration_seconds, summary}` | 子代理结束（成功/失败/超时） |

超时语义：daemon 线程 + `join(timeout)` 的"放弃等待"——超时后主循环继续，子代理结果被丢弃（其线程可能仍在后台运行至进程退出）。

### 3. 防递归保护（双层）

1. Task 工具注册检查 `enable_subagent_task` 配置；
2. `Role.create_subagent()` 强制子代理 Config 的 `subagent_task_enabled=False`——子代理的注册表里不会有 Task 工具，物理上无法再派生孙代理。

---

## 🔗 相关文档

- [会话持久化](./session-persistence-guide.md) - 保存子代理会话
- [可观测性](./observability-guide.md) - 追踪子代理执行

---

## ❓ 常见问题

**Q: 子代理会污染主 Agent 的历史吗？**

A: 不会。子代理是全新实例，有独立的历史与注册表；主 Agent 只接收子代理的最终结果（截断后）作为工具返回值。

**Q: 如何禁用子代理机制？**

A: 设置 `subagent_task_enabled=False`，主循环不再注册 Task 工具：
```python
config = Config(subagent_task_enabled=False)
```

**Q: 子代理可以访问主 Agent 的工具吗？**

A: 子代理有独立的 ToolRegistry，只包含其角色白名单内的工具：
- `explorer` / `reviewer`：只读工具（Read, Glob, Grep, LS, Git 只读系列, 受限 Bash）
- `tester`：Read, Write, Edit, Bash, GitDiff

**Q: 子代理能再派生子代理吗？**

A: 不能。防递归保护强制子代理的 `subagent_task_enabled=False`。

**Q: 子代理的成本如何计算？**

A: 子代理独立计费：
```python
# 主 Agent Token: 10,000
# 子 Agent 1 Token: 2,000
# 子 Agent 2 Token: 1,500
# 总计: 13,500 tokens
```

---

## 📈 性能指标

### 上下文隔离效果

| 场景         | 无隔离（共享历史） | 有隔离（子代理）  |
| ------------ | ------------------ | ----------------- |
| 历史长度     | 100+ 条消息        | 主 20 + 子 10     |
| 上下文清晰度 | 混乱               | 清晰              |
| Token 消耗   | 50,000             | 15,000（节省70%） |

### 成本优化效果

| 模型组合               | 成本（1M tokens） | 节省比例 |
| ---------------------- | ----------------- | -------- |
| 全部 GPT-4             | $30               | 0%       |
| 主 GPT-4 + 子 GPT-3.5  | $12               | 60%      |
| 主 GPT-4 + 子 DeepSeek | $9.7              | 68%      |

---

**最后更新**: 2026-08-20
