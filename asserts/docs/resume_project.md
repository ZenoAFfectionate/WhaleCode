# 简历 — 项目经验

---

## 项目：基于 HA 框架的编程智能体 WhaleCode

**项目地址**：https://github.com/ZenoAFfectionate/WhaleCode

**基础框架**：HelloAgent 1.0.0（https://github.com/jjyaoao/HelloAgents）—— 通用多智能体框架，提供 Agent 基类、ReAct 推理循环、12 个原子化编程工具、三层上下文压缩引擎、熔断器与乐观锁、会话持久化等 16 项核心能力

**角色**：项目负责人

**技术栈**：Python 3.12、HelloAgent 框架、OpenAI 函数调用、vLLM 推理引擎、Qwen3.5 / Gemma / DeepSeek 等多模型适配

---

### 概况

在 HA 通用框架之上，围绕编程场景实现了高安全性工具链与完整评测体系（新增约 18,000 行代码），构建了可投入实际使用的编程智能体 WhaleCode。该智能体能够在真实代码仓库中自主完成代码阅读、搜索定位、精准编辑、命令执行和任务规划等完整工作流，在多个基座模型上完成了系统性评测验证。

---

### 亮点一：实现多种原子化编程工具并实现沙箱隔离以确保工作目录不越界，编写专用提示词并实现上下文智能压缩

- **原子化编程工具集成与沙箱隔离**：将 HA 提供的多种原子化编程工具（Read、Write、Edit、Delete、LS、Glob、Grep、Bash 等）统一注册到 CodeAgent，为每个文件操作和搜索工具注入 `project_root` 和 `working_dir` 参数，通过 `resolve_path` 实现工作区路径校验，确保工具调用始终限定在项目目录内而不会越界访问系统文件。

- **编程智能体专用提示词**：编写了约 280 行的 `prompts/system_prompt.md`，覆盖工具调用规则（修改前必须先 Read、优先使用专用工具而非 Bash 裸命令）、并行调用策略、文件操作安全约束等，使模型行为符合编程场景的规范要求。

- **三层上下文压缩引擎**：利用并配置了 HA 的上下文压缩体系——Layer 1（micro_compact）每轮仅保留最近 N 轮的工具调用结果，旧结果替换为截断摘要；Layer 2（auto_compact）在 token 超阈值时自动生成结构化摘要并重建消息；Layer 3（manual_compact）支持手动触发并指定压缩焦点。配合增量 token 计数与工具输出截断，使智能体在长程编程任务中可持续运行而不会因上下文膨胀而失效。

---

### 亮点二：采用"多轮受控提交 + 双层安全沙箱"，从零搭建针对多种编程数据集的评测体系

- **多轮受控提交机制**：设计并实现了 `_run_controlled_submission_rounds()` 闭环方法——智能体提交代码 → 沙箱安全评测 → 构建结构化反馈注入下一轮提示 → 智能体根据反馈迭代优化 → 再次提交。将评测从"一次性执行"升级为"多轮迭代反馈"的完整闭环。各 benchmark 可自定义评测逻辑、反馈构建策略和步数预算控制，适配不同评测场景。

- **4 种评测适配器统一异构格式**：针对不同 benchmark 的输出格式差异，实现了 `PythonAssertionAdapter`（MBPP+ 断言式）、`PythonVerifierAdapter`（HEVP/CLEV 隐藏 unittest）、`PythonStdinAdapter`（LCB stdin/stdout 式）、`PythonFunctionalAdapter`（LCB 函数调用式），将 6 类数据集的评测统一为同一套执行接口。

- **双层安全沙箱**：第一层 **AST 静态检查**——在代码执行前解析 AST，拦截 `subprocess / socket / importlib / pickle / ctypes` 等 10 个高危模块导入，以及 `exec / eval / open / __import__` 等内置危险调用；第二层 **OS 级资源隔离**——通过 `preexec_fn` + `prlimit` 设置内核级硬限制（CPU 时间、内存地址空间、最大进程数、文件大小），子进程在独立进程组中运行，超时时 `SIGKILL` 杀整个进程树，配合最小化环境变量剥离敏感信息。

- 覆盖 6 类标准 benchmark（MBPP+ 378 题 / HumanEval+ 164 题 / ClassEval 100 题 / LiveCodeBench v6 / AIME 24-26 / SWE-bench Verified 500 题），支持断点续测、任务级超时保护和轨迹持久化。

---

### 亮点三：实现交互式 CLI 与 Web 控制台，构建完整的工程化使用工具链

- **交互式 CLI**（约 2,200 行）：基于 Rich 终端框架实现了 13 个 slash 命令（`/help`、`/info`、`/tools`、`/cd`、`/history`、`/save`、`/load`、`/compact` 等），覆盖会话持久化与恢复、环境感知与工作目录导航、历史查看与上下文手动压缩等功能，支持彩色终端与纯文本两种模式。

- **Web 控制台**（约 2,000 行）：基于 Python 标准库实现了零依赖 Web 服务——通过 SSE 实时推送 Agent 执行事件流，支持 vLLM 进程远程启停与模型切换，支持 benchmark 远程触发并流式返回结果。前端提供一步式操作：选择模型 → 启动 vLLM → 输入任务 → 实时观察执行过程。

---

### 发现 Qwen3.5 模型工具调用 Bug 并向 vLLM 提交 PR

使用 Qwen3.5 进行评测时，发现同时启用 `--reasoning-parser qwen3` 和 `--tool-call-parser qwen3_coder` 时，模型会将工具调用嵌入到 `<think>...</think>` 标签内部而非放到正确位置，导致 vLLM 无法解析到有效工具调用，模型表现为卡死。定位根因后向 vLLM 提交了修复 PR（https://github.com/vllm-project/vllm/pull/39055），通过在 `qwen3_reasoning_parser` 中将 reasoning 内的有效 XML tool-call block 提升到 content 中解决。

---

### 实测效果

- 能够在真实代码仓库中自主完成代码阅读、搜索定位、精准编辑、命令执行和任务规划的完整闭环
- 在 Gemma-4-31B-IT 上完成系统性评测，AIME 2024/2025/2026 共 90 题达成 87/90 正确率（24: 29/30, 25: 28/30, 26: 30/30）
- 双层安全沙箱有效拦截 AI 生成代码中的危险操作，评测环境零安全事故
- 在多模型（Gemma、Qwen3.5 等）上验证了框架的基座模型兼容性
