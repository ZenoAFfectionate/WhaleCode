"""subagent (AgentOrchestra) 最小示例 — 主从式多智能体编排.

用法::

    python scripts/orchestra_demo.py "你的任务描述"

不传任务时使用内置示例任务。需要 vLLM 已运行（默认 http://localhost:8000/v1）。
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# 与 scripts/cli.py 相同的包引导方式
pkg = types.ModuleType("hello_agents")
pkg.__path__ = [str(PROJECT_ROOT / "code")]
pkg.__file__ = str(PROJECT_ROOT / "code" / "__init__.py")
sys.modules["hello_agents"] = pkg

from hello_agents.agents.code_agent import CodeAgent  # noqa: E402
from hello_agents.agents.orchestra import AgentOrchestra  # noqa: E402
from hello_agents.core.config import Config  # noqa: E402
from hello_agents.core.llm import HelloAgentsLLM  # noqa: E402
from hello_agents.tools.registry import ToolRegistry  # noqa: E402

DEFAULT_TASK = (
    "探索 /home/kemove/CodeingAgent/WhaleCode 项目的目录结构，"
    "输出一份简洁的代码结构报告（只读操作即可）。"
)


async def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    print(f"任务: {task}\n")

    config = Config.from_env()
    llm = HelloAgentsLLM()
    registry = ToolRegistry(config=config, verbose=False)
    workspace = PROJECT_ROOT
    agent = CodeAgent(
        name="main-agent",
        llm=llm,
        tool_registry=registry,
        project_root=str(workspace),
        working_dir=str(workspace),
        config=config,
        max_steps=20,
        register_default_tools=True,
    )

    orchestra = AgentOrchestra(agent)

    # 1) 分解任务 → 执行计划（LLM 生成，可能降级为单个 explorer 子任务）
    plan = await orchestra.decompose(task)
    print(f"执行计划: mode={plan.mode.value}, {len(plan.subtasks)} 个子任务")
    for i, st in enumerate(plan.subtasks, 1):
        print(f"  [{i}] [{st.role}] {st.id}: {st.description[:80]}")

    # 2) 调度子 agent 执行
    results = await orchestra.execute(plan)
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"\n=== 子任务 {r.subtask_id} [{r.metadata.get('role', '?')}] {status} ===")
        print((r.summary or r.full_result or "(无输出)")[:400])

    # 3) 汇总为最终答案
    answer = await orchestra.aggregate(plan, results)
    print(f"\n=== 最终汇总 ===\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
