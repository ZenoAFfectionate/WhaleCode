"""Test bootstrap: expose the local ``code/`` tree as the ``hello_agents`` package.

Mirrors ``run_cli.bootstrap_package`` so tests can ``import hello_agents.*``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"

if "hello_agents" not in sys.modules:
    pkg = types.ModuleType("hello_agents")
    pkg.__path__ = [str(CODE_DIR)]
    pkg.__file__ = str(CODE_DIR / "__init__.py")
    sys.modules["hello_agents"] = pkg

# ── Orchestra / Roles / Review 共享 fixture ──

import json
import time
from typing import List
from unittest.mock import MagicMock

import pytest

from hello_agents.core.llm import HelloAgentsLLM


@pytest.fixture
def mock_llm():
    """MagicMock LLM — 仅用于满足构造签名 (角色/隔离测试不触发真实调用)."""
    llm = MagicMock(spec=HelloAgentsLLM)
    llm.model = "test-model"
    return llm


@pytest.fixture
def plan_llm(mock_llm):
    """Mock llm.invoke 返回预制 plan JSON (decompose 测试)."""
    response = MagicMock()
    response.content = json.dumps(
        {
            "subtasks": [
                {"id": "exp-1", "description": "explore the repo", "role": "explorer", "dependencies": []},
            ],
            "mode": "hybrid",
            "stages": [["exp-1"]],
        }
    )
    mock_llm.invoke.return_value = response
    return mock_llm


class StubSubAgent:
    """假子Agent: run() 立即返回预制文本, 并提供元数据/摘要方法 (Orchestra 调度测试)."""

    def __init__(self, result_text: str = "stub-result", delay: float = 0.0):
        self.result_text = result_text
        self.delay = delay
        self.run_prompts: List[str] = []

    def run(self, prompt: str, **kwargs) -> str:
        self.run_prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        return self.result_text

    def _get_subagent_metadata(self, duration, error):
        metadata = {
            "steps": 1,
            "tokens": 10,
            "duration_seconds": round(duration, 2),
            "tools_used": [],
        }
        if error:
            metadata["error"] = error
        return metadata

    def _generate_subagent_summary(self, task, result, metadata):
        return f"任务: {task}\n结果: {result}"


@pytest.fixture
def stub_subagent_factory(monkeypatch):
    """monkeypatch AgentOrchestra._create_subagent → 返回可控 StubSubAgent.

    用法: created = stub_subagent_factory(orchestra, factory=lambda role: StubSubAgent(...))
    """
    created: List[StubSubAgent] = []

    def _install(orchestra, factory=None):
        def _create(role_name: str):
            stub = factory(role_name) if factory else StubSubAgent()
            created.append(stub)
            return stub

        monkeypatch.setattr(orchestra, "_create_subagent", _create)
        return created

    return _install
