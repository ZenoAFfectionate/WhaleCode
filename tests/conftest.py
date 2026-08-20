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

# ── Roles / Review / Task tool 共享 fixture ──

from unittest.mock import MagicMock

import pytest

from hello_agents.core.llm import HelloAgentsLLM


@pytest.fixture
def mock_llm():
    """MagicMock LLM — 仅用于满足构造签名 (角色/隔离测试不触发真实调用)."""
    llm = MagicMock(spec=HelloAgentsLLM)
    llm.model = "test-model"
    return llm
