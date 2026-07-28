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
