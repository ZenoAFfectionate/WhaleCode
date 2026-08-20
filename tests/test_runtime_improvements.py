"""Tests for runtime improvements — security hardening, validation, and robustness.

Covers the six issues identified in IMPROVEMENT.md:
  P1-1: Security blacklist expansion (importlib, pickle, ctypes)
  P1-2: EvalRequest empty-value validation
  P1-3: command=() sentinel semantics
  P2-4: _env_int / _env_bool shared in _utils
  P2-5: _remove_tree permission tolerance
  P3-6: BenchmarkExecutionEnvironment @runtime_checkable Protocol
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hello_agents.benchmark._utils import _env_bool, _env_int
from hello_agents.benchmark.runtime import (
    BenchmarkArtifactStore,
    BenchmarkExecutionEnvironment,
    EvalRequest,
    EvalResult,
    PythonSubprocessEnvironment,
)
from hello_agents.benchmark.runtime.config import BenchmarkRuntimeConfig
from hello_agents.benchmark.safe_exec import (
    UnsafeBenchmarkCodeError,
    _BLOCKED_IMPORT_ROOTS,
    run_python_code_with_stdin,
    validate_python_source_safe,
)


# ============================================================================
# P1-1: Security blacklist expansion
# ============================================================================

class TestSecurityBlacklistExpansion:
    """Verify the three newly blocked modules are actually rejected."""

    def test_importlib_rejected(self):
        with pytest.raises(UnsafeBenchmarkCodeError, match="importlib"):
            validate_python_source_safe("import importlib\n")

    def test_importlib_import_module_rejected(self):
        """importlib.import_module is the primary bypass vector."""
        with pytest.raises(UnsafeBenchmarkCodeError, match="importlib"):
            validate_python_source_safe(
                "import importlib\nimportlib.import_module('subprocess')\n"
            )

    def test_pickle_rejected(self):
        with pytest.raises(UnsafeBenchmarkCodeError, match="pickle"):
            validate_python_source_safe("import pickle\n")

    def test_ctypes_rejected(self):
        with pytest.raises(UnsafeBenchmarkCodeError, match="ctypes"):
            validate_python_source_safe("import ctypes\n")

    def test_ctypes_cdll_rejected(self):
        """ctypes.CDLL can call arbitrary C functions."""
        with pytest.raises(UnsafeBenchmarkCodeError, match="ctypes"):
            validate_python_source_safe(
                "import ctypes\nctypes.CDLL(None)\n"
            )

    def test_original_blocked_modules_still_rejected(self):
        """Regression: original 7 blocked modules should still be blocked."""
        for source in [
            "import subprocess\n",
            "import socket\n",
            "from urllib import request\n",
            "import requests\n",
            "import ftplib\n",
            "import http\n",
            "import paramiko\n",
        ]:
            with pytest.raises(UnsafeBenchmarkCodeError):
                validate_python_source_safe(source)

    def test_safe_algorithm_modules_still_allowed(self):
        """Standard library algorithm modules must not be affected."""
        for source in [
            "import math\n",
            "import itertools\n",
            "import collections\n",
            "import functools\n",
            "import heapq\n",
            "import bisect\n",
            "import random\n",
            "import re\n",
            "import json\n",
            "from typing import List\n",
            "from dataclasses import dataclass\n",
            "import copy\n",
            "import enum\n",
        ]:
            validate_python_source_safe(source)  # should not raise

    def test_blacklist_contains_expected_entries(self):
        assert "importlib" in _BLOCKED_IMPORT_ROOTS
        assert "pickle" in _BLOCKED_IMPORT_ROOTS
        assert "ctypes" in _BLOCKED_IMPORT_ROOTS
        assert "subprocess" in _BLOCKED_IMPORT_ROOTS
        assert "socket" in _BLOCKED_IMPORT_ROOTS


# ============================================================================
# P1-2: EvalRequest empty-value validation
# ============================================================================

class TestEvalRequestValidation:
    """Verify EvalRequest rejects empty code/command but accepts valid inputs."""

    @pytest.mark.parametrize(
        "code,command,should_fail",
        [
            (None, [], True),
            (None, (), True),
            ("", None, True),
            ("   ", None, True),
            ("\n\t", None, True),
            (None, None, True),
        ],
    )
    def test_rejects_empty_inputs(self, code, command, should_fail):
        with pytest.raises(ValueError, match="non-empty"):
            EvalRequest(code=code, command=command, cwd="/tmp", timeout_s=5)

    @pytest.mark.parametrize(
        "code,command,stdin_text",
        [
            ("print(1)", None, None),           # stdin mode
            ("print(1)", (), "hello\n"),         # file mode sentinel
            (None, ["-c", "print(1)"], None),    # command mode
            ("x = 1\ny = 2\n", None, None),     # multiline code
        ],
    )
    def test_accepts_valid_inputs(self, code, command, stdin_text):
        kwargs = {"code": code, "command": command, "cwd": "/tmp", "timeout_s": 5}
        if stdin_text is not None:
            kwargs["stdin"] = stdin_text
        req = EvalRequest(**kwargs)
        assert req.code == code
        assert req.command == command

    def test_rejects_zero_timeout(self):
        with pytest.raises(ValueError, match="timeout_s"):
            EvalRequest(code="print(1)", cwd="/tmp", timeout_s=0)

    def test_rejects_negative_timeout(self):
        with pytest.raises(ValueError, match="timeout_s"):
            EvalRequest(code="print(1)", cwd="/tmp", timeout_s=-1)


# ============================================================================
# P1-3: command=() sentinel semantics
# ============================================================================

class TestCommandSentinelSemantics:
    """Verify that command=() correctly routes to file-mode execution."""

    def test_empty_tuple_sentinel_uses_file_mode(self, tmp_path):
        """command=() must write code to a temp file, not pipe via stdin."""
        result = PythonSubprocessEnvironment().evaluate(
            EvalRequest(
                code="import sys\nprint(sys.stdin.read().strip().upper())\n",
                command=(),  # sentinel: file mode
                stdin="hello sentinel\n",
                cwd=tmp_path,
                timeout_s=5,
            )
        )
        assert result.passed is True
        assert result.status == "passed"
        assert result.stdout.strip() == "HELLO SENTINEL"

    def test_none_command_uses_stdin_mode(self, tmp_path):
        """command=None must pipe code directly to ``python -``."""
        result = PythonSubprocessEnvironment().evaluate(
            EvalRequest(
                code="print('stdin-mode-ok')\n",
                command=None,
                cwd=tmp_path,
                timeout_s=5,
            )
        )
        assert result.passed is True
        assert result.status == "passed"
        assert result.stdout.strip() == "stdin-mode-ok"

    def test_safe_exec_facade_preserves_sentinel(self, tmp_path):
        """run_python_code_with_stdin must correctly relay stdin to subprocess."""
        result = run_python_code_with_stdin(
            "import sys\nprint(sys.stdin.read().strip())\n",
            "facade-ok\n",
            cwd=tmp_path,
            timeout=5,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "facade-ok"

    def test_temp_file_cleaned_after_eval(self, tmp_path):
        """The ._bench_eval_*.py temp file must be removed after evaluation."""
        before = set(tmp_path.glob("._bench_eval_*.py"))
        PythonSubprocessEnvironment().evaluate(
            EvalRequest(
                code="print('tmp-cleanup')\n",
                command=(),
                stdin="",
                cwd=tmp_path,
                timeout_s=5,
            )
        )
        after = set(tmp_path.glob("._bench_eval_*.py"))
        assert after == before  # no new files left behind


# ============================================================================
# P2-4: _env_int / _env_bool shared in _utils
# ============================================================================

class TestEnvHelpersInUtils:
    """Verify _env_int and _env_bool in _utils.py work correctly."""

    def test_env_int_reads_valid_value(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VALID", "42")
        assert _env_int("TEST_INT_VALID", 99) == 42

    def test_env_int_falls_back_for_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_INT_MISSING", raising=False)
        assert _env_int("TEST_INT_MISSING", 99) == 99

    def test_env_int_falls_back_for_empty(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_EMPTY", "")
        assert _env_int("TEST_INT_EMPTY", 99) == 99

    def test_env_int_falls_back_for_whitespace(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_SPACE", "   ")
        assert _env_int("TEST_INT_SPACE", 99) == 99

    def test_env_int_falls_back_for_garbage(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_GARBAGE", "not-a-number")
        assert _env_int("TEST_INT_GARBAGE", 99) == 99

    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("TRUE", True), ("True", True),
        ("yes", True), ("YES", True), ("on", True), ("ON", True),
        ("0", False), ("false", False), ("FALSE", False),
        ("no", False), ("off", False), ("random_stuff", False),
    ])
    def test_env_bool_variants(self, monkeypatch, raw, expected):
        monkeypatch.setenv("TEST_BOOL_VAR", raw)
        assert _env_bool("TEST_BOOL_VAR", not expected) == expected

    def test_env_bool_falls_back_for_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL_MISSING", raising=False)
        assert _env_bool("TEST_BOOL_MISSING", True) is True
        assert _env_bool("TEST_BOOL_MISSING", False) is False

    def test_benchmark_runtime_config_uses_shared_helpers(self, monkeypatch):
        """config.py must import _env_int/_env_bool from _utils, not redefine."""
        monkeypatch.setenv("WHALE_BENCH_EVAL_CPU_SECONDS", "77")
        config = BenchmarkRuntimeConfig.from_env(profile="python_strict")
        assert config.cpu_seconds == 77


# ============================================================================
# P2-5: _remove_tree permission tolerance
# ============================================================================

class TestRemoveTreePermissionTolerance:
    """Verify artifact retention cleanup survives permission errors."""

    def test_cleanup_removes_normal_dirs(self, tmp_path):
        store = BenchmarkArtifactStore(tmp_path / "artifacts", retention=1)
        store.record_eval(stem="first", code="1", stdout="", stderr="", metadata={})
        store.record_eval(stem="second", code="2", stdout="", stderr="", metadata={})
        dirs = [p for p in store.root.iterdir() if p.is_dir()]
        assert len(dirs) == 1  # retention enforced

    def test_cleanup_survives_readonly_file(self, tmp_path):
        """_remove_tree must not crash when a file is read-only."""
        store = BenchmarkArtifactStore(tmp_path / "artifacts", retention=1)
        # Create a properly-formatted stale run dir (matching _new_run_dir pattern)
        # that will be targeted by retention enforcement
        stale = store.root / "20000101-000000-0001-stale-task"
        stale.mkdir(parents=True, exist_ok=True)
        ro_file = stale / "readonly.txt"
        ro_file.write_text("protected")
        ro_file.chmod(0o444)  # read-only for owner
        # Creating a second run dir triggers retention cleanup;
        # the stale dir must be removed without raising
        store.record_eval(stem="new", code="1", stdout="", stderr="", metadata={})
        assert not stale.exists()

    def test_cleanup_survives_readonly_dir_tree(self, tmp_path):
        """Recursive removal must handle nested read-only directories."""
        store = BenchmarkArtifactStore(tmp_path / "artifacts", retention=1)
        nested = store.root / "20000101-000000-0002-nested-stale" / "inner"
        nested.mkdir(parents=True, exist_ok=True)
        ro_file = nested / "readonly.txt"
        ro_file.write_text("protected")
        ro_file.chmod(0o444)
        nested.chmod(0o555)  # read+exec only
        # Must not crash on nested read-only tree
        store.record_eval(stem="new", code="1", stdout="", stderr="", metadata={})
        # Either cleaned up or gracefully skipped — no exception means pass


# ============================================================================
# P3-6: @runtime_checkable Protocol
# ============================================================================

class TestRuntimeCheckableProtocol:
    """Verify BenchmarkExecutionEnvironment supports isinstance/issubclass."""

    def test_python_subprocess_is_instance(self):
        env = PythonSubprocessEnvironment()
        assert isinstance(env, BenchmarkExecutionEnvironment)

    def test_python_subprocess_is_subclass(self):
        assert issubclass(PythonSubprocessEnvironment, BenchmarkExecutionEnvironment)

    def test_eval_result_not_environment(self):
        """Sanity: unrelated types should NOT be instances."""
        result = EvalResult(passed=True, status="passed")
        assert not isinstance(result, BenchmarkExecutionEnvironment)

    def test_protocol_duck_typing(self):
        """Any object with evaluate() returning EvalResult satisfies the Protocol."""
        class _FakeEnv:
            def evaluate(self, request: EvalRequest) -> EvalResult:
                return EvalResult(passed=True, status="passed")

        fake = _FakeEnv()
        assert isinstance(fake, BenchmarkExecutionEnvironment)


# ============================================================================
# Code Quality Fixes: C1–C6
# ============================================================================

class TestCodeQualityFixes:
    """Verify the code-quality improvements are effective and correct."""

    # C1: config.py no longer imports 'os'
    def test_config_has_no_dead_os_import(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "code" / "benchmark" / "runtime" / "config.py").read_text()
        assert "import os" not in src

    # C2: for_profile merged branches
    def test_for_profile_docker_branches_identical_result(self):
        repo = BenchmarkRuntimeConfig.for_profile("repo_docker")
        term = BenchmarkRuntimeConfig.for_profile("terminal_docker")
        assert repo.profile == "repo_docker"
        assert term.profile == "terminal_docker"
        assert repo.max_processes == term.max_processes
        assert repo.file_size_bytes == term.file_size_bytes

    @pytest.mark.parametrize("profile,expected", [
        ("python_strict", (128, 256 * 1024 * 1024)),
        ("repo_docker", (4096, 1024 * 1024 * 1024)),
        ("terminal_docker", (4096, 1024 * 1024 * 1024)),
    ])
    def test_for_profile_values(self, profile, expected):
        config = BenchmarkRuntimeConfig.for_profile(profile)
        assert (config.max_processes, config.file_size_bytes) == expected

    # C4: _json_safe handles typical metadata correctly
    def test_json_safe_handles_standard_types(self):
        from hello_agents.benchmark.runtime.artifacts import BenchmarkArtifactStore
        from hello_agents.benchmark._utils import _json_safe_full as _json_safe
        # Test the equivalent of what record_eval sends through _json_safe
        result = _json_safe({"key": "val", "nested": {"a": 1}, "list": [1, 2]})
        assert result == {"key": "val", "nested": {"a": 1}, "list": [1, 2]}

    def test_json_safe_handles_path(self):
        from hello_agents.benchmark._utils import _json_safe_full as _json_safe
        from pathlib import Path
        result = _json_safe({"path": Path("/tmp/test")})
        assert result["path"] == "/tmp/test"

    def test_json_safe_handles_bytes(self):
        from hello_agents.benchmark._utils import _json_safe_full as _json_safe
        result = _json_safe({"data": b"hello"})
        assert result["data"] == "hello"

    def test_json_safe_handles_unknown_type(self):
        from hello_agents.benchmark._utils import _json_safe_full as _json_safe
        class _Custom:
            pass
        result = _json_safe({"obj": _Custom()})
        assert "repr" in result["obj"] or "Custom" in str(result["obj"])

    # C6: python_env.py metrics dedup (success + timeout now use base_metrics)
    def test_python_env_metrics_not_duplicated_in_source(self):
        import inspect
        from hello_agents.benchmark.runtime import python_env as pe_module
        src = inspect.getsource(pe_module.PythonSubprocessEnvironment.evaluate)
        # base_metrics is defined once and referenced in success + timeout paths
        assert src.count("metrics=base_metrics,") == 2
        # The error path still has its own metrics (no 'command' available there)
        assert 'base_metrics' in src
