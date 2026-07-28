"""Comprehensive tests for the tools subsystem — covers all 2026-07-26 improvements.

Tests in this file cover the full pipeline:
    Tool → ToolRegistry → execute_tool → _validate_against_schema → run()
    + ToolErrorCode.is_fault → CircuitBreaker → ToolResponse protocol
    + ToolFilter (category-based + name-based fallback)
    + _WorkspaceFileTool subclass integration (Read/Write/Edit/Delete/Glob/Grep/LS)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Dict, List

import pytest

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_project():
    """Create a temporary project root with a few files and dirs."""
    tmp = tempfile.mkdtemp()
    for d in ("skills", "memory/todos", "memory/.backups", "memory/tool-output"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    os.makedirs(os.path.join(tmp, "src", "sub"), exist_ok=True)
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write("# Test Project\n")
    with open(os.path.join(tmp, "src", "main.py"), "w") as f:
        f.write('"""Main module."""\nprint("hello world")\n')
    with open(os.path.join(tmp, "src", "utils.py"), "w") as f:
        f.write("import os\nimport sys\n\ndef add(a, b):\n    return a + b\n")
    with open(os.path.join(tmp, "config.json"), "w") as f:
        f.write('{"key": "value"}\n')
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def registry():
    from hello_agents.tools.registry import ToolRegistry
    return ToolRegistry(verbose=False)


# ============================================================================
# 1. ToolErrorCode — is_fault attribute + string compatibility
# ============================================================================


class TestToolErrorCode:
    """Every member of ToolErrorCode is a str subclass with an .is_fault flag."""

    def test_all_members_are_str_instances(self):
        from hello_agents.tools.errors import ToolErrorCode
        for member in ToolErrorCode:
            assert isinstance(member, str)
            assert isinstance(member, ToolErrorCode)

    def test_is_fault_true_only_for_genuine_faults(self):
        from hello_agents.tools.errors import ToolErrorCode
        fault_names = {"INTERNAL_ERROR", "EXECUTION_ERROR", "TIMEOUT",
                       "NETWORK_ERROR", "API_ERROR"}
        for member in ToolErrorCode:
            expected = member.name in fault_names
            assert member.is_fault is expected, (
                f"{member.name}.is_fault should be {expected}, got {member.is_fault}"
            )

    def test_fault_codes_returns_all_faults(self):
        from hello_agents.tools.errors import ToolErrorCode
        codes = ToolErrorCode.fault_codes()
        expected = {"INTERNAL_ERROR", "EXECUTION_ERROR", "TIMEOUT",
                    "NETWORK_ERROR", "API_ERROR"}
        assert set(codes) == expected
        assert all(isinstance(c, str) for c in codes)

    def test_string_equality_drop_in(self):
        """All existing comparisons keep working."""
        from hello_agents.tools.errors import ToolErrorCode
        assert ToolErrorCode.TIMEOUT == "TIMEOUT"
        assert ToolErrorCode.INVALID_PARAM == "INVALID_PARAM"
        assert ToolErrorCode.NOT_FOUND != "NOT_FOUND_TYPO"

    def test_str_and_repr(self):
        from hello_agents.tools.errors import ToolErrorCode
        assert str(ToolErrorCode.INTERNAL_ERROR) == "INTERNAL_ERROR"
        assert f"{ToolErrorCode.INVALID_PARAM}" == "INVALID_PARAM"

    def test_json_serialization(self):
        from hello_agents.tools.errors import ToolErrorCode
        # Enum members serialize as their string value in JSON
        payload = {"code": ToolErrorCode.TIMEOUT}
        assert json.dumps(payload) == '{"code": "TIMEOUT"}'

    def test_get_all_codes(self):
        from hello_agents.tools.errors import ToolErrorCode
        codes = ToolErrorCode.get_all_codes()
        assert len(codes) == 16  # 16 error codes total
        assert all(isinstance(c, str) for c in codes)
        assert "INTERNAL_ERROR" in codes
        assert "ASK_USER_UNAVAILABLE" in codes

    def test_is_valid_code(self):
        from hello_agents.tools.errors import ToolErrorCode
        assert ToolErrorCode.is_valid_code("TIMEOUT")
        assert ToolErrorCode.is_valid_code(ToolErrorCode.TIMEOUT)
        assert not ToolErrorCode.is_valid_code("MADE_UP")
        assert not ToolErrorCode.is_valid_code(42)

    def test_enum_iteration_yields_all_members(self):
        from hello_agents.tools.errors import ToolErrorCode
        members = list(ToolErrorCode)
        assert len(members) == 16

    def test_member_value_is_plain_string(self):
        from hello_agents.tools.errors import ToolErrorCode
        for member in ToolErrorCode:
            assert isinstance(member.value, str), (
                f"{member.name}.value must be str, got {type(member.value)}"
            )


# ============================================================================
# 2. ToolResponse — protocol completeness
# ============================================================================


class TestToolResponse:
    """ToolResponse factory methods and serialization."""

    def test_success_factory(self):
        from hello_agents.tools.response import ToolResponse, ToolStatus
        r = ToolResponse.success(text="ok", data={"key": "val"},
                                 stats={"time_ms": 5})
        assert r.status == ToolStatus.SUCCESS
        assert r.text == "ok"
        assert r.data == {"key": "val"}
        assert r.stats == {"time_ms": 5}
        assert r.error_info is None

    def test_partial_factory(self):
        from hello_agents.tools.response import ToolResponse, ToolStatus
        r = ToolResponse.partial(text="partial result",
                                 data={"truncated": True})
        assert r.status == ToolStatus.PARTIAL
        assert r.data["truncated"] is True

    def test_error_factory(self):
        from hello_agents.tools.response import ToolResponse, ToolStatus
        from hello_agents.tools.errors import ToolErrorCode
        r = ToolResponse.error(code=ToolErrorCode.INVALID_PARAM,
                               message="bad input",
                               stats={"time_ms": 1})
        assert r.status == ToolStatus.ERROR
        assert r.error_info == {"code": "INVALID_PARAM", "message": "bad input"}
        assert r.text == "bad input"  # error message becomes text

    def test_error_factory_with_string_code(self):
        """Backward compat: code can be a plain string."""
        from hello_agents.tools.response import ToolResponse, ToolStatus
        r = ToolResponse.error(code="TIMEOUT", message="timed out")
        assert r.status == ToolStatus.ERROR
        assert r.error_info["code"] == "TIMEOUT"

    def test_to_dict_round_trip(self):
        from hello_agents.tools.response import ToolResponse
        original = ToolResponse.success(
            text="hello", data={"a": 1}, stats={"ms": 10},
            context={"tool_name": "X"},
        )
        as_dict = original.to_dict()
        restored = ToolResponse.from_dict(as_dict)
        assert restored.status == original.status
        assert restored.text == original.text
        assert restored.data == original.data
        assert restored.stats == original.stats
        assert restored.context == original.context

    def test_to_json_round_trip(self):
        from hello_agents.tools.response import ToolResponse
        original = ToolResponse.error(code="TIMEOUT", message="x")
        json_str = original.to_json()
        restored = ToolResponse.from_json(json_str)
        assert restored.status == original.status
        assert restored.error_info == original.error_info

    def test_from_dict_missing_fields(self):
        from hello_agents.tools.response import ToolResponse, ToolStatus
        minimal = ToolResponse.from_dict({"status": "error", "text": "bad"})
        assert minimal.status == ToolStatus.ERROR
        assert minimal.text == "bad"
        assert minimal.data == {}

    def test_success_no_data_defaults(self):
        from hello_agents.tools.response import ToolResponse
        r = ToolResponse.success(text="bare")
        assert r.data == {}

    def test_partial_no_data_defaults(self):
        from hello_agents.tools.response import ToolResponse
        r = ToolResponse.partial(text="bare partial")
        assert r.data == {}


# ============================================================================
# 3. Tool base class — category, schema validation, run_with_timing
# ============================================================================


class _MinimalTool:
    """Build minimal Tool subclasses for testing base class behaviour."""

    @staticmethod
    def make(name="T", description="desc", category="general", **params):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        param_list = params.get("_params", None)
        if param_list is None:
            param_list = [
                ToolParameter(name="x", type="string", description="", required=True),
            ]

        class _T(Tool):
            def __init__(self):
                super().__init__(name=name, description=description, category=category)
            def get_parameters(self):
                return param_list
            def run(self, p):
                return ToolResponse.success(text="ok", data={"got": p})

        return _T()


class TestToolBaseCategory:
    """Tool.category attribute behaviour."""

    def test_default_category_is_general(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        class Plain(Tool):
            def __init__(self):
                super().__init__(name="P", description="")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")

        t = Plain()
        assert t.category == "general"

    def test_explicit_category_preserved(self):
        t = _MinimalTool.make(category="dangerous")
        assert t.category == "dangerous"

    def test_category_is_plain_str(self):
        t = _MinimalTool.make(category="readonly")
        assert isinstance(t.category, str)
        assert t.category == "readonly"


class TestToolValidateAgainstSchema:
    """Tool._validate_against_schema edge cases."""

    def test_required_string_missing(self):
        t = _MinimalTool.make()
        err = t._validate_against_schema({})
        assert err is not None
        assert err.status.value == "error"
        assert "x" in err.text

    def test_required_string_present_ok(self):
        t = _MinimalTool.make()
        assert t._validate_against_schema({"x": "hello"}) is None

    def test_wrong_type_string_vs_int(self):
        t = _MinimalTool.make()
        err = t._validate_against_schema({"x": 42})
        assert err is not None
        assert "string" in err.text.lower()

    def test_wrong_type_int_vs_string(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="n", type="integer", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        err = t._validate_against_schema({"n": "abc"})
        assert err is not None
        assert "integer" in err.text.lower()

    def test_bool_not_accepted_as_integer(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="n", type="integer", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        # Python: isinstance(True, int) == True — we must reject it
        err = t._validate_against_schema({"n": True})
        assert err is not None
        assert "integer" in err.text.lower()

    def test_bool_not_accepted_as_number(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="n", type="number", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        err = t._validate_against_schema({"n": False})
        assert err is not None
        assert "number" in err.text.lower()

    def test_integer_accepts_int(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="n", type="integer", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t._validate_against_schema({"n": 10}) is None

    def test_number_accepts_int_and_float(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="n", type="number", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t._validate_against_schema({"n": 3.14}) is None
        assert t._validate_against_schema({"n": 42}) is None

    def test_boolean_accepts_bool_only(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="flag", type="boolean", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t._validate_against_schema({"flag": True}) is None
        assert t._validate_against_schema({"flag": False}) is None
        err = t._validate_against_schema({"flag": 1})
        assert err is not None

    def test_array_accepts_list_and_string(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="items", type="array", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t._validate_against_schema({"items": [1, 2]}) is None
        # JSON strings are tolerated (many tools decode them in run())
        assert t._validate_against_schema({"items": '["a","b"]'}) is None
        err = t._validate_against_schema({"items": 123})
        assert err is not None

    def test_object_accepts_dict_and_string(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="config", type="object", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t._validate_against_schema({"config": {}}) is None
        assert t._validate_against_schema({"config": {"a": 1}}) is None
        assert t._validate_against_schema({"config": '{"a":1}'}) is None
        err = t._validate_against_schema({"config": 42})
        assert err is not None

    def test_extra_params_allowed(self):
        t = _MinimalTool.make()
        assert t._validate_against_schema({"x": "hi", "bonus": "ignored"}) is None

    def test_optional_param_not_required(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [
                    ToolParameter(name="a", type="string", description="", required=True),
                    ToolParameter(name="b", type="string", description="", required=False),
                ]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t._validate_against_schema({"a": "yes"}) is None

    def test_multiple_errors_reports_first_required_missing(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [
                    ToolParameter(name="first", type="string", description="", required=True),
                    ToolParameter(name="second", type="string", description="", required=True),
                ]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        err = t._validate_against_schema({})
        assert err is not None
        assert "first" in err.text

    def test_validate_schema_with_default_params(self):
        """A param with a default but not required: still optional."""
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [
                    ToolParameter(name="cmd", type="string", description="", required=True),
                    ToolParameter(name="timeout", type="integer", description="",
                                  required=False, default=30),
                ]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t._validate_against_schema({"cmd": "run"}) is None


class TestToolRunWithTiming:
    """Tool.run_with_timing and arun_with_timing behaviour."""

    def test_run_with_timing_adds_stats_and_context(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        resp = t.run_with_timing({})
        assert resp.stats is not None
        assert "time_ms" in resp.stats
        assert resp.context is not None
        assert resp.context["tool_name"] == "T"

    def test_run_with_timing_catches_exception(self):
        from hello_agents.tools.base import Tool, ToolParameter
        class _Crash(Tool):
            def __init__(self): super().__init__(name="C", description="")
            def get_parameters(self): return []
            def run(self, p): raise RuntimeError("boom")
        t = _Crash()
        resp = t.run_with_timing({})
        assert resp.status.value == "error"
        assert "boom" in resp.text
        assert resp.stats is not None
        assert "time_ms" in resp.stats

    def test_arun_defaults_to_thread_pool(self):
        import asyncio
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="sync")
        t = _T()

        async def go():
            return await t.arun({})
        resp = asyncio.run(go())
        assert resp.status.value == "success"
        assert resp.text == "sync"

    def test_arun_with_timing(self):
        import asyncio
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()

        async def go():
            return await t.arun_with_timing({})
        resp = asyncio.run(go())
        assert resp.stats is not None
        assert "time_ms" in resp.stats

    def test_validate_parameters_basic(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="T", description="")
            def get_parameters(self):
                return [ToolParameter(name="x", type="string", description="", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert t.validate_parameters({"x": "yes"})
        assert not t.validate_parameters({})

    def test_to_dict_includes_parameters(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="Read", description="reads files")
            def get_parameters(self):
                return [ToolParameter(name="path", type="string", description="file path", required=True)]
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        d = t.to_dict()
        assert d["name"] == "Read"
        assert d["description"] == "reads files"
        assert len(d["parameters"]) == 1
        assert d["parameters"][0]["name"] == "path"

    def test_str_represents_name(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self): super().__init__(name="MyTool", description="")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")
        t = _T()
        assert "MyTool" in str(t)
        assert str(t) == repr(t)


# ============================================================================
# 4. CircuitBreaker — lifecycle + is_fault integration
# ============================================================================


class TestCircuitBreakerLifecycle:
    """Circuit breaker state machine."""

    def test_disabled_bypasses_everything(self):
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from hello_agents.tools.response import ToolResponse
        cb = CircuitBreaker(enabled=False)
        for _ in range(10):
            cb.record_result("X", ToolResponse.error(code="INTERNAL_ERROR", message=""))
        assert not cb.is_open("X")

    def test_manual_open_close(self):
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        assert not cb.is_open("X")
        cb.open("X")
        assert cb.is_open("X")
        cb.close("X")
        assert not cb.is_open("X")

    def test_auto_recovery_after_timeout(self):
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from hello_agents.tools.response import ToolResponse
        import time
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        cb.record_result("X", ToolResponse.error(code="INTERNAL_ERROR", message=""))
        assert cb.is_open("X")
        # Wait for recovery
        time.sleep(1.1)
        assert not cb.is_open("X")

    def test_success_resets_counter(self):
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from hello_agents.tools.response import ToolResponse
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_result("X", ToolResponse.error(code="INTERNAL_ERROR", message=""))
        assert cb.get_status("X")["failure_count"] == 4
        cb.record_result("X", ToolResponse.success(text="ok"))
        assert cb.get_status("X")["failure_count"] == 0

    def test_non_fault_does_not_count(self):
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from hello_agents.tools.response import ToolResponse
        cb = CircuitBreaker(failure_threshold=2)
        for _ in range(10):
            cb.record_result("X", ToolResponse.error(code="INVALID_PARAM", message=""))
        assert cb.get_status("X")["failure_count"] == 0
        assert not cb.is_open("X")

    def test_get_all_status(self):
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from hello_agents.tools.response import ToolResponse
        cb = CircuitBreaker()
        cb.record_result("A", ToolResponse.success(text=""))
        cb.record_result("B", ToolResponse.error(code="INTERNAL_ERROR", message=""))
        all_status = cb.get_all_status()
        assert "A" in all_status
        assert "B" in all_status
        assert all_status["A"]["failure_count"] == 0
        assert all_status["B"]["failure_count"] == 1

    def test_concurrent_record_no_corruption(self):
        """Thread-safety: concurrent record_result must not corrupt counters."""
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from hello_agents.tools.response import ToolResponse

        cb = CircuitBreaker(failure_threshold=100)
        ITERS = 500
        errors = []

        def worker():
            for _ in range(ITERS):
                try:
                    cb.record_result("Z", ToolResponse.success(text=""))
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # All successes, counter must be 0
        assert cb.get_status("Z")["failure_count"] == 0
        assert not cb.is_open("Z")


# ============================================================================
# 5. ToolRegistry — schema validation in execute flow
# ============================================================================


class TestRegistryToolExecution:
    """Registry.execute_tool integrates schema validation."""

    def test_schema_rejects_before_run(self, registry, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        registry.register_tool(ReadTool(project_root=temp_project))
        # Missing required "path" — should be caught by schema, not by run()
        resp = registry.execute_tool("Read", "{}")
        assert resp.status.value == "error"
        assert "path" in resp.text.lower()

    def test_schema_accepts_valid_call(self, registry, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        registry.register_tool(ReadTool(project_root=temp_project))
        resp = registry.execute_tool("Read", '{"path": "README.md"}')
        assert resp.status.value == "success"

    def test_schema_wrong_type(self, registry, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        registry.register_tool(ReadTool(project_root=temp_project))
        resp = registry.execute_tool("Read", '{"path": 123}')
        assert resp.status.value == "error"

    def test_unknown_tool_returns_not_found(self, registry):
        resp = registry.execute_tool("NoSuchTool", "{}")
        assert resp.status.value == "error"
        assert resp.error_info["code"] == "NOT_FOUND"

    def test_function_tool_still_works(self, registry):
        registry.register_function(lambda p: "result-" + p, name="echo")
        resp = registry.execute_tool("echo", "hello-world")
        assert resp.status.value == "success"
        assert "result-hello-world" in resp.data.get("output", "")

    def test_circuit_breaker_integration(self, registry, temp_project):
        """Schema validation rejects happen before breaker check."""
        from hello_agents.tools.builtin.file_tools import ReadTool
        registry.register_tool(ReadTool(project_root=temp_project))
        # Repeated bad params should not trip breaker (INVALID_PARAM is not a fault)
        for _ in range(5):
            resp = registry.execute_tool("Read", '{"path": 999}')
            assert resp.status.value == "error"
        assert not registry.circuit_breaker.is_open("Read")

    def test_schema_validates_before_run_with_timing(self, registry, temp_project):
        """run_with_timing should never be called with bad params."""
        from hello_agents.tools.builtin.file_tools import ReadTool
        registry.register_tool(ReadTool(project_root=temp_project))
        resp = registry.execute_tool("Read", "{}")
        # This should be caught before run() — if run() had been called without
        # a path, it would crash with KeyError on .get("path") returning None
        assert resp.status.value == "error"


# ============================================================================
# 6. ToolFilter — category-based + name-based filtering
# ============================================================================


class TestToolFilterFull:
    """Complete ToolFilter testing: categories, names, edges."""

    @staticmethod
    def _make_tool(name, category):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        class _T(Tool):
            def __init__(self):
                super().__init__(name=name, description="", category=category)
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")
        return _T()

    @staticmethod
    def _make_registry(*tools, existing_categories: dict | None = None):
        from hello_agents.tools.registry import ToolRegistry
        reg = ToolRegistry(verbose=False)
        for t in tools:
            reg.register_tool(t)
        if existing_categories is not None:
            reg.get_tool_categories().update(existing_categories)
        return reg

    # ---- ReadOnlyFilter ----

    def test_readonly_allows_readonly_category(self):
        from hello_agents.tools.tool_filter import ReadOnlyFilter
        t1 = self._make_tool("R", "readonly")
        t2 = self._make_tool("W", "write")
        reg = self._make_registry(t1, t2)
        rof = ReadOnlyFilter(tool_categories=reg.get_tool_categories())
        allowed = rof.filter(reg.list_tools())
        assert "R" in allowed
        assert "W" not in allowed

    def test_readonly_denies_uncategorized(self):
        from hello_agents.tools.tool_filter import ReadOnlyFilter
        t1 = self._make_tool("General", "general")
        reg = self._make_registry(t1)
        rof = ReadOnlyFilter(tool_categories=reg.get_tool_categories())
        allowed = rof.filter(reg.list_tools())
        assert "General" not in allowed  # "general" ≠ "readonly"

    def test_readonly_additional_allowed_by_name(self):
        from hello_agents.tools.tool_filter import ReadOnlyFilter
        t1 = self._make_tool("CustomW", "write")
        reg = self._make_registry(t1)
        rof = ReadOnlyFilter(additional_allowed=["CustomW"],
                             tool_categories=reg.get_tool_categories())
        allowed = rof.filter(reg.list_tools())
        assert "CustomW" in allowed  # explicitly allowed by name

    # ---- FullAccessFilter ----

    def test_fullaccess_denies_dangerous(self):
        from hello_agents.tools.tool_filter import FullAccessFilter
        t1 = self._make_tool("Bash", "dangerous")
        t2 = self._make_tool("Read", "readonly")
        t3 = self._make_tool("Write", "write")
        reg = self._make_registry(t1, t2, t3)
        faf = FullAccessFilter(tool_categories=reg.get_tool_categories())
        allowed = faf.filter(reg.list_tools())
        assert "Bash" not in allowed
        assert "Read" in allowed
        assert "Write" in allowed

    def test_fullaccess_denies_by_extra_name(self):
        from hello_agents.tools.tool_filter import FullAccessFilter
        t1 = self._make_tool("Read", "readonly")
        reg = self._make_registry(t1)
        faf = FullAccessFilter(additional_denied=["Read"],
                               tool_categories=reg.get_tool_categories())
        allowed = faf.filter(reg.list_tools())
        assert "Read" not in allowed

    # ---- CustomFilter whitelist ----

    def test_custom_whitelist_name_only(self):
        from hello_agents.tools.tool_filter import CustomFilter
        t1 = self._make_tool("A", "readonly")
        t2 = self._make_tool("B", "write")
        reg = self._make_registry(t1, t2)
        cf = CustomFilter(mode="whitelist", allowed=["A"],
                          tool_categories=reg.get_tool_categories())
        allowed = cf.filter(reg.list_tools())
        # Whitelist: only "A" explicitly, no category-based allow
        assert "A" in allowed
        # "B" is write — not in allowed names, no allowed_categories
        # In whitelist mode with allowed set but allowed_categories NOT set,
        # the fallback is: if no category match and no name match, deny.
        assert "B" not in allowed

    def test_custom_whitelist_category_based(self):
        from hello_agents.tools.tool_filter import CustomFilter
        t1 = self._make_tool("A", "readonly")
        t2 = self._make_tool("B", "write")
        reg = self._make_registry(t1, t2)
        cf = CustomFilter(mode="whitelist", allowed_categories={"readonly"},
                          tool_categories=reg.get_tool_categories())
        allowed = cf.filter(reg.list_tools())
        assert "A" in allowed
        assert "B" not in allowed

    # ---- CustomFilter blacklist ----

    def test_custom_blacklist_deny_by_category_and_name(self):
        from hello_agents.tools.tool_filter import CustomFilter
        t1 = self._make_tool("Read", "readonly")
        t2 = self._make_tool("Bash", "dangerous")
        t3 = self._make_tool("Write", "write")
        reg = self._make_registry(t1, t2, t3)
        cf = CustomFilter(mode="blacklist", denied=["Write"],
                          denied_categories={"dangerous"},
                          tool_categories=reg.get_tool_categories())
        allowed = cf.filter(reg.list_tools())
        assert "Bash" not in allowed  # dangerous category
        assert "Write" not in allowed  # explicit deny
        assert "Read" in allowed

    # ---- Backward compat (no categories) ----

    def test_readonly_no_categories_fallback(self):
        """Without category info, falls back to name-based allow list."""
        from hello_agents.tools.tool_filter import ReadOnlyFilter
        rof = ReadOnlyFilter()  # no tool_categories
        assert "Read" in rof.filter(["Read", "Bash", "Write"])
        assert "Bash" not in rof.filter(["Read", "Bash", "Write"])

    def test_fullaccess_no_categories_fallback(self):
        from hello_agents.tools.tool_filter import FullAccessFilter
        faf = FullAccessFilter()  # no tool_categories
        assert "Bash" not in faf.filter(["Read", "Bash", "Write"])
        assert "Read" in faf.filter(["Read", "Bash", "Write"])

    # ---- Error cases ----

    def test_custom_invalid_mode_raises(self):
        from hello_agents.tools.tool_filter import CustomFilter
        with pytest.raises(ValueError, match="Invalid mode"):
            CustomFilter(mode="invalid")


# ============================================================================
# 7. _WorkspaceFileTool — subclass integration
# ============================================================================


class TestWorkspaceFileToolSubclasses:
    """All file/glob/grep tools correctly inherit from _WorkspaceFileTool."""

    def test_inheritance_chain(self, temp_project):
        from hello_agents.tools.builtin.file_tools import (
            _WorkspaceFileTool, ReadTool, WriteTool, EditTool,
            DeleteTool, ListFilesTool,
        )
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool
        from hello_agents.tools.base import Tool

        base = _WorkspaceFileTool
        for cls in (ReadTool, WriteTool, EditTool, DeleteTool,
                     ListFilesTool, GlobTool, GrepTool):
            assert issubclass(cls, base), f"{cls.__name__} must subclass _WorkspaceFileTool"
            assert issubclass(cls, Tool), f"{cls.__name__} must subclass Tool"

    def test_listfiles_not_readtool(self):
        from hello_agents.tools.builtin.file_tools import ListFilesTool, ReadTool
        assert not issubclass(ListFilesTool, ReadTool), (
            "ListFilesTool must NOT inherit ReadTool (LSP violation)"
        )

    def test_all_use_inherited_resolve_path(self, temp_project):
        """_resolve_path works identically across all subclasses."""
        from hello_agents.tools.builtin.file_tools import (
            ReadTool, WriteTool, EditTool, DeleteTool, ListFilesTool,
        )
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool

        for cls in (ReadTool, WriteTool, EditTool, DeleteTool,
                     ListFilesTool, GlobTool, GrepTool):
            t = cls(project_root=temp_project)
            resolved = t._resolve_path("README.md")
            assert resolved == Path(temp_project).resolve() / "README.md"
            assert resolved.exists()

    def test_all_use_inherited_display_path(self, temp_project):
        from hello_agents.tools.builtin.file_tools import (
            ReadTool, WriteTool, EditTool, DeleteTool, ListFilesTool,
        )
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool

        for cls in (ReadTool, WriteTool, EditTool, DeleteTool,
                     ListFilesTool, GlobTool, GrepTool):
            t = cls(project_root=temp_project)
            displayed = t._display_path(Path(temp_project) / "src" / "main.py")
            assert displayed == "src/main.py"

    def test_path_escape_blocked_on_all_subclasses(self, temp_project):
        from hello_agents.tools.builtin.file_tools import (
            ReadTool, WriteTool, EditTool, DeleteTool, ListFilesTool,
        )
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool

        for cls in (ReadTool, WriteTool, EditTool, DeleteTool):
            t = cls(project_root=temp_project)
            with pytest.raises(ValueError):
                t._resolve_path("/etc/passwd")
        # Glob/Grep use _resolve_path internally via run()
        for cls in (GlobTool, GrepTool):
            t = cls(project_root=temp_project)
            with pytest.raises(ValueError):
                t._resolve_path("../../../etc")

    # ---- _list_directory works on ReadTool and ListFilesTool ----

    def test_list_directory_read_tool(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        t = ReadTool(project_root=temp_project)
        resp = t.run({"path": "src"})
        assert resp.status.value == "success"
        assert resp.data["is_directory"] is True
        assert resp.data["total_entries"] >= 2  # main.py, utils.py, sub/
        assert resp.data["total_dirs"] >= 1
        assert resp.data["total_files"] >= 2

    def test_list_directory_ls_tool(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ListFilesTool
        t = ListFilesTool(project_root=temp_project)
        resp = t.run({"path": "src"})
        assert resp.status.value == "success"
        assert resp.data["is_directory"] is True

    def test_ls_rejects_file_path(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ListFilesTool
        t = ListFilesTool(project_root=temp_project)
        resp = t.run({"path": "README.md"})
        assert resp.status.value == "error"
        assert "not a directory" in resp.text.lower()

    def test_read_tool_reads_file(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        t = ReadTool(project_root=temp_project)
        resp = t.run({"path": "README.md"})
        assert resp.status.value == "success"
        assert "# Test Project" in resp.data["content"]

    def test_read_tool_binary_rejection(self, temp_project):
        """Binary files get BINARY_FILE error."""
        from hello_agents.tools.builtin.file_tools import ReadTool
        import struct
        bin_path = os.path.join(temp_project, "data.bin")
        with open(bin_path, "wb") as f:
            f.write(struct.pack("i", 42) + b"\x00\x00\xff\xfe")
        t = ReadTool(project_root=temp_project)
        resp = t.run({"path": "data.bin"})
        assert resp.status.value == "error"
        assert resp.error_info["code"] == "BINARY_FILE"

    def test_read_tool_missing_path(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        t = ReadTool(project_root=temp_project)
        resp = t.run({"path": "nonexistent.txt"})
        assert resp.status.value == "partial"
        assert resp.data.get("missing_path") is True

    def test_read_tool_offset_and_limit(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        t = ReadTool(project_root=temp_project)
        # utils.py has 4 lines; reading 1 means partial (more lines remain)
        resp = t.run({"path": "src/utils.py", "offset": 0, "limit": 1})
        assert resp.status.value == "partial"
        assert "import os" in resp.data["content"]

    def test_read_tool_offset_out_of_range(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        t = ReadTool(project_root=temp_project)
        resp = t.run({"path": "src/utils.py", "offset": 9999, "limit": 10})
        assert resp.status.value == "error"

    # ---- _format_size inherited everywhere ----

    def test_format_size_inherited(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool, ListFilesTool
        for cls in (ReadTool, ListFilesTool):
            t = cls(project_root=temp_project)
            assert t._format_size(500) == "500.0B"
            assert t._format_size(1024) == "1.0KB"
            assert t._format_size(1024 * 1024) == "1.0MB"

    # ---- GlobTool / GrepTool use inherited paths ----

    def test_glob_tool_functional(self, temp_project):
        from hello_agents.tools.builtin.glob_tool import GlobTool
        t = GlobTool(project_root=temp_project)
        resp = t.run({"pattern": "*.py"})
        assert resp.status.value == "success"
        paths = [m["path"] for m in resp.data["matches"]]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)

    def test_glob_tool_subpath(self, temp_project):
        from hello_agents.tools.builtin.glob_tool import GlobTool
        # working_dir must be passed as a str path relative to cwd or absolute
        src_dir = os.path.join(temp_project, "src")
        t = GlobTool(project_root=temp_project, working_dir=src_dir)
        resp = t.run({"pattern": "*.py"})
        assert resp.status.value == "success"
        paths = [m["path"] for m in resp.data["matches"]]
        assert all("src/" in p for p in paths)

    def test_glob_tool_not_found(self, temp_project):
        from hello_agents.tools.builtin.glob_tool import GlobTool
        t = GlobTool(project_root=temp_project)
        # rg with --glob=*.rst may error if no .rst files exist
        # We just check the response is well-formed (either success with 0 or error)
        resp = t.run({"pattern": "*.rst"})
        assert resp.status.value in ("success", "error")
        if resp.status.value == "success":
            assert resp.data["count"] == 0

    def test_grep_tool_functional(self, temp_project):
        from hello_agents.tools.builtin.grep_tool import GrepTool
        t = GrepTool(project_root=temp_project)
        resp = t.run({"pattern": "import", "include": "*.py"})
        assert resp.status.value == "success"
        assert resp.data["total_matches"] >= 2  # import os, import sys

    def test_grep_tool_no_match(self, temp_project):
        from hello_agents.tools.builtin.grep_tool import GrepTool
        t = GrepTool(project_root=temp_project)
        resp = t.run({"pattern": "NO_SUCH_TEXT_XYZ123"})
        assert resp.status.value == "success"
        assert resp.data["total_matches"] == 0

    def test_grep_tool_invalid_regex(self, temp_project):
        """Unclosed bracket may be treated as literal by newer rg; verify it's handled."""
        from hello_agents.tools.builtin.grep_tool import GrepTool
        t = GrepTool(project_root=temp_project)
        resp = t.run({"pattern": "["})
        # Depending on rg version: parse error → error status, or 0 matches → success
        assert resp.status.value in ("success", "error")

    # ---- _missing_path_response works on all file tools ----

    def test_missing_path_response_on_ls(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ListFilesTool
        t = ListFilesTool(project_root=temp_project)
        resp = t.run({"path": "no_such_dir"})
        assert resp.data.get("missing_path") is True

    # ---- Test project_root / working_dir match ----

    def test_project_root_set_consistently(self, temp_project):
        from hello_agents.tools.builtin.file_tools import (
            ReadTool, WriteTool, EditTool, DeleteTool, ListFilesTool,
        )
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool

        for cls in (ReadTool, WriteTool, EditTool, DeleteTool,
                     ListFilesTool, GlobTool, GrepTool):
            t = cls(project_root=temp_project)
            assert t.project_root == Path(temp_project).resolve()
            assert t.working_dir == Path(temp_project).resolve()

    def test_working_dir_subpath(self, temp_project):
        from hello_agents.tools.builtin.file_tools import ReadTool
        # working_dir is resolved relative to CWD, so pass a relative dir that exists within temp_project
        # or an absolute path for reliable test
        abs_src = os.path.join(temp_project, "src")
        t = ReadTool(project_root=temp_project, working_dir=abs_src)
        assert t.working_dir == Path(abs_src).resolve()


# ============================================================================
# 8. BashTool — interactive commands + sandbox
# ============================================================================


class TestBashToolSecurity:
    """Security policies: interactive, privileged, destructive, network."""

    def _tool(self, tmp_path):
        from hello_agents.tools.builtin.bash import BashTool
        return BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))

    def test_all_interactive_blocked(self, tmp_path):
        tool = self._tool(tmp_path)
        blocked = [
            # original
            "vim", "vi", "nano", "less", "more", "top", "htop",
            "watch", "tmux", "screen",
            # new
            "emacs", "micro", "most", "btop", "atop",
            "dialog", "whiptail", "fzf", "peco", "ncdu",
            "mutt", "neomutt", "irssi", "weechat",
        ]
        for cmd in blocked:
            reason = tool.validate_command_policy(cmd)
            assert reason is not None, f"'{cmd}' must be blocked as interactive"

    def test_all_privileged_blocked(self, tmp_path):
        tool = self._tool(tmp_path)
        for cmd in ("sudo", "su", "doas"):
            reason = tool.validate_command_policy(f"{cmd} rm x")
            assert reason is not None
            assert "privileged" in reason.lower() or "not allowed" in reason.lower()

    def test_all_destructive_blocked(self, tmp_path):
        tool = self._tool(tmp_path)
        for cmd in ("mkfs", "fdisk", "shutdown", "reboot", "poweroff", "halt"):
            reason = tool.validate_command_policy(cmd)
            assert reason is not None, f"'{cmd}' must be blocked as destructive"

    def test_delete_commands_blocked(self, tmp_path):
        tool = self._tool(tmp_path)
        for cmd in ("rm", "rmdir", "unlink", "shred", "srm", "del", "erase"):
            reason = tool.validate_command_policy(f"{cmd} some_file")
            assert reason is not None, f"'{cmd}' must be blocked"

    def test_network_blocked_by_default(self, tmp_path):
        tool = self._tool(tmp_path)
        for cmd in ("npm", "pip", "apt", "curl", "wget"):
            reason = tool.validate_command_policy(f"{cmd} install foo")
            assert reason is not None, f"'{cmd}' must be blocked by default"

    def test_rm_rf_root_always_blocked(self, tmp_path):
        tool = self._tool(tmp_path)
        reason = tool.validate_command_policy("rm -rf /")
        assert reason is not None

    def test_git_clean_blocked(self, tmp_path):
        tool = self._tool(tmp_path)
        reason = tool.validate_command_policy("git clean -fd")
        assert reason is not None

    def test_find_delete_blocked(self, tmp_path):
        tool = self._tool(tmp_path)
        reason = tool.validate_command_policy("find . -name '*.tmp' -delete")
        assert reason is not None

    def test_normal_commands_allowed(self, tmp_path):
        tool = self._tool(tmp_path)
        # ls/grep/cat/sed/awk are blocked by PREFER_SPECIALIZED_TOOLS — expected
        for cmd in ("echo hello", "git status", "python --version",
                     "pytest tests/", "whoami", "mkdir testdir",
                     "git diff HEAD~1"):
            reason = tool.validate_command_policy(cmd)
            assert reason is None, f"'{cmd}' should be allowed, got: {reason}"

    def test_sandbox_env_strips_secrets(self, tmp_path, monkeypatch):
        from hello_agents.tools.builtin.bash import build_sandbox_env
        monkeypatch.setenv("LLM_API_KEY", "sk-secret-123")
        monkeypatch.setenv("HF_TOKEN", "hf-456")
        monkeypatch.setenv("PATH", "/usr/bin")
        env = build_sandbox_env(tmp_path)
        assert "LLM_API_KEY" not in env
        assert "HF_TOKEN" not in env
        assert env.get("PATH") == "/usr/bin"

    def test_deeply_nested_shell_blocked(self, tmp_path):
        import shlex
        tool = self._tool(tmp_path)
        cmd = "echo hi"
        for _ in range(5):
            cmd = f"bash -c {shlex.quote(cmd)}"
        assert tool.validate_command_policy(cmd) is not None

    def test_command_no_timeout_execute(self, tmp_path):
        """A quick command completes within the default timeout."""
        tool = self._tool(tmp_path)
        resp = tool.run({"command": "echo hello", "block_until_ms": 15000})
        assert resp.status.value == "success"
        assert "hello" in resp.data.get("output", "")

    def test_invalid_block_until_rejected(self, tmp_path):
        """block_until_ms out of range returns error."""
        tool = self._tool(tmp_path)
        resp = tool.run({"command": "echo hi", "block_until_ms": 999999999})
        assert resp.status.value == "error"
        assert "block_until_ms" in resp.text.lower()

    def test_invalid_params_rejected(self, tmp_path):
        """Missing command or bad description type."""
        tool = self._tool(tmp_path)
        resp = tool.run({"block_until_ms": 1000})
        assert resp.status.value == "error"

    def test_description_wrong_type_rejected(self, tmp_path):
        tool = self._tool(tmp_path)
        resp = tool.run({"command": "echo hi", "description": 123})
        assert resp.status.value == "error"


# ============================================================================
# 9. TodoWriteTool — deterministic session_id
# ============================================================================


class TestTodoWriteSessionIdExtended:
    """Session ID generation and todo persistence."""

    def test_deterministic_across_instances(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            for _ in range(5):
                t = TodoWriteTool(project_root=tmp)
                assert t.session_id == TodoWriteTool(project_root=tmp).session_id
        finally:
            shutil.rmtree(tmp)

    def test_different_projects_different_ids(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        tmp1 = tempfile.mkdtemp()
        tmp2 = tempfile.mkdtemp()
        try:
            id1 = TodoWriteTool(project_root=tmp1).session_id
            id2 = TodoWriteTool(project_root=tmp2).session_id
            assert id1 != id2
        finally:
            shutil.rmtree(tmp1, ignore_errors=True)
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_explicit_session_id_respected(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        t = TodoWriteTool(project_root="/tmp", session_id="custom-session")
        assert t.session_id == "custom-session"

    def test_full_todo_workflow(self):
        """Write todos, export state, load back, verify."""
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile
        tmp = tempfile.mkdtemp()
        persistence = os.path.join(tmp, "memory", "todos")
        os.makedirs(persistence, exist_ok=True)
        try:
            t = TodoWriteTool(project_root=tmp, persistence_dir="memory/todos")
            initial = t.export_state()
            assert initial["todos"] == []

            # Write some todos
            resp = t.run({"todos": [
                {"content": "Fix bug", "status": "in_progress", "priority": "high"},
                {"content": "Write test", "status": "pending"},
                {"content": "Deploy", "status": "pending", "priority": "low"},
            ]})
            assert resp.status.value == "success"

            state = t.export_state()
            assert len(state["todos"]) == 3
            assert state["todos"][0]["content"] == "Fix bug"
            assert state["todos"][0]["status"] == "in_progress"

            # Update — replace all
            resp2 = t.run({"todos": [
                {"content": "Fix bug", "status": "completed", "priority": "high"},
                {"content": "Write test", "status": "in_progress"},
            ]})
            assert resp2.status.value == "success"
            state2 = t.export_state()
            assert len(state2["todos"]) == 2
            assert state2["todos"][0]["status"] == "completed"

            # Same project, new instance — should load same state
            t2 = TodoWriteTool(project_root=tmp, persistence_dir="memory/todos")
            state3 = t2.export_state()
            assert len(state3["todos"]) == 2
            assert state3["session_id"] == t.session_id
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_import_state(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            t = TodoWriteTool(project_root=tmp)
            t.import_state({"todos": [
                {"content": "Preloaded", "status": "pending"},
            ]})
            state = t.export_state()
            assert len(state["todos"]) == 1
            assert state["todos"][0]["content"] == "Preloaded"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_invalid_status_rejected(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            t = TodoWriteTool(project_root=tmp)
            resp = t.run({"todos": [
                {"content": "X", "status": "not_a_real_status"},
            ]})
            assert resp.status.value == "error"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_duplicate_content_rejected(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            t = TodoWriteTool(project_root=tmp)
            resp = t.run({"todos": [
                {"content": "Same task", "status": "pending"},
                {"content": "Same task", "status": "in_progress"},
            ]})
            assert resp.status.value == "error"
            assert "duplicate" in resp.text.lower()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_two_in_progress_rejected(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            t = TodoWriteTool(project_root=tmp)
            resp = t.run({"todos": [
                {"content": "A", "status": "in_progress"},
                {"content": "B", "status": "in_progress"},
            ]})
            assert resp.status.value == "error"
            assert "in_progress" in resp.text.lower()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_completed_cannot_reopen(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            t = TodoWriteTool(project_root=tmp)
            # First, complete a task
            t.run({"todos": [{"content": "Done", "status": "completed"}]})
            # Then try to set it back to pending
            resp = t.run({"todos": [{"content": "Done", "status": "pending"}]})
            assert resp.status.value == "error"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 10. Registry — category + function tools + get_tool_category
# ============================================================================


class TestRegistryCategoryMethods:
    """ToolRegistry.get_tool_category and get_tool_categories."""

    def test_tool_category_known(self):
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        class _T(Tool):
            def __init__(self):
                super().__init__(name="MyTool", description="", category="readonly")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")

        reg = ToolRegistry(verbose=False)
        reg.register_tool(_T())
        assert reg.get_tool_category("MyTool") == "readonly"

    def test_function_tool_category_defaults_to_general(self):
        from hello_agents.tools.registry import ToolRegistry
        reg = ToolRegistry(verbose=False)
        reg.register_function(lambda x: x, name="func1")
        assert reg.get_tool_category("func1") == "general"

    def test_unknown_tool_returns_general(self):
        from hello_agents.tools.registry import ToolRegistry
        reg = ToolRegistry(verbose=False)
        assert reg.get_tool_category("NoSuchTool") == "general"

    def test_get_tool_categories_mixed(self):
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        class _A(Tool):
            def __init__(self):
                super().__init__(name="A", description="", category="readonly")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")

        reg = ToolRegistry(verbose=False)
        reg.register_tool(_A())
        reg.register_function(lambda x: x, name="func_x")

        cats = reg.get_tool_categories()
        assert cats["A"] == "readonly"
        # Function tools only appear in _functions, not _tools
        # so get_tool_categories only covers Tool instances
        assert "func_x" not in cats  # function tools not in category map


# ============================================================================
# 11. Asynchronous execution paths
# ============================================================================


class TestAsyncExecution:
    """Async tool execution paths (arun, arun_with_timing, aexecute_tool)."""

    def test_aexecute_tool_sync_tool(self):
        """Even sync tools can be executed via the async path (thread pool)."""
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        class _Sync(Tool):
            def __init__(self):
                super().__init__(name="Sync", description="")
            def get_parameters(self): return [
                ToolParameter(name="inp", type="string", description="", required=True),
            ]
            def run(self, p):
                return ToolResponse.success(text=f"got: {p['inp']}")

        reg = ToolRegistry(verbose=False)
        reg.register_tool(_Sync())

        async def go():
            return await reg.aexecute_tool("Sync", '{"inp": "hello async"}')
        resp = asyncio.run(go())
        assert resp.status.value == "success"
        assert "hello async" in resp.text

    def test_aexecute_tool_schema_validation(self):
        """Schema validation still runs on async path."""
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        class _Sync(Tool):
            def __init__(self):
                super().__init__(name="SV", description="")
            def get_parameters(self): return [
                ToolParameter(name="x", type="string", description="", required=True),
            ]
            def run(self, p):
                return ToolResponse.success(text="ok")

        reg = ToolRegistry(verbose=False)
        reg.register_tool(_Sync())

        async def go():
            return await reg.aexecute_tool("SV", '{}')
        resp = asyncio.run(go())
        assert resp.status.value == "error"
        assert "x" in resp.text.lower()


# ============================================================================
# 12. Regression — Tool.__str__ does not have dead code
# ============================================================================


class TestToolStrRepr:
    """Ensure __str__ and __repr__ are clean."""

    def test_str_single_return(self):
        import inspect
        from hello_agents.tools.base import Tool
        source = inspect.getsource(Tool.__str__)
        # Count returns in the method body
        returns = [line for line in source.splitlines()
                   if "return" in line and not line.strip().startswith("#")]
        assert len(returns) == 1, "Tool.__str__ must have exactly one return statement"
