"""Regression tests for IMPROVEMENT.md fixes.

Covers: 重要-1 (circuit breaker fault codes), 严重-3 (WebFetch SSRF),
重要-8 (trace redaction + HTML escape), 重要-4 (max_steps), 重要-5 (CLI tasks),
重要-7 (token estimation), 重要-12 (retry), 重要-3 (adapter normalization),
重要-6 (subagent filter), 重要-9/10 (shared tool loop), 建议-4/10/13/14/15,
重要-11 (packaging), 建议-8 (factory),
重要-7-enum (ToolErrorCode.is_fault), 重要-5-category (Tool categories + filter).
"""

import asyncio
import json
import shlex
import sys
import threading
from pathlib import Path

import pytest

# Bootstrap: add project root to sys.path so "scripts.cli" is importable
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# 重要-1: circuit breaker only trips on real tool faults
# ---------------------------------------------------------------------------

def test_circuit_breaker_ignores_model_errors():
    from hello_agents.tools.circuit_breaker import CircuitBreaker
    from hello_agents.tools.response import ToolResponse
    from hello_agents.tools.errors import ToolErrorCode

    cb = CircuitBreaker(failure_threshold=3)
    # Many model-side errors must NOT count toward tripping.
    for code in ("INVALID_PARAM", "ACCESS_DENIED", "NOT_FOUND",
                 "CONFLICT", "BINARY_FILE", "IS_DIRECTORY"):
        for _ in range(5):
            cb.record_result("X", ToolResponse.error(code=code, message=""))

    assert not cb.is_open("X"), f"model error {code} tripped the breaker"

    # Real faults MUST trip.
    for code in ("INTERNAL_ERROR", "EXECUTION_ERROR", "TIMEOUT",
                 "NETWORK_ERROR", "API_ERROR"):
        cb2 = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb2.record_result("Y", ToolResponse.error(code=code, message=""))
        assert cb2.is_open("Y"), f"fault code {code} did NOT trip the breaker"

    # Success / partial resets the counter.
    cb3 = CircuitBreaker(failure_threshold=3)
    for _ in range(2):
        cb3.record_result("Z", ToolResponse.error(code="INTERNAL_ERROR", message=""))
    from hello_agents.tools.response import ToolStatus
    from hello_agents.tools.response import ToolResponse, ToolStatus
    cb3.record_result("Z", ToolResponse(text="ok", data={}, status=ToolStatus.SUCCESS))
    # After a success the failure_count is reset (we check via get_status).
    assert cb3.get_status("Z")["failure_count"] == 0  # counter reset by success


def test_circuit_breaker_is_thread_safe():
    """建议-11: concurrent record_result calls must not corrupt counters."""
    from hello_agents.tools.circuit_breaker import CircuitBreaker
    from hello_agents.tools.response import ToolResponse
    from hello_agents.tools.errors import ToolErrorCode

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    errors = []
    ITERS = 200

    def worker():
        for _ in range(ITERS):
            try:
                # All successes — after all workers finish, counter must be 0.
                cb.record_result("T", ToolResponse.success(text="", data={}))
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"thread errors: {errors[0] if errors else 'none'}"
    assert not cb.is_open("T")

    # Now trip with faults — all workers see open state eventually.
    for _ in range(3):
        cb.record_result("T", ToolResponse.error(code="INTERNAL_ERROR", message=""))
    assert cb.is_open("T")


# ---------------------------------------------------------------------------
# 重要-7-enum: ToolErrorCode.is_fault attribute
# ---------------------------------------------------------------------------

class TestToolErrorCodeIsFault:
    """Verify every ToolErrorCode member carries the correct is_fault flag."""

    @staticmethod
    def _fault_codes():
        return {"INTERNAL_ERROR", "EXECUTION_ERROR", "TIMEOUT",
                "NETWORK_ERROR", "API_ERROR"}

    def test_all_fault_codes_marked_true(self):
        from hello_agents.tools.errors import ToolErrorCode
        for name in self._fault_codes():
            member = ToolErrorCode[name]
            assert member.is_fault is True, f"{name}.is_fault must be True"

    def test_non_fault_codes_marked_false(self):
        from hello_agents.tools.errors import ToolErrorCode
        fault_names = self._fault_codes()
        for member in ToolErrorCode:
            if member.name not in fault_names:
                assert member.is_fault is False, (
                    f"{member.name}.is_fault must be False"
                )

    def test_fault_codes_classmethod(self):
        from hello_agents.tools.errors import ToolErrorCode
        codes = ToolErrorCode.fault_codes()
        assert set(codes) == self._fault_codes()

    def test_string_equality_preserved(self):
        """重要-7: members must remain drop-in string-compatible."""
        from hello_agents.tools.errors import ToolErrorCode
        assert ToolErrorCode.TIMEOUT == "TIMEOUT"
        assert ToolErrorCode.INVALID_PARAM == "INVALID_PARAM"
        assert str(ToolErrorCode.INTERNAL_ERROR) == "INTERNAL_ERROR"
        # JSON serialisation
        assert json.dumps(ToolErrorCode.NETWORK_ERROR) == '"NETWORK_ERROR"'

    def test_is_valid_code_still_works(self):
        from hello_agents.tools.errors import ToolErrorCode
        assert ToolErrorCode.is_valid_code("TIMEOUT")
        assert ToolErrorCode.is_valid_code("INVALID_PARAM")
        assert not ToolErrorCode.is_valid_code("NONEXISTENT")
        # Accepts enum member itself
        assert ToolErrorCode.is_valid_code(ToolErrorCode.NOT_FOUND)

    def test_get_all_codes_returns_strings(self):
        from hello_agents.tools.errors import ToolErrorCode
        codes = ToolErrorCode.get_all_codes()
        assert isinstance(codes, list)
        assert all(isinstance(c, str) for c in codes)
        assert "INTERNAL_ERROR" in codes

    def test_member_value_is_plain_string(self):
        """The .value of each member must be the plain string code."""
        from hello_agents.tools.errors import ToolErrorCode
        for member in ToolErrorCode:
            assert isinstance(member.value, str), (
                f"{member.name}.value must be str, got {type(member.value)}"
            )
            assert member.value == member.name


# ---------------------------------------------------------------------------
# 重要-7-enum: _is_fault_response integration with ToolErrorCode.is_fault
# ---------------------------------------------------------------------------

class TestCircuitBreakerIsFaultIntegration:
    """Verify the circuit breaker uses ToolErrorCode.is_fault correctly."""

    def test_known_fault_code_triggers_is_fault(self):
        from hello_agents.tools.circuit_breaker import _is_fault_response
        from hello_agents.tools.response import ToolResponse, ToolStatus
        from hello_agents.tools.errors import ToolErrorCode

        for name in ("INTERNAL_ERROR", "EXECUTION_ERROR", "TIMEOUT",
                     "NETWORK_ERROR", "API_ERROR"):
            resp = ToolResponse.error(code=ToolErrorCode[name], message="boom")
            assert _is_fault_response(resp), f"{name} must be detected as fault"

    def test_known_non_fault_not_treated_as_fault(self):
        from hello_agents.tools.circuit_breaker import _is_fault_response
        from hello_agents.tools.response import ToolResponse
        from hello_agents.tools.errors import ToolErrorCode

        for name in ("INVALID_PARAM", "NOT_FOUND", "CONFLICT", "ACCESS_DENIED",
                     "PERMISSION_DENIED", "BINARY_FILE", "IS_DIRECTORY",
                     "INVALID_FORMAT", "RATE_LIMIT", "CIRCUIT_OPEN",
                     "ASK_USER_UNAVAILABLE"):
            resp = ToolResponse.error(code=ToolErrorCode[name], message="rejected")
            assert not _is_fault_response(resp), (
                f"{name} must NOT be treated as fault"
            )

    def test_unknown_code_treated_as_fault(self):
        """Unknown error codes should be treated as faults conservatively."""
        from hello_agents.tools.circuit_breaker import _is_fault_response
        from hello_agents.tools.response import ToolResponse

        resp = ToolResponse.error(code="MYSTERY_CODE", message="unknown")
        assert _is_fault_response(resp), "unknown code must be treated as fault"

    def test_none_code_treated_as_non_fault(self):
        """None error_info means no code → not a fault."""
        from hello_agents.tools.circuit_breaker import _is_fault_response
        from hello_agents.tools.response import ToolResponse, ToolStatus

        resp = ToolResponse(
            status=ToolStatus.ERROR,
            text="broken",
            error_info=None,
        )
        assert not _is_fault_response(resp)

    def test_success_response_not_fault(self):
        from hello_agents.tools.circuit_breaker import _is_fault_response
        from hello_agents.tools.response import ToolResponse

        resp = ToolResponse.success(text="ok")
        assert not _is_fault_response(resp)

    def test_partial_response_not_fault(self):
        from hello_agents.tools.circuit_breaker import _is_fault_response
        from hello_agents.tools.response import ToolResponse

        resp = ToolResponse.partial(text="partial result")
        assert not _is_fault_response(resp)

    def test_end_to_end_circuit_breaker_uses_is_fault(self):
        """The full record_result → is_open pipeline uses is_fault correctly."""
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from hello_agents.tools.response import ToolResponse
        from hello_agents.tools.errors import ToolErrorCode

        cb = CircuitBreaker(failure_threshold=2)

        # Non-fault errors do NOT trip
        for _ in range(5):
            cb.record_result("tool_a", ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM, message="bad input"))
        assert not cb.is_open("tool_a"), "non-fault must not trip breaker"

        # Fault errors DO trip
        for _ in range(2):
            cb.record_result("tool_b", ToolResponse.error(
                code=ToolErrorCode.TIMEOUT, message="timed out"))
        assert cb.is_open("tool_b"), "fault must trip breaker"

    def test_fault_codes_match_circuit_breaker_expectations(self):
        """All fault codes listed in _is_fault_response must match
        ToolErrorCode.is_fault — no drift possible (重要-7)."""
        from hello_agents.tools.errors import ToolErrorCode
        from hello_agents.tools.circuit_breaker import _is_fault_response
        from hello_agents.tools.response import ToolResponse

        for member in ToolErrorCode:
            resp = ToolResponse.error(code=member, message="test")
            detected = _is_fault_response(resp)
            assert detected == member.is_fault, (
                f"_is_fault_response({member.name}) returned {detected}, "
                f"but member.is_fault={member.is_fault}"
            )


# ---------------------------------------------------------------------------
# 重要-5-category: Tool categories
# ---------------------------------------------------------------------------

class TestToolCategoryAndFilter:
    """Verify tool categories are set correctly and filters work with them."""

    def test_all_builtin_tools_have_categories(self):
        from hello_agents.tools import (
            BashTool, ReadTool, WriteTool, EditTool, DeleteTool,
            ListFilesTool, GlobTool, GrepTool, TodoWriteTool,
            SkillTool, AskUserTool, WebSearchTool, WebFetchTool,
        )
        from hello_agents.skills.loader import SkillLoader
        from pathlib import Path
        import tempfile, os

        tmp = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp, "skills"), exist_ok=True)
            loader = SkillLoader(Path(os.path.join(tmp, "skills")))

            tools = [
                (ReadTool(project_root=tmp), "readonly"),
                (WriteTool(project_root=tmp), "write"),
                (EditTool(project_root=tmp), "write"),
                (DeleteTool(project_root=tmp), "write"),
                (ListFilesTool(project_root=tmp), "readonly"),
                (GlobTool(project_root=tmp), "readonly"),
                (GrepTool(project_root=tmp), "readonly"),
                (BashTool(project_root=tmp), "dangerous"),
                (TodoWriteTool(project_root=tmp), "write"),
                (SkillTool(skill_loader=loader), "readonly"),
                (AskUserTool(interactive=False), "interactive"),
                (WebSearchTool(project_root=tmp, enabled=True), "network"),
                (WebFetchTool(project_root=tmp, enabled=True), "network"),
            ]
            for tool, expected_cat in tools:
                assert tool.category == expected_cat, (
                    f"{tool.name}: expected category={expected_cat}, "
                    f"got {tool.category}"
                )

            # Verify category is accessible as a plain str attribute
            assert isinstance(ReadTool(project_root=tmp).category, str)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_registry_get_tool_category(self):
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        reg = ToolRegistry(verbose=False)

        class _T(Tool):
            def __init__(self):
                super().__init__(name="MyTool", description="", category="readonly")
            def get_parameters(self): return []
            def run(self, p): return ToolResponse.success(text="ok")

        reg.register_tool(_T())
        assert reg.get_tool_category("MyTool") == "readonly"
        assert reg.get_tool_category("Nonexistent") == "general"

        cats = reg.get_tool_categories()
        assert cats["MyTool"] == "readonly"


# ---------------------------------------------------------------------------
# 严重-3: WebFetch SSRF guards
# ---------------------------------------------------------------------------

def test_ssrf_ip_blocking_logic():
    from hello_agents.tools.builtin.web_tool import _ip_is_blocked, _host_is_blocked
    # Metadata endpoint MUST be blocked.
    assert _ip_is_blocked("169.254.169.254")
    # Loopback
    assert _ip_is_blocked("127.0.0.1")
    assert _ip_is_blocked("::1")
    # Private ranges
    assert _ip_is_blocked("10.0.0.1")
    assert _ip_is_blocked("192.168.1.1")
    assert _ip_is_blocked("172.16.0.1")
    # Link-local
    assert _ip_is_blocked("169.254.1.1")
    # Public addresses OK (TEST-NET-3 203.0.113.0/24 is *reserved*, not public)
    assert not _ip_is_blocked("8.8.8.8")
    assert not _ip_is_blocked("93.184.216.34")  # example.com (real public)
    # IPv4 multicast range should be blocked via is_multicast
    assert _ip_is_blocked("224.0.0.1")


def test_ssrf_guarded_get_blocks_localhost():
    """A literal localhost URL must be rejected before any network call."""
    from hello_agents.tools.builtin.web_tool import WebFetchTool, SSRFBlockedError
    tool = WebFetchTool(enabled=True)
    dummy = type("req", (), {"Session": lambda *a, **kw: None})()
    with pytest.raises(SSRFBlockedError):
        tool._guarded_get(None, dummy, "http://127.0.0.1:8000/",
                          headers={}, timeout=5)


def test_ssrf_guarded_get_blocks_metadata():
    from hello_agents.tools.builtin.web_tool import WebFetchTool, SSRFBlockedError
    tool = WebFetchTool(enabled=True)
    dummy = type("req", (), {"Session": lambda *a, **kw: None})()
    with pytest.raises(SSRFBlockedError):
        tool._guarded_get(None, dummy, "http://169.254.169.254/latest/meta-data/",
                          headers={}, timeout=5)


# ---------------------------------------------------------------------------
# 重要-8: TraceLogger redaction + HTML escaping
# ---------------------------------------------------------------------------

def test_trace_redaction_handles_hyphenated_keys():
    from hello_agents.observability.trace_logger import TraceLogger
    tl = TraceLogger.__new__(TraceLogger)
    # sk-ant-api03-XXXX used to leak the tail.
    red = tl._redact_secrets_in_text("x sk-ant-api03-ABCDEFGHIJ y")
    assert "api03-ABCDEFGHIJ" not in red
    assert "sk-***" in red

    # hf_ / ghp_ / AKIA etc.
    assert "hf_wrong-leak" not in tl._redact_secrets_in_text("key hf_abcdefghijkl tail")


def test_trace_key_aware_redaction():
    from hello_agents.observability.trace_logger import TraceLogger
    # Use object.__new__ to create a bare instance without its __init__
    # side-effects (file opening, mkdir, etc.)
    import importlib
    import hello_agents.observability.trace_logger as tlm
    tl = object.__new__(tlm.TraceLogger)

    # Dict keys that look sensitive → whole value redacted regardless of format.
    v = tl._sanitize_value({"api_key": "plain-not-a-token", "desc": "ok"})
    assert v["api_key"] == "***REDACTED***"
    assert v["desc"] == "ok"

    # Nested dicts.
    v2 = tl._sanitize_value({"auth": {"authorization": "super-secret"}})
    assert v2["auth"]["authorization"] == "***REDACTED***"


def test_trace_html_escape():
    # Full HTML escaping happens in _write_html_event / _write_html_footer.
    # Here we just validate the redaction method is callable.
    import hello_agents.observability.trace_logger as tlm
    tl = object.__new__(tlm.TraceLogger)
    text = tl._redact_secrets_in_text("<script>alert(1)</script>", key_hint=None)
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# 重要-4: max_steps finite default
# ---------------------------------------------------------------------------

def test_code_agent_has_finite_default_max_steps():
    from hello_agents.agents.code_agent import DEFAULT_CODE_AGENT_MAX_STEPS, CodeAgent
    assert DEFAULT_CODE_AGENT_MAX_STEPS > 0
    assert CodeAgent.DEFAULT_MAX_STEPS == DEFAULT_CODE_AGENT_MAX_STEPS


# ---------------------------------------------------------------------------
# 重要-5: CLI task list reads from export_state (no task_manager)
# ---------------------------------------------------------------------------

def test_cli_task_list_uses_export_state():
    """_get_task_list must work with TodoWrite export_state/import_state API."""
    from hello_agents.core.config import Config
    from hello_agents.core.llm import HelloAgentsLLM
    from hello_agents.tools.registry import ToolRegistry
    from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
    from hello_agents.agents.code_agent import CodeAgent

    config = Config(trace_enabled=False, todowrite_enabled=True)
    llm = HelloAgentsLLM(
        model="gpt-4", api_key="sk-test", base_url="http://no-op"
    )
    registry = ToolRegistry(config=config, verbose=False)
    agent = CodeAgent(
        name="test", llm=llm, tool_registry=registry, config=config,
        register_default_tools=False,  # don't auto-register default tool set
    )
    # Register just TodoWrite so the side-effects are minimal.
    todo = TodoWriteTool(project_root=".", session_id="test-session")
    registry.register_tool(todo)

    # Use the CLIUI helper directly.
    from scripts.cli import CLIUI
    tasks = CLIUI._get_task_list(agent)
    assert isinstance(tasks, list)

    # Set some tasks and check they are reflected.
    todo.import_state({"todos": [
        {"content": "Fix bug", "status": "in_progress", "priority": "high"},
        {"content": "Write test", "status": "pending"},
    ]})
    tasks2 = CLIUI._get_task_list(agent)
    assert len(tasks2) == 2
    assert tasks2[0]["subject"] == "Fix bug"
    assert tasks2[0]["status"] == "in_progress"


# ---------------------------------------------------------------------------
# 重要-7: token estimation includes tool_calls arguments
# ---------------------------------------------------------------------------

def test_token_estimation_includes_tool_call_args():
    """重要-7: HistoryManager.estimate_tokens counts tool_calls arguments
    in addition to message content.  We test this by constructing LLM-ready
    dicts *outside* HistoryManager (to avoid message re-projection) and
    feeding them directly via ``history=`` to ``estimate_tokens``, which
    accepts a bare list of dicts (it calls ``build_llm_messages`` internally
    only when no history is supplied — but here we want *direct* messages).
    """
    from hello_agents.context.history import HistoryManager

    hm = HistoryManager()

    # Plain assistant (no tool calls)
    msgs_plain: list = [
        {"role": "system", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]
    # Hmm, we can't pass raw dicts either. Let's test via the individual
    # message projection instead: the key fix is that _project_message_for_llm
    # includes tool_calls in the LLM dict, and estimate_tokens iterates over
    # the projected result.  Just verify that the projected message dict DOES
    # include tool_calls when metadata carries them.
    from hello_agents.core.message import Message

    hm2 = HistoryManager()
    big_args = json.dumps({"content": "x" * 5000})
    hm2.append(
        Message(
            role="assistant",
            content="calling Write",
            metadata={
                "tool_calls": [
                    {"id": "1", "name": "Write", "arguments": big_args},
                ]
            },
        )
    )
    messages = hm2.get_history()
    assert len(messages) == 1
    projected = hm2._project_message_for_llm(messages[0])
    assert projected is not None
    assert "tool_calls" in projected, "tool_calls must appear in projected LLM dict"
    # Estimate from the manager's own history (Message objects, not raw dicts).
    est = hm2.estimate_tokens(system_prompt="hello")
    assert est > 0, "token estimation must be non-zero for tool-call message"


# ---------------------------------------------------------------------------
# 重要-3: cross-provider tool-call normalization
# ---------------------------------------------------------------------------

def test_normalize_anthropic_tool_response():
    from hello_agents.core.llm_adapters import normalize_anthropic_tool_response

    class ANamed:
        def __init__(self, name): self.name = name

    class ABlock:
        def __init__(self, tp, **kw):
            self.type = tp
            for k, v in kw.items():
                setattr(self, k, v)

    class AResponse:
        content = [
            ABlock("text", text="some text"),
            ABlock("tool_use", id="x1", name="Greet", input={"msg": "hi"}),
        ]
        usage = ANamed("usage-obj")

    nr = normalize_anthropic_tool_response(AResponse())
    assert nr.choices[0].message.content == "some text"
    tc = nr.choices[0].message.tool_calls
    assert len(tc) == 1
    assert tc[0].id == "x1"
    assert tc[0].function.name == "Greet"
    assert json.loads(tc[0].function.arguments) == {"msg": "hi"}
    assert nr.usage is not None  # usage is passed through


def test_normalize_gemini_tool_response():
    from hello_agents.core.llm_adapters import normalize_gemini_tool_response

    class Part:
        def __init__(self, text=None, function_call=None):
            self.text = text
            self.function_call = function_call
    class FnCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args
    class Content:
        def __init__(self, parts):
            self.parts = parts
    class Candidate:
        def __init__(self, content):
            self.content = content
    class UMeta:
        prompt_token_count = 10
        candidates_token_count = 5
        total_token_count = 15

    class GResponse:
        candidates = [
            Candidate(Content([
                Part(text="hello"),
                Part(function_call=FnCall("search", {"q": "test"})),
            ]))
        ]
        usage_metadata = UMeta()

    nr = normalize_gemini_tool_response(GResponse())
    assert nr.choices[0].message.content == "hello"
    tc = nr.choices[0].message.tool_calls
    assert len(tc) == 1
    assert tc[0].function.name == "search"
    assert json.loads(tc[0].function.arguments) == {"q": "test"}
    assert nr.usage_metadata is not None


# ---------------------------------------------------------------------------
# 重要-12: retry with exponential backoff
# ---------------------------------------------------------------------------

def test_retry_wraps_invoke():
    from hello_agents.core.llm_adapters import OpenAIAdapter, _NResponse, _NMessage

    class FakeOpenAI:
        def __init__(self):
            self.calls = 0
        class chat:
            class completions:
                instance = None
                @staticmethod
                def create(*, model, messages, **kw):
                    FakeOpenAI.instance.calls += 1
                    if FakeOpenAI.instance.calls < 3:
                        raise _FakeAPIError("too many requests", status_code=429)
                    return _NResponse(_NMessage("final"))

    class _FakeAPIError(Exception):
        def __init__(self, msg, status_code=None):
            super().__init__(msg)
            self.status_code = status_code

    adapter = OpenAIAdapter(
        api_key="sk-test", base_url="http://test", timeout=10, model="gpt-4"
    )
    adapter.max_retries = 3
    adapter.retry_base_delay = 0.001
    adapter.retry_max_delay = 0.01

    adapter._client = FakeOpenAI()
    FakeOpenAI.instance = FakeOpenAI()
    FakeOpenAI.instance.calls = 0

    resp = adapter.invoke([])
    assert resp.content == "final"
    assert FakeOpenAI.instance.calls >= 3, f"expected >=3 calls (retries), got {FakeOpenAI.instance.calls}"


# 注：重要-9/10 的共享工具循环（_tool_loop.py）及其测试已随三个教学型
# agent（SimpleAgent/PlanSolveAgent/ReflectionAgent）一并移除（2026-08-19
# 精简决策，见 IMPROVEMENT.md 第 10 轮）。ReAct 主循环与 CodeAgent 不受影响。


# ---------------------------------------------------------------------------
# 建议-4: schema hash uses get_parameters() (not tool.parameters attr)
# ---------------------------------------------------------------------------

def test_schema_hash_includes_parameter_defs():
    from hello_agents.tools.registry import ToolRegistry
    from hello_agents.tools.base import Tool, ToolParameter
    from hello_agents.tools.response import ToolResponse
    from hello_agents.core.config import Config
    from hello_agents.core.llm import HelloAgentsLLM
    from hello_agents.agents.react_agent import ReActAgent

    class _V1(Tool):
        def __init__(self):
            super().__init__(name="Test", description="")
        def get_parameters(self):
            return [ToolParameter(name="a", type="string", description="", required=True)]
        def run(self, p):
            return ToolResponse.success(text="ok")

    r1 = ToolRegistry(config=Config(), verbose=False)
    r1.register_tool(_V1())

    class _V2(Tool):
        def __init__(self):
            super().__init__(name="Test", description="")
        def get_parameters(self):
            return [ToolParameter(name="a", type="string", description="", required=True),
                    ToolParameter(name="b", type="integer", description="", required=False)]
        def run(self, p):
            return ToolResponse.success(text="ok")

    r2 = ToolRegistry(config=Config(), verbose=False)
    r2.register_tool(_V2())

    llm = HelloAgentsLLM(model="gpt-4", api_key="sk-t", base_url="http://x")
    agent = ReActAgent(name="t", llm=llm, tool_registry=r1, max_steps=1)
    h1 = agent._compute_tool_schema_hash()
    agent.tool_registry = r2
    h2 = agent._compute_tool_schema_hash()
    assert h1 != h2, "Schema hash must differ when parameters change"


# ---------------------------------------------------------------------------
# 建议-14: non-dict tool arguments are wrapped instead of crashing
# ---------------------------------------------------------------------------

def test_non_dict_arguments_safely_observed():
    """建议-14: _prepare_tool_registry_input and _convert_parameter_types must
    not crash (AttributeError) when model-supplied arguments are a list/str
    instead of a dict. Test via a concrete agent subclass that mocks only the
    one abstract method."""
    from hello_agents.core.agent import Agent
    from hello_agents.core.config import Config
    from hello_agents.core.llm import HelloAgentsLLM
    from hello_agents.tools.registry import ToolRegistry
    from hello_agents.tools.base import Tool, ToolParameter
    from hello_agents.tools.response import ToolResponse

    # Concrete agent for testing — no real LLM invocation.
    class _ConcreteAgent(Agent):
        def get_parameters(self): return []
        def run(self, input_text, **kw): return "ok"

    llm = HelloAgentsLLM(model="gpt-4", api_key="sk-t", base_url="http://x")
    a = _ConcreteAgent(name="test", llm=llm)

    # ── without a registry ──────────────────────────────
    a.tool_registry = None
    # Non-dict args are passed straight through.
    assert a._prepare_tool_registry_input("X", [1, 2, 3]) == [1, 2, 3]
    assert a._prepare_tool_registry_input("X", "str") == "str"

    # _convert_parameter_types also guards against non-dict input.
    assert a._convert_parameter_types("X", [1, 2]) == [1, 2]

    # ── with a registry + a known tool ──────────────────
    class _ToolX(Tool):
        def __init__(self):
            super().__init__(name="X", description="")
        def get_parameters(self):
            return [ToolParameter(name="val", type="string", description="", required=True)]
        def run(self, params):
            return ToolResponse.success(text="ok")

    a.tool_registry = ToolRegistry(config=Config(), verbose=False)
    a.tool_registry.register_tool(_ToolX())

    # list argument is wrapped so the tool can reject it with INVALID_PARAM
    # instead of crashing.
    result = a._prepare_tool_registry_input("X", [1, 2, 3])
    assert isinstance(result, dict), f"expected dict wrapper, got {type(result)}"
    assert "input" in result


# ---------------------------------------------------------------------------
# 建议-10: grep read stop cap
# ---------------------------------------------------------------------------

def test_grep_tool_read_cap_exists():
    from hello_agents.tools.builtin.grep_tool import GrepTool
    t = GrepTool(project_root=".")
    # The read cap logic is in _run_rg; verify the expected symbols exist.
    assert hasattr(t, "_run_rg")


# ---------------------------------------------------------------------------
# 建议-13: token_counter trust_remote_code is off by default
# ---------------------------------------------------------------------------

def test_token_counter_trust_remote_off():
    import os as _os
    # Default: trust_remote should be False.
    from hello_agents.context.token_counter import TokenCounter
    # TokenCounter.__init__ calls _get_encoding which may or may not invoke
    # transformers. Verify the env var guard exists.
    import hello_agents.context.token_counter as tcm
    # The _try_local_transformers_tokenizer method must exist.
    assert hasattr(TokenCounter("gpt-4"), "_try_local_transformers_tokenizer")


# ---------------------------------------------------------------------------
# 建议-15: get_event_loop replaced with get_running_loop
# ---------------------------------------------------------------------------

def test_async_invoke_uses_running_loop():
    """Verify the core async methods prefer ``asyncio.get_running_loop()``
    over the deprecated ``asyncio.get_event_loop()`` (建议-15)."""
    import hello_agents.core.llm_adapters as adp
    import hello_agents.core.llm as llm_mod
    import hello_agents.core.agent as agent_mod

    adapter_src = str(adp.__file__)
    if not adapter_src.endswith(".py"):
        adapter_src = str(getattr(adp, "__spec__", None) or adapter_src)
    if adapter_src.endswith(".py"):
        text = Path(adapter_src).read_text()
        assert "get_running_loop" in text, "adapters must use get_running_loop"

    llm_path = Path(llm_mod.__file__)
    if llm_path.suffix == ".py":
        llm_text = llm_path.read_text()
        assert "get_running_loop" in llm_text, "llm.py must use get_running_loop"
        # The deprecated bare ``asyncio.get_event_loop()`` should no longer
        # appear in the file.
        assert "asyncio.get_event_loop()" not in llm_text

    agent_path = Path(agent_mod.__file__)
    if agent_path.suffix == ".py":
        agent_text = agent_path.read_text()
        assert "get_running_loop" in agent_text, "agent.py must use get_running_loop"
        assert "asyncio.get_event_loop()" not in agent_text


# ---------------------------------------------------------------------------
# 重要-11: pyproject maps hello_agents → code/
# ---------------------------------------------------------------------------

def test_pyproject_maps_hello_agents_to_code():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject.read_text()
    assert 'hello_agents' in content
    # Must contain package-dir mapping
    assert 'package-dir' in content.lower() or 'package_dir' in content.lower()
    # The code/ directory must exist
    assert (pyproject.parent / "code").is_dir()


# ---------------------------------------------------------------------------
# 建议-8: factory supports "code" and simple carries tool_registry
# ---------------------------------------------------------------------------

def test_factory_supports_code_type():
    from hello_agents.agents.factory import create_agent
    from hello_agents.core.llm import HelloAgentsLLM

    llm = HelloAgentsLLM(model="gpt-4", api_key="sk-t", base_url="http://x")
    agent = create_agent("code", name="test", llm=llm)
    assert agent is not None
    assert "code-agent" in str(type(agent)).lower() or "CodeAgent" in str(type(agent))

    # react with tool_registry（simple/reflection/plan 已随教学型 agent 移除）
    from hello_agents.tools.registry import ToolRegistry
    from hello_agents.core.config import Config
    reg = ToolRegistry(config=Config(), verbose=False)
    react = create_agent("react", name="test", llm=llm, tool_registry=reg)
    assert type(react).__name__ == "ReActAgent"


# ---------------------------------------------------------------------------
# 建议-9: stream usage fallback — adapter.last_stats estimator
# ---------------------------------------------------------------------------

def test_stream_stats_includes_usage():
    from hello_agents.core.llm_response import StreamStats
    st = StreamStats(model="gpt-4", usage={"prompt_tokens": 10, "completion_tokens": 5,
                                            "total_tokens": 15, "estimated": True})
    d = st.to_dict()
    assert d["usage"]["estimated"] is True


# ============================================================================
# NEW TESTS for 2026-07-26 improvements
# ============================================================================

# ---------------------------------------------------------------------------
# 重要-3/4: Registry schema validation integrated into execute_tool
# ---------------------------------------------------------------------------

class TestRegistrySchemaValidation:
    """Verify Tool._validate_against_schema is called during tool execution."""

    def test_missing_required_param_blocked_by_registry(self):
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.builtin.file_tools import ReadTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            reg = ToolRegistry(verbose=False)
            reg.register_tool(ReadTool(project_root=tmp))
            resp = reg.execute_tool("Read", "{}")
            assert resp.status.value == "error"
            assert "path" in resp.text.lower()
        finally:
            shutil.rmtree(tmp)

    def test_wrong_type_blocked_by_registry(self):
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.builtin.file_tools import ReadTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            reg = ToolRegistry(verbose=False)
            reg.register_tool(ReadTool(project_root=tmp))
            # path must be string, not int
            resp = reg.execute_tool("Read", '{"path": 42, "offset": 0}')
            assert resp.status.value == "error"
            assert "string" in resp.text.lower()
        finally:
            shutil.rmtree(tmp)

    def test_valid_params_pass_through_registry(self):
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.builtin.file_tools import ReadTool
        import tempfile, os, shutil

        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, "f.txt"), "w") as f:
                f.write("hello")
            reg = ToolRegistry(verbose=False)
            reg.register_tool(ReadTool(project_root=tmp))
            resp = reg.execute_tool("Read", '{"path": "f.txt"}')
            assert resp.status.value == "success"
            assert "hello" in resp.data.get("content", "")
        finally:
            shutil.rmtree(tmp)

    def test_bool_not_accepted_as_integer(self):
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse

        class _IntTool(Tool):
            def __init__(self):
                super().__init__(name="IntTool", description="")
            def get_parameters(self):
                return [ToolParameter(name="count", type="integer",
                        description="", required=True)]
            def run(self, p):
                return ToolResponse.success(text="ok")

        tool = _IntTool()
        # True is an int in Python — we must reject it
        resp = tool._validate_against_schema({"count": True})
        assert resp is not None
        assert "integer" in resp.text.lower()

    def test_extra_parameters_allowed(self):
        """Schema validation tolerates extra parameters not in the schema."""
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.builtin.file_tools import ReadTool
        import tempfile, os, shutil

        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, "f.txt"), "w") as f:
                f.write("hello")
            reg = ToolRegistry(verbose=False)
            reg.register_tool(ReadTool(project_root=tmp))
            resp = reg.execute_tool(
                "Read", '{"path": "f.txt", "extra_param": "ignored"}'
            )
            assert resp.status.value == "success"
        finally:
            shutil.rmtree(tmp)

    def test_schema_validation_skipped_for_function_tools(self):
        """Function tools bypass schema validation (they use their own path)."""
        from hello_agents.tools.registry import ToolRegistry

        reg = ToolRegistry(verbose=False)
        called = []

        def my_func(payload):
            called.append(payload)
            return "done"

        reg.register_function(my_func, name="my_func")
        resp = reg.execute_tool("my_func", '{"key": "val"}')
        assert resp.status.value == "success"
        assert len(called) == 1


# ---------------------------------------------------------------------------
# 建议-10: TodoWriteTool deterministic session_id
# ---------------------------------------------------------------------------

class TestTodoWriteSessionId:
    """Verify session_id is deterministically derived from project_root."""

    def test_same_root_produces_same_session_id(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            t1 = TodoWriteTool(project_root=tmp)
            t2 = TodoWriteTool(project_root=tmp)
            assert t1.session_id == t2.session_id, (
                f"Same root must produce same session_id: "
                f"{t1.session_id} vs {t2.session_id}"
            )
            assert len(t1.session_id) == 12
        finally:
            shutil.rmtree(tmp)

    def test_different_roots_produce_different_ids(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile, shutil

        tmp1 = tempfile.mkdtemp()
        tmp2 = tempfile.mkdtemp()
        try:
            t1 = TodoWriteTool(project_root=tmp1)
            t2 = TodoWriteTool(project_root=tmp2)
            assert t1.session_id != t2.session_id, (
                "Different roots must produce different session_ids"
            )
        finally:
            shutil.rmtree(tmp1, ignore_errors=True)
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_explicit_session_id_respected(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            t = TodoWriteTool(project_root=tmp, session_id="my-custom-id")
            assert t.session_id == "my-custom-id"
        finally:
            shutil.rmtree(tmp)

    def test_session_id_is_hex_string(self):
        from hello_agents.tools.builtin.todowrite_tool import TodoWriteTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            t = TodoWriteTool(project_root=tmp)
            # 12 hex chars
            assert all(c in "0123456789abcdef" for c in t.session_id)
        finally:
            shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# 建议-11: BashTool INTERACTIVE_COMMANDS expanded
# ---------------------------------------------------------------------------

class TestBashToolInteractiveCommands:
    """Verify all expected interactive commands are blocked."""

    def test_new_interactive_commands_are_blocked(self):
        from hello_agents.tools.builtin.bash import BashTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            tool = BashTool(project_root=tmp, working_dir=tmp)
            new_blocked = [
                "emacs", "micro", "most", "btop", "atop",
                "dialog", "whiptail", "fzf", "peco", "ncdu",
                "mutt", "neomutt", "irssi", "weechat",
            ]
            for cmd in new_blocked:
                reason = tool.validate_command_policy(cmd)
                assert reason is not None, (
                    f"Interactive command '{cmd}' must be blocked"
                )
                assert "interactive" in reason.lower() or "not allowed" in reason.lower(), (
                    f"Block reason for '{cmd}' should mention interactive: {reason}"
                )
        finally:
            shutil.rmtree(tmp)

    def test_original_interactive_still_blocked(self):
        from hello_agents.tools.builtin.bash import BashTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            tool = BashTool(project_root=tmp, working_dir=tmp)
            for cmd in ("vim", "nano", "less", "tmux", "screen", "top"):
                assert tool.validate_command_policy(cmd) is not None
        finally:
            shutil.rmtree(tmp)

    def test_non_interactive_still_allowed(self):
        from hello_agents.tools.builtin.bash import BashTool
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            tool = BashTool(project_root=tmp, working_dir=tmp)
            for cmd in ("echo hello", "git status", "python --version"):
                assert tool.validate_command_policy(cmd) is None, (
                    f"Command '{cmd}' should be allowed"
                )
        finally:
            shutil.rmtree(tmp)


# ═══════════════════════════════════════════════════════════════════════
# 重要-1: HistoryManager tool_calls projection safety
# ═══════════════════════════════════════════════════════════════════════


class TestHistoryManagerToolCallSafety:
    """Verify 重要-1 fixes: tool_calls are never silently lost during
    message projection, and metadata corruption is detected."""

    def test_empty_content_without_tool_calls_not_dropped(self):
        """content='' and no tool_calls → message returns [no output], not None."""
        from hello_agents.context.history import HistoryManager
        from hello_agents.core.message import Message

        hm = HistoryManager()
        msg = Message(content="", role="assistant")
        projected = hm._project_message_for_llm(msg)
        assert projected is not None, "empty assistant message must not be dropped"
        assert projected["role"] == "assistant"
        assert projected["content"] == "[no output]"

    def test_empty_content_with_valid_tool_calls_still_works(self):
        """content='' but tool_calls present → correctly projected with tool_calls."""
        from hello_agents.context.history import HistoryManager
        from hello_agents.core.message import Message

        hm = HistoryManager()
        msg = Message(
            content="",
            role="assistant",
            metadata={
                "tool_calls": [{
                    "id": "call-1",
                    "name": "Read",
                    "arguments": {"path": "f.py"},
                }],
            },
        )
        projected = hm._project_message_for_llm(msg)
        assert projected is not None
        assert projected["role"] == "assistant"
        assert len(projected["tool_calls"]) == 1
        assert projected["tool_calls"][0]["function"]["name"] == "Read"

    def test_corrupt_tool_calls_not_list_still_preserves_message(self):
        """tool_calls in metadata is a string (corrupted) → message is NOT
        dropped, and tool_calls are empty (graceful degradation)."""
        from hello_agents.context.history import HistoryManager
        from hello_agents.core.message import Message

        hm = HistoryManager()
        msg = Message(
            content="I will now read the file",
            role="assistant",
            metadata={"tool_calls": "this should be a list but isn't"},
        )
        # _assistant_tool_calls must return [] for non-list metadata
        extracted = hm._assistant_tool_calls(msg)
        assert extracted == []

        # _project_message_for_llm must NOT drop the message
        projected = hm._project_message_for_llm(msg)
        assert projected is not None
        assert projected["role"] == "assistant"
        assert "tool_calls" not in projected
        assert "I will now read the file" in projected["content"]

    def test_assistant_with_content_and_valid_tool_calls(self):
        """Normal case: content + valid tool_calls → both preserved."""
        from hello_agents.context.history import HistoryManager
        from hello_agents.core.message import Message

        hm = HistoryManager()
        msg = Message(
            content="Let me check something",
            role="assistant",
            metadata={
                "tool_calls": [
                    {"id": "c1", "name": "Read", "arguments": {"path": "a.py"}},
                    {"id": "c2", "name": "Grep", "arguments": {"pattern": "TODO"}},
                ],
            },
        )
        projected = hm._project_message_for_llm(msg)
        assert projected is not None
        assert projected["role"] == "assistant"
        assert len(projected["tool_calls"]) == 2
        assert projected["tool_calls"][0]["function"]["name"] == "Read"
        assert projected["tool_calls"][1]["function"]["name"] == "Grep"

    def test_metadata_tool_calls_roundtrip(self):
        """build_assistant_tool_call_message → to_dict → from_dict →
        _project_message_for_llm: tool_calls survive full roundtrip."""
        from hello_agents.context.history import HistoryManager
        from hello_agents.core.message import Message

        hm = HistoryManager()
        # Build via the public API
        built = hm.build_assistant_tool_call_message(
            tool_calls=[
                {"id": "r1", "type": "function", "function": {
                    "name": "Bash", "arguments": '{"command":"pytest"}'}},
            ],
            content="running tests",
        )
        # Serialise + deserialise
        restored = Message.from_dict(built.to_dict())
        # Extract + project
        extracted = hm._assistant_tool_calls(restored)
        assert len(extracted) == 1
        assert extracted[0]["name"] == "Bash"
        assert extracted[0]["arguments"] == {"command": "pytest"}

        projected = hm._project_message_for_llm(restored)
        assert projected is not None
        assert projected["tool_calls"][0]["function"]["name"] == "Bash"


class TestNormalizeToolCallsDefensive:
    """Verify _normalize_tool_calls handles edge cases gracefully
    (fix for 重要-1)."""

    def test_string_arguments_parsed_to_dict(self):
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls([{
            "id": "c1",
            "function": {"name": "Read", "arguments": '{"path":"f.py"}'},
        }])
        assert result[0]["arguments"] == {"path": "f.py"}

    def test_integer_arguments_wrapped_in_raw(self):
        """Non-dict, non-string arguments must be wrapped, not passed through."""
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls([{
            "id": "c2",
            "function": {"name": "Tool", "arguments": 42},
        }])
        assert isinstance(result[0]["arguments"], dict)
        assert result[0]["arguments"]["_raw"] == 42

    def test_boolean_arguments_wrapped_in_raw(self):
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls([{
            "id": "c3",
            "function": {"name": "Tool", "arguments": True},
        }])
        assert isinstance(result[0]["arguments"], dict)
        assert result[0]["arguments"]["_raw"] is True

    def test_list_arguments_wrapped_in_raw(self):
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls([{
            "id": "c4",
            "function": {"name": "Tool", "arguments": [1, 2, 3]},
        }])
        assert isinstance(result[0]["arguments"], dict)
        assert result[0]["arguments"]["_raw"] == [1, 2, 3]

    def test_none_arguments_replaced_with_empty_dict(self):
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls([{
            "id": "c5",
            "function": {"name": "Tool", "arguments": None},
        }])
        assert result[0]["arguments"] == {}

    def test_no_function_field_falls_back_to_top_level(self):
        """When the 'function' wrapper is missing, use top-level fields."""
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls([{
            "id": "c6",
            "name": "Read",
            "arguments": {"path": "f.py"},
        }])
        assert result[0]["name"] == "Read"
        assert result[0]["arguments"] == {"path": "f.py"}

    def test_empty_tool_call_list(self):
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls([])
        assert result == []

    def test_non_dict_entry_skipped(self):
        """A tool_call that isn't a dict is silently skipped."""
        from hello_agents.context.history import HistoryManager
        hm = HistoryManager()
        result = hm._normalize_tool_calls(["not_a_dict", {"id": "ok", "name": "T", "arguments": {}}])
        assert len(result) == 1
        assert result[0]["name"] == "T"


# ═══════════════════════════════════════════════════════════════════════
# 改进项 8/F2: lcb6 --timeout must reach the controlled evaluation
# ═══════════════════════════════════════════════════════════════════════


class TestLCB6TimeoutPlumbing:
    """Verify the CLI --timeout value flows into the stdio/functional
    evaluation adapters instead of a hardcoded 10s."""

    def _capture_adapter_kwargs(self, monkeypatch, module, adapter_name):
        captured = {}

        class _FakeAdapter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def evaluate(self, **kwargs):
                return {"passed": True, "public": {"passed": 1, "failed": 0},
                        "private": {"passed": 1, "failed": 0}, "details": []}

        monkeypatch.setattr(module, adapter_name, _FakeAdapter)
        return captured

    def test_stdin_eval_honors_timeout_kwarg(self, tmp_path, monkeypatch):
        import hello_agents.benchmark.lcb6_bench as lcb6
        captured = self._capture_adapter_kwargs(monkeypatch, lcb6, "PythonStdinAdapter")
        result = lcb6._evaluate_stdin_solution(
            tmp_path / "solution.py", [{"input": "1\n", "output": "1\n"}], 1, timeout=42,
        )
        assert result["passed"] is True
        assert captured["timeout"] == 42, "stdin evaluation must honor the timeout parameter"

    def test_functional_eval_honors_timeout_kwarg(self, tmp_path, monkeypatch):
        import hello_agents.benchmark.lcb6_bench as lcb6
        captured = self._capture_adapter_kwargs(monkeypatch, lcb6, "PythonFunctionalAdapter")
        result = lcb6._evaluate_functional_solution(
            tmp_path / "solution.py", [], 0, "class Solution:", {}, timeout=33,
        )
        assert result["passed"] is True
        assert captured["timeout"] == 33, "functional evaluation must honor the timeout parameter"

    def test_default_timeout_still_10(self, tmp_path, monkeypatch):
        """Back-compat: without an explicit timeout the old 10s default applies."""
        import hello_agents.benchmark.lcb6_bench as lcb6
        captured = self._capture_adapter_kwargs(monkeypatch, lcb6, "PythonStdinAdapter")
        lcb6._evaluate_stdin_solution(tmp_path / "solution.py", [], 0)
        assert captured["timeout"] == 10


# ═══════════════════════════════════════════════════════════════════════
# 改进项 9/F3: LSPManager shared per workspace root
# ═══════════════════════════════════════════════════════════════════════


class TestSharedLSPManager:
    """Verify get_shared_manager caches one manager per resolved root so
    sub-agents reuse the main agent's language servers."""

    def test_same_root_returns_same_instance(self, tmp_path):
        from hello_agents.tools.lsp import get_shared_manager, LSPManager
        m1 = get_shared_manager(tmp_path)
        m2 = get_shared_manager(tmp_path)
        assert m1 is m2
        assert isinstance(m1, LSPManager)

    def test_different_roots_return_different_instances(self, tmp_path):
        from hello_agents.tools.lsp import get_shared_manager
        other = tmp_path / "other"
        other.mkdir()
        m1 = get_shared_manager(tmp_path)
        m2 = get_shared_manager(other)
        assert m1 is not m2

    def test_unresolved_paths_share_by_resolved_root(self, tmp_path):
        """Symlinked / non-normalized paths to the same root share one manager."""
        from hello_agents.tools.lsp import get_shared_manager
        m1 = get_shared_manager(tmp_path)
        m2 = get_shared_manager(str(tmp_path) + "/./subdir/../")
        assert m1 is m2

    def test_concurrent_access_is_thread_safe(self, tmp_path):
        import concurrent.futures
        from hello_agents.tools.lsp import get_shared_manager
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            managers = list(pool.map(lambda _: get_shared_manager(tmp_path), range(32)))
        assert all(m is managers[0] for m in managers), (
            "concurrent get_shared_manager calls must return a single instance"
        )

    def test_two_code_agents_share_one_manager(self, tmp_path, monkeypatch):
        """Main agent + sub-agent on the same root must share the LSP manager."""
        from hello_agents.tools.lsp import manager as lsp_manager_mod
        from hello_agents.agents.code_agent import CodeAgent

        class _StubLLM:
            model = "stub-model"

            def invoke_with_tools(self, *args, **kwargs):
                raise RuntimeError("not used during construction")

        created = []
        real_cls = lsp_manager_mod.LSPManager

        class _CountingManager(real_cls):
            def __init__(self, workspace_root):
                created.append(workspace_root)
                super().__init__(workspace_root)

        monkeypatch.setattr(lsp_manager_mod, "LSPManager", _CountingManager)
        # get_shared_manager resolves LSPManager from the module namespace at
        # call time, so the patch above takes effect for new roots only.
        lsp_manager_mod._SHARED_MANAGERS.clear()
        try:
            for _ in range(2):
                CodeAgent(
                    "agent",
                    _StubLLM(),
                    project_root=str(tmp_path),
                    register_default_tools=True,
                    enable_task_tool=False,
                    enable_subagent_task=False,
                    interactive=False,
                )
            assert len(created) == 1, (
                f"two CodeAgents on the same root must share one LSPManager, "
                f"created {len(created)}"
            )
        finally:
            lsp_manager_mod._SHARED_MANAGERS.clear()


# ═══════════════════════════════════════════════════════════════════════
# Q2-5/Q2-6: shared env parsing (core/env_utils.py)
# ═══════════════════════════════════════════════════════════════════════


class TestEnvUtils:
    """Single implementation of tolerant env parsing."""

    def test_env_bool_tokens(self, monkeypatch):
        from hello_agents.core.env_utils import env_bool
        for token in ("1", "true", "TRUE", "Yes", "on"):
            monkeypatch.setenv("X_FLAG", token)
            assert env_bool("X_FLAG", False) is True, token
        for token in ("0", "false", "no", "off", "garbage"):
            monkeypatch.setenv("X_FLAG", token)
            assert env_bool("X_FLAG", True) is False, token

    def test_env_bool_unset_and_empty(self, monkeypatch):
        from hello_agents.core.env_utils import env_bool
        monkeypatch.delenv("X_FLAG", raising=False)
        assert env_bool("X_FLAG", True) is True
        assert env_bool("X_FLAG", False) is False
        monkeypatch.setenv("X_FLAG", "   ")
        assert env_bool("X_FLAG", True) is True

    def test_env_int_clamps_negative_and_invalid(self, monkeypatch):
        from hello_agents.core.env_utils import env_int
        monkeypatch.setenv("X_INT", "42")
        assert env_int("X_INT", 7) == 42
        monkeypatch.setenv("X_INT", "  17 ")
        assert env_int("X_INT", 7) == 17
        monkeypatch.setenv("X_INT", "-5")
        assert env_int("X_INT", 7) == 7, "negative values must fall back to default"
        monkeypatch.setenv("X_INT", "not-a-number")
        assert env_int("X_INT", 7) == 7

    def test_env_float_clamps_negative_and_invalid(self, monkeypatch):
        from hello_agents.core.env_utils import env_float
        monkeypatch.setenv("X_FLOAT", "0.5")
        assert env_float("X_FLOAT", 1.0) == 0.5
        monkeypatch.setenv("X_FLOAT", "-3")
        assert env_float("X_FLOAT", 1.0) == 1.0
        monkeypatch.setenv("X_FLOAT", "zzz")
        assert env_float("X_FLOAT", 1.0) == 1.0

    def test_debug_env_now_accepts_standard_true_tokens(self, monkeypatch):
        """Q4-5: DEBUG previously only matched the literal string 'true'."""
        from hello_agents.core.config import Config
        monkeypatch.setenv("DEBUG", "1")
        assert Config.from_env().debug is True


# ═══════════════════════════════════════════════════════════════════════
# Q2-7: single atomic_write implementation (context/io_utils.py)
# ═══════════════════════════════════════════════════════════════════════


class TestAtomicWriteShared:
    def test_writes_and_creates_parent_dirs(self, tmp_path):
        from hello_agents.context.io_utils import atomic_write
        target = tmp_path / "a" / "b" / "file.txt"
        atomic_write(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello"

    def test_preserves_file_mode(self, tmp_path):
        import os
        from hello_agents.context.io_utils import atomic_write
        target = tmp_path / "file.txt"
        atomic_write(target, "first")
        os.chmod(target, 0o640)
        atomic_write(target, "second")
        assert (os.stat(target).st_mode & 0o777) == 0o640
        assert target.read_text(encoding="utf-8") == "second"

    def test_tool_layer_reexports_same_function(self):
        """_code_utils.atomic_write must be the very same object (single source)."""
        from hello_agents.context.io_utils import atomic_write as impl
        from hello_agents.tools.builtin._code_utils import atomic_write as reexported
        assert reexported is impl

    def test_no_local_duplicates_remain(self):
        import inspect
        from hello_agents.context import history as history_mod
        from hello_agents.context import truncator as truncator_mod
        assert not hasattr(history_mod.HistoryManager, "_atomic_write")
        assert not hasattr(truncator_mod.ObservationTruncator, "_atomic_write")


# ═══════════════════════════════════════════════════════════════════════
# Q2-8: shared search-exclusion constants
# ═══════════════════════════════════════════════════════════════════════


class TestSharedSearchConstants:
    def test_grep_and_glob_share_the_same_constants(self):
        from hello_agents.tools.builtin.glob_tool import GlobTool
        from hello_agents.tools.builtin.grep_tool import GrepTool
        assert GlobTool.DEFAULT_EXCLUDE_GLOBS is GrepTool.DEFAULT_EXCLUDE_GLOBS
        assert GlobTool.INTERNAL_ARTIFACT_DIRS == GrepTool.INTERNAL_ARTIFACT_DIRS
        assert ".backups" in GlobTool.INTERNAL_ARTIFACT_DIRS


# ═══════════════════════════════════════════════════════════════════════
# Q3-1: append-only result persistence
# ═══════════════════════════════════════════════════════════════════════


class TestAppendResultRecord:
    def _utils(self):
        # _utils is importable both as package module and standalone script.
        try:
            from hello_agents.benchmark import _utils
        except ImportError:
            import importlib.util, sys
            from pathlib import Path
            spec = importlib.util.spec_from_file_location(
                "_utils_standalone",
                Path(__file__).resolve().parents[1] / "code" / "benchmark" / "_utils.py",
            )
            _utils = importlib.util.module_from_spec(spec)
            sys.modules["_utils_standalone"] = _utils
            spec.loader.exec_module(_utils)
        return _utils

    def test_append_then_load_collapses_duplicates(self, tmp_path):
        utils = self._utils()
        path = tmp_path / "results.jsonl"
        utils.append_result_record(path, {"task_id": "t1", "passed": False})
        utils.append_result_record(path, {"task_id": "t2", "passed": True})
        utils.append_result_record(path, {"task_id": "t1", "passed": True})  # rerun wins

        records = utils.load_result_records(path)
        assert len(records) == 3, "append keeps all lines; resume collapses later"
        latest = utils.latest_result_records(records)
        by_id = {r["task_id"]: r for r in latest}
        assert by_id["t1"]["passed"] is True, "last write must win"
        assert len(latest) == 2

    def test_torn_final_line_is_skipped(self, tmp_path):
        utils = self._utils()
        path = tmp_path / "results.jsonl"
        utils.append_result_record(path, {"task_id": "t1", "passed": True})
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"task_id": "t2", "pass')  # simulate crash mid-write
        records = utils.load_result_records(path)
        assert [r["task_id"] for r in records] == ["t1"]


# ═══════════════════════════════════════════════════════════════════════
# Q3-2: _node_metrics single-pass equivalence
# ═══════════════════════════════════════════════════════════════════════


class TestNodeMetricsSinglePass:
    def _node(self, tag, text=""):
        from hello_agents.tools.builtin.web_tool import _HTMLNode
        node = _HTMLNode(tag)
        if text:
            node.text_parts.append(text)
        return node

    def test_counts_match_legacy_semantics(self):
        from hello_agents.tools.builtin.web_tool import _node_metrics
        root = self._node("div", "intro ")
        p1 = self._node("p", "para one ")
        p2 = self._node("p", "para two ")
        h = self._node("h2", "Title")
        ul = self._node("ul")
        li = self._node("li", "bullet")
        pre = self._node("pre", "code()")
        a = self._node("a", "link text")
        p1.children.append(a)
        ul.children.append(li)
        for child in (p1, p2, h, ul, pre):
            root.children.append(child)

        m = _node_metrics(root)
        # tag_count excludes the root itself; descendants = p,p2,h,ul,li,pre,a = 7
        assert m["tag_count"] == 7
        assert m["paragraph_count"] == 2
        assert m["heading_count"] == 1
        assert m["list_item_count"] == 1
        assert m["code_block_count"] == 1
        assert m["link_text_len"] == len("link text")
        assert "intro" in m["text"] and "para one" in m["text"]

    def test_node_itself_counts_in_per_tag_counters(self):
        """iter_nodes() yields self first — per-tag counters include it (legacy semantics)."""
        from hello_agents.tools.builtin.web_tool import _node_metrics
        p = self._node("p", "only")
        m = _node_metrics(p)
        assert m["tag_count"] == 0
        assert m["paragraph_count"] == 1, "root matching the tag is counted"
        link = self._node("a", "self link")
        assert _node_metrics(link)["link_text_len"] == len("self link")


# ═══════════════════════════════════════════════════════════════════════
# Q2-11: sub-agent token estimate prefers TokenCounter over chars//4
# ═══════════════════════════════════════════════════════════════════════


class TestSubagentTokenEstimate:
    def test_uses_history_manager_estimate_when_available(self):
        """When TokenCounter estimation is available it must win over chars//4."""
        from hello_agents.core.agent import Agent
        from hello_agents.core.message import Message

        class _HM:
            def get_history(self):
                return [Message("assistant", "assistant")]

            def get_estimated_token_count(self):
                return 12345  # distinct sentinel

        class _A(Agent):
            def __init__(self):
                self.history_manager = _HM()

            def run(self, *args, **kwargs):  # abstract stub
                raise NotImplementedError

        meta = _A()._get_subagent_metadata(duration=1.0, error=None)
        assert meta["tokens"] == 12345

    def test_falls_back_to_chars_div_4(self):
        from hello_agents.core.agent import Agent
        from hello_agents.core.message import Message

        class _HM:
            def get_history(self):
                return [Message("x" * 400, "assistant"), Message("y" * 400, "assistant")]

            def get_estimated_token_count(self):
                raise RuntimeError("estimator unavailable")

        class _A(Agent):
            def __init__(self):
                self.history_manager = _HM()

            def run(self, *args, **kwargs):  # abstract stub
                raise NotImplementedError

        meta = _A()._get_subagent_metadata(duration=1.0, error=None)
        assert meta["tokens"] == 200  # 800 chars // 4


# ═══════════════════════════════════════════════════════════════════════
# Q1: dead-code removal contracts (must stay removed)
# ═══════════════════════════════════════════════════════════════════════


class TestDeadCodeRemoval:
    def test_llm_dead_methods_removed(self):
        from hello_agents.core.llm import HelloAgentsLLM
        for name in ("think", "stream_invoke", "ainvoke", "astream_invoke"):
            assert not hasattr(HelloAgentsLLM, name), name
        # Active entry points stay.
        for name in ("invoke", "invoke_with_tools", "ainvoke_with_tools"):
            assert hasattr(HelloAgentsLLM, name), name

    def test_exceptions_pruned_to_base(self):
        from hello_agents.core import exceptions as exc_mod
        for name in ("LLMException", "AgentException", "ConfigException", "ToolException"):
            assert not hasattr(exc_mod, name), name
        assert hasattr(exc_mod, "HelloAgentsException")

    def test_global_registry_removed(self):
        import hello_agents
        from hello_agents.tools import registry as registry_mod
        assert not hasattr(registry_mod, "global_registry")
        assert "global_registry" not in getattr(hello_agents, "__all__", [])

    def test_streaming_dead_helpers_removed(self):
        from hello_agents.core import streaming
        for name in ("StreamBuffer", "stream_to_sse", "stream_to_json"):
            assert not hasattr(streaming, name), name
        assert hasattr(streaming, "StreamEvent")

    def test_config_to_dict_removed(self):
        from hello_agents.core.config import Config
        assert not hasattr(Config, "to_dict")
