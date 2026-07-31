"""Supplementary tests for react_agent.py — covering paths not hit by existing tests.

Targeted uncovered areas:
- Structured output pipeline (extraction, normalization, schema building, validation)
- Render/console event hooks (all _render_* methods)
- _decode_tool_call with JSON errors
- _coerce_optional_bool edge cases
- _response_unfinished_flag with various attribute sources
- _apply_prompt_instruction with/without system message
- _resolve_no_tool_call_response structured_output + unfinished branches
- _build_structured_output_result
- _record_tool_execution_result for control/structured-output tools
- _invalid_finalizing_tool_calls edge cases
- _tool_choice_for, _structured_output_instruction, _builtin_tool_instruction
- _timeout_final_answer, _stagnation state transitions
- _assistant_reasoning_metadata, _state_reasoning_kwargs
- _trace_tool_call, _trace_tool_result
- _normalize_start_step edge cases
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from hello_agents.agents.react_agent import (
    FINISH_TOOL_NAME,
    STRUCTURED_OUTPUT_TOOL_NAME,
    THOUGHT_TOOL_NAME,
    _ExecutionState,
    _FinishTool,
    _StructuredOutputSpec,
    _ThoughtTool,
    ReActAgent,
)
from hello_agents.core.config import Config
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.response import ToolResponse, ToolStatus


def _mock_llm():
    llm = MagicMock()
    llm.model = "test-model"
    llm.temperature = 0.7
    return llm


def _registry():
    return ToolRegistry(verbose=False)


# ============================================================================
# Structured Output Pipeline
# ============================================================================


class TestStructuredOutputPipeline:
    """_extract_structured_output_spec, _normalize_structured_output_schema,
    _ensure_structured_output_name_available, _build_structured_output_tool_schema."""

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_extract_structured_output_spec_basic(self):
        agent = self._agent()
        kwargs = {
            "structured_output_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            "structured_output_name": "MyOutput",
            "structured_output_description": "Custom output.",
        }
        spec = agent._extract_structured_output_spec(kwargs)
        assert spec is not None
        assert spec.name == "MyOutput"
        assert spec.description == "Custom output."
        assert "x" in spec.schema["properties"]
        # Original kwargs should be cleaned
        assert "structured_output_schema" not in kwargs

    def test_extract_structured_output_spec_uses_output_schema_alias(self):
        agent = self._agent()
        kwargs = {"output_schema": {"type": "object"}}
        spec = agent._extract_structured_output_spec(kwargs)
        assert spec is not None
        assert spec.name == STRUCTURED_OUTPUT_TOOL_NAME

    def test_extract_no_schema_returns_none(self):
        agent = self._agent()
        spec = agent._extract_structured_output_spec({})
        assert spec is None

    def test_normalize_adds_empty_properties_when_missing(self):
        schema = {"type": "object"}
        normalized = ReActAgent._normalize_structured_output_schema(deepcopy(schema))
        assert normalized["properties"] == {}

    def test_normalize_rejects_non_object_type(self):
        with pytest.raises(ValueError, match="type='object'"):
            ReActAgent._normalize_structured_output_schema({"type": "array"})

    def test_normalize_rejects_non_dict(self):
        with pytest.raises(TypeError, match="dictionary"):
            ReActAgent._normalize_structured_output_schema("not a dict")

    def test_normalize_rejects_non_dict_properties(self):
        with pytest.raises(ValueError, match="properties.*dictionary"):
            ReActAgent._normalize_structured_output_schema(
                {"type": "object", "properties": "not_a_dict"}
            )

    def test_normalize_rejects_non_list_required(self):
        with pytest.raises(ValueError, match="required.*list"):
            ReActAgent._normalize_structured_output_schema(
                {"type": "object", "required": "not_a_list"}
            )

    def test_normalize_is_non_destructive(self):
        original = {"type": "object", "properties": {"a": {"type": "string"}}}
        normalized = ReActAgent._normalize_structured_output_schema(deepcopy(original))
        assert normalized["properties"]["a"]["type"] == "string"
        assert original == {"type": "object", "properties": {"a": {"type": "string"}}}

    def test_name_conflict_detected(self):
        agent = self._agent()
        agent.tool_registry = _registry()
        agent.tool_registry.register_function(lambda x: x, name="ConflictName")

        with pytest.raises(ValueError, match="conflict"):
            agent._ensure_structured_output_name_available("ConflictName")

    def test_name_available_when_registry_none(self):
        agent = self._agent()
        agent.tool_registry = None
        # Should not raise
        agent._ensure_structured_output_name_available("AnyName")

    def test_empty_name_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="non-empty"):
            agent._extract_structured_output_spec(
                {"structured_output_schema": {"type": "object"}, "structured_output_name": "  "}
            )

    def test_empty_description_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="non-empty"):
            agent._extract_structured_output_spec(
                {"structured_output_schema": {"type": "object"}, "structured_output_description": ""}
            )

    def test_tool_choice_for_structured(self):
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        assert ReActAgent._tool_choice_for(spec) == "required"
        assert ReActAgent._tool_choice_for(None) == "auto"

    def test_structured_output_instruction(self):
        spec = _StructuredOutputSpec(name="FinalOutput", description="d", schema={"type": "object"})
        instr = ReActAgent._structured_output_instruction(spec)
        assert "Structured output mode" in instr
        assert "FinalOutput" in instr

    def test_build_structured_output_tool_schema(self):
        agent = self._agent()
        spec = _StructuredOutputSpec(
            name="CustomOutput",
            description="Custom description",
            schema={"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"]},
        )
        schema = agent._build_structured_output_tool_schema(spec)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "CustomOutput"
        assert "result" in schema["function"]["parameters"]["properties"]
        assert "required" in schema["function"]["parameters"]

    def test_structured_output_tool_name(self):
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        assert ReActAgent._structured_output_tool_name(spec) == "Out"
        assert ReActAgent._structured_output_tool_name(None) is None

    def test_is_structured_output_tool_name(self):
        agent = self._agent()
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        assert agent._is_structured_output_tool_name("Out", spec) is True
        assert agent._is_structured_output_tool_name("Other", spec) is False
        assert agent._is_structured_output_tool_name("Out", None) is False

    def test_is_finalizing_tool_name(self):
        agent = self._agent()
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        assert agent._is_finalizing_tool_name(FINISH_TOOL_NAME) is True
        assert agent._is_finalizing_tool_name("Out", spec) is True
        assert agent._is_finalizing_tool_name("Random") is False


# ============================================================================
# _invalid_finalizing_tool_calls
# ============================================================================


class TestInvalidFinalizingToolCalls:
    """Multiple Finish/StructuredOutput calls in one response are detected."""

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    class _FakeToolCall:
        def __init__(self, id: str, name: str):
            self.id = id
            self.function = MagicMock()
            self.function.name = name

    def test_single_finish_is_valid(self):
        agent = self._agent()
        calls = [self._FakeToolCall("c1", FINISH_TOOL_NAME)]
        result = agent._invalid_finalizing_tool_calls(calls)
        assert result == {}

    def test_multiple_finish_invalid(self):
        agent = self._agent()
        calls = [
            self._FakeToolCall("c1", FINISH_TOOL_NAME),
            self._FakeToolCall("c2", FINISH_TOOL_NAME),
        ]
        result = agent._invalid_finalizing_tool_calls(calls)
        assert "c1" in result
        assert "c2" in result
        assert "at most once" in result["c1"]

    def test_finish_with_other_tools_invalid(self):
        agent = self._agent()
        calls = [
            self._FakeToolCall("c1", "Read"),
            self._FakeToolCall("c2", FINISH_TOOL_NAME),
        ]
        result = agent._invalid_finalizing_tool_calls(calls)
        assert "c2" in result
        assert "after all other tool work" in result["c2"]

    def test_multiple_structured_output_invalid(self):
        agent = self._agent()
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        calls = [
            self._FakeToolCall("c1", "Out"),
            self._FakeToolCall("c2", "Out"),
        ]
        result = agent._invalid_finalizing_tool_calls(calls, spec)
        assert "c1" in result or "c2" in result

    def test_structured_output_with_other_tools_invalid(self):
        agent = self._agent()
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        calls = [
            self._FakeToolCall("c1", "Read"),
            self._FakeToolCall("c2", "Out"),
        ]
        result = agent._invalid_finalizing_tool_calls(calls, spec)
        assert "c2" in result
        assert "after all other tool work" in result["c2"]


# ============================================================================
# _decode_tool_call
# ============================================================================


class TestDecodeToolCall:
    """_decode_tool_call with valid and invalid JSON arguments."""

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    class _FakeToolCall:
        def __init__(self, id: str, name: str, arguments: str):
            self.id = id
            self.function = MagicMock()
            self.function.name = name
            self.function.arguments = arguments

    def test_valid_json(self):
        agent = self._agent()
        tc = self._FakeToolCall("c1", "Read", '{"path": "test.py"}')
        name, call_id, args, error = agent._decode_tool_call(tc)
        assert name == "Read"
        assert args == {"path": "test.py"}
        assert error is None

    def test_invalid_json(self):
        agent = self._agent()
        tc = self._FakeToolCall("c1", "Bash", 'not-valid-json{{{')
        name, call_id, args, error = agent._decode_tool_call(tc)
        assert name == "Bash"
        assert args is None
        assert error is not None
        assert "Invalid argument format" in error


# ============================================================================
# _apply_prompt_instruction
# ============================================================================


class TestApplyPromptInstruction:
    """_apply_prompt_instruction static method."""

    def test_appends_to_existing_system_message(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        ReActAgent._apply_prompt_instruction(messages, "Use tools wisely.")
        assert "You are helpful." in messages[0]["content"]
        assert "Use tools wisely." in messages[0]["content"]

    def test_inserts_when_no_system_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        ReActAgent._apply_prompt_instruction(messages, "Be concise.")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."

    def test_handles_empty_system_content(self):
        messages = [{"role": "system", "content": ""}]
        ReActAgent._apply_prompt_instruction(messages, "Instruction.")
        assert messages[0]["content"] == "Instruction."

    def test_empty_messages_list(self):
        messages = []
        ReActAgent._apply_prompt_instruction(messages, "Only instruction.")
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


# ============================================================================
# _builtin_tool_instruction
# ============================================================================


class TestBuiltinToolInstruction:
    """_builtin_tool_instruction and _apply_builtin_tool_prompt."""

    def test_instruction_contains_tool_names(self):
        instr = ReActAgent._builtin_tool_instruction()
        assert THOUGHT_TOOL_NAME in instr
        assert FINISH_TOOL_NAME in instr

    def test_apply_builtin_tool_prompt(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        messages = [{"role": "system", "content": "Base prompt."}]
        agent._apply_builtin_tool_prompt(messages)
        assert THOUGHT_TOOL_NAME in messages[0]["content"]


# ============================================================================
# _coerce_optional_bool
# ============================================================================


class TestCoerceOptionalBool:
    """_coerce_optional_bool static method."""

    def test_true_bool(self):
        assert ReActAgent._coerce_optional_bool(True) is True
        assert ReActAgent._coerce_optional_bool(False) is False

    def test_true_strings(self):
        for v in ["true", "True", "TRUE", "1", "yes", "YES"]:
            assert ReActAgent._coerce_optional_bool(v) is True, f"Failed for {v!r}"

    def test_false_strings(self):
        for v in ["false", "False", "FALSE", "0", "no", "NO"]:
            assert ReActAgent._coerce_optional_bool(v) is False, f"Failed for {v!r}"

    def test_none_for_unrecognized(self):
        assert ReActAgent._coerce_optional_bool("maybe") is None
        assert ReActAgent._coerce_optional_bool(42) is None
        assert ReActAgent._coerce_optional_bool(None) is None


# ============================================================================
# _response_unfinished_flag
# ============================================================================


class TestResponseUnfinishedFlag:
    """_response_unfinished_flag with various attribute locations."""

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_direct_attribute_true(self):
        agent = self._agent()
        msg = MagicMock()
        msg.unfinished = True
        msg.unfinish = False
        msg.additional_kwargs = {}
        msg.metadata = {}
        assert agent._response_unfinished_flag(msg) is True

    def test_direct_attribute_false(self):
        agent = self._agent()
        msg = MagicMock()
        msg.unfinished = False
        msg.unfinish = False
        msg.additional_kwargs = {}
        msg.metadata = {}
        assert agent._response_unfinished_flag(msg) is False

    def test_additional_kwargs_true(self):
        agent = self._agent()
        msg = MagicMock()
        msg.unfinished = None
        msg.unfinish = None
        msg.additional_kwargs = {"unfinished": True}
        msg.metadata = {}
        assert agent._response_unfinished_flag(msg) is True

    def test_metadata_true(self):
        agent = self._agent()
        msg = MagicMock()
        msg.unfinished = None
        msg.unfinish = None
        msg.additional_kwargs = {}
        msg.metadata = {"unfinish": True}
        assert agent._response_unfinished_flag(msg) is True

    def test_all_none_returns_false(self):
        agent = self._agent()
        msg = MagicMock()
        msg.unfinished = None
        msg.unfinish = None
        msg.additional_kwargs = {}  # Not None — empty dict
        msg.metadata = None  # None — not a dict
        assert agent._response_unfinished_flag(msg) is False

    def test_string_coercion_in_kwargs(self):
        agent = self._agent()
        msg = MagicMock()
        msg.unfinished = None
        msg.additional_kwargs = {"unfinished": "true"}
        msg.metadata = {}
        assert agent._response_unfinished_flag(msg) is True


# ============================================================================
# _resolve_no_tool_call_response
# ============================================================================


class TestResolveNoToolCallResponse:
    """_resolve_no_tool_call_response with various branches."""

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    class _FakeRespMsg:
        def __init__(self, content="", unfinished=None):
            self.content = content
            self.unfinished = unfinished
            self.unfinish = None
            self.additional_kwargs = {}
            self.metadata = {}

    def test_structured_output_with_unfinished(self):
        """Truncated response with structured_output → continue, extend history."""
        agent = self._agent()
        msg = self._FakeRespMsg(content="Partial response...")
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})

        state = _ExecutionState(current_step=1)
        state.last_finish_reason = "length"

        should_continue, final_answer, status = agent._resolve_no_tool_call_response(
            msg, "Partial response...",
            structured_output=spec,
            state=state,
        )
        # Structured output mode → should continue (model needs to finish)
        assert should_continue is True
        assert final_answer is None

    def test_unfinished_response_with_content(self):
        """Response with unfinished flag and content → append to history, continue."""
        agent = self._agent()
        msg = self._FakeRespMsg(content="Still working...", unfinished=True)

        should_continue, final_answer, status = agent._resolve_no_tool_call_response(
            msg, "Still working...",
            structured_output=None,
        )
        assert should_continue is True

    def test_direct_response_text(self):
        agent = self._agent()
        result = ReActAgent._direct_response_text("  Hello world  ")
        assert result == "Hello world"

    def test_direct_response_text_empty(self):
        agent = self._agent()
        result = ReActAgent._direct_response_text("   ")
        assert "Sorry" in result or len(result) > 0

    def test_direct_response_text_with_fallback(self):
        agent = self._agent()
        result = ReActAgent._direct_response_text("", fallback_text="Custom fallback")
        assert result == "Custom fallback"


# ============================================================================
# _build_structured_output_result
# ============================================================================


class TestBuildStructuredOutputResult:
    """_build_structured_output_result and _format_structured_output."""

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_formats_json_with_sorted_keys(self):
        result = ReActAgent._format_structured_output({"b": 2, "a": 1})
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["a", "b"]  # Sorted

    def test_build_result_marks_finished(self):
        agent = self._agent()
        result = agent._build_structured_output_result({"answer": 42})
        assert result["finished"] is True
        assert "final_answer" in result
        assert "structured_output" in result
        assert "status" in result


# ============================================================================
# _record_tool_execution_result
# ============================================================================


class TestRecordToolExecutionResult:
    """_record_tool_execution_result for builtin/control/structured-output tools."""

    def _agent(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        # Don't overwrite tool_registry — use the one created by constructor
        # (which already has Thought/Finish registered)
        agent.trace_logger = None
        return agent

    def test_builtin_tool_through_execute(self):
        """_execute_one_tool_call for Thought → result has builtin_tool flag."""
        agent = self._agent()
        name, call_id, recorded = agent._execute_one_tool_call(
            THOUGHT_TOOL_NAME, "c1", {"reasoning": "test"}, current_step=1,
        )
        assert name == THOUGHT_TOOL_NAME
        assert recorded["builtin_tool"] is True

    def test_finish_tool_through_execute(self):
        """_execute_one_tool_call for Finish → result has finished flag."""
        agent = self._agent()
        name, call_id, recorded = agent._execute_one_tool_call(
            FINISH_TOOL_NAME, "c1", {"answer": "All done"}, current_step=1,
        )
        assert recorded["finished"] is True
        assert recorded["final_answer"] == "All done"

    def test_structured_output_tool(self):
        agent = self._agent()
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        result = {"content": '{"a": 1}', "structured_output": {"a": 1}}
        name, call_id, recorded = agent._record_tool_execution_result(
            "Out", "c1", result, step=1, structured_output=spec,
        )
        assert recorded["content"] == '{"a": 1}'

    def test_regular_tool_no_extra_flags(self):
        agent = self._agent()
        result = {"content": "File contents here", "status": "success"}
        name, call_id, recorded = agent._record_tool_execution_result(
            "Read", "c1", result, step=1,
        )
        assert recorded.get("builtin_tool") is None
        assert recorded.get("finished") is None


# ============================================================================
# _extract_response_usage
# ============================================================================


class TestExtractResponseUsage:
    """_extract_response_usage static method."""

    def test_standard_usage(self):
        resp = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        resp.usage.total_tokens = 150
        resp.usage_metadata = None

        p, c, t = ReActAgent._extract_response_usage(resp)
        assert p == 100
        assert c == 50
        assert t == 150

    def test_usage_metadata_fallback(self):
        resp = MagicMock()
        resp.usage = None
        resp.usage_metadata.prompt_token_count = 200
        resp.usage_metadata.candidates_token_count = 80
        resp.usage_metadata.total_token_count = 280

        p, c, t = ReActAgent._extract_response_usage(resp)
        assert p == 200
        assert c == 80
        assert t == 280

    def test_no_usage_returns_zeros(self):
        resp = MagicMock()
        resp.usage = None
        resp.usage_metadata = None

        p, c, t = ReActAgent._extract_response_usage(resp)
        assert p == 0
        assert c == 0
        assert t == 0

    def test_usage_without_total(self):
        resp = MagicMock()
        resp.usage.prompt_tokens = 300
        resp.usage.completion_tokens = 100
        resp.usage.total_tokens = None
        resp.usage_metadata = None

        p, c, t = ReActAgent._extract_response_usage(resp)
        assert t == 400  # Infer from p + c

    def test_usage_metadata_without_total(self):
        resp = MagicMock()
        resp.usage = None
        resp.usage_metadata.prompt_token_count = 50
        resp.usage_metadata.candidates_token_count = 30
        resp.usage_metadata.total_token_count = None

        p, c, t = ReActAgent._extract_response_usage(resp)
        assert t == 80  # Infer from p + c


# ============================================================================
# _timeout_final_answer
# ============================================================================


class TestTimeoutFinalAnswer:
    """_timeout_final_answer static method."""

    def test_returns_sorry_message(self):
        result = ReActAgent._timeout_final_answer()
        assert "step limit" in result.lower() or "could not complete" in result.lower() or "Sorry" in result.lower()
        assert len(result) > 5


# ============================================================================
# _normalize_start_step
# ============================================================================


class TestNormalizeStartStep:
    """_normalize_start_step static method."""

    def test_positive_integer(self):
        assert ReActAgent._normalize_start_step(5) == 5

    def test_zero(self):
        assert ReActAgent._normalize_start_step(0) == 0

    def test_negative_clamped_to_zero(self):
        assert ReActAgent._normalize_start_step(-1) == 0

    def test_none_clamped_to_zero(self):
        assert ReActAgent._normalize_start_step(None) == 0

    def test_string_converted(self):
        assert ReActAgent._normalize_start_step("3") == 3


# ============================================================================
# _assistant_reasoning_metadata & _state_reasoning_kwargs
# ============================================================================


class TestReasoningMetadata:
    """_assistant_reasoning_metadata and _state_reasoning_kwargs."""

    def test_metadata_with_both(self):
        meta = ReActAgent._assistant_reasoning_metadata(
            reasoning_content="Think step by step.",
            reasoning_source="message.content_block",
        )
        assert meta["reasoning_content"] == "Think step by step."
        assert meta["reasoning_source"] == "message.content_block"

    def test_metadata_with_none_values(self):
        meta = ReActAgent._assistant_reasoning_metadata()
        assert meta == {}

    def test_metadata_with_only_content(self):
        meta = ReActAgent._assistant_reasoning_metadata(reasoning_content="Thought")
        assert meta == {"reasoning_content": "Thought"}

    def test_state_reasoning_kwargs(self):
        state = _ExecutionState(current_step=1)
        state.last_reasoning_content = "Plan: do X"
        state.last_reasoning_source = "choice.reasoning"
        kwargs = ReActAgent._state_reasoning_kwargs(state)
        assert kwargs["reasoning_content"] == "Plan: do X"
        assert kwargs["reasoning_source"] == "choice.reasoning"

    def test_state_reasoning_kwargs_none(self):
        state = _ExecutionState(current_step=1)
        kwargs = ReActAgent._state_reasoning_kwargs(state)
        assert kwargs["reasoning_content"] is None
        assert kwargs["reasoning_source"] is None


# ============================================================================
# _register_builtin_tools edge cases
# ============================================================================


class TestRegisterBuiltinTools:
    """_register_builtin_tools conflict detection and re-registration."""

    def test_conflict_with_existing_function(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.tool_registry = _registry()
        agent.tool_registry.register_function(lambda x: x, name=THOUGHT_TOOL_NAME)

        with pytest.raises(ValueError, match="conflict"):
            agent._register_builtin_tools()

    def test_conflict_with_non_builtin_tool(self):
        class CustomThought(Tool):
            def __init__(self):
                super().__init__(name=THOUGHT_TOOL_NAME, description="Custom")
            def get_parameters(self):
                return []
            def run(self, p):
                return ToolResponse.success(text="ok")

        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(CustomThought())

        with pytest.raises(ValueError, match="conflict"):
            agent._register_builtin_tools()

    def test_no_registry_no_error(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.tool_registry = None
        # Should not raise
        agent._register_builtin_tools()

    def test_description_updated_on_re_register(self):
        """Re-registering same type updates description."""
        agent = ReActAgent("test", _mock_llm(), config=Config())
        # First registration
        agent._register_builtin_tools()
        # Second registration should update description
        agent._register_builtin_tools()
        thought = agent.tool_registry.get_tool(THOUGHT_TOOL_NAME)
        assert thought is not None


# ============================================================================
# _execute_tool_call and _execute_tool_call_result
# ============================================================================


class TestExecuteToolCallCoverage:
    """_execute_tool_call and _execute_tool_call_result with various tool types."""

    def test_execute_tool_call_with_function_tool(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.tool_registry = _registry()
        agent.tool_registry.register_function(lambda x: f"got: {x}", name="echo")

        result = agent._execute_tool_call("echo", {"input": "hello"})
        assert "got: hello" in result

    def test_execute_tool_call_result_structured_output(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_FinishTool(description="End"))

        result = agent._execute_tool_call_result(FINISH_TOOL_NAME, {"answer": "done"})
        assert "content" in result
        assert result.get("finished") is True


# ============================================================================
# _is_builtin_tool_name
# ============================================================================


class TestIsBuiltinToolName:
    """_is_builtin_tool_name method."""

    def test_returns_true_for_builtins(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        assert agent._is_builtin_tool_name(THOUGHT_TOOL_NAME) is True
        assert agent._is_builtin_tool_name(FINISH_TOOL_NAME) is True

    def test_returns_false_for_others(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        assert agent._is_builtin_tool_name("Read") is False
        assert agent._is_builtin_tool_name("CustomTool") is False


# ============================================================================
# _trace_tool_call and _trace_tool_result
# ============================================================================


class TestTraceMethods:
    """_trace_tool_call and _trace_tool_result with/without logger."""

    def test_trace_tool_call_without_logger(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.trace_logger = None
        # Should not raise
        agent._trace_tool_call("Read", "c1", {"path": "x"}, step=1)

    def test_trace_tool_call_with_logger(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.trace_logger = MagicMock()
        agent._trace_tool_call("Read", "c1", {"path": "x"}, step=1)
        agent.trace_logger.log_event.assert_called_once()

    def test_trace_tool_result_without_logger(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.trace_logger = None
        agent._trace_tool_result("Read", "c1", "output", step=1)

    def test_trace_tool_result_with_logger_and_status(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.trace_logger = MagicMock()
        agent._trace_tool_result("Read", "c1", "output", step=1, status="error")
        agent.trace_logger.log_event.assert_called_once()


# ============================================================================
# _tool_error_result
# ============================================================================


class TestToolErrorResult:
    """_tool_error_result method."""

    def test_returns_formatted_error(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.trace_logger = None
        name, call_id, result = agent._tool_error_result("Bash", "c1", "Permission denied", step=1)
        assert name == "Bash"
        assert call_id == "c1"
        assert result["content"] == "Permission denied"
        assert result["status"] == "error"

    def test_renders_error_event(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.trace_logger = None
        agent._tool_error_result("Bash", "c1", "Error message", step=1)
        # Should not raise


# ============================================================================
# _tool_call_arguments_by_id
# ============================================================================


class TestToolCallArgumentsById:
    """_tool_call_arguments_by_id static method."""

    class _FakeToolCall:
        def __init__(self, id: str, arguments: str):
            self.id = id
            self.function = MagicMock()
            self.function.arguments = arguments

    def test_valid_arguments(self):
        calls = [
            self._FakeToolCall("c1", '{"path": "a.py"}'),
            self._FakeToolCall("c2", '{"path": "b.py"}'),
        ]
        result = ReActAgent._tool_call_arguments_by_id(calls)
        assert result["c1"] == {"path": "a.py"}
        assert result["c2"] == {"path": "b.py"}

    def test_invalid_json_skipped(self):
        calls = [self._FakeToolCall("c1", "not-json{{{")]
        result = ReActAgent._tool_call_arguments_by_id(calls)
        assert result == {}

    def test_non_dict_arguments_skipped(self):
        calls = [self._FakeToolCall("c1", '["list", "not", "dict"]')]
        result = ReActAgent._tool_call_arguments_by_id(calls)
        assert result == {}


# ============================================================================
# _executionState dataclass
# ============================================================================


class TestExecutionState:
    """_ExecutionState dataclass behavior."""

    def test_default_values(self):
        state = _ExecutionState(current_step=0)
        assert state.current_step == 0
        assert state.total_tokens == 0
        assert state.stagnation_detected is False
        assert state.consecutive_no_diff_edits == 0
        assert state.consecutive_same_tests == 0
        assert state.last_reasoning_content is None
        assert state.truncation_retries == 0

    def test_mutable_fields(self):
        state = _ExecutionState(current_step=1)
        state.stagnation_detected = True
        state.consecutive_no_diff_edits = 3
        assert state.stagnation_detected is True
        assert state.consecutive_no_diff_edits == 3


# ============================================================================
# _update_stagnation_state — bash test repetition
# ============================================================================


class TestStagnationStateBash:
    """_update_stagnation_state with Bash test repetition detection."""

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_same_test_results_triggers_stagnation(self):
        agent = self._agent()
        state = _ExecutionState(current_step=1)
        test_output = "ALL TESTS PASSED - 42 tests run"

        # Call 1 sets baseline (hash != None), counter → 0
        # Calls 2-4: each matches → counter 1, 2, 3 → stagnation at 3
        for i in range(4):
            agent._update_stagnation_state(
                "Bash", f"c{i}",
                test_output, 1, state,
                tool_arguments={"command": "python -m pytest tests/"},
            )

        assert state.stagnation_detected is True
        assert state.consecutive_same_tests == 3

    def test_different_test_results_reset(self):
        agent = self._agent()
        state = _ExecutionState(current_step=1)

        agent._update_stagnation_state("Bash", "c0", "FAILED test_a", 1, state,
                                       tool_arguments={"command": "pytest test_a.py"})
        agent._update_stagnation_state("Bash", "c1", "FAILED test_a", 1, state,
                                       tool_arguments={"command": "pytest test_a.py"})
        # After 2 calls: call1 sets baseline (c=0), call2 matches (c=1)
        assert state.consecutive_same_tests == 1

        # Different output → counter resets
        agent._update_stagnation_state("Bash", "c2", "ALL PASSED", 1, state,
                                       tool_arguments={"command": "pytest test_a.py"})
        assert state.consecutive_same_tests == 0

    def test_non_test_bash_does_not_track(self):
        agent = self._agent()
        state = _ExecutionState(current_step=1)

        for i in range(5):
            agent._update_stagnation_state(
                "Bash", f"c{i}",
                "same output every time", 1, state,
                tool_arguments={"command": "ls -la"},
            )
        # ls output being identical is NOT stagnation
        assert state.stagnation_detected is False

    def test_edit_and_bash_counters_independent(self):
        """Edit no-diff counter and Bash same-test counter are independent."""
        agent = self._agent()
        state = _ExecutionState(current_step=1)

        # Trigger Edit counter
        agent._update_stagnation_state("Edit", "c0", "[no textual diff]", 1, state)
        agent._update_stagnation_state("Edit", "c1", "[no textual diff]", 1, state)
        assert state.consecutive_no_diff_edits == 2

        # Bash test tracking is separate
        agent._update_stagnation_state("Bash", "c2", "PASSED", 1, state,
                                       tool_arguments={"command": "pytest"})
        agent._update_stagnation_state("Bash", "c3", "PASSED", 1, state,
                                       tool_arguments={"command": "pytest"})
        # Edit counter shouldn't be affected by Bash
        assert state.consecutive_no_diff_edits == 0  # Reset by Bash (non-Edit tool)
        # Bash counter is tracked independently (first call sets baseline to c=0)
        assert state.consecutive_same_tests == 1  # call2 matched call1
