"""Unit tests for code/agents/ and code/core/agent.py.

Covers:
- Agent base: _build_tool_schemas, _convert_parameter_types, _map_parameter_type,
  _prepare_tool_registry_input, _format_tool_response_text, _tool_history_metadata
- ReActAgent: Thought/Finish tools, _build_structured_output_tool_schema,
  stagnation detection, tool execution
- CodeAgent: construction, register_default_tools, set_working_dir,
  _get_context_system_prompt, _create_subagent, compact
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from code.agents.code_agent import CodeAgent
from code.agents.react_agent import (
    FINISH_TOOL_NAME,
    THOUGHT_TOOL_NAME,
    _FinishTool,
    _ThoughtTool,
    ReActAgent,
)
from code.core.config import Config
from code.core.llm import HelloAgentsLLM
from code.core.message import Message
from code.tools.base import Tool, ToolParameter
from code.tools.registry import ToolRegistry
from code.tools.response import ToolResponse, ToolStatus


def _mock_llm():
    llm = MagicMock(spec=HelloAgentsLLM)
    llm.model = "test-model"
    llm.temperature = 0.7
    return llm


def _registry():
    return ToolRegistry(verbose=False)


class _SimpleTool(Tool):
    def __init__(self, name="test_tool", description="Test tool"):
        super().__init__(name=name, description=description)

    def get_parameters(self):
        return [
            ToolParameter(name="path", type="string", description="A path", required=True),
            ToolParameter(name="count", type="integer", description="A count", required=False, default=0),
        ]

    def run(self, parameters):
        return ToolResponse.success(text="ok")


# ────────────────────────────────────────────────────────────────────────────
# Agent base — tested via ReActAgent
# ────────────────────────────────────────────────────────────────────────────


class TestAgentBuildToolSchemas:

    @pytest.fixture
    def agent(self):
        """Concrete ReActAgent for testing base class methods."""
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_no_registry_raises_or_handles_gracefully(self):
        """ReActAgent always creates a registry; verify builtin tools are included."""
        agent = ReActAgent("test", _mock_llm(), config=Config())
        schemas = agent._build_tool_schemas()
        # ReActAgent auto-registers Skill, TodoWrite, Thought, Finish
        names = {s["function"]["name"] for s in schemas}
        assert "Thought" in names
        assert "Finish" in names

    def test_empty_registry_no_builtins(self, agent):
        """After unregistering all tools, schema is empty."""
        for name in list(agent.tool_registry.list_tools()):
            agent.tool_registry.unregister(name)
        schemas = agent._build_tool_schemas()
        assert schemas == []

    def test_tool_object_schema(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        schemas = agent._build_tool_schemas()
        assert len(schemas) == 1
        f = schemas[0]["function"]
        assert f["name"] == "test_tool"
        assert "path" in f["parameters"]["properties"]
        assert f["parameters"]["required"] == ["path"]

    def test_function_tool_schema(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_function(lambda x: x, name="my_func", description="f")
        schemas = agent._build_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "my_func" in names

    def test_multiple_tools(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool(name="a"))
        agent.tool_registry.register_tool(_SimpleTool(name="b"))
        assert len(agent._build_tool_schemas()) == 2

    def test_broken_get_parameters(self, agent):
        class Broken(Tool):
            def __init__(self):
                super().__init__(name="b", description="b")
            def get_parameters(self):
                raise RuntimeError("boom")
            def run(self, p):
                return ToolResponse.success(text="ok")

        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(Broken())
        schemas = agent._build_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["parameters"]["properties"] == {}


class TestAgentParameterConversion:

    @pytest.fixture
    def agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_map_known(self):
        for t in ("string", "integer", "number", "boolean", "array", "object"):
            assert ReActAgent._map_parameter_type(t) == t

    def test_map_unknown(self):
        assert ReActAgent._map_parameter_type("unknown") == "string"

    def test_convert_integer(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        assert agent._convert_parameter_types("test_tool", {"count": "42"})["count"] == 42

    def test_convert_float(self, agent):
        class FT(Tool):
            def __init__(self):
                super().__init__(name="ft", description="f")
            def get_parameters(self):
                return [ToolParameter(name="val", type="number", description="v", required=True)]
            def run(self, p):
                return ToolResponse.success(text="ok")

        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(FT())
        r = agent._convert_parameter_types("ft", {"val": "3.14"})
        assert isinstance(r["val"], float) and r["val"] == 3.14

    def test_convert_bool(self, agent):
        class BT(Tool):
            def __init__(self):
                super().__init__(name="bt", description="b")
            def get_parameters(self):
                return [ToolParameter(name="flag", type="boolean", description="f", required=True)]
            def run(self, p):
                return ToolResponse.success(text="ok")

        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(BT())
        c = agent._convert_parameter_types
        assert c("bt", {"flag": "true"})["flag"] is True
        assert c("bt", {"flag": "false"})["flag"] is False
        assert c("bt", {"flag": True})["flag"] is True

    def test_non_dict_passthrough(self, agent):
        assert agent._convert_parameter_types("any", "not_dict") == "not_dict"

    def test_unknown_tool_passthrough(self, agent):
        assert agent._convert_parameter_types("nonexistent", {"x": 1}) == {"x": 1}

    def test_invalid_value_kept(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        assert agent._convert_parameter_types("test_tool", {"count": "abc"})["count"] == "abc"


class TestAgentPrepareToolRegistryInput:

    @pytest.fixture
    def agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_dict_input(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        r = agent._prepare_tool_registry_input("test_tool", {"path": "/x", "count": "5"})
        assert r["count"] == 5

    def test_non_dict_wrapped(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        assert agent._prepare_tool_registry_input("test_tool", "str") == {"input": "str"}

    def test_function_tool(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_function(lambda x: x, name="f1")
        assert agent._prepare_tool_registry_input("f1", {"input": "payload"}) == "payload"

    def test_no_registry_passthrough(self, agent):
        assert agent._prepare_tool_registry_input("any", {"x": 1}) == {"x": 1}


class TestAgentToolResponseFormat:

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_success(self):
        text = self._agent()._format_tool_response_text("Read", ToolResponse.success(text="Done"))
        assert "Done" in text and "❌" not in text

    def test_error(self):
        text = self._agent()._format_tool_response_text("Bash", ToolResponse.error(code=None, message="Fail"))
        assert "❌" in text

    def test_partial(self):
        text = self._agent()._format_tool_response_text("Read", ToolResponse.partial(text="Partial"))
        assert "⚠️" in text

    def test_history_metadata(self):
        resp = ToolResponse.success(text="OK", data={"full_output_path": "/tmp/o.json", "stderr": "err"})
        meta = self._agent()._tool_history_metadata("Bash", resp)
        assert meta["tool_name"] == "Bash"
        assert meta["full_output_path"] == "/tmp/o.json"
        assert "stderr" not in (meta.get("tool_data") or {})

    def test_compactable_set(self):
        for t in ("Read", "Bash", "Grep", "Glob", "Edit", "Write"):
            assert t in ReActAgent.COMPACTABLE_TOOL_OUTPUTS


# ────────────────────────────────────────────────────────────────────────────
# Thought & Finish Tools
# ────────────────────────────────────────────────────────────────────────────


class TestThoughtFinishTools:

    def test_thought_records(self):
        tool = _ThoughtTool(description="Think")
        assert tool.name == THOUGHT_TOOL_NAME
        resp = tool.run({"reasoning": "Read first"})
        assert resp.status == ToolStatus.SUCCESS and "Read first" in resp.text

    def test_thought_empty(self):
        resp = _ThoughtTool(description="Think").run({"reasoning": ""})
        assert "[empty reasoning]" in resp.text

    def test_finish(self):
        tool = _FinishTool(description="End")
        assert tool.name == FINISH_TOOL_NAME
        resp = tool.run({"answer": "42"})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["finished"] is True
        assert "42" in resp.text


# ────────────────────────────────────────────────────────────────────────────
# CodeAgent
# ────────────────────────────────────────────────────────────────────────────


class TestCodeAgent:

    @pytest.fixture
    def tmp(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_construction(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, register_default_tools=False)
        assert a.name == "c"
        assert a.project_root == Path(tmp).resolve()
        assert a.max_steps == CodeAgent.DEFAULT_MAX_STEPS

    def test_context_prompt(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, register_default_tools=False)
        p = a._get_context_system_prompt()
        assert "Workspace root:" in p and "Current working directory:" in p

    def test_custom_prompt_merged(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, system_prompt="Custom.", register_default_tools=False)
        assert "Custom." in a._get_context_system_prompt()

    def test_config_isolation(self, tmp):
        config = Config()
        a = CodeAgent("c", _mock_llm(), project_root=tmp, config=config, register_default_tools=False)
        assert a.config is not config

    def test_register_default_tools(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, register_default_tools=True,
                      enable_task_tool=False, interactive=False)
        tools = a.tool_registry.list_tools()
        for t in ("Read", "Write", "Edit", "Bash", "Grep", "Glob", "LS", "Delete"):
            assert t in tools

    def test_set_working_dir(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, register_default_tools=True,
                      enable_task_tool=False, interactive=False)
        sub = Path(tmp) / "sub"
        sub.mkdir()
        a.set_working_dir(str(sub))
        assert a.working_dir == sub.resolve()
        for t in a.tool_registry.get_all_tools():
            if hasattr(t, "working_dir"):
                assert t.working_dir == sub.resolve()

    def test_create_subagent(self, tmp):
        a = CodeAgent("main", _mock_llm(), project_root=tmp, register_default_tools=False)
        sub = a._create_subagent("x")
        assert sub.name == "main-x-subagent"
        assert sub.project_root == a.project_root

    def test_compact_empty(self, tmp):
        assert CodeAgent("c", _mock_llm(), project_root=tmp, register_default_tools=False).compact() == "Nothing to compact."


# ────────────────────────────────────────────────────────────────────────────
# Session & lifecycle
# ────────────────────────────────────────────────────────────────────────────


class TestReActAgentSession:

    @pytest.fixture
    def agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_session_id(self, agent):
        assert agent.session_id and len(agent.session_id) > 0

    def test_add_get_history(self, agent):
        agent.add_message(Message("hello", "user"))
        assert len(agent.get_history()) == 1

    def test_history_is_copy(self, agent):
        agent.add_message(Message("hello", "user"))
        h = agent.get_history()
        h.append(Message("extra", "user"))
        assert len(agent.get_history()) == 1

    def test_clear_history(self, agent):
        agent.add_message(Message("hello", "user"))
        agent.clear_history()
        assert len(agent.get_history()) == 0

    def test_build_messages(self, agent):
        agent.add_message(Message("earlier", "user"))
        agent.system_prompt = "SP"
        msgs = agent._build_messages(input_text="new")
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["content"] == "new"

    def test_schema_hash(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        h1, h2 = agent._compute_tool_schema_hash(), agent._compute_tool_schema_hash()
        assert h1 == h2 and len(h1) == 16

    def test_schema_hash_no_registry(self):
        """_compute_tool_schema_hash returns 'no-tools' when registry is None."""
        from code.agents.react_agent import ReActAgent
        a = ReActAgent("t", _mock_llm(), config=Config())
        a.tool_registry = None
        assert a._compute_tool_schema_hash() == "no-tools"

    def test_schema_hash_with_tools(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        h = agent._compute_tool_schema_hash()
        assert isinstance(h, str) and len(h) == 16


# ────────────────────────────────────────────────────────────────────────────
# Tool execution
# ────────────────────────────────────────────────────────────────────────────


class TestReActAgentToolExecution:

    @pytest.fixture
    def agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_execute_tool_success(self, agent):
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool())
        result = agent._execute_tool_call("test_tool", {"path": "a.py"})
        assert "ok" in result

    def test_execute_nonexistent(self, agent):
        agent.tool_registry = _registry()
        result = agent._execute_tool_call("nonexistent", {})
        assert "Error" in result or "❌" in result or "NOT_FOUND" in result

    def test_execute_no_registry(self, agent):
        result = agent._execute_tool_call("anything", {})
        assert "Error" in result or "❌" in result

    def test_execute_response_no_registry(self, agent):
        assert agent._execute_tool_response("any", {}).status == ToolStatus.ERROR


# ────────────────────────────────────────────────────────────────────────────
# Subagent & tool filter
# ────────────────────────────────────────────────────────────────────────────


class TestReActAgentSubagent:

    def test_tool_filter_swaps_and_restores(self):
        agent = ReActAgent("test", _mock_llm(), config=Config())
        agent.tool_registry = _registry()
        agent.tool_registry.register_tool(_SimpleTool(name="a"))
        agent.tool_registry.register_tool(_SimpleTool(name="b"))
        agent.tool_registry.register_tool(_SimpleTool(name="c"))

        from code.tools.tool_filter import CustomFilter

        f = CustomFilter(allowed=["a", "b"], mode="whitelist")
        orig = agent._apply_tool_filter(f)
        assert orig is not None and "a" in agent.tool_registry._tools
        assert "c" not in agent.tool_registry._tools
        agent._restore_tools(orig)
        assert "c" in agent.tool_registry._tools


# ────────────────────────────────────────────────────────────────────────────
# ReActAgent — structured output
# ────────────────────────────────────────────────────────────────────────────


class TestReActAgentStructuredOutput:

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_structured_output_schema(self):
        from code.agents.react_agent import _StructuredOutputSpec
        spec = _StructuredOutputSpec(
            name="CustomOutput",
            description="Custom structured output",
            schema={"type": "object", "properties": {"result": {"type": "string"}}},
        )
        schema = self._agent()._build_structured_output_tool_schema(spec)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "CustomOutput"
        assert "result" in schema["function"]["parameters"]["properties"]

    def test_tool_choice_for_structured_output(self):
        from code.agents.react_agent import _StructuredOutputSpec
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        assert ReActAgent._tool_choice_for(spec) == "required"
        assert ReActAgent._tool_choice_for(None) == "auto"

    def test_structured_output_instruction(self):
        from code.agents.react_agent import _StructuredOutputSpec
        spec = _StructuredOutputSpec(name="Out", description="d", schema={"type": "object"})
        instr = ReActAgent._structured_output_instruction(spec)
        assert "Structured output" in instr
        assert "Out" in instr


# ────────────────────────────────────────────────────────────────────────────
# Agent — tool response slimming / observation / result building
# ────────────────────────────────────────────────────────────────────────────


class TestAgentToolObservation:

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_slim_non_compactable_tool(self):
        a = self._agent()
        resp = ToolResponse.success(text="full text here")
        # "AskUser" is not in COMPACTABLE_TOOL_OUTPUTS
        result = a._slim_tool_response_text("AskUser", resp)
        assert result == "full text here"

    def test_build_tool_execution_result_success(self):
        a = self._agent()
        resp = ToolResponse.success(text="Done", data={"key": "val"})
        r = a._build_tool_execution_result("Read", resp)
        assert r["status"] == "success"
        assert "content" in r
        assert "metadata" in r

    def test_build_tool_execution_result_error(self):
        a = self._agent()
        resp = ToolResponse.error(code=None, message="Failed")
        r = a._build_tool_execution_result("Bash", resp)
        assert r["status"] == "error"

    def test_tool_observation_source_text_bash(self):
        a = self._agent()
        resp = ToolResponse.success(
            text="output text",
            data={"command": "ls -la", "description": "List files", "exit_code": 0},
        )
        source, meta = a._tool_observation_source_text("Bash", resp)
        assert "ls -la" in source
        assert "List files" in source
        assert "Exit code: 0" in source
        assert meta["tool_name"] == "Bash"

    def test_tool_observation_source_text_bash_failure(self):
        a = self._agent()
        resp = ToolResponse.success(
            text="error output",
            data={"command": "bad cmd", "exit_code": 1},
        )
        source, _meta = a._tool_observation_source_text("Bash", resp)
        assert "bad cmd" in source
        assert "failure" in source.lower() or "exit code: 1" in source.lower()

    def test_tool_observation_source_text_non_bash(self):
        a = self._agent()
        resp = ToolResponse.success(text="plain text result")
        source, meta = a._tool_observation_source_text("Read", resp)
        assert source == "plain text result"
        assert meta["tool_name"] == "Read"


# ────────────────────────────────────────────────────────────────────────────
# Agent — subagent helpers
# ────────────────────────────────────────────────────────────────────────────


class TestAgentSubagentHelpers:

    def _agent(self):
        return ReActAgent("test", _mock_llm(), config=Config())

    def test_generate_subagent_summary_basic(self):
        a = self._agent()
        summary = a._generate_subagent_summary(
            "do something",
            "result text here",
            {"steps": 3, "tokens": 100, "duration_seconds": 1.5, "tools_used": ["Read", "Edit"]},
        )
        assert "do something" in summary
        assert "3" in summary  # steps
        assert "Read" in summary

    def test_generate_subagent_summary_with_error(self):
        a = self._agent()
        summary = a._generate_subagent_summary(
            "task",
            "failed",
            {"steps": 1, "tokens": 10, "duration_seconds": 0.1, "tools_used": [], "error": "timeout"},
        )
        assert "timeout" in summary

    def test_generate_subagent_summary_truncates_long_result(self):
        a = self._agent()
        long_result = "x" * 600
        summary = a._generate_subagent_summary(
            "task", long_result,
            {"steps": 1, "tokens": 10, "duration_seconds": 0.1, "tools_used": []},
        )
        assert "..." in summary
        assert len(summary) < len(long_result) + 200

    def test_extract_tools_from_history(self):
        a = self._agent()
        history = [
            Message(
                "Action: Read[/tmp/x.py]\nAction: Write[/tmp/y.py]",
                "assistant",
            ),
        ]
        tools = a._extract_tools_from_history(history)
        assert "Read" in tools
        assert "Write" in tools

    def test_extract_tools_from_history_react_format(self):
        a = self._agent()
        history = [
            Message("", "assistant", metadata={}),
            Message("Action: Read[/tmp/x.py]\nAction: Write[/tmp/y.py]", "assistant"),
        ]
        tools = a._extract_tools_from_history(history)
        # ReAct text format detection uses regex
        assert isinstance(tools, list)

    def test_get_agent_config(self):
        a = ReActAgent("my_agent", _mock_llm(), config=Config())
        cfg = a._get_agent_config()
        assert cfg["name"] == "my_agent"
        assert cfg["agent_type"] == "ReActAgent"
        assert "llm_model" in cfg


# ────────────────────────────────────────────────────────────────────────────
# Agent — map_parameter_type edge cases
# ────────────────────────────────────────────────────────────────────────────


class TestAgentMapParameterTypeEdgeCases:

    def test_map_none(self):
        assert ReActAgent._map_parameter_type(None) == "string"

    def test_map_empty(self):
        assert ReActAgent._map_parameter_type("") == "string"

    def test_map_case_insensitive(self):
        assert ReActAgent._map_parameter_type("STRING") == "string"
        # "Boolean" → "boolean" (lowercased to known type)
        assert ReActAgent._map_parameter_type("Boolean") == "boolean"

    def test_map_float_alias(self):
        # "float" is not in the recognized set → falls back to "string"
        assert ReActAgent._map_parameter_type("float") == "string"

    def test_map_number(self):
        assert ReActAgent._map_parameter_type("number") == "number"
        assert ReActAgent._map_parameter_type("integer") == "integer"
        assert ReActAgent._map_parameter_type("array") == "array"
        assert ReActAgent._map_parameter_type("object") == "object"


# ────────────────────────────────────────────────────────────────────────────
# CodeAgent — additional construction and tool registration edge cases
# ────────────────────────────────────────────────────────────────────────────


class TestCodeAgentEdgeCases:

    @pytest.fixture
    def tmp(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_working_dir_relative(self, tmp):
        """working_dir outside project_root should raise."""
        import pytest
        with pytest.raises(ValueError):
            CodeAgent("c", _mock_llm(), project_root=tmp, working_dir="/etc",
                      register_default_tools=False)

    def test_tool_registry_passed_in(self, tmp):
        """When tool_registry is passed, it is used instead of creating new one."""
        reg = _registry()
        reg.register_tool(_SimpleTool(name="pre_registered"))
        a = CodeAgent("c", _mock_llm(), project_root=tmp, tool_registry=reg,
                      register_default_tools=False)
        assert a.tool_registry is reg
        assert "pre_registered" in a.tool_registry.list_tools()

    def test_max_steps_override(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, max_steps=42,
                      register_default_tools=False)
        assert a.max_steps == 42

    def test_interactive_disabled(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, interactive=False,
                      register_default_tools=True, enable_task_tool=False)
        # AskUser should still be registered but with interactive=False
        ask = a.tool_registry.get_tool("AskUser")
        assert ask is not None
        assert ask._interactive is False

    def test_todowrite_registered_when_enabled(self, tmp):
        a = CodeAgent("c", _mock_llm(), project_root=tmp, register_default_tools=True,
                      enable_task_tool=True, interactive=False)
        assert a.tool_registry.get_tool("TodoWrite") is not None
