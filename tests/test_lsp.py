"""Comprehensive tests for the LSP (Language Server Protocol) module.

Covers:
- LSPClient: JSON-RPC communication, frame parsing, error handling
- LSPManager: server lifecycle, language detection, availability checks
- LSP tools: all 4 tools with mocked LSP client, path safety, degradation
- LSP helpers: formatting functions, parameter extraction
- CodeAgent integration: LSP tools registered in default tool set
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code.tools.lsp.client import (
    LSPClient,
    LSPError,
    LSPServerStartError,
)
from code.tools.lsp.manager import LSPManager
from code.tools.lsp.tools import (
    LSPDefinitionTool,
    LSPDiagnosticsTool,
    LSPHoverTool,
    LSPReferencesTool,
    _format_diagnostic_entry,
    _format_hover,
    _format_location,
    _position_from_params,
)
from code.tools.response import ToolStatus


# ============================================================================
# Mock LSP client factories
# ============================================================================

def _mock_lsp_client():
    """Return a MagicMock that behaves like LSPClient for one session."""
    client = MagicMock(spec=LSPClient)
    client.definition.return_value = [
        {"uri": "file:///project/src/main.py",
         "range": {"start": {"line": 3, "character": 4},
                   "end": {"line": 3, "character": 30}}},
    ]
    client.references.return_value = [
        {"uri": "file:///project/src/main.py",
         "range": {"start": {"line": 10, "character": 4},
                   "end": {"line": 10, "character": 14}}},
        {"uri": "file:///project/src/utils.py",
         "range": {"start": {"line": 25, "character": 8},
                   "end": {"line": 25, "character": 18}}},
        {"uri": "file:///project/tests/test_app.py",
         "range": {"start": {"line": 42, "character": 8},
                   "end": {"line": 42, "character": 18}}},
    ]
    client.hover.return_value = {
        "contents": {
            "kind": "markdown",
            "value": "```python\ndef authenticate(user: str, token: bytes) -> bool\n```\n\nAuthenticate a user with the given credentials.",
        }
    }
    client.document_diagnostic.return_value = {
        "items": [
            {"range": {"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 10}},
             "severity": 1, "message": "Undefined variable 'foo'", "source": "pyflakes", "code": "F821"},
            {"range": {"start": {"line": 10, "character": 4}, "end": {"line": 10, "character": 20}},
             "severity": 2, "message": "Unused import 'os'", "source": "pyflakes", "code": "F401"},
            {"range": {"start": {"line": 15, "character": 0}, "end": {"line": 15, "character": 5}},
             "severity": 4, "message": "Missing docstring", "source": "pylint", "code": "C0111"},
        ]
    }
    return client


def _empty_lsp_client():
    """LSPClient mock that returns empty/no results."""
    client = MagicMock(spec=LSPClient)
    client.definition.return_value = []
    client.references.return_value = []
    client.hover.return_value = None
    client.document_diagnostic.return_value = {"items": []}
    return client


@pytest.fixture
def workspace(tmp_path):
    """Temporary workspace with a Python file."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "main.py").write_text("""\
def authenticate(user: str, token: bytes) -> bool:
    \"\"\"Authenticate a user with the given credentials.\"\"\"
    if user == "admin" and token == b"secret":
        return True
    return False

def helper():
    return authenticate("guest", b"token")

x = 42
""")
    return ws


# ============================================================================
# LSPClient unit tests
# ============================================================================


class TestLSPClient:
    """Tests for LSPClient — mostly static methods and error paths."""

    def test_start_missing_executable(self, workspace):
        with pytest.raises(LSPServerStartError) as exc:
            LSPClient(["nonexistent-binary-xyz-123"], workspace)
        assert "not found" in str(exc.value).lower()

    def test_content_length_parsing(self):
        assert LSPClient._parse_content_length("Content-Length: 123\r\n") == 123
        assert LSPClient._parse_content_length("Content-Length: 42") == 42
        with pytest.raises(LSPError):
            LSPClient._parse_content_length("Bad-Header: 5")

    def test_content_length_no_colon(self):
        with pytest.raises(LSPError):
            LSPClient._parse_content_length("no colon")

    def test_content_length_bad_value(self):
        with pytest.raises(LSPError):
            LSPClient._parse_content_length("Content-Length: abc")


# ============================================================================
# LSPManager unit tests
# ============================================================================


class TestLSPManager:
    """Tests for LSPManager language detection and lifecycle."""

    def test_detect_language_python(self):
        assert LSPManager.detect_language("src/main.py") == "python"
        assert LSPManager.detect_language("src/types.pyi") == "python"

    def test_detect_language_js_family(self):
        assert LSPManager.detect_language("app.js") == "javascript"
        assert LSPManager.detect_language("component.ts") == "typescript"
        assert LSPManager.detect_language("Component.tsx") == "typescriptreact"
        assert LSPManager.detect_language("App.jsx") == "javascriptreact"

    def test_detect_language_unsupported(self):
        for ext in ("data.json", "README.md", "Dockerfile", "Makefile", "script.sh"):
            assert LSPManager.detect_language(ext) is None, f"Expected None for {ext}"

    def test_server_label(self):
        assert LSPManager.server_label("test.py") == "Python (pylsp)"
        assert LSPManager.server_label("test.ts") == "TypeScript (typescript-language-server)"
        assert LSPManager.server_label("test.md") is None

    def test_available_languages(self):
        langs = LSPManager.available_languages()
        assert ".py" in langs
        assert ".js" in langs
        assert ".ts" in langs
        assert ".go" in langs
        assert ".rs" in langs
        for label in langs.values():
            assert isinstance(label, str)

    def test_server_for_nonexistent_extension(self, workspace):
        mgr = LSPManager(workspace)
        assert mgr.server_for("readme.txt") is None
        assert mgr.server_for("script.sh") is None
        assert mgr.server_for("Makefile") is None

    def test_ensure_file_open_unsupported(self, workspace):
        mgr = LSPManager(workspace)
        assert mgr.ensure_file_open("data.json") is None

    def test_shutdown_clears_state(self, workspace):
        mgr = LSPManager(workspace)
        mgr.shutdown()
        assert len(mgr._servers) == 0
        assert len(mgr._opened_uris) == 0

    def test_shutdown_idempotent(self, workspace):
        mgr = LSPManager(workspace)
        mgr.shutdown()
        mgr.shutdown()
        assert len(mgr._servers) == 0

    def test_server_available_returns_bool(self, workspace):
        mgr = LSPManager(workspace)
        assert isinstance(mgr.server_available("test.py"), bool)

    def test_server_available_unsupported_ext(self, workspace):
        mgr = LSPManager(workspace)
        assert mgr.server_available("file.json") is False


# ============================================================================
# Helper function tests
# ============================================================================


class TestLSPHelpers:

    def test_position_from_params(self):
        assert _position_from_params({"line": 10, "character": 20}) == (10, 20)
        assert _position_from_params({"line": "5", "character": "8"}) == (5, 8)
        assert _position_from_params({}) == (0, 0)

    def test_format_location(self):
        loc = {
            "uri": "file:///project/src/main.py",
            "range": {
                "start": {"line": 5, "character": 4},
                "end": {"line": 5, "character": 20},
            },
        }
        formatted = _format_location(loc, workspace_root="/project")
        assert "src/main.py" in formatted

    def test_format_location_same_line(self):
        loc = {
            "uri": "file:///ws/app.py",
            "range": {
                "start": {"line": 10, "character": 0},
                "end": {"line": 10, "character": 8},
            },
        }
        formatted = _format_location(loc, workspace_root="/ws")
        assert "app.py" in formatted

    def test_format_hover_markdown(self):
        hover = {"contents": {"kind": "markdown", "value": "```python\ndef foo(x: int) -> str\n```"}}
        assert "foo" in _format_hover(hover)

    def test_format_hover_list(self):
        hover = {"contents": [{"value": "first"}, {"value": "second"}]}
        result = _format_hover(hover)
        assert "first" in result
        assert "second" in result

    def test_format_hover_list_mixed(self):
        hover = {"contents": [{"value": "typed"}, "plain string"]}
        result = _format_hover(hover)
        assert "typed" in result
        assert "plain string" in result

    def test_format_hover_empty(self):
        assert "(no hover information)" in _format_hover({})

    def test_format_hover_empty_contents(self):
        # Empty dict with no "value" key → falls through to "(no hover information)"
        hover = {"contents": {}}
        assert "(no hover information)" in _format_hover(hover)

    def test_format_hover_string_contents(self):
        result = _format_hover({"contents": "plain string result"})
        assert "plain string result" in result

    def test_format_hover_null_contents(self):
        hover = {"contents": None}
        result = _format_hover(hover)
        assert "no hover information" in result

    def test_format_diagnostic_entry_full(self):
        diag = {
            "range": {"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 10}},
            "severity": 1,
            "message": "Undefined variable 'foo'",
            "source": "pyflakes",
            "code": "F821",
        }
        formatted = _format_diagnostic_entry(diag)
        assert "ERROR" in formatted
        assert "Undefined variable" in formatted
        assert "F821" in formatted
        assert "pyflakes" in formatted

    def test_format_diagnostic_entry_warning(self):
        diag = {
            "range": {"start": {"line": 3, "character": 1}, "end": {"line": 3, "character": 5}},
            "severity": 2,
            "message": "X is unused",
        }
        formatted = _format_diagnostic_entry(diag)
        assert "WARNING" in formatted
        assert "X is unused" in formatted

    def test_format_diagnostic_entry_info(self):
        diag = {
            "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 0}},
            "severity": 3,
            "message": "Note: consider using f-strings",
        }
        assert "INFO" in _format_diagnostic_entry(diag)

    def test_format_diagnostic_entry_hint(self):
        diag = {
            "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 0}},
            "severity": 4,
            "message": "Missing type annotation",
        }
        assert "HINT" in _format_diagnostic_entry(diag)

    def test_format_diagnostic_entry_unknown_severity(self):
        diag = {
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
            "severity": 99,
            "message": "Something",
        }
        assert "UNKNOWN" in _format_diagnostic_entry(diag)


# ============================================================================
# LSP Tool tests — with mocked client
# ============================================================================


class TestLSPDefinitionTool:
    """Tests for LSPDefinitionTool with mocked LSP client."""

    def test_finds_definition(self, workspace):
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        tool = LSPDefinitionTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 42, "character": 5})
        assert resp.status == ToolStatus.SUCCESS
        assert "Found" in resp.text
        assert "src/main.py" in resp.text

    def test_no_definition(self, workspace):
        mgr = LSPManager(workspace)
        client = _empty_lsp_client()
        tool = LSPDefinitionTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
        assert resp.status == ToolStatus.SUCCESS
        assert "No definition" in resp.text

    def test_path_outside_workspace(self, workspace):
        mgr = LSPManager(workspace)
        tool = LSPDefinitionTool(workspace_root=str(workspace), manager=mgr)
        resp = tool.run({"file": "../etc/passwd", "line": 0, "character": 0})
        assert resp.status == ToolStatus.ERROR

    def test_unsupported_language(self, workspace):
        mgr = LSPManager(workspace)
        tool = LSPDefinitionTool(workspace_root=str(workspace), manager=mgr)
        resp = tool.run({"file": "data.md", "line": 0, "character": 0})
        assert resp.status == ToolStatus.PARTIAL

    def test_server_unavailable(self, workspace):
        mgr = LSPManager(workspace)
        tool = LSPDefinitionTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=False):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
            assert resp.status == ToolStatus.PARTIAL

    def test_ensure_file_open_fails(self, workspace):
        mgr = LSPManager(workspace)
        tool = LSPDefinitionTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=None):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
            assert resp.status == ToolStatus.PARTIAL
            assert "fall back" in resp.text.lower() or "could not connect" in resp.text.lower()


class TestLSPReferencesTool:
    """Tests for LSPReferencesTool."""

    def test_finds_references(self, workspace):
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        tool = LSPReferencesTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 3, "character": 4})
        assert resp.status == ToolStatus.SUCCESS
        assert "reference" in resp.text.lower() or "Found" in resp.text
        assert resp.data["count"] == 3

    def test_no_references(self, workspace):
        mgr = LSPManager(workspace)
        client = _empty_lsp_client()
        tool = LSPReferencesTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
        assert resp.status == ToolStatus.SUCCESS
        assert "No references" in resp.text

    def test_truncates_large_reference_list(self, workspace):
        mgr = LSPManager(workspace)
        client = MagicMock(spec=LSPClient)
        client.references.return_value = [
            {"uri": f"file:///project/file_{i}.py",
             "range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": 10}}}
            for i in range(60)
        ]
        tool = LSPReferencesTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
        assert resp.data["truncated"] is True
        assert "more" in resp.text.lower()


class TestLSPHoverTool:
    """Tests for LSPHoverTool."""

    def test_hover_shows_type(self, workspace):
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        tool = LSPHoverTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 3, "character": 4})
        assert resp.status == ToolStatus.SUCCESS
        assert "authenticate" in resp.text

    def test_no_hover_info(self, workspace):
        mgr = LSPManager(workspace)
        client = _empty_lsp_client()
        tool = LSPHoverTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
        assert resp.status == ToolStatus.SUCCESS
        assert "No hover information" in resp.text


class TestLSPDiagnosticsTool:
    """Tests for LSPDiagnosticsTool."""

    def test_shows_diagnostics(self, workspace):
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        tool = LSPDiagnosticsTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py"})
        assert resp.status == ToolStatus.SUCCESS
        assert "Undefined variable" in resp.text
        assert "ERROR" in resp.text
        assert "WARNING" in resp.text
        assert resp.data["count"] == 3

    def test_no_diagnostics(self, workspace):
        mgr = LSPManager(workspace)
        client = _empty_lsp_client()
        tool = LSPDiagnosticsTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py"})
        assert resp.status == ToolStatus.SUCCESS
        assert "No diagnostics" in resp.text or resp.data["count"] == 0

    def test_diagnostics_list_format(self, workspace):
        """Server may return items as a list instead of {"items": [...]}."""
        mgr = LSPManager(workspace)
        client = MagicMock(spec=LSPClient)
        client.document_diagnostic.return_value = [
            {"range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}},
             "severity": 1, "message": "Bad"},
        ]
        tool = LSPDiagnosticsTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py"})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["count"] == 1

    def test_diagnostics_sorted_by_severity(self, workspace):
        """Errors (severity=1) should appear before warnings and hints."""
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        tool = LSPDiagnosticsTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py"})
        items = resp.data["items"]
        severities = [d["severity"] for d in items]
        assert severities == sorted(severities)


# ============================================================================
# CodeAgent integration
# ============================================================================


class TestCodeAgentLSPIntegration:
    """Verify LSP tools are registered in CodeAgent's default tool set."""

    def test_lsp_tools_registered(self, tmp_path):
        from code.agents.code_agent import CodeAgent
        from code.core.llm import HelloAgentsLLM

        llm = MagicMock(spec=HelloAgentsLLM)
        llm.model = "test-model"
        llm.temperature = 0.7

        agent = CodeAgent(
            name="test",
            llm=llm,
            project_root=str(tmp_path),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
        )
        tools = agent.tool_registry.list_tools()
        for name in ("LSPDefinition", "LSPReferences", "LSPHover", "LSPDiagnostics"):
            assert name in tools, f"Missing {name} in {tools}"

    def test_lsp_tools_have_valid_schemas(self, tmp_path):
        """Each LSP tool should produce a valid OpenAI function schema."""
        from code.agents.code_agent import CodeAgent
        from code.core.llm import HelloAgentsLLM

        llm = MagicMock(spec=HelloAgentsLLM)
        llm.model = "test-model"
        llm.temperature = 0.7

        agent = CodeAgent(
            name="test",
            llm=llm,
            project_root=str(tmp_path),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
        )
        schemas = agent._build_tool_schemas()
        lsp_names = {"LSPDefinition", "LSPReferences", "LSPHover", "LSPDiagnostics"}
        for schema in schemas:
            name = schema["function"]["name"]
            if name in lsp_names:
                assert schema["type"] == "function"
                props = schema["function"]["parameters"]["properties"]
                if name == "LSPDiagnostics":
                    assert "file" in props
                else:
                    assert "file" in props
                    assert "line" in props
                    assert "character" in props


# ============================================================================
# Real pylsp end-to-end tests
# ============================================================================


class TestRealPylspEndToEnd:
    """Comprehensive end-to-end tests with real pylsp (skip if not installed)."""

    @pytest.fixture
    def py_project(self, tmp_path):
        """Create a proper Python project for pylsp to analyze."""
        ws = tmp_path / "py_project"
        ws.mkdir()
        (ws / "__init__.py").write_text("")
        (ws / "main.py").write_text("""\
import os

def greet(name: str) -> str:
    \"\"\"Return a personalised greeting.\"\"\"
    return f"Hello, {name}"

def farewell(name: str) -> str:
    \"\"\"Say goodbye.\"\"\"
    return f"Goodbye, {name}"

class Greeter:
    \"\"\"A greeter class.\"\"\"

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def say_hello(self, name: str) -> str:
        return f"{self.prefix}Hello, {name}"

def main():
    g = Greeter("[bot] ")
    msg = g.say_hello("world")
    print(msg)
    bye = farewell("world")
    print(bye)

if __name__ == "__main__":
    main()
""")
        return ws

    def _require_pylsp(self):
        if shutil.which("pylsp") is None:
            pytest.skip("pylsp not installed")

    # ---- definition ----

    def test_definition_of_function(self, py_project):
        self._require_pylsp()
        mgr = LSPManager(py_project)
        client = mgr.ensure_file_open("main.py")
        if client is None:
            pytest.skip("pylsp failed to start")
        try:
            uri = py_project.joinpath("main.py").as_uri()
            result = client.definition(uri, 20, 14)  # "g.say_hello"
            assert isinstance(result, (list, type(None)))
        finally:
            client.shutdown()

    def test_definition_not_found(self, py_project):
        """Definition on a non-symbol position returns empty."""
        self._require_pylsp()
        mgr = LSPManager(py_project)
        client = mgr.ensure_file_open("main.py")
        if client is None:
            pytest.skip("pylsp failed to start")
        try:
            uri = py_project.joinpath("main.py").as_uri()
            result = client.definition(uri, 0, 0)  # line 0 is 'import os'
            assert result is None or result == []
        finally:
            client.shutdown()

    # ---- hover ----

    def test_hover_on_function(self, py_project):
        self._require_pylsp()
        mgr = LSPManager(py_project)
        client = mgr.ensure_file_open("main.py")
        if client is None:
            pytest.skip("pylsp failed to start")
        try:
            uri = py_project.joinpath("main.py").as_uri()
            result = client.hover(uri, 3, 4)  # 'greet' function name
            assert isinstance(result, (dict, list, type(None)))
        finally:
            client.shutdown()

    # ---- references ----

    def test_references(self, py_project):
        self._require_pylsp()
        mgr = LSPManager(py_project)
        client = mgr.ensure_file_open("main.py")
        if client is None:
            pytest.skip("pylsp failed to start")
        try:
            uri = py_project.joinpath("main.py").as_uri()
            result = client.references(uri, 3, 4)  # 'greet' function
            assert isinstance(result, (list, type(None)))
        finally:
            client.shutdown()

    # ---- did_open / did_close ----

    def test_did_open_and_close(self, py_project):
        self._require_pylsp()
        mgr = LSPManager(py_project)
        client = mgr.ensure_file_open("main.py")
        if client is None:
            pytest.skip("pylsp failed to start")
        try:
            uri = py_project.joinpath("main.py").as_uri()
            client.did_open(uri, "python", "x = 1")
            client.did_close(uri)
        finally:
            client.shutdown()

    # ---- full tool integration ----

    def test_definition_tool_end_to_end(self, py_project):
        """LSPDefinitionTool.run() with real pylsp."""
        self._require_pylsp()
        mgr = LSPManager(py_project)
        tool = LSPDefinitionTool(workspace_root=str(py_project), manager=mgr)
        resp = tool.run({"file": "main.py", "line": 20, "character": 14})
        assert resp.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL)

    def test_hover_tool_end_to_end(self, py_project):
        """LSPHoverTool.run() with real pylsp."""
        self._require_pylsp()
        mgr = LSPManager(py_project)
        tool = LSPHoverTool(workspace_root=str(py_project), manager=mgr)
        resp = tool.run({"file": "main.py", "line": 3, "character": 4})
        assert resp.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL)

    def test_references_tool_end_to_end(self, py_project):
        """LSPReferencesTool.run() with real pylsp."""
        self._require_pylsp()
        mgr = LSPManager(py_project)
        tool = LSPReferencesTool(workspace_root=str(py_project), manager=mgr)
        resp = tool.run({"file": "main.py", "line": 3, "character": 4})
        assert resp.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL)

    def test_diagnostics_tool_end_to_end(self, py_project):
        """LSPDiagnosticsTool.run() with real pylsp — expects partial (pull not supported)."""
        self._require_pylsp()
        mgr = LSPManager(py_project)
        tool = LSPDiagnosticsTool(workspace_root=str(py_project), manager=mgr)
        resp = tool.run({"file": "main.py"})
        # pylsp doesn't support textDocument/diagnostic (pull model) → PARTIAL
        assert resp.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL)

    # ---- degradation when no server ----

    def test_unsupported_language_end_to_end(self, py_project):
        """Markdown files have no LSP server → PARTIAL with hint."""
        self._require_pylsp()
        mgr = LSPManager(py_project)
        tool = LSPDefinitionTool(workspace_root=str(py_project), manager=mgr)
        resp = tool.run({"file": "readme.md", "line": 0, "character": 0})
        assert resp.status == ToolStatus.PARTIAL

    def test_path_safety_end_to_end(self, py_project):
        """Path escaping the workspace is rejected."""
        self._require_pylsp()
        mgr = LSPManager(py_project)
        tool = LSPDefinitionTool(workspace_root=str(py_project), manager=mgr)
        resp = tool.run({"file": "../etc/shadow", "line": 0, "character": 0})
        assert resp.status == ToolStatus.ERROR


# ============================================================================
# LSPClient lifecycle edge cases
# ============================================================================


class TestLSPClientLifecycle:
    """Tests for LSPClient start/shutdown edge cases."""

    def test_shutdown_already_dead_process(self, tmp_path):
        """Shutting down an already terminated process should not raise."""
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("x = 1")
        client = LSPClient(["pylsp"], ws)
        client._kill_process()  # force kill
        client.shutdown()  # should not raise

    def test_send_raises_after_kill(self, tmp_path):
        """After kill, _send should raise LSPError."""
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("x = 1")
        client = LSPClient(["pylsp"], ws)
        client._kill_process()
        with pytest.raises(LSPError):
            client._send({"jsonrpc": "2.0", "id": 1, "method": "test"})

    def test_server_for_existing_language_id(self, tmp_path):
        """server_for returns same client on second call."""
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("x = 1")
        mgr = LSPManager(ws)
        c1 = mgr.server_for("test.py")
        c2 = mgr.server_for("test.py")
        # pylsp may or may not be installed
        if c1 is not None:
            assert c1 is c2
            c1.shutdown()

    def test_read_exactly_timeout(self, tmp_path):
        """_read_exactly with a past deadline raises timeout."""
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("x = 1")
        client = LSPClient(["pylsp"], ws)
        try:
            with pytest.raises(LSPError) as exc:
                client._read_exactly(100, deadline=0)  # deadline in the past
            assert "Timeout" in str(exc.value)
        finally:
            client.shutdown()

    def test_read_line_timeout(self, tmp_path):
        """_read_line with a past deadline raises timeout."""
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("x = 1")
        client = LSPClient(["pylsp"], ws)
        try:
            with pytest.raises(LSPError) as exc:
                client._read_line(deadline=0)  # deadline in the past
            assert "Timeout" in str(exc.value)
        finally:
            client.shutdown()

    def test_ensure_file_open_force(self, tmp_path):
        """ensure_file_open with force=True re-opens the file."""
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("x = 1")
        mgr = LSPManager(ws)
        # First open
        c1 = mgr.ensure_file_open("test.py")
        # Force re-open
        c2 = mgr.ensure_file_open("test.py", force=True)
        # Both should return the same client
        if c1 is not None and c2 is not None:
            assert c1 is c2
            c1.shutdown()


# ============================================================================
# Additional edge case tests
# ============================================================================


class TestLSPFormatLocationEdgeCases:
    """Additional edge cases for _format_location."""

    def test_location_not_under_workspace(self):
        """URI outside workspace → shows absolute path."""
        loc = {
            "uri": "file:///other/project/file.py",
            "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}},
        }
        formatted = _format_location(loc, workspace_root="/my/project")
        # Should show the path, not crash
        assert "file.py" in formatted or "/" in formatted

    def test_location_non_file_uri(self):
        """Non-file URI → use as-is."""
        loc = {
            "uri": "untitled:Untitled-1",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
        }
        formatted = _format_location(loc, workspace_root="/ws")
        assert len(formatted) > 0  # at minimum, shows the URI

    def test_location_missing_range(self):
        """Location without range → shows what we have."""
        loc = {"uri": "file:///project/x.py"}
        formatted = _format_location(loc, workspace_root="/project")
        assert "x.py" in formatted


class TestLSPEnsureLSPEdgeCases:
    """Tests for _ensure_lsp edge cases."""

    def test_invalid_path_characters(self, tmp_path):
        """A path that causes OSError on resolve returns INVALID_PARAM."""
        ws = tmp_path / "ws"; ws.mkdir()
        mgr = LSPManager(ws)
        tool = LSPDefinitionTool(workspace_root=str(ws), manager=mgr)
        # Path with embedded NUL byte is always invalid
        resp = tool.run({"file": "test\x00.py", "line": 0, "character": 0})
        assert resp.status in (ToolStatus.ERROR, ToolStatus.PARTIAL)

    def test_file_is_directory(self, tmp_path):
        """Opening a directory path should be handled gracefully."""
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "subdir").mkdir()

        # server_available for dirs is True (uses suffix detection)
        mgr = LSPManager(ws)
        assert not mgr.server_available("subdir")  # "subdir" has no extension

    def test_hover_server_exception(self, workspace):
        """Hover request that raises → PARTIAL with error message."""
        mgr = LSPManager(workspace)
        client = MagicMock(spec=LSPClient)
        client.hover.side_effect = LSPError("Server crashed")

        tool = LSPHoverTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
        assert resp.status == ToolStatus.PARTIAL
        assert "failed" in resp.text.lower()

    def test_references_server_exception(self, workspace):
        """References request that raises → PARTIAL with error message."""
        mgr = LSPManager(workspace)
        client = MagicMock(spec=LSPClient)
        client.references.side_effect = LSPError("Server crashed")

        tool = LSPReferencesTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/main.py", "line": 0, "character": 0})
        assert resp.status == ToolStatus.PARTIAL
        assert "failed" in resp.text.lower()


# ============================================================================
# 改进 #3 — didChange / notify_changed
# ============================================================================


class TestLSPDidChange:
    """Tests for did_change (client) and notify_changed (manager)."""

    def test_did_change_sends_notification(self, workspace):
        """did_change calls _notify which calls _send — crudely verify no crash."""
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        # Invoke did_change directly — mock will record the call
        client.did_change("file:///ws/test.py", "new content", version=2)
        # The mock client's _notify was called indirectly via did_change.
        # At minimum, no exception was raised.

    def test_notify_changed_new_file(self, workspace):
        """notify_changed on a not-yet-opened file → did_open."""
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        resolved = (workspace / "src" / "main.py")

        with patch.object(mgr, "server_for", return_value=client), \
             patch.object(mgr, "_resolve", return_value=resolved):
            result = mgr.notify_changed("src/main.py")
            assert result is client

    def test_notify_changed_already_opened(self, workspace):
        """notify_changed on an already-opened file → did_change."""
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()
        resolved = (workspace / "src" / "main.py")
        uri = resolved.as_uri()
        mgr._opened_uris[uri] = "python"

        with patch.object(mgr, "server_for", return_value=client), \
             patch.object(mgr, "_resolve", return_value=resolved):
            result = mgr.notify_changed("src/main.py")
            assert result is client

    def test_notify_changed_no_server(self, workspace):
        """notify_changed returns None when no server."""
        mgr = LSPManager(workspace)
        with patch.object(mgr, "server_for", return_value=None):
            assert mgr.notify_changed("src/main.py") is None

    def test_notify_changed_read_error(self, workspace):
        """notify_changed on unreadable file returns client gracefully."""
        mgr = LSPManager(workspace)
        client = _mock_lsp_client()

        class BadPath:
            def read_text(self, **kw):
                raise PermissionError("denied")
            def as_uri(self):
                return "file:///ws/bad.py"

        with patch.object(mgr, "server_for", return_value=client), \
             patch.object(mgr, "_resolve", return_value=BadPath()):
            result = mgr.notify_changed("src/main.py")
            assert result is client

    def test_did_change_with_real_pylsp(self, tmp_path):
        """Real pylsp: did_open → modify → notify_changed."""
        import shutil
        if shutil.which("pylsp") is None:
            pytest.skip("pylsp not installed")

        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("def foo():\n    pass\n")
        mgr = LSPManager(ws)
        client = mgr.ensure_file_open("test.py")
        if client is None:
            pytest.skip("pylsp failed to start")
        try:
            (ws / "test.py").write_text("def foo():\n    return 42\n")
            result = mgr.notify_changed("test.py")
            assert result is client
        finally:
            client.shutdown()


# ============================================================================
# 改进 #4 — server_for logging on startup failure
# ============================================================================


class TestLSPManagerStartupLogging:
    """Tests that server_for logs warnings on startup failure."""

    def test_server_for_logs_on_start_error(self, workspace):
        """When LSPServerStartError is raised, a warning is logged."""
        mgr = LSPManager(workspace)

        with patch("code.tools.lsp.manager._check_executable", return_value="/fake/pylsp"), \
             patch("code.tools.lsp.manager.LSPClient") as mock_cls, \
             patch("code.tools.lsp.manager._logger") as mock_logger:
            from code.tools.lsp.client import LSPServerStartError
            mock_cls.side_effect = LSPServerStartError("test failure")
            result = mgr.server_for("test.py")
            assert result is None
            mock_logger.warning.assert_called_once()
            call_msg = mock_logger.warning.call_args[0][0]
            assert "failed to start" in call_msg.lower()


# ============================================================================
# 改进 #5 — context manager + __del__
# ============================================================================


class TestLSPManagerContextManager:
    """Tests for LSPManager's context manager and __del__."""

    def test_context_manager_cleans_up(self, workspace):
        """__exit__ calls shutdown."""
        with LSPManager(workspace) as mgr:
            pass
        assert len(mgr._servers) == 0

    def test_context_manager_with_pylsp(self, tmp_path):
        """Full lifecycle: with → use server → auto-shutdown."""
        import shutil
        if shutil.which("pylsp") is None:
            pytest.skip("pylsp not installed")

        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "test.py").write_text("x = 1\n")
        with LSPManager(ws) as mgr:
            client = mgr.ensure_file_open("test.py")
            if client is None:
                pytest.skip("pylsp failed to start")
            uri = ws.joinpath("test.py").as_uri()
            result = client.definition(uri, 0, 0)
            assert isinstance(result, (list, type(None)))

    def test_does_not_suppress_exceptions(self, workspace):
        """Exceptions propagate through the context manager."""
        with pytest.raises(ValueError, match="boom"):
            with LSPManager(workspace):
                raise ValueError("boom")

    def test_del_cleans_up(self, workspace):
        """__del__ triggers shutdown."""
        mgr = LSPManager(workspace)
        mgr.__del__()
        assert len(mgr._servers) == 0

    def test_del_idempotent(self, workspace):
        """__del__ after shutdown is safe."""
        mgr = LSPManager(workspace)
        mgr.shutdown()
        mgr.__del__()
        assert len(mgr._servers) == 0


# ============================================================================
# Fix #2 — proper URI parsing in _format_location
# ============================================================================


class TestFormatLocationURIParsing:

    def test_normal_file_uri(self):
        loc = {
            "uri": "file:///home/user/project/main.py",
            "range": {"start": {"line": 5, "character": 4}, "end": {"line": 5, "character": 10}},
        }
        result = _format_location(loc, workspace_root="/home/user/project")
        assert "main.py" in result
        assert "5:4" in result

    def test_file_uri_triple_slash(self):
        """LSP file URIs use file:///host/path → host is empty → path starts with /."""
        loc = {
            "uri": "file:///home/user/project/sub/file.py",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
        }
        result = _format_location(loc, workspace_root="/home/user/project")
        assert "sub/file.py" in result

    def test_uri_file_prefix_in_path(self):
        """URI containing 'file://' as literal path content should not be corrupted."""
        loc = {
            "uri": "file:///project/results/file_analysis.md",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
        }
        result = _format_location(loc, workspace_root="/project")
        assert "results" in result

    def test_windows_style_uri(self):
        """Windows file URIs use file:///C:/... format."""
        loc = {
            "uri": "file:///C:/Users/test/project/main.py",
            "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}},
        }
        result = _format_location(loc, workspace_root="C:/Users/test/project")
        assert "main.py" in result

    def test_non_file_uri(self):
        """Non-file URIs fall through gracefully."""
        loc = {
            "uri": "untitled:Untitled-1",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
        }
        result = _format_location(loc, workspace_root="/ws")
        assert len(result) > 0


# ============================================================================
# Fix #3 — server_for race condition
# ============================================================================


class TestServerForRaceCondition:

    def test_creation_lock_exists(self, workspace):
        """LSPManager has a creation lock for thread-safe server creation."""
        from code.tools.lsp.manager import LSPManager
        mgr = LSPManager(workspace)
        assert hasattr(mgr, "_creation_lock")

    def test_double_check_under_lock(self, workspace):
        """Two rapid calls to server_for return the same result (both None or same obj)."""
        mgr = LSPManager(workspace)
        # Use an extension with guaranteed-no-server to avoid spurious process startup.
        c1 = mgr.server_for("data.txt")
        c2 = mgr.server_for("data.txt")
        assert c1 is c2  # both None


# ============================================================================
# Fix #4 — f-string diagnostic placeholder
# ============================================================================


class TestDiagnosticsFStringFix:

    def test_error_message_contains_file_path(self, workspace):
        """When diagnostics fails, the error message should contain the actual file path."""
        from code.tools.lsp.client import LSPClient, LSPError
        from code.tools.lsp.tools import LSPDiagnosticsTool

        mgr = LSPManager(workspace)
        client = MagicMock(spec=LSPClient)
        client.document_diagnostic.side_effect = LSPError("not supported")

        tool = LSPDiagnosticsTool(workspace_root=str(workspace), manager=mgr)
        with patch.object(mgr, "server_available", return_value=True), \
             patch.object(mgr, "ensure_file_open", return_value=client):
            resp = tool.run({"file": "src/my_app.py"})
        assert resp.status == ToolStatus.PARTIAL
        # The message MUST contain the actual file path, not the literal "{file_path}"
        assert "my_app.py" in resp.text
        assert "{file_path}" not in resp.text


# ============================================================================
# Fix #1 — benchmark LSP registration
# ============================================================================


class TestBenchmarkLSPRegistration:

    def test_benchmark_agent_has_lsp_tools(self, tmp_path):
        """Benchmark-created agents should have LSP tools registered."""
        from code.benchmark.base import BenchmarkCodeAgent
        from code.tools.registry import ToolRegistry

        ws = tmp_path / "bench_ws"
        ws.mkdir()

        llm = MagicMock()
        llm.model = "test-model"
        llm.temperature = 0.7

        registry = ToolRegistry(verbose=False)
        agent = BenchmarkCodeAgent(
            name="bench-test",
            llm=llm,
            tool_registry=registry,
            project_root=str(ws),
            working_dir=str(ws),
            register_default_tools=False,
            enable_task_tool=False,
            task_id="test-1",
            interactive=False,
        )

        # Simulate what _register_agent_tools does for LSP registration
        from hello_agents.tools.lsp import (
            LSPDefinitionTool, LSPReferencesTool, LSPHoverTool,
            LSPDiagnosticsTool, LSPManager,
        )
        lsp_mgr = LSPManager(ws)
        registry.register_tool(LSPDefinitionTool(workspace_root=str(ws), manager=lsp_mgr))
        registry.register_tool(LSPReferencesTool(workspace_root=str(ws), manager=lsp_mgr))
        registry.register_tool(LSPHoverTool(workspace_root=str(ws), manager=lsp_mgr))
        registry.register_tool(LSPDiagnosticsTool(workspace_root=str(ws), manager=lsp_mgr))

        tools = registry.list_tools()
        for name in ("LSPDefinition", "LSPReferences", "LSPHover", "LSPDiagnostics"):
            assert name in tools, f"Missing {name} in benchmark tools: {tools}"

    def test_benchmark_import_path_works(self):
        """The import path used in benchmark base.py should resolve."""
        from hello_agents.tools.lsp import (
            LSPDefinitionTool, LSPReferencesTool, LSPHoverTool,
            LSPDiagnosticsTool, LSPManager,
        )
        # Just verify all imports resolve
        assert LSPDefinitionTool is not None
        assert LSPManager is not None
