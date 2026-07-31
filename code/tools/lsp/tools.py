"""LSP tools for WhaleCode agents.

Four tools are provided:

- ``LSPDefinitionTool`` — go to symbol definition
- ``LSPReferencesTool`` — find all references
- ``LSPHoverTool`` — type info + documentation on hover
- ``LSPDiagnosticsTool`` — file-level error/warning diagnostics

All tools degrade gracefully when no LSP server is installed for the
target language, returning a ``ToolResponse.partial`` with a hint about
how to install the server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from .manager import LSPManager

if TYPE_CHECKING:
    from .client import LSPClient

# Shared across all LSP tools created for the same workspace.
# Owned by the tool instances and cleaned up via the agent lifecycle.
# In practice each agent creates tools once, so this is fine.


def _format_location(
    loc: Dict[str, Any],
    *,
    workspace_root: str,
    base_index: int = 0,
) -> str:
    """Format one LSP Location into a one-line string."""
    uri = loc.get("uri", "")
    range_info = loc.get("range", {})
    start = range_info.get("start", {})
    line = start.get("line", 0)
    char = start.get("character", 0)
    end = range_info.get("end", {})
    end_line = end.get("line", 0)
    end_char = end.get("character", 0)

    # Convert URI back to a workspace-relative path using proper URL parsing.
    try:
        parsed = urlparse(uri)
        file_path = Path(parsed.path)
        try:
            rel = file_path.relative_to(workspace_root)
        except ValueError:
            rel = file_path
    except Exception:
        rel = uri

    return (
        f"{rel}:{line + base_index}:{char}-{end_line + base_index}:{end_char}"
    )


def _format_hover(hover: Dict[str, Any]) -> str:
    """Format a ``textDocument/hover`` result into readable text."""
    contents = hover.get("contents", {})
    if isinstance(contents, dict):
        value = contents.get("value")
        if value:
            return str(value)
        # Empty dict or dict without "value" → fall through
        if not value and not contents:
            return "(no hover information)"
        return str(contents)
    if isinstance(contents, list):
        parts: List[str] = []
        for item in contents:
            if isinstance(item, dict):
                v = item.get("value")
                parts.append(str(v) if v else str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(no hover information)"
    if contents:
        return str(contents)
    return "(no hover information)"


def _format_diagnostic_entry(diag: Dict[str, Any], base_index: int = 0) -> str:
    """Format one LSP Diagnostic into a one-line string."""
    severity_map = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}
    severity = severity_map.get(diag.get("severity", 0), "UNKNOWN")
    range_info = diag.get("range", {})
    start = range_info.get("start", {})
    line = start.get("line", 0)
    char = start.get("character", 0)
    message = diag.get("message", "").split("\n")[0]
    code = diag.get("code", "")
    source = diag.get("source", "")
    parts = [f"  L{line + base_index}:C{char}"]
    if source:
        parts.append(f"[{source}]")
    parts.append(f"{severity}: {message}")
    if code:
        parts.append(f"({code})")
    return " ".join(parts)


def _position_from_params(parameters: Dict[str, Any]) -> tuple[int, int]:
    """Extract (line, character) from tool parameters, both 0-indexed."""
    line = int(parameters.get("line", 0))
    character = int(parameters.get("character", 0))
    return line, character


def _ensure_lsp(
    file_path: str,
    workspace_root: str,
    manager: LSPManager,
) -> tuple[Optional[LSPClient], Optional[ToolResponse], str]:
    """Common prelude for all LSP tools.

    Returns ``(client, error_response, resolved_path)``.  When *client*
    is ``None``, *error_response* is populated and the caller should
    return it immediately.
    """
    # Path safety
    try:
        resolved = (Path(workspace_root) / file_path).resolve()
        if not str(resolved).startswith(str(Path(workspace_root).resolve())):
            return None, ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path '{file_path}' is outside the project root",
            ), ""
    except (ValueError, OSError):
        return None, ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message=f"Invalid path: {file_path}",
        ), ""

    # Check server availability
    if not manager.server_available(file_path):
        label = LSPManager.server_label(file_path)
        if label is None:
            ext = Path(file_path).suffix
            supported = ", ".join(
                f"{e} ({l})" for e, l in sorted(LSPManager.available_languages().items())
            )
            return None, ToolResponse.partial(
                text=(
                    f"No LSP server is registered for files with extension '{ext}'.\n"
                    f"Supported languages: {supported}\n"
                    "Fall back to Grep/Read for code exploration."
                ),
            ), ""
        return None, ToolResponse.partial(
            text=(
                f"LSP server for {label} is not installed.\n"
                f"Install it and retry, or use Grep/Read instead."
            ),
        ), ""

    # Open file in server
    client = manager.ensure_file_open(file_path)
    if client is None:
        return None, ToolResponse.partial(
            text=(
                f"Could not connect to LSP server for '{file_path}'. "
                "Fall back to Grep/Read for code exploration."
            ),
        ), ""

    return client, None, str(resolved)


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------


class LSPDefinitionTool(Tool):
    """Go to the definition of a symbol at a given position.

    Returns the exact file, line, and column where the symbol is defined.
    For Python this typically jumps to the ``def`` or ``class`` line.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        manager: Optional[LSPManager] = None,
    ):
        super().__init__(
            name="LSPDefinition",
            description=(
                "Go to the definition of a symbol at a specific file position. "
                "Returns the exact location (file, line, column) where the symbol "
                "is defined. Use this instead of Grep when you know the symbol name "
                "and want to find its precise definition site."
            ),
            category="readonly",
        )
        self._workspace_root = workspace_root
        self._manager = manager or LSPManager(Path(workspace_root).resolve())

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file",
                type="string",
                description="File path relative to the project root containing the symbol.",
                required=True,
            ),
            ToolParameter(
                name="line",
                type="integer",
                description="0-indexed line number where the symbol appears.",
                required=True,
            ),
            ToolParameter(
                name="character",
                type="integer",
                description="0-indexed column (character offset) within the line.",
                required=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        file_path = parameters.get("file", "")
        line, character = _position_from_params(parameters)

        client, error, resolved = _ensure_lsp(
            file_path, self._workspace_root, self._manager,
        )
        if error is not None:
            return error

        try:
            uri = Path(resolved).as_uri()
            result = client.definition(uri, line, character)  # type: ignore[union-attr]
        except Exception as exc:
            return ToolResponse.partial(
                text=f"LSP definition request failed: {exc}",
            )

        if not result:
            return ToolResponse.success(
                text=f"No definition found at {file_path}:{line}:{character}",
                data={"locations": [], "count": 0},
            )

        locations = result if isinstance(result, list) else [result]
        ws_root = str(Path(self._workspace_root).resolve())
        lines = [
            f"Found {len(locations)} definition(s) for {file_path}:{line}:{character}:",
        ]
        for loc in locations:
            lines.append(_format_location(loc, workspace_root=ws_root))

        return ToolResponse.success(
            text="\n".join(lines),
            data={"locations": locations, "count": len(locations)},
        )


class LSPReferencesTool(Tool):
    """Find all references to a symbol at a given position.

    Returns every location in the project that references the symbol,
    including calls, attribute accesses, and imports.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        manager: Optional[LSPManager] = None,
    ):
        super().__init__(
            name="LSPReferences",
            description=(
                "Find all references to a symbol at a specific file position. "
                "Returns a list of every location (file, line, column) where the "
                "symbol is used — including calls, imports, and attribute accesses. "
                "Use this when you need to understand the impact of changing a function, "
                "class, or variable."
            ),
            category="readonly",
        )
        self._workspace_root = workspace_root
        self._manager = manager or LSPManager(Path(workspace_root).resolve())

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file",
                type="string",
                description="File path relative to the project root.",
                required=True,
            ),
            ToolParameter(
                name="line",
                type="integer",
                description="0-indexed line number where the symbol appears.",
                required=True,
            ),
            ToolParameter(
                name="character",
                type="integer",
                description="0-indexed column (character offset) within the line.",
                required=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        file_path = parameters.get("file", "")
        line, character = _position_from_params(parameters)

        client, error, resolved = _ensure_lsp(
            file_path, self._workspace_root, self._manager,
        )
        if error is not None:
            return error

        try:
            uri = Path(resolved).as_uri()
            result = client.references(uri, line, character)  # type: ignore[union-attr]
        except Exception as exc:
            return ToolResponse.partial(
                text=f"LSP references request failed: {exc}",
            )

        if not result:
            return ToolResponse.success(
                text=f"No references found for {file_path}:{line}:{character}",
                data={"locations": [], "count": 0},
            )

        locations = result if isinstance(result, list) else [result]
        ws_root = str(Path(self._workspace_root).resolve())
        max_display = 50
        lines = [f"Found {len(locations)} reference(s):"]
        for loc in locations[:max_display]:
            lines.append(_format_location(loc, workspace_root=ws_root))
        if len(locations) > max_display:
            lines.append(f"... and {len(locations) - max_display} more (use LSPReferences with a narrower scope)")

        return ToolResponse.success(
            text="\n".join(lines),
            data={
                "locations": locations,
                "count": len(locations),
                "truncated": len(locations) > max_display,
            },
        )


class LSPHoverTool(Tool):
    """Show type information and documentation for a symbol.

    Returns the type signature, docstring, and/or inferred type of the
    symbol at the given position.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        manager: Optional[LSPManager] = None,
    ):
        super().__init__(
            name="LSPHover",
            description=(
                "Show type information and documentation for a symbol at a specific "
                "file position. Returns the inferred type, function signature, and/or "
                "docstring. Use this when you need to understand what type a variable "
                "has, or what parameters a function accepts."
            ),
            category="readonly",
        )
        self._workspace_root = workspace_root
        self._manager = manager or LSPManager(Path(workspace_root).resolve())

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file",
                type="string",
                description="File path relative to the project root.",
                required=True,
            ),
            ToolParameter(
                name="line",
                type="integer",
                description="0-indexed line number where the symbol appears.",
                required=True,
            ),
            ToolParameter(
                name="character",
                type="integer",
                description="0-indexed column (character offset) within the line.",
                required=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        file_path = parameters.get("file", "")
        line, character = _position_from_params(parameters)

        client, error, resolved = _ensure_lsp(
            file_path, self._workspace_root, self._manager,
        )
        if error is not None:
            return error

        try:
            uri = Path(resolved).as_uri()
            result = client.hover(uri, line, character)  # type: ignore[union-attr]
        except Exception as exc:
            return ToolResponse.partial(
                text=f"LSP hover request failed: {exc}",
            )

        if result is None:
            return ToolResponse.success(
                text=f"No hover information at {file_path}:{line}:{character}",
                data={"hover": None},
            )

        formatted = _format_hover(result)
        return ToolResponse.success(
            text=f"Hover at {file_path}:{line}:{character}:\n{formatted}",
            data={"hover": result},
        )


class LSPDiagnosticsTool(Tool):
    """Get all diagnostics (errors, warnings) for a file.

    Returns the linter/type-checker output for the current file, grouped
    by severity.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        manager: Optional[LSPManager] = None,
    ):
        super().__init__(
            name="LSPDiagnostics",
            description=(
                "Get all diagnostics (errors, warnings, hints) for a file from the "
                "language server. Returns linter and type-checker output. "
                "Use this after editing a file to check for syntax errors, type "
                "mismatches, or other issues."
            ),
            category="readonly",
        )
        self._workspace_root = workspace_root
        self._manager = manager or LSPManager(Path(workspace_root).resolve())

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file",
                type="string",
                description="File path relative to the project root to check.",
                required=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        file_path = parameters.get("file", "")

        client, error, resolved = _ensure_lsp(
            file_path, self._workspace_root, self._manager,
        )
        if error is not None:
            return error

        try:
            uri = Path(resolved).as_uri()
            result = client.document_diagnostic(uri)  # type: ignore[union-attr]
        except Exception:
            # Fallback: older servers use textDocument/publishDiagnostics (push),
            # which our client doesn't collect. Return a helpful message.
            return ToolResponse.partial(
                text=(
                    f"LSP diagnostics request failed or returned no results.\n"
                    f"The language server may not support pull-based diagnostics (textDocument/diagnostic).\n"
                    f"Try running a linter or type-checker via Bash instead "
                    f"(e.g., `pylint {file_path}`, `mypy {file_path}`)."
                ),
            )

        items = result.get("items", []) if isinstance(result, dict) else []
        if not items and isinstance(result, list):
            items = result

        if not items:
            return ToolResponse.success(
                text=f"No diagnostics for {file_path}",
                data={"items": [], "count": 0},
            )

        # Sort by severity (error first) then by line
        items = sorted(items, key=lambda d: (
            d.get("severity", 4),
            d.get("range", {}).get("start", {}).get("line", 0),
        ))

        severity_counts: Dict[str, int] = {}
        lines: List[str] = [f"Diagnostics for {file_path} ({len(items)} issue(s)):"]
        for diag in items:
            sev_name = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}.get(
                diag.get("severity", 0), "UNKNOWN"
            )
            severity_counts[sev_name] = severity_counts.get(sev_name, 0) + 1
            lines.append(_format_diagnostic_entry(diag))

        summary = "  ".join(
            f"{v} {k}(s)" for k, v in sorted(severity_counts.items())
        )
        lines.insert(1, f"Summary: {summary}")

        return ToolResponse.success(
            text="\n".join(lines),
            data={
                "items": items,
                "count": len(items),
                "severity_counts": severity_counts,
            },
        )
