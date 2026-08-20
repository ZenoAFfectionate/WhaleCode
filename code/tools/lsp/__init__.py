"""LSP (Language Server Protocol) tools for WhaleCode agents.

Provides four tools:

- ``LSPDefinitionTool`` — go to symbol definition
- ``LSPReferencesTool`` — find all references
- ``LSPHoverTool`` — type info + documentation on hover
- ``LSPDiagnosticsTool`` — file-level error/warning diagnostics

All tools degrade gracefully when no LSP server is installed.
"""

from .client import LSPClient, LSPError, LSPServerStartError
from .manager import LSPManager, get_shared_manager
from .tools import (
    LSPDefinitionTool,
    LSPDiagnosticsTool,
    LSPHoverTool,
    LSPReferencesTool,
)

__all__ = [
    "LSPClient",
    "LSPError",
    "LSPServerStartError",
    "LSPManager",
    "get_shared_manager",
    "LSPDefinitionTool",
    "LSPReferencesTool",
    "LSPHoverTool",
    "LSPDiagnosticsTool",
]
