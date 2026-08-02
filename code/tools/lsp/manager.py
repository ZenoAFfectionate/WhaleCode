"""LSP server lifecycle manager — one per workspace.

An ``LSPManager`` instance is created once per agent workspace and owns
the LSP server subprocesses for all languages used during that session.
Servers are started lazily (on first access to that language) and shut
down together.
"""

from __future__ import annotations

import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import LSPClient, LSPServerStartError, LSPError

_logger = logging.getLogger(__name__)

# File extension → LSP language identifier + server command.
# Each value is (language_id, server_command, human_label).
# ``server_command`` may be a plain executable name (looked up on PATH)
# or an absolute path.  When the executable is not found at startup time
# the manager returns ``None`` and the tool falls back gracefully.
_LANGUAGE_REGISTRY: Dict[str, tuple] = {
    ".py": (
        "python",
        ["pylsp"],
        "Python (pylsp)",
    ),
    ".pyi": (
        "python",
        ["pylsp"],
        "Python stub (pylsp)",
    ),
    ".js": (
        "javascript",
        ["typescript-language-server", "--stdio"],
        "JavaScript (typescript-language-server)",
    ),
    ".ts": (
        "typescript",
        ["typescript-language-server", "--stdio"],
        "TypeScript (typescript-language-server)",
    ),
    ".tsx": (
        "typescriptreact",
        ["typescript-language-server", "--stdio"],
        "TypeScript React (typescript-language-server)",
    ),
    ".jsx": (
        "javascriptreact",
        ["typescript-language-server", "--stdio"],
        "JavaScript React (typescript-language-server)",
    ),
    ".rs": (
        "rust",
        ["rust-analyzer"],
        "Rust (rust-analyzer)",
    ),
    ".go": (
        "go",
        ["gopls"],
        "Go (gopls)",
    ),
}


def _detect_language(file_path: str) -> Optional[str]:
    """Return the LSP language id for *file_path*, or None."""
    suffix = Path(file_path).suffix.lower()
    info = _LANGUAGE_REGISTRY.get(suffix)
    return info[0] if info else None


def _server_command(file_path: str) -> Optional[List[str]]:
    """Return the server command list for *file_path*, or None."""
    suffix = Path(file_path).suffix.lower()
    info = _LANGUAGE_REGISTRY.get(suffix)
    return info[1] if info else None


def _server_label(file_path: str) -> Optional[str]:
    """Return the human-readable server label, or None."""
    suffix = Path(file_path).suffix.lower()
    info = _LANGUAGE_REGISTRY.get(suffix)
    return info[2] if info else None


def _check_executable(command: List[str]) -> Optional[str]:
    """Return the resolved path to *command[0]*, or None if not found.

    PATH 查找失败时回退到当前 Python 环境的 bin 目录——服务进程可能从
    任意 PATH 启动（如 nohup / 绝对路径），而 pylsp 等服务器常装在解释器
    同目录的 bin 下，仅查 PATH 会导致"已安装却找不到"。
    """
    resolved = shutil.which(command[0])
    if resolved:
        return resolved
    env_bin = Path(sys.executable).parent / command[0]
    return str(env_bin) if env_bin.is_file() else None


class LSPManager:
    """Per-workspace LSP server manager.

    Lifecycle::

        manager = LSPManager(workspace_root)
        manager.open_file("/src/main.py")
        client = manager.server_for("/src/main.py")
        if client is not None:
            locations = client.definition(uri, 10, 5)
        manager.shutdown()
    """

    def __init__(self, workspace_root: Path):
        self._workspace_root = workspace_root.resolve()
        # language_id → LSPClient
        self._servers: Dict[str, LSPClient] = {}
        # {uri} — tracks files that have been "opened" via didOpen
        self._opened_uris: Dict[str, str] = {}  # uri → language_id
        # Prevent race when two threads create the same language server.
        self._creation_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def server_for(self, file_path: str) -> Optional[LSPClient]:
        """Return the LSP client for *file_path*'s language, or None.

        Creates the server on first access.  Returns ``None`` when no
        language server is configured for this file type OR when the
        required server executable is not installed.
        """
        command = _server_command(file_path)
        if command is None:
            return None

        language_id = _detect_language(file_path)
        if language_id is None:
            return None

        if language_id not in self._servers:
            with self._creation_lock:
                if language_id in self._servers:
                    return self._servers[language_id]
                resolved = _check_executable(command)
                if resolved is None:
                    _logger.debug(
                        "LSP server %r not found on PATH; tools will fall back",
                        command[0],
                    )
                    return None
                try:
                    # 用解析后的绝对路径启动：subprocess.Popen 不会继承
                    # shutil.which 的结果，若 PATH 缺该 bin 目录会启动失败。
                    resolved_command = [resolved, *command[1:]]
                    self._servers[language_id] = LSPClient(resolved_command, self._workspace_root)
                except LSPServerStartError as exc:
                    _logger.warning(
                        "LSP server %r failed to start: %s. "
                        "Tools will fall back to Grep/Read.",
                        command[0],
                        exc,
                    )
                    return None

        return self._servers[language_id]

    def ensure_file_open(
        self, file_path: str, *, force: bool = False,
    ) -> Optional[LSPClient]:
        """Open *file_path* in the LSP server and return the client.

        Returns None when no server is available for this file type.
        """
        client = self.server_for(file_path)
        if client is None:
            return None

        resolved = self._resolve(file_path)
        uri = resolved.as_uri()
        language_id = _detect_language(file_path)

        if force or uri not in self._opened_uris:
            try:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                text = ""
            if language_id:
                client.did_open(uri, language_id, text)
                self._opened_uris[uri] = language_id

        return client

    def server_available(self, file_path: str) -> bool:
        """Check whether an LSP server is *available* (no side effects)."""
        command = _server_command(file_path)
        if command is None:
            return False
        if _detect_language(file_path) is None:
            return False
        return _check_executable(command) is not None

    def notify_changed(self, file_path: str) -> Optional[LSPClient]:
        """Notify the LSP server that *file_path*'s content has changed.

        Reads the current file content from disk and sends a
        ``textDocument/didChange`` notification (full-document sync).
        Returns the client, or None when no server is available.

        Call this after Edit/Write operations to keep the LSP server's
        view of the file in sync with the workspace.
        """
        client = self.server_for(file_path)
        if client is None:
            return None

        resolved = self._resolve(file_path)
        uri = resolved.as_uri()
        language_id = _detect_language(file_path)
        if language_id is None:
            return client

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            return client

        # Use did_change if the file is already tracked; otherwise did_open.
        if uri in self._opened_uris:
            version = 2  # simple increment; LSP servers mostly ignore exact version
            client.did_change(uri, text, version=version)
        else:
            client.did_open(uri, language_id, text)
            self._opened_uris[uri] = language_id

        return client

    def shutdown(self) -> None:
        """Shut down all managed LSP servers."""
        for client in self._servers.values():
            try:
                client.shutdown()
            except Exception:
                pass
        self._servers.clear()
        self._opened_uris.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LSPManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
        return False  # do not suppress exceptions

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(self, file_path: str) -> Path:
        return (self._workspace_root / file_path).resolve()

    # ------------------------------------------------------------------
    # Static helpers (exposed for tool use)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_language(file_path: str) -> Optional[str]:
        return _detect_language(file_path)

    @staticmethod
    def server_label(file_path: str) -> Optional[str]:
        return _server_label(file_path)

    @staticmethod
    def available_languages() -> Dict[str, str]:
        """Return {extension: human_label} for all registered languages."""
        return {
            ext: info[2]
            for ext, info in _LANGUAGE_REGISTRY.items()
        }
