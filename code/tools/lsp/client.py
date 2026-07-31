"""Minimal JSON-RPC LSP client over subprocess stdio.

Implements the client side of the Language Server Protocol (LSP v3.17),
using zero external dependencies — only ``subprocess``, ``json``, and
``threading`` from the standard library.

Frame format (LSP spec):
    Content-Length: <bytes>\r\n
    \r\n
    <JSON payload>
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# LSP shutdown is a two-step handshake (shutdown → exit).
# We give the server this many seconds to respond before force-killing.
_SHUTDOWN_GRACE_SECONDS = 2.0

# Cap response payload at 1 MiB to protect against runaway servers.
_MAX_RESPONSE_BYTES = 1_048_576


class LSPError(Exception):
    """Raised when the LSP server returns an error response."""


class LSPServerStartError(Exception):
    """Raised when the LSP server process cannot be started."""


class LSPClient:
    """JSON-RPC 2.0 client for a single LSP server subprocess.

    Thread-safe: sends are serialised by an internal lock so the agent
    main thread and tool-parallel workers can coexist.

    Usage::

        client = LSPClient(["pylsp"], workspace_root=Path("/project"))
        result = client.definition("file:///project/main.py", 10, 5)
        client.shutdown()
    """

    # LSP initialisation timeout (seconds).  Complex language servers
    # (jdtls, rust-analyzer) may need more; pylsp typically < 1 s.
    INIT_TIMEOUT = 10.0

    def __init__(self, command: List[str], workspace_root: Path):
        self._command = list(command)
        self._workspace_root = workspace_root
        self._id = 0
        self._lock = threading.Lock()
        self._request_id_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None

        try:
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)  # avoid leaking venv into pylsp
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except FileNotFoundError:
            raise LSPServerStartError(
                f"LSP server not found: {self._command[0]!r}. "
                "Install it (e.g. `pip install python-lsp-server`) "
                "or use Grep/Read instead."
            )
        except Exception as exc:
            raise LSPServerStartError(
                f"Failed to start LSP server {self._command!r}: {exc}"
            )

        try:
            self._initialize()
        except Exception:
            self._kill_process()
            raise

    # ------------------------------------------------------------------
    # Public API — LSP methods
    # ------------------------------------------------------------------

    def definition(
        self, uri: str, line: int, character: int
    ) -> List[Dict[str, Any]]:
        """``textDocument/definition`` — go to definition."""
        return self._request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

    def references(
        self, uri: str, line: int, character: int, *, include_declaration: bool = False,
    ) -> List[Dict[str, Any]]:
        """``textDocument/references`` — find all references."""
        return self._request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_declaration},
        })

    def hover(self, uri: str, line: int, character: int) -> Optional[Dict[str, Any]]:
        """``textDocument/hover`` — type info + docstring."""
        return self._request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

    def document_diagnostic(
        self, uri: str,
    ) -> Dict[str, Any]:
        """``textDocument/diagnostic`` — get diagnostics for a file."""
        return self._request("textDocument/diagnostic", {
            "textDocument": {"uri": uri},
        })

    def did_open(self, uri: str, language_id: str, text: str, version: int = 1) -> None:
        """``textDocument/didOpen`` — notify server the file is open."""
        self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": version,
                "text": text,
            },
        })

    def did_change(self, uri: str, text: str, version: int = 2) -> None:
        """``textDocument/didChange`` — notify server the file content changed.

        Sends the full document content (``TextDocumentSyncKind.Full``).
        Incremental sync would require computing diffs, which is overkill
        for the Agent's use case where the model already has the full text
        after Read/Edit/Write.
        """
        self._notify("textDocument/didChange", {
            "textDocument": {
                "uri": uri,
                "version": version,
            },
            "contentChanges": [{"text": text}],
        })

    def did_close(self, uri: str) -> None:
        """``textDocument/didClose`` — notify server the file is closed."""
        self._notify("textDocument/didClose", {
            "textDocument": {"uri": uri},
        })

    def shutdown(self) -> None:
        """Graceful LSP shutdown handshake."""
        if self._process is None or self._process.stdin is None:
            return
        try:
            with self._lock:
                self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "shutdown"})
                self._read_response()
                # LSP exit notification has no params (must not be ``null``).
                self._send({"jsonrpc": "2.0", "method": "exit"})
        except Exception:
            pass
        finally:
            self._kill_process()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """LSP handshake: initialize → initialized."""
        result = self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": self._workspace_root.as_uri(),
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "hover": {
                        "dynamicRegistration": False,
                        "contentFormat": ["markdown", "plaintext"],
                    },
                    "diagnostic": {"dynamicRegistration": False},
                },
            },
        }, timeout=self.INIT_TIMEOUT)
        _logger.debug("LSP server initialized: %s", result.get("serverInfo", {}))
        self._notify("initialized", {})

    # ------------------------------------------------------------------
    # JSON-RPC core
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        with self._request_id_lock:
            self._id += 1
            return self._id

    def _request(
        self, method: str, params: Any, *, timeout: Optional[float] = None,
    ) -> Any:
        """Send a JSON-RPC request and return ``result``."""
        rid = self._next_id()
        payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        with self._lock:
            self._send(payload)
            response = self._read_response(timeout=timeout)
        if "error" in response:
            err = response["error"]
            raise LSPError(
                f"LSP {method}: {err.get('message', 'unknown error')} "
                f"(code={err.get('code', -1)})"
            )
        return response.get("result")

    def _notify(self, method: str, params: Any) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        with self._lock:
            self._send(payload)

    def _send(self, payload: Dict[str, Any]) -> None:
        """Encode a JSON-RPC message onto the wire."""
        if self._process is None or self._process.stdin is None:
            raise LSPError("LSP server process is not running")
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body_bytes)}\r\n\r\n".encode("ascii")
        try:
            self._process.stdin.write(header)
            self._process.stdin.write(body_bytes)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise LSPError(f"LSP server communication lost: {exc}") from exc

    def _read_response(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Read one LSP response frame.

        Frame format::

            Content-Length: <N>\r\n
            \r\n
            <N bytes of JSON>
        """
        if self._process is None or self._process.stdout is None:
            raise LSPError("LSP server process is not running")

        deadline = (time.monotonic() + timeout) if timeout is not None else None
        line = self._read_line(deadline)
        content_length = self._parse_content_length(line)

        # Additional headers (Content-Type, etc.) may follow; skip until empty CRLF.
        while True:
            line = self._read_line(deadline)
            if not line or line == "\r\n":
                break

        if content_length > _MAX_RESPONSE_BYTES:
            raise LSPError(
                f"LSP response too large: {content_length} bytes "
                f"(max {_MAX_RESPONSE_BYTES})"
            )

        body_bytes = self._read_exactly(content_length, deadline)
        try:
            return json.loads(body_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LSPError(f"Invalid JSON from LSP server: {exc}") from exc

    def _read_line(self, deadline: Optional[float]) -> str:
        """Read CRLF-terminated line from server stdout with optional timeout.

        The underlying ``subprocess.PIPE`` is a binary stream, so we must
        compare against bytes (``b"\\r"``, ``b"\\n"``) and decode at the end.
        """
        if self._process is None or self._process.stdout is None:
            raise LSPError("LSP server process is not running")

        parts: List[bytes] = []
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise LSPError("Timeout waiting for LSP server response")
            ch = self._process.stdout.read(1)  # bytes
            if not ch:
                raise LSPError("LSP server closed stdout unexpectedly")
            parts.append(ch)
            if len(parts) >= 2 and parts[-2] == b"\r" and parts[-1] == b"\n":
                break
        return b"".join(parts).decode("ascii")

    @staticmethod
    def _parse_content_length(line: str) -> int:
        """Extract Content-Length from a header line."""
        line = line.strip()
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        raise LSPError(f"Bad LSP header: expected Content-Length, got {line!r}")

    def _read_exactly(self, nbytes: int, deadline: Optional[float]) -> bytes:
        """Read exactly *nbytes* from server stdout."""
        if self._process is None or self._process.stdout is None:
            raise LSPError("LSP server process is not running")

        buf = bytearray()
        while len(buf) < nbytes:
            if deadline is not None and time.monotonic() >= deadline:
                raise LSPError("Timeout reading LSP response body")
            chunk = self._process.stdout.read(min(65536, nbytes - len(buf)))
            if not chunk:
                raise LSPError("LSP server closed stdout mid-response")
            buf.extend(chunk)
        return bytes(buf)

    def _kill_process(self) -> None:
        """Force-terminate the server subprocess."""
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    self._process.kill()
        except Exception:
            pass
        self._process = None
