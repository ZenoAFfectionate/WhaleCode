"""Shared helpers for coding-oriented built-in tools."""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".DS_Store",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
}


def ensure_working_dir(project_root: str | Path, working_dir: str | Path | None) -> Path:
    """Resolve and validate a working directory within the project root."""
    root = Path(project_root).expanduser().resolve()
    candidate = Path(working_dir).expanduser().resolve() if working_dir else root
    candidate.relative_to(root)
    return candidate


def resolve_path(
    project_root: str | Path,
    working_dir: str | Path | None,
    raw_path: str | Path | None,
) -> Path:
    """Resolve a user path and ensure it stays inside the project root."""
    root = Path(project_root).expanduser().resolve()
    base = ensure_working_dir(root, working_dir)
    requested = Path(raw_path or ".").expanduser()
    candidate = requested.resolve() if requested.is_absolute() else (base / requested).resolve()
    candidate.relative_to(root)
    return candidate


def relative_display(project_root: str | Path, target: str | Path) -> str:
    """Return a project-relative POSIX path for display."""
    root = Path(project_root).expanduser().resolve()
    path = Path(target).expanduser().resolve()
    try:
        rel = path.relative_to(root)
        text = rel.as_posix()
        return text if text else "."
    except ValueError:
        return path.as_posix()


def is_binary_file(path: str | Path, sample_size: int = 8192) -> bool:
    """Best-effort binary-file detection."""
    file_path = Path(path)
    with open(file_path, "rb") as handle:
        sample = handle.read(sample_size)

    if not sample:
        return False
    if b"\x00" in sample:
        return True

    text_bytes = bytearray({7, 8, 9, 10, 12, 13, 27})
    text_bytes.extend(range(0x20, 0x100))
    non_text = sample.translate(None, bytes(text_bytes))
    return len(non_text) / max(len(sample), 1) > 0.30


def read_text_file(path: str | Path) -> Tuple[str, str]:
    """Read a text file with simple encoding fallback."""
    file_path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding, newline="") as handle:
                return handle.read(), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Failed to decode file with supported encodings: {last_error}",
    )


def format_numbered_lines(content: str, start_line: int = 1) -> str:
    """Format file content with 1-based line numbers."""
    if content == "":
        return "[empty file]"

    lines = content.splitlines()
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(
        f"{line_number:>{width}} | {line}"
        for line_number, line in enumerate(lines, start=start_line)
    )


def make_diff_preview(
    old_text: str,
    new_text: str,
    path: str,
    max_lines: int = 120,
    max_chars: int = 12000,
) -> tuple[str, bool]:
    """Create a bounded unified diff preview."""
    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=3,
        )
    )
    if not diff_lines:
        return "[no textual diff]", False

    preview: List[str] = []
    total_chars = 0
    truncated = False
    for line in diff_lines:
        line_chars = len(line) + 1
        if len(preview) >= max_lines or total_chars + line_chars > max_chars:
            truncated = True
            break
        preview.append(line)
        total_chars += line_chars

    if truncated:
        preview.append("[diff truncated]")

    return "\n".join(preview), truncated


def prune_walk_dirs(dirnames: List[str], include_hidden: bool, extra_ignores: Sequence[str] | None = None) -> None:
    """In-place prune for os.walk dir lists."""
    ignore_set = set(DEFAULT_IGNORES)
    if extra_ignores:
        ignore_set.update(extra_ignores)

    dirnames[:] = [
        name
        for name in dirnames
        if name not in ignore_set and (include_hidden or not name.startswith("."))
    ]


def normalize_ignore_patterns(value) -> List[str]:
    """Normalize ignore/include patterns from user input."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def safe_decode_output(payload: bytes, limit_bytes: int = 30000) -> tuple[str, bool]:
    """Decode subprocess output and cap the byte length for prompt safety."""
    truncated = len(payload) > limit_bytes
    if truncated:
        payload = payload[:limit_bytes]
    return payload.decode("utf-8", errors="replace"), truncated


def apply_line_limit(text: str, max_lines: int = 200) -> tuple[str, bool]:
    """Cap text by number of lines."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, False
    return "\n".join(lines[:max_lines]) + "\n[output truncated]", True


def atomic_write(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write a file atomically via a temporary sibling path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temp_path.write_text(content, encoding=encoding)
    os.replace(temp_path, target)


def detect_line_ending(content: str) -> str:
    """Detect the dominant line ending of existing content."""
    if "\r\n" in content:
        return "\r\n"
    if "\r" in content:
        return "\r"
    return "\n"


def normalize_line_endings(content: str, newline: str) -> str:
    """Normalize all line endings in content to the target newline style."""
    if newline not in {"\n", "\r\n", "\r"}:
        newline = "\n"
    normalized = re.sub(r"\r\n|\r|\n", "\n", content)
    return normalized.replace("\n", newline)
