"""Shared helpers for coding-oriented built-in tools."""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".backups",
    ".delete_trash",
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

SUPPORTED_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


@dataclass(frozen=True)
class TextWindow:
    """Bounded text window read result."""

    content: str
    encoding: str
    shown_lines: int
    total_lines: int
    truncated: bool
    truncated_by_bytes: bool
    truncated_long_lines: bool
    next_offset: int | None


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
    for encoding in SUPPORTED_TEXT_ENCODINGS:
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


def read_text_window(
    path: str | Path,
    offset: int = 0,
    limit: int = 2000,
    max_bytes: int = 50 * 1024,
    max_line_length: int = 2000,
) -> TextWindow:
    """Read a bounded window of text without loading the full file into memory."""
    file_path = Path(path)
    last_error: Exception | None = None
    for encoding in SUPPORTED_TEXT_ENCODINGS:
        try:
            return _read_text_window_with_encoding(
                file_path=file_path,
                encoding=encoding,
                offset=offset,
                limit=limit,
                max_bytes=max_bytes,
                max_line_length=max_line_length,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Failed to decode file with supported encodings: {last_error}",
    )


def _read_text_window_with_encoding(
    file_path: Path,
    encoding: str,
    offset: int,
    limit: int,
    max_bytes: int,
    max_line_length: int,
) -> TextWindow:
    suffix = f"... (line truncated to {max_line_length} chars)"
    selected: List[str] = []
    total_lines = 0
    bytes_used = 0
    truncated_by_bytes = False
    truncated_long_lines = False
    has_more_lines = False

    with open(file_path, "r", encoding=encoding, newline="") as handle:
        for raw_line in handle:
            total_lines += 1
            line_index = total_lines - 1
            if line_index < offset:
                continue

            if len(selected) >= limit:
                has_more_lines = True
                continue

            line = raw_line.rstrip("\r\n")
            if len(line) > max_line_length:
                line = line[:max_line_length] + suffix
                truncated_long_lines = True

            line_size = len(line.encode("utf-8")) + (1 if selected else 0)
            if bytes_used + line_size > max_bytes:
                truncated_by_bytes = True
                has_more_lines = True
                continue

            selected.append(line)
            bytes_used += line_size

    if offset > total_lines and not (offset == 0 and total_lines == 0):
        raise ValueError(f"Offset {offset} is out of range for this file ({total_lines} lines)")

    next_offset = offset + len(selected) if has_more_lines else None
    return TextWindow(
        content="\n".join(selected),
        encoding=encoding,
        shown_lines=len(selected),
        total_lines=total_lines,
        truncated=has_more_lines or truncated_by_bytes or truncated_long_lines,
        truncated_by_bytes=truncated_by_bytes,
        truncated_long_lines=truncated_long_lines,
        next_offset=next_offset,
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
    temp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    original_mode = None
    if target.exists():
        try:
            original_mode = target.stat().st_mode
        except OSError:
            original_mode = None

    try:
        with open(temp_path, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
        if original_mode is not None:
            os.chmod(temp_path, original_mode)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


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


class EditMatchError(ValueError):
    """Base exception for flexible edit matching."""


class EditNotFoundError(EditMatchError):
    """Raised when a match cannot be found."""


class EditAmbiguousError(EditMatchError):
    """Raised when more than one candidate match exists."""


@dataclass(frozen=True)
class MatchCandidate:
    """A concrete substring candidate inside a file."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ReplaceResult:
    """Successful replacement output."""

    content: str
    strategy: str
    replacements: int


def replace_with_flexible_match(
    content: str,
    old_text: str,
    new_text: str,
    *,
    replace_all: bool = False,
) -> ReplaceResult:
    if old_text == new_text:
        raise EditMatchError("No changes to apply: old_string and new_string are identical.")

    exact_matches = _exact_matches(content, old_text)
    if exact_matches:
        if replace_all:
            return ReplaceResult(
                content=content.replace(old_text, new_text),
                strategy="exact",
                replacements=len(exact_matches),
            )
        if len(exact_matches) == 1:
            candidate = exact_matches[0]
            return ReplaceResult(
                content=content[: candidate.start] + new_text + content[candidate.end :],
                strategy="exact",
                replacements=1,
            )
        raise EditAmbiguousError(
            f"old_string must match file content uniquely. Found {len(exact_matches)} matches."
        )

    if replace_all:
        raise EditNotFoundError("old_string not found in file content.")

    strategies: Sequence[tuple[str, Callable[[str, str], List[MatchCandidate]], bool]] = (
        ("line_trimmed", _line_trimmed_matches, True),
        ("indentation_flexible", _indentation_flexible_matches, True),
        ("whitespace_normalized", _whitespace_normalized_matches, False),
        ("trimmed_boundary", _trimmed_boundary_matches, True),
    )

    for strategy_name, finder, allow_reindent in strategies:
        matches = _dedupe(finder(content, old_text))
        if not matches:
            continue
        if len(matches) > 1:
            raise EditAmbiguousError(
                f"old_string matched multiple locations via {strategy_name}. Provide more surrounding context."
            )

        candidate = matches[0]
        replacement = (
            _reindent_replacement(old_text, candidate.text, new_text)
            if allow_reindent
            else new_text
        )
        return ReplaceResult(
            content=content[: candidate.start] + replacement + content[candidate.end :],
            strategy=strategy_name,
            replacements=1,
        )

    raise EditNotFoundError("old_string not found in file content.")


def _exact_matches(content: str, old_text: str) -> List[MatchCandidate]:
    matches: List[MatchCandidate] = []
    start = 0
    while True:
        index = content.find(old_text, start)
        if index == -1:
            break
        matches.append(MatchCandidate(index, index + len(old_text), old_text))
        start = index + len(old_text)
    return matches


def _line_trimmed_matches(content: str, old_text: str) -> List[MatchCandidate]:
    old_lines = old_text.splitlines()
    if not old_lines:
        return []
    candidates = []
    for candidate in _line_window_candidates(content, len(old_lines)):
        candidate_lines = candidate.text.splitlines()
        if len(candidate_lines) != len(old_lines):
            continue
        if all(a.strip() == b.strip() for a, b in zip(candidate_lines, old_lines)):
            candidates.append(candidate)
    return candidates


def _indentation_flexible_matches(content: str, old_text: str) -> List[MatchCandidate]:
    old_lines = old_text.splitlines()
    if not old_lines:
        return []
    normalized_old = _dedent_preserving_structure(old_text)
    candidates = []
    for candidate in _line_window_candidates(content, len(old_lines)):
        if _dedent_preserving_structure(candidate.text) == normalized_old:
            candidates.append(candidate)
    return candidates


def _whitespace_normalized_matches(content: str, old_text: str) -> List[MatchCandidate]:
    normalized_old = _collapse_whitespace(old_text)
    if not normalized_old:
        return []
    old_lines = old_text.splitlines()
    if not old_lines:
        return []
    candidates = []
    for candidate in _line_window_candidates(content, len(old_lines)):
        if _collapse_whitespace(candidate.text) == normalized_old:
            candidates.append(candidate)
    return candidates


def _trimmed_boundary_matches(content: str, old_text: str) -> List[MatchCandidate]:
    normalized_old = old_text.strip()
    if not normalized_old:
        return []
    old_lines = old_text.splitlines()
    if not old_lines:
        return []
    candidates = []
    for candidate in _line_window_candidates(content, len(old_lines)):
        if candidate.text.strip() == normalized_old:
            candidates.append(candidate)
    return candidates


def _line_window_candidates(content: str, line_count: int) -> Iterable[MatchCandidate]:
    if line_count <= 0:
        return []
    lines = content.splitlines(keepends=True)
    if not lines:
        return []
    starts = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)

    candidates = []
    for index in range(0, len(lines) - line_count + 1):
        start = starts[index]
        end = starts[index + line_count] if index + line_count < len(starts) else len(content)
        text = content[start:end]
        candidates.append(MatchCandidate(start, end, text))
    return candidates


def _dedupe(matches: Iterable[MatchCandidate]) -> List[MatchCandidate]:
    deduped = {}
    for match in matches:
        deduped[(match.start, match.end)] = match
    return list(deduped.values())


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _common_indent(lines: Sequence[str]) -> str:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    indents = [_leading_whitespace(line) for line in non_empty]
    indent = indents[0]
    for item in indents[1:]:
        max_prefix = min(len(indent), len(item))
        common_length = 0
        while common_length < max_prefix and indent[common_length] == item[common_length]:
            common_length += 1
        indent = indent[:common_length]
        if not indent:
            break
    return indent


def _strip_indent(line: str, indent: str) -> str:
    if indent and line.startswith(indent):
        return line[len(indent) :]
    return line


def _dedent_preserving_structure(text: str) -> str:
    lines = text.splitlines()
    indent = _common_indent(lines)
    return "\n".join(_strip_indent(line, indent) for line in lines)


def _reindent_replacement(old_text: str, candidate_text: str, new_text: str) -> str:
    old_lines = old_text.splitlines()
    candidate_lines = candidate_text.splitlines()
    new_lines = new_text.splitlines()
    if not old_lines or not candidate_lines or not new_lines:
        return new_text

    old_indent = _common_indent(old_lines)
    candidate_indent = _common_indent(candidate_lines)
    if old_indent == candidate_indent:
        return new_text

    trailing_newline = ""
    if new_text.endswith("\r\n"):
        trailing_newline = "\r\n"
    elif new_text.endswith("\n"):
        trailing_newline = "\n"
    elif new_text.endswith("\r"):
        trailing_newline = "\r"

    adjusted_lines = []
    for line in new_lines:
        if not line.strip():
            adjusted_lines.append(line)
            continue
        stripped = _strip_indent(line, old_indent)
        adjusted_lines.append(candidate_indent + stripped)
    return "\n".join(adjusted_lines) + trailing_newline


POST_TOOL_TIMEOUT_SECONDS = 20
MAX_DIAGNOSTICS = 20

PYTHON_EXTENSIONS = {".py", ".pyi"}
WEB_EXTENSIONS = {
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".json",
    ".jsonc",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".html",
    ".htm",
    ".md",
    ".mdx",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Diagnostic:
    """Single diagnostic issue."""

    source: str
    severity: str
    message: str
    line: int
    column: int
    code: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FormatterResult:
    """Formatter execution result."""

    attempted: bool
    available: bool
    success: bool
    tool: Optional[str] = None
    command: Optional[List[str]] = None
    changed: bool = False
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiagnosticsResult:
    """Diagnostics execution result."""

    attempted: bool
    available: bool
    success: bool
    tool: Optional[str] = None
    command: Optional[List[str]] = None
    diagnostics: List[Diagnostic] = field(default_factory=list)
    total: int = 0
    truncated: bool = False
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["diagnostics"] = [item.to_dict() for item in self.diagnostics]
        return payload


@dataclass(frozen=True)
class FormatterSpec:
    name: str
    executable: str
    extensions: set[str]
    command_builder: Callable[[Path], List[str]]


@dataclass(frozen=True)
class DiagnosticsSpec:
    name: str
    executable: str
    extensions: set[str]
    command_builder: Callable[[Path], List[str]]
    parser: Callable[[subprocess.CompletedProcess[str]], DiagnosticsResult]


FORMATTERS: Sequence[FormatterSpec] = (
    FormatterSpec(
        name="ruff format",
        executable="ruff",
        extensions=PYTHON_EXTENSIONS,
        command_builder=lambda path: ["ruff", "format", str(path)],
    ),
    FormatterSpec(
        name="black",
        executable="black",
        extensions=PYTHON_EXTENSIONS,
        command_builder=lambda path: ["black", "--quiet", str(path)],
    ),
    FormatterSpec(
        name="prettier",
        executable="prettier",
        extensions=WEB_EXTENSIONS,
        command_builder=lambda path: ["prettier", "--write", str(path)],
    ),
    FormatterSpec(
        name="gofmt",
        executable="gofmt",
        extensions={".go"},
        command_builder=lambda path: ["gofmt", "-w", str(path)],
    ),
    FormatterSpec(
        name="rustfmt",
        executable="rustfmt",
        extensions={".rs"},
        command_builder=lambda path: ["rustfmt", str(path)],
    ),
)


def _parse_ruff_diagnostics(result: subprocess.CompletedProcess[str]) -> DiagnosticsResult:
    issues = json.loads(result.stdout or "[]")
    diagnostics = [
        Diagnostic(
            source="ruff",
            severity="error",
            message=item.get("message", ""),
            line=int(item.get("location", {}).get("row", 1)),
            column=int(item.get("location", {}).get("column", 1)),
            code=item.get("code"),
        )
        for item in issues
    ]
    truncated = len(diagnostics) > MAX_DIAGNOSTICS
    return DiagnosticsResult(
        attempted=True,
        available=True,
        success=True,
        tool="ruff check",
        command=result.args,
        diagnostics=diagnostics[:MAX_DIAGNOSTICS],
        total=len(diagnostics),
        truncated=truncated,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _parse_pyright_diagnostics(result: subprocess.CompletedProcess[str]) -> DiagnosticsResult:
    payload = json.loads(result.stdout or "{}")
    issues = payload.get("generalDiagnostics", []) or []
    diagnostics = [
        Diagnostic(
            source="pyright",
            severity=item.get("severity", "error"),
            message=item.get("message", ""),
            line=int(item.get("range", {}).get("start", {}).get("line", 0)) + 1,
            column=int(item.get("range", {}).get("start", {}).get("character", 0)) + 1,
            code=item.get("rule"),
        )
        for item in issues
    ]
    truncated = len(diagnostics) > MAX_DIAGNOSTICS
    return DiagnosticsResult(
        attempted=True,
        available=True,
        success=True,
        tool="pyright",
        command=result.args,
        diagnostics=diagnostics[:MAX_DIAGNOSTICS],
        total=len(diagnostics),
        truncated=truncated,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _parse_eslint_diagnostics(result: subprocess.CompletedProcess[str]) -> DiagnosticsResult:
    payload = json.loads(result.stdout or "[]")
    issues = []
    for file_result in payload:
        issues.extend(file_result.get("messages", []))
    diagnostics = [
        Diagnostic(
            source="eslint",
            severity="error" if int(item.get("severity", 2)) >= 2 else "warning",
            message=item.get("message", ""),
            line=int(item.get("line", 1)),
            column=int(item.get("column", 1)),
            code=item.get("ruleId"),
        )
        for item in issues
    ]
    truncated = len(diagnostics) > MAX_DIAGNOSTICS
    return DiagnosticsResult(
        attempted=True,
        available=True,
        success=True,
        tool="eslint",
        command=result.args,
        diagnostics=diagnostics[:MAX_DIAGNOSTICS],
        total=len(diagnostics),
        truncated=truncated,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


DIAGNOSTICS: Sequence[DiagnosticsSpec] = (
    DiagnosticsSpec(
        name="ruff check",
        executable="ruff",
        extensions=PYTHON_EXTENSIONS,
        command_builder=lambda path: ["ruff", "check", "--output-format=json", str(path)],
        parser=_parse_ruff_diagnostics,
    ),
    DiagnosticsSpec(
        name="pyright",
        executable="pyright",
        extensions=PYTHON_EXTENSIONS,
        command_builder=lambda path: ["pyright", "--outputjson", str(path)],
        parser=_parse_pyright_diagnostics,
    ),
    DiagnosticsSpec(
        name="eslint",
        executable="eslint",
        extensions={".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"},
        command_builder=lambda path: ["eslint", "-f", "json", str(path)],
        parser=_parse_eslint_diagnostics,
    ),
)


def _run_command(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=POST_TOOL_TIMEOUT_SECONDS,
        check=False,
    )


def _matching_specs(specs: Iterable, file_path: Path):
    suffix = file_path.suffix.lower()
    for spec in specs:
        if suffix in spec.extensions:
            yield spec


def run_formatter(file_path: Path, project_root: Path) -> FormatterResult:
    before = file_path.read_bytes()
    matched_specs = list(_matching_specs(FORMATTERS, file_path))
    if not matched_specs:
        return FormatterResult(
            attempted=False,
            available=False,
            success=False,
            skipped_reason="No formatter is configured for this file type.",
        )

    for spec in matched_specs:
        if shutil.which(spec.executable) is None:
            continue
        command = spec.command_builder(file_path)
        try:
            result = _run_command(command, project_root)
        except subprocess.TimeoutExpired:
            return FormatterResult(
                attempted=True,
                available=True,
                success=False,
                tool=spec.name,
                command=command,
                skipped_reason=f"Formatter timed out after {POST_TOOL_TIMEOUT_SECONDS} seconds.",
            )

        success = result.returncode == 0
        after = file_path.read_bytes()
        return FormatterResult(
            attempted=True,
            available=True,
            success=success,
            tool=spec.name,
            command=command,
            changed=before != after,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            skipped_reason=None if success else "Formatter execution failed.",
        )

    return FormatterResult(
        attempted=False,
        available=False,
        success=False,
        skipped_reason="No applicable formatter executable was found on PATH.",
    )


def run_diagnostics(file_path: Path, project_root: Path) -> DiagnosticsResult:
    matched_specs = list(_matching_specs(DIAGNOSTICS, file_path))
    if not matched_specs:
        return DiagnosticsResult(
            attempted=False,
            available=False,
            success=False,
            skipped_reason="No diagnostics runner is configured for this file type.",
        )

    for spec in matched_specs:
        if shutil.which(spec.executable) is None:
            continue
        command = spec.command_builder(file_path)
        try:
            result = _run_command(command, project_root)
        except subprocess.TimeoutExpired:
            return DiagnosticsResult(
                attempted=True,
                available=True,
                success=False,
                tool=spec.name,
                command=command,
                skipped_reason=f"Diagnostics timed out after {POST_TOOL_TIMEOUT_SECONDS} seconds.",
            )

        try:
            parsed = spec.parser(result)
        except Exception as exc:
            return DiagnosticsResult(
                attempted=True,
                available=True,
                success=False,
                tool=spec.name,
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                skipped_reason=f"Failed to parse diagnostics output: {exc}",
            )
        return parsed

    return DiagnosticsResult(
        attempted=False,
        available=False,
        success=False,
        skipped_reason="No applicable diagnostics executable was found on PATH.",
    )
