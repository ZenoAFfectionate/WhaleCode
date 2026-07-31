"""Git-native tools for the coding agent.

Five atomic tools backed by the local ``git`` binary:

- ``GitStatusTool`` — structured ``git status --porcelain=v2 --branch``
- ``GitDiffTool``   — structured diff (numstat + name-status + patch)
- ``GitLogTool``    — structured commit history
- ``GitBlameTool``  — per-line attribution (``--line-porcelain``)
- ``GitCommitTool`` — guarded ``git commit`` wrapper

All subprocess execution goes through ``_run_git`` which reuses the Bash
sandbox environment (secret-stripped env, see ``bash.build_sandbox_env``),
disables interactive credential prompts and applies a hard timeout. Output
parsers are module-level pure functions so they can be unit-tested without
spawning git.
"""

from __future__ import annotations

import codecs
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from ...context.truncator import ObservationTruncator
from ..base import ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse
from ._code_utils import is_binary_file
from .bash import build_sandbox_env
from .file_tools import _WorkspaceFileTool

GIT_TIMEOUT_SECONDS = 30
DEFAULT_LOG_COUNT = 20
MAX_LOG_COUNT = 100
MAX_BLAME_LINES = 500
DIFF_PREVIEW_MAX_LINES = 400
DIFF_PREVIEW_MAX_BYTES = 32_000
MAX_COMMIT_MESSAGE_CHARS = 2000
COMMIT_SUBJECT_SOFT_LIMIT = 72
_TEXT_PREVIEW_ENTRIES = 20

_LOG_FIELD_SEP = "\x1f"
_LOG_RECORD_SEP = "\x1e"
_LOG_FORMAT = "%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s%x1e"
_HEAD_FORMAT = "%H%x1f%h%x1f%s"

_ENABLE_VALUES = {"1", "true", "yes", "on"}

_BLAME_HEADER_RE = re.compile(r"^([0-9a-f]{40,64}) (\d+) (\d+)(?: (\d+))?$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_TEST_DIR_NAMES = {"test", "tests", "testing", "__tests__", "spec"}
_DOC_DIR_NAMES = {"doc", "docs", "documentation"}
_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}


class _GitFailure(Exception):
    """Internal carrier for fatal git execution problems (missing binary, timeout)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _run_git(args: List[str], cwd: Path, project_root: Path) -> Tuple[int, str, str]:
    """Run ``git <args>`` in ``cwd`` with the sandboxed (secret-stripped) env.

    Returns ``(returncode, stdout, stderr)``. Raises ``_GitFailure`` for fatal
    infrastructure problems (git binary missing, hard timeout).
    """
    env = build_sandbox_env(project_root)
    env["GIT_TERMINAL_PROMPT"] = "0"  # never block on credential prompts
    env["GIT_OPTIONAL_LOCKS"] = "0"   # read-only commands skip the index lock
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        if shutil.which("git") is None:
            raise _GitFailure(
                ToolErrorCode.EXECUTION_ERROR,
                "git executable not found on PATH. Install git to use the Git tools.",
            )
        raise _GitFailure(
            ToolErrorCode.EXECUTION_ERROR,
            f"Working directory for git does not exist: {cwd}",
        )
    except subprocess.TimeoutExpired:
        raise _GitFailure(
            ToolErrorCode.TIMEOUT,
            f"`git {args[0]}` timed out after {GIT_TIMEOUT_SECONDS}s.",
        )
    return proc.returncode, proc.stdout, proc.stderr


def _find_repo_root(cwd: Path, project_root: Path) -> Optional[Path]:
    """Locate the git work-tree root containing ``cwd`` (None when not a repo)."""
    rc, stdout, _ = _run_git(["rev-parse", "--show-toplevel"], cwd, project_root)
    if rc != 0:
        return None
    text = stdout.strip()
    return Path(text).resolve() if text else None


def _ref_exists(ref: str, cwd: Path, project_root: Path) -> bool:
    """True when ``ref`` resolves to a commit (quiet verify)."""
    rc, _, _ = _run_git(
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd, project_root
    )
    return rc == 0


def _ref_validation_error(value: str, label: str) -> Optional[str]:
    """Reject option-like or whitespace-containing refs (option-injection guard).

    Refs are appended as positional argv entries; a value starting with ``-``
    would be consumed by git as an *option* (e.g. ``--output=/tmp/x``), which
    is an injection vector. Real refs never start with ``-`` or contain
    whitespace.
    """
    if value.startswith("-"):
        return f"{label} must be a git ref, not an option-like value: {value!r}"
    if any(ch.isspace() for ch in value):
        return f"{label} must not contain whitespace: {value!r}"
    return None


# ---------------------------------------------------------------------------
# Pure parsers (unit-tested without git)
# ---------------------------------------------------------------------------


def _unquote_path(path: str) -> str:
    """Decode git's C-style quoted path (octal escapes) when present.

    Only quoted paths that actually contain escape sequences are decoded, so a
    file legitimately named ``"foo"`` (literal quotes) is left untouched.
    """
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        body = path[1:-1]
        if "\\" not in body:
            return path
        try:
            return codecs.decode(body, "unicode_escape").encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return body
    return path


def parse_status_porcelain_v2(output: str) -> Dict[str, Any]:
    """Parse ``git status --porcelain=v2 --branch --show-stash`` output."""
    branch: Dict[str, Any] = {
        "name": "",
        "oid": "",
        "upstream": None,
        "ahead": 0,
        "behind": 0,
        "detached": False,
        "initial": False,
    }
    staged: List[Dict[str, Any]] = []
    unstaged: List[Dict[str, Any]] = []
    untracked: List[str] = []
    conflicted: List[Dict[str, Any]] = []
    stash_count = 0

    for line in output.splitlines():
        if not line:
            continue
        if line.startswith("# "):
            body = line[2:]
            if body.startswith("branch.oid "):
                oid = body[len("branch.oid "):].strip()
                if oid == "(initial)":
                    branch["initial"] = True
                else:
                    branch["oid"] = oid
            elif body.startswith("branch.head "):
                head = body[len("branch.head "):].strip()
                if head == "(detached)":
                    branch["detached"] = True
                else:
                    branch["name"] = head
            elif body.startswith("branch.upstream "):
                branch["upstream"] = body[len("branch.upstream "):].strip()
            elif body.startswith("branch.ab "):
                match = re.match(r"branch\.ab \+(\d+) -(\d+)", body)
                if match:
                    branch["ahead"] = int(match.group(1))
                    branch["behind"] = int(match.group(2))
            elif body.startswith("stash "):
                try:
                    stash_count = int(body[len("stash "):].strip())
                except ValueError:
                    stash_count = 0
            continue

        tag = line[0]
        if tag == "?":
            untracked.append(_unquote_path(line[2:]))
        elif tag == "!":
            continue  # ignored entries (only shown with --ignored)
        elif tag == "u":
            parts = line.split(" ", 10)
            if len(parts) < 11:
                continue
            conflicted.append({"path": _unquote_path(parts[10]), "xy": parts[1]})
        elif tag == "2":
            field_part, _, orig = line.partition("\t")
            parts = field_part.split(" ", 9)
            if len(parts) < 10:
                continue
            xy = parts[1]
            entry = {
                "path": _unquote_path(parts[9]),
                "xy": xy,
                "orig_path": _unquote_path(orig) if orig else None,
            }
            if xy[0] != ".":
                staged.append(entry)
            if len(xy) > 1 and xy[1] != ".":
                unstaged.append(entry)
        elif tag == "1":
            parts = line.split(" ", 8)
            if len(parts) < 9:
                continue
            xy = parts[1]
            entry = {"path": _unquote_path(parts[8]), "xy": xy, "orig_path": None}
            if xy[0] != ".":
                staged.append(entry)
            if len(xy) > 1 and xy[1] != ".":
                unstaged.append(entry)

    return {
        "branch": branch,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "conflicted": conflicted,
        "stash_count": stash_count,
        "is_clean": not (staged or unstaged or untracked or conflicted),
    }


def parse_numstat_z(output: str) -> List[Dict[str, Any]]:
    """Parse ``git diff --numstat -z`` output.

    Records are NUL-terminated: ``<added>\\t<deleted>\\t<path>\\0``; renames
    use ``<added>\\t<deleted>\\t\\0<old>\\0<new>\\0``. Binary files report
    ``-`` for both counts.
    """
    records = output.split("\0")
    files: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(records):
        record = records[idx]
        idx += 1
        if not record:
            continue
        fields = record.split("\t")
        if len(fields) < 3:
            continue
        added_raw, deleted_raw = fields[0], fields[1]
        binary = added_raw == "-" or deleted_raw == "-"
        try:
            additions = 0 if binary else int(added_raw)
            deletions = 0 if binary else int(deleted_raw)
        except ValueError:
            continue
        path_field = fields[2]
        orig_path: Optional[str] = None
        if path_field == "":
            # rename/copy: old and new paths follow as separate NUL records
            if idx + 1 >= len(records):
                break
            orig_path = records[idx]
            path = records[idx + 1]
            idx += 2
        else:
            path = path_field
        files.append(
            {
                "path": path,
                "additions": additions,
                "deletions": deletions,
                "binary": binary,
                "orig_path": orig_path,
            }
        )
    return files


def parse_name_status_z(output: str) -> Dict[str, Dict[str, Any]]:
    """Parse ``git diff --name-status -z`` into ``{path: status_info}``."""
    records = [record for record in output.split("\0") if record]
    result: Dict[str, Dict[str, Any]] = {}
    idx = 0
    while idx < len(records):
        token = records[idx]
        idx += 1
        code = token[0] if token else ""
        if code in ("R", "C"):
            if idx + 1 >= len(records):
                break
            orig_path, path = records[idx], records[idx + 1]
            idx += 2
            result[path] = {
                "status": code,
                "score": token[1:] or None,
                "orig_path": orig_path,
            }
        else:
            if idx >= len(records):
                break
            path = records[idx]
            idx += 1
            result[path] = {"status": code, "score": None, "orig_path": None}
    return result


def merge_diff_files(
    numstat_files: List[Dict[str, Any]],
    name_status: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Join numstat entries with name-status entries on the (new) path."""
    merged: List[Dict[str, Any]] = []
    for item in numstat_files:
        status_info = name_status.get(item["path"], {})
        merged.append(
            {
                "path": item["path"],
                "status": status_info.get("status") or ("R" if item["orig_path"] else "M"),
                "additions": item["additions"],
                "deletions": item["deletions"],
                "binary": item["binary"],
                "orig_path": item["orig_path"] or status_info.get("orig_path"),
            }
        )
    return merged


def parse_log(output: str) -> List[Dict[str, Any]]:
    """Parse ``git log --pretty=format:%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s%x1e``."""
    commits: List[Dict[str, Any]] = []
    for record in output.split(_LOG_RECORD_SEP):
        record = record.strip("\n")
        if not record.strip():
            continue
        fields = record.split(_LOG_FIELD_SEP)
        if len(fields) < 6:
            continue
        commits.append(
            {
                "hash": fields[0],
                "short_hash": fields[1],
                "author": fields[2],
                "email": fields[3],
                "date": fields[4],
                "subject": fields[5].strip(),
            }
        )
    return commits


def _parse_tz_offset(tz: Optional[str]) -> timezone:
    if not tz or not re.fullmatch(r"[+-]\d{4}", tz):
        return timezone.utc
    sign = 1 if tz[0] == "+" else -1
    return timezone(sign * timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])))


def _format_blame_date(unix_ts: Optional[str], tz: Optional[str]) -> str:
    if not unix_ts:
        return ""
    try:
        timestamp = int(unix_ts)
    except ValueError:
        return ""
    try:
        return datetime.fromtimestamp(timestamp, tz=_parse_tz_offset(tz)).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def parse_blame(output: str) -> List[Dict[str, Any]]:
    """Parse ``git blame --line-porcelain`` into per-line entries.

    Each block starts with ``<sha> <orig_line> <final_line> [<count>]``,
    followed by metadata lines (author, author-time, summary, ...) and one
    TAB-prefixed content line per source line in the group.
    """
    commits: Dict[str, Dict[str, Any]] = {}
    lines: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    content_index = 0

    for raw in output.splitlines():
        header = _BLAME_HEADER_RE.match(raw)
        if header:
            sha = header.group(1)
            current = {"hash": sha, "final_line": int(header.group(3))}
            content_index = 0
            commits.setdefault(sha, {})
            continue
        if current is None:
            continue
        if raw.startswith("\t"):
            meta = commits[current["hash"]]
            lines.append(
                {
                    "line": current["final_line"] + content_index,
                    "hash": current["hash"],
                    "short_hash": current["hash"][:7],
                    "author": meta.get("author", ""),
                    "email": meta.get("author-mail", "").strip("<>"),
                    "date": _format_blame_date(meta.get("author-time"), meta.get("author-tz")),
                    "summary": meta.get("summary", ""),
                    "content": raw[1:],
                }
            )
            content_index += 1
            continue
        key, _, value = raw.partition(" ")
        commits[current["hash"]][key] = value

    return lines


# ---------------------------------------------------------------------------
# Commit message helpers
# ---------------------------------------------------------------------------


def sanitize_commit_message(message: str) -> str:
    """Strip control characters (newlines preserved) and surrounding whitespace."""
    return _CONTROL_CHARS_RE.sub("", message).strip()


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    if any(part.lower() in _TEST_DIR_NAMES for part in parts[:-1]):
        return True
    stem = parts[-1].lower().rsplit(".", 1)[0]
    return stem.startswith("test_") or stem.endswith("_test") or stem in {"test", "spec"}


def _is_doc_path(path: str) -> bool:
    parts = path.split("/")
    if any(part.lower() in _DOC_DIR_NAMES for part in parts[:-1]):
        return True
    name = parts[-1].lower()
    ext = "." + name.rsplit(".", 1)[1] if "." in name else ""
    return ext in _DOC_EXTENSIONS


def _common_dir(paths: List[str]) -> str:
    """Longest common directory prefix of repo-relative paths ('' when none)."""
    dir_parts: List[List[str]] = []
    for path in paths:
        parent = PurePosixPath(path).parent
        dir_parts.append([] if str(parent) == "." else list(parent.parts))
    prefix: List[str] = []
    for chunk in zip(*dir_parts):
        if all(part == chunk[0] for part in chunk):
            prefix.append(chunk[0])
        else:
            break
    return "/".join(prefix)


def generate_commit_message(files: List[Dict[str, Any]]) -> str:
    """Deterministically derive a conventional-commit style message.

    ``files`` are merged diff entries (path/status/additions/deletions).
    """
    if not files:
        return "chore: update repository"

    paths = [item["path"] for item in files]
    statuses = {item["status"] for item in files}

    if all(_is_test_path(p) for p in paths):
        prefix = "test"
    elif all(_is_doc_path(p) for p in paths):
        prefix = "docs"
    elif statuses == {"A"}:
        prefix = "feat"
    else:
        prefix = ""

    if len(files) == 1:
        name = PurePosixPath(paths[0]).name
        if statuses == {"D"}:
            return f"chore: remove {name}"
        if statuses == {"R"}:
            orig = PurePosixPath(files[0]["orig_path"] or "").name
            return f"refactor: rename {orig} to {name}"
        if prefix == "feat":
            return f"feat: add {name}"
        if prefix:
            return f"{prefix}: update {name}"
        return f"update {name}"

    common = _common_dir(paths)
    location = f" in {common}" if common else " across the repository"
    count = len(files)
    if statuses == {"D"}:
        return f"chore: remove {count} files{location}"
    if statuses == {"R"}:
        return f"refactor: rename {count} files{location}"
    if prefix == "feat":
        return f"feat: add {count} files{location}"
    if prefix:
        return f"{prefix}: update {count} files{location}"
    return f"update {count} files{location}"


# ---------------------------------------------------------------------------
# Text renderers
# ---------------------------------------------------------------------------


def _render_status_text(status: Dict[str, Any]) -> str:
    branch = status["branch"]
    if branch["detached"]:
        head = f"HEAD detached at {branch['oid'][:7] or 'unknown'}"
    elif branch["initial"]:
        head = f"On branch {branch['name']} (no commits yet)"
    else:
        head = f"On branch {branch['name']}"
    if branch["upstream"]:
        head += f" | upstream {branch['upstream']} (ahead {branch['ahead']}, behind {branch['behind']})"

    lines = [head]
    if status["stash_count"]:
        lines.append(f"Stash entries: {status['stash_count']}")
    if status["is_clean"]:
        lines.append("Working tree clean.")
        return "\n".join(lines)

    lines.append(
        f"Staged: {len(status['staged'])} | Unstaged: {len(status['unstaged'])} | "
        f"Untracked: {len(status['untracked'])} | Conflicted: {len(status['conflicted'])}"
    )

    def _entries(label: str, entries: List[Dict[str, Any]]) -> None:
        shown = entries[:_TEXT_PREVIEW_ENTRIES]
        for entry in shown:
            rename = f" (from {entry['orig_path']})" if entry.get("orig_path") else ""
            lines.append(f"  {label:<9} {entry['xy']} {entry['path']}{rename}")
        if len(entries) > len(shown):
            lines.append(f"  ... and {len(entries) - len(shown)} more {label.strip()} entries")

    _entries("staged", status["staged"])
    _entries("unstaged", status["unstaged"])
    for path in status["untracked"][:_TEXT_PREVIEW_ENTRIES]:
        lines.append(f"  untracked ?? {path}")
    if len(status["untracked"]) > _TEXT_PREVIEW_ENTRIES:
        lines.append(f"  ... and {len(status['untracked']) - _TEXT_PREVIEW_ENTRIES} more untracked entries")
    _entries("conflict", status["conflicted"])
    return "\n".join(lines)


def _render_diff_text(data: Dict[str, Any], stat_only: bool) -> str:
    header = f"Diff mode: {data['mode']}"
    if data.get("ref"):
        header += f" (ref: {data['ref']})"
    lines = [header, f"Scope: {data['scope']}"]

    if data["total_files"] == 0:
        lines.append("No changes.")
        return "\n".join(lines)

    lines.append(
        f"Files changed: {data['total_files']} "
        f"(+{data['total_additions']} -{data['total_deletions']})"
    )
    for item in data["files"][:_TEXT_PREVIEW_ENTRIES]:
        rename = f" (from {item['orig_path']})" if item["orig_path"] else ""
        counts = " [binary]" if item["binary"] else f" (+{item['additions']} -{item['deletions']})"
        lines.append(f"  {item['status']} {item['path']}{rename}{counts}")
    if data["total_files"] > _TEXT_PREVIEW_ENTRIES:
        lines.append(f"  ... and {data['total_files'] - _TEXT_PREVIEW_ENTRIES} more files")

    if not stat_only and data.get("patch"):
        lines.extend(["", "--- patch ---", data["patch"]])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _GitBaseTool(_WorkspaceFileTool):
    """Shared git execution / repo-resolution helpers for the Git tools."""

    ENABLE_ENV = "GIT_TOOLS_ENABLED"

    def __init__(
        self,
        *,
        name: str,
        description: str,
        project_root: str = ".",
        working_dir: Optional[str] = None,
        config: Any = None,
        output_truncator: Optional[ObservationTruncator] = None,
        category: str = "readonly",
    ):
        super().__init__(
            name=name,
            description=description,
            project_root=project_root,
            working_dir=working_dir,
            config=config,
            category=category,
        )
        if output_truncator is not None:
            self.output_truncator = output_truncator
        else:
            self.output_truncator = ObservationTruncator(
                max_lines=DIFF_PREVIEW_MAX_LINES,
                max_bytes=DIFF_PREVIEW_MAX_BYTES,
                truncate_direction="head",
                output_dir=str(self.project_root / "memory" / "tool-output"),
            )

    @classmethod
    def is_enabled_by_default(cls) -> bool:
        raw = os.getenv(cls.ENABLE_ENV)
        if raw is None or not raw.strip():
            return True
        return raw.strip().lower() in _ENABLE_VALUES

    def _git(self, args: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        return _run_git(args, cwd or self.working_dir, self.project_root)

    @staticmethod
    def _failure_response(exc: _GitFailure) -> ToolResponse:
        return ToolResponse.error(code=exc.code, message=exc.message)

    def _require_repo(self) -> Tuple[Optional[Path], Optional[ToolResponse]]:
        """Return ``(repo_root, None)`` or ``(None, error_response)``."""
        try:
            repo_root = _find_repo_root(self.working_dir, self.project_root)
        except _GitFailure as exc:
            return None, self._failure_response(exc)
        if repo_root is None:
            return None, ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=(
                    f"Not a git repository: {self._display_path(self.working_dir)}. "
                    "Git tools require the workspace to be inside a git repository."
                ),
            )
        return repo_root, None

    def _repo_relative(
        self, repo_root: Path, raw_path: str
    ) -> Tuple[Optional[str], Optional[ToolResponse]]:
        """Resolve a workspace-relative path to a repo-relative POSIX path."""
        try:
            resolved = self._resolve_path(raw_path)
        except ValueError:
            return None, ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"Path escapes the workspace root: {raw_path}",
            )
        try:
            rel = resolved.relative_to(repo_root)
        except ValueError:
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Path is outside the git repository: {raw_path}",
            )
        text = rel.as_posix()
        return (text if text else "."), None

    @staticmethod
    def _pathspec(repo_rel: str) -> str:
        """Prefix with ``./`` so git always treats the value as a literal path.

        Without the prefix a path starting with ``:`` (e.g. ``:!x.py``) could
        be interpreted as pathspec magic by some git versions.
        """
        return f"./{repo_rel}"

    def _truncate_patch(self, patch: str) -> Tuple[str, bool, Optional[str]]:
        # Deliberately reuse the truncator's own limits: the agent injects a
        # shared instance whose thresholds stay consistent with Bash/WebFetch.
        truncation = self.output_truncator.truncate(
            tool_name="git_diff",
            output=patch,
            truncate_direction="head",
        )
        preview = truncation.get("display_preview", truncation.get("preview", patch))
        return (
            preview,
            bool(truncation.get("truncated", False)),
            truncation.get("full_output_path"),
        )

    def _diff_files(self, repo_root: Path, base_args: List[str]) -> List[Dict[str, Any]]:
        """Merged numstat + name-status entries for ``base_args`` (git diff ...).

        Raises ``_GitFailure`` when either git invocation fails — silently
        returning an empty list would masquerade a real error as "no changes".
        """
        full_args = ["-c", "core.quotePath=false"] + base_args
        rc, numstat_out, stderr = self._git(full_args + ["--numstat", "-z"], cwd=repo_root)
        if rc != 0:
            raise _GitFailure(
                ToolErrorCode.EXECUTION_ERROR,
                f"git {base_args[0]} --numstat failed (exit {rc}): {stderr.strip() or '[no output]'}",
            )
        rc, name_status_out, stderr = self._git(full_args + ["--name-status", "-z"], cwd=repo_root)
        if rc != 0:
            raise _GitFailure(
                ToolErrorCode.EXECUTION_ERROR,
                f"git {base_args[0]} --name-status failed (exit {rc}): {stderr.strip() or '[no output]'}",
            )
        return merge_diff_files(parse_numstat_z(numstat_out), parse_name_status_z(name_status_out))


# ---------------------------------------------------------------------------
# GitStatus
# ---------------------------------------------------------------------------


class GitStatusTool(_GitBaseTool):
    """Structured ``git status`` (branch, staged, unstaged, untracked, conflicts)."""

    def __init__(
        self,
        name: str = "GitStatus",
        project_root: str = ".",
        working_dir: Optional[str] = None,
        config: Any = None,
        output_truncator: Optional[ObservationTruncator] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Show the git working tree status as structured data: current branch, "
                "upstream ahead/behind counts, staged, unstaged, untracked and conflicted "
                "files. Prefer this over running `git status` in Bash."
            ),
            project_root=project_root,
            working_dir=working_dir,
            config=config,
            output_truncator=output_truncator,
            category="readonly",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description=(
                    "Optional workspace-relative file or directory to limit the status to. "
                    "Defaults to the whole repository."
                ),
                required=False,
                default=".",
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        raw_path = parameters.get("path", ".")
        if raw_path is None:
            raw_path = "."
        if not isinstance(raw_path, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"path must be a string when provided, got {type(raw_path).__name__}.",
            )

        repo_root, error = self._require_repo()
        if error:
            return error

        args = ["-c", "core.quotePath=false", "status", "--porcelain=v2", "--branch", "--show-stash"]
        scope = "."
        if raw_path and raw_path != ".":
            repo_rel, error = self._repo_relative(repo_root, raw_path)
            if error:
                return error
            args += ["--", self._pathspec(repo_rel)]
            scope = repo_rel

        try:
            rc, stdout, stderr = self._git(args, cwd=repo_root)
        except _GitFailure as exc:
            return self._failure_response(exc)
        if rc != 0:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"git status failed (exit {rc}): {stderr.strip() or stdout.strip() or '[no output]'}",
            )

        parsed = parse_status_porcelain_v2(stdout)
        parsed["repo_root"] = str(repo_root)
        parsed["scope"] = scope

        # Fallback: older git (< 2.35) silently ignores --show-stash, so
        # stash_count stays 0 even when stashes exist.  Count via `git stash
        # list` when the porcelain output didn't include a stash line.
        if parsed["stash_count"] == 0:
            try:
                _rc, stash_out, _stderr = self._git(
                    ["stash", "list"], cwd=repo_root,
                )
                if _rc == 0:
                    # Each stash entry is one line: "stash@{0}: ..."
                    parsed["stash_count"] = len(
                        [l for l in stash_out.splitlines() if l.strip()]
                    )
            except _GitFailure:
                pass  # stay with 0 from porcelain

        text = _render_status_text(parsed)
        if scope != ".":
            text = f"Scope: {scope}\n{text}"
        return ToolResponse.success(text=text, data=parsed)


# ---------------------------------------------------------------------------
# GitDiff
# ---------------------------------------------------------------------------


class GitDiffTool(_GitBaseTool):
    """Structured diff between worktree / index / arbitrary commits."""

    def __init__(
        self,
        name: str = "GitDiff",
        project_root: str = ".",
        working_dir: Optional[str] = None,
        config: Any = None,
        output_truncator: Optional[ObservationTruncator] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Show changes as structured data: per-file status (A/M/D/R), addition and "
                "deletion counts, plus the unified patch (truncated with a recoverable "
                "full-output file when large). Use staged=true to inspect what would be "
                "committed, or commit=<ref> to compare against a commit or range. "
                "Untracked files are not included; use GitStatus to discover them."
            ),
            project_root=project_root,
            working_dir=working_dir,
            config=config,
            output_truncator=output_truncator,
            category="readonly",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="staged",
                type="boolean",
                description="Compare the index against HEAD (what would be committed). Mutually exclusive with `commit`.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="commit",
                type="string",
                description="Compare the working tree against a commit-ish or range (e.g. HEAD~1, main...feature). Mutually exclusive with `staged`.",
                required=False,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Limit the diff to a workspace-relative file or directory.",
                required=False,
            ),
            ToolParameter(
                name="context_lines",
                type="integer",
                description="Number of context lines in the patch (0-20).",
                required=False,
                default=3,
            ),
            ToolParameter(
                name="stat_only",
                type="boolean",
                description="Only return per-file statistics, skip the patch text.",
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        staged = parameters.get("staged", False)
        commit = parameters.get("commit")
        raw_path = parameters.get("path")
        context_lines = parameters.get("context_lines", 3)
        stat_only = parameters.get("stat_only", False)

        if not isinstance(staged, bool):
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="staged must be a boolean.")
        if not isinstance(stat_only, bool):
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="stat_only must be a boolean.")
        if commit is not None and not isinstance(commit, str):
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="commit must be a string when provided.")
        if raw_path is not None and not isinstance(raw_path, str):
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="path must be a string when provided.")
        if isinstance(context_lines, bool) or not isinstance(context_lines, int) or not 0 <= context_lines <= 20:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"context_lines must be an integer between 0 and 20, got {context_lines!r}.",
            )
        if staged and commit:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="staged and commit are mutually exclusive; pass only one.",
            )
        if commit is not None and not commit.strip():
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="commit must be a non-empty ref when provided.")
        if commit is not None:
            ref_error = _ref_validation_error(commit.strip(), "commit")
            if ref_error:
                return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message=ref_error)

        repo_root, error = self._require_repo()
        if error:
            return error

        repo_rel: Optional[str] = None
        if raw_path:
            repo_rel, error = self._repo_relative(repo_root, raw_path)
            if error:
                return error

        base = ["-c", "core.quotePath=false", "diff", "--no-color", "-M"]
        mode = "worktree"
        ref: Optional[str] = None
        if staged:
            base.append("--cached")
            mode = "staged"
            ref = "HEAD"
        elif commit:
            base.append(commit.strip())
            mode = "commit"
            ref = commit.strip()
        path_args = ["--", self._pathspec(repo_rel)] if repo_rel else []

        try:
            rc, numstat_out, stderr = self._git(base + ["--numstat", "-z"] + path_args, cwd=repo_root)
            if rc != 0:
                return self._diff_command_error(mode, ref, rc, stderr)
            rc, name_status_out, stderr = self._git(base + ["--name-status", "-z"] + path_args, cwd=repo_root)
            if rc != 0:
                return self._diff_command_error(mode, ref, rc, stderr)
            patch = ""
            if not stat_only:
                rc, patch, stderr = self._git(base + [f"-U{context_lines}"] + path_args, cwd=repo_root)
                if rc != 0:
                    return self._diff_command_error(mode, ref, rc, stderr)
        except _GitFailure as exc:
            return self._failure_response(exc)

        files = merge_diff_files(parse_numstat_z(numstat_out), parse_name_status_z(name_status_out))
        preview, truncated, full_output_path = ("", False, None)
        if not stat_only and patch:
            preview, truncated, full_output_path = self._truncate_patch(patch)

        data: Dict[str, Any] = {
            "mode": mode,
            "ref": ref,
            "scope": repo_rel or ".",
            "files": files,
            "total_files": len(files),
            "total_additions": sum(item["additions"] for item in files),
            "total_deletions": sum(item["deletions"] for item in files),
            "context_lines": context_lines,
            "patch": preview,
            "patch_truncated": truncated,
            "full_output_path": full_output_path,
        }
        text = _render_diff_text(data, stat_only)
        if truncated:
            return ToolResponse.partial(text=text, data=data)
        return ToolResponse.success(text=text, data=data)

    @staticmethod
    def _diff_command_error(mode: str, ref: Optional[str], rc: int, stderr: str) -> ToolResponse:
        detail = stderr.strip() or "[no output]"
        if mode == "commit":
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"git diff failed for ref {ref!r} (exit {rc}). The ref may be invalid.\n{detail}",
            )
        return ToolResponse.error(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=f"git diff failed (exit {rc}): {detail}",
        )


# ---------------------------------------------------------------------------
# GitLog
# ---------------------------------------------------------------------------


class GitLogTool(_GitBaseTool):
    """Structured commit history."""

    def __init__(
        self,
        name: str = "GitLog",
        project_root: str = ".",
        working_dir: Optional[str] = None,
        config: Any = None,
        output_truncator: Optional[ObservationTruncator] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Show commit history as structured data (hash, author, email, ISO date, "
                "subject). Supports count, path, author and message-grep filters."
            ),
            project_root=project_root,
            working_dir=working_dir,
            config=config,
            output_truncator=output_truncator,
            category="readonly",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="count",
                type="integer",
                description=f"Maximum commits to return (1-{MAX_LOG_COUNT}).",
                required=False,
                default=DEFAULT_LOG_COUNT,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Only commits touching this workspace-relative file or directory.",
                required=False,
            ),
            ToolParameter(
                name="author",
                type="string",
                description="Filter by author (forwarded to git --author).",
                required=False,
            ),
            ToolParameter(
                name="grep",
                type="string",
                description="Only commits whose message matches this pattern.",
                required=False,
            ),
            ToolParameter(
                name="ref",
                type="string",
                description="Starting ref (branch, tag or commit). Defaults to HEAD.",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        count = parameters.get("count", DEFAULT_LOG_COUNT)
        raw_path = parameters.get("path")
        author = parameters.get("author")
        grep = parameters.get("grep")
        ref = parameters.get("ref")

        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_LOG_COUNT:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"count must be an integer between 1 and {MAX_LOG_COUNT}, got {count!r}.",
            )
        for label, value in (("path", raw_path), ("author", author), ("grep", grep), ("ref", ref)):
            if value is not None and not isinstance(value, str):
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"{label} must be a string when provided.",
                )
        if ref is not None and not ref.strip():
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="ref must be non-empty when provided.")
        if ref is not None:
            ref_error = _ref_validation_error(ref.strip(), "ref")
            if ref_error:
                return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message=ref_error)

        repo_root, error = self._require_repo()
        if error:
            return error

        repo_rel: Optional[str] = None
        if raw_path:
            repo_rel, error = self._repo_relative(repo_root, raw_path)
            if error:
                return error

        start_ref = ref.strip() if ref else "HEAD"
        try:
            if not _ref_exists(start_ref, repo_root, self.project_root):
                if ref:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_PARAM,
                        message=f"Unknown ref: {ref!r}. It does not resolve to a commit.",
                    )
                data = {"commits": [], "count": 0, "has_more": False, "ref": start_ref, "scope": repo_rel or "."}
                return ToolResponse.success(
                    text="No commits yet in this repository.",
                    data=data,
                )

            args = ["log", "--no-color", f"--pretty=format:{_LOG_FORMAT}", "-n", str(count + 1)]
            if ref:
                args.append(start_ref)
            if author:
                args.append(f"--author={author}")
            if grep:
                args.append(f"--grep={grep}")
            if repo_rel:
                args += ["--", self._pathspec(repo_rel)]

            rc, stdout, stderr = self._git(args, cwd=repo_root)
        except _GitFailure as exc:
            return self._failure_response(exc)
        if rc != 0:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"git log failed (exit {rc}): {stderr.strip() or '[no output]'}",
            )

        commits = parse_log(stdout)
        has_more = len(commits) > count
        commits = commits[:count]

        data = {
            "commits": commits,
            "count": len(commits),
            "has_more": has_more,
            "ref": start_ref,
            "scope": repo_rel or ".",
        }

        lines = [f"Commits: {len(commits)} (ref: {start_ref}, has_more: {str(has_more).lower()})"]
        if not commits:
            lines.append("No matching commits.")
        for item in commits:
            lines.append(f"{item['short_hash']} {item['date']} {item['author']}: {item['subject']}")
        return ToolResponse.success(text="\n".join(lines), data=data)


# ---------------------------------------------------------------------------
# GitBlame
# ---------------------------------------------------------------------------


class GitBlameTool(_GitBaseTool):
    """Per-line attribution via ``git blame --line-porcelain``."""

    def __init__(
        self,
        name: str = "GitBlame",
        project_root: str = ".",
        working_dir: Optional[str] = None,
        config: Any = None,
        output_truncator: Optional[ObservationTruncator] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Show per-line git blame attribution (commit, author, date, content) for a "
                f"tracked file. Limited to {MAX_BLAME_LINES} lines per call; use start_line/"
                "end_line to page through large files."
            ),
            project_root=project_root,
            working_dir=working_dir,
            config=config,
            output_truncator=output_truncator,
            category="readonly",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Workspace-relative file to blame (must be tracked by git).",
                required=True,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="First line to blame (1-based).",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="Last line to blame (inclusive). Defaults to the end of the file.",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        raw_path = parameters.get("path")
        start_line = parameters.get("start_line", 1)
        end_line = parameters.get("end_line")

        if not raw_path or not isinstance(raw_path, str):
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="path must be a non-empty string.")
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"start_line must be an integer >= 1, got {start_line!r}.",
            )
        if end_line is not None and (isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < 1):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"end_line must be an integer >= 1, got {end_line!r}.",
            )
        if end_line is not None and end_line < start_line:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"end_line ({end_line}) must be >= start_line ({start_line}).",
            )

        repo_root, error = self._require_repo()
        if error:
            return error

        repo_rel, error = self._repo_relative(repo_root, raw_path)
        if error:
            return error

        resolved = self._resolve_path(raw_path)
        if not resolved.is_file():
            return ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"File not found: {raw_path}",
            )
        if is_binary_file(resolved):
            return ToolResponse.error(
                code=ToolErrorCode.BINARY_FILE,
                message=f"Cannot blame a binary file: {raw_path}",
            )

        try:
            rc, _, stderr = self._git(["ls-files", "--error-unmatch", "--", self._pathspec(repo_rel)], cwd=repo_root)
        except _GitFailure as exc:
            return self._failure_response(exc)
        if rc != 0:
            return ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"File is not tracked by git: {repo_rel}. git blame only works on tracked files.",
            )

        total_lines = 0
        with open(resolved, "rb") as handle:
            for _ in handle:
                total_lines += 1
        if total_lines == 0:
            data = {"path": repo_rel, "start_line": start_line, "end_line": 0, "total_lines": 0, "lines": []}
            return ToolResponse.success(text=f"{repo_rel}: file is empty.", data=data)
        if start_line > total_lines:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"start_line {start_line} is beyond the end of the file ({total_lines} lines).",
            )

        effective_end = min(end_line, total_lines) if end_line is not None else total_lines
        if effective_end - start_line + 1 > MAX_BLAME_LINES:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"Requested {effective_end - start_line + 1} lines exceeds the "
                    f"{MAX_BLAME_LINES}-line limit. Narrow the range with start_line/end_line."
                ),
            )

        try:
            rc, stdout, stderr = self._git(
                ["blame", "--line-porcelain", "-L", f"{start_line},{effective_end}", "--", self._pathspec(repo_rel)],
                cwd=repo_root,
            )
        except _GitFailure as exc:
            return self._failure_response(exc)
        if rc != 0:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"git blame failed (exit {rc}): {stderr.strip() or '[no output]'}",
            )

        entries = parse_blame(stdout)
        data = {
            "path": repo_rel,
            "start_line": start_line,
            "end_line": effective_end,
            "total_lines": total_lines,
            "lines": entries,
        }

        lines = [f"Blame {repo_rel} (lines {start_line}-{effective_end} of {total_lines}):"]
        for item in entries[:100]:
            content = item["content"]
            if len(content) > 120:
                content = content[:117] + "..."
            lines.append(f"  {item['line']:>5} {item['short_hash']} {item['author']:<16} {item['date'][:10]}  {content}")
        if len(entries) > 100:
            lines.append(f"  ... and {len(entries) - 100} more lines (see structured data)")
        return ToolResponse.success(text="\n".join(lines), data=data)


# ---------------------------------------------------------------------------
# GitCommit
# ---------------------------------------------------------------------------


class GitCommitTool(_GitBaseTool):
    """Guarded ``git commit`` wrapper.

    Deliberately offers no push / force / reset capabilities. ``--amend`` and
    ``--no-verify`` require explicit opt-in flags.
    """

    def __init__(
        self,
        name: str = "GitCommit",
        project_root: str = ".",
        working_dir: Optional[str] = None,
        config: Any = None,
        output_truncator: Optional[ObservationTruncator] = None,
    ):
        super().__init__(
            name=name,
            description=(
                "Create a git commit. Optionally stages the given workspace-relative paths "
                "first, validates the message, and can deterministically generate a "
                "conventional-commit style message from the staged changes "
                "(auto_message=true). Only commit when the user explicitly asks. "
                "amend/no_verify require explicit opt-in. Never pushes."
            ),
            project_root=project_root,
            working_dir=working_dir,
            config=config,
            output_truncator=output_truncator,
            category="write",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                type="string",
                description=(
                    f"Commit message (max {MAX_COMMIT_MESSAGE_CHARS} chars; control characters "
                    "are stripped). Required unless auto_message=true."
                ),
                required=False,
            ),
            ToolParameter(
                name="paths",
                type="array",
                description=(
                    "Workspace-relative files/directories to stage (`git add -A --`) before "
                    "committing. All paths are validated before any staging happens."
                ),
                required=False,
            ),
            ToolParameter(
                name="auto_message",
                type="boolean",
                description="Generate the commit message deterministically from the staged changes. Ignored when `message` is provided.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="amend",
                type="boolean",
                description="Amend the HEAD commit. Only use when explicitly requested and the commit has NOT been pushed.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="no_verify",
                type="boolean",
                description="Skip commit hooks (--no-verify). Only use when explicitly requested.",
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        message = parameters.get("message")
        paths = parameters.get("paths") or []
        auto_message = parameters.get("auto_message", False)
        amend = parameters.get("amend", False)
        no_verify = parameters.get("no_verify", False)

        if message is not None and not isinstance(message, str):
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="message must be a string when provided.")
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or any(not isinstance(p, str) or not p.strip() for p in paths):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="paths must be a list of non-empty workspace-relative path strings.",
            )
        for label, value in (("auto_message", auto_message), ("amend", amend), ("no_verify", no_verify)):
            if not isinstance(value, bool):
                return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message=f"{label} must be a boolean.")

        repo_root, error = self._require_repo()
        if error:
            return error

        # Validate every path BEFORE any side effect (no partial staging).
        repo_rel_paths: List[str] = []
        for raw in paths:
            repo_rel, error = self._repo_relative(repo_root, raw.strip())
            if error:
                return error
            repo_rel_paths.append(repo_rel)

        cleaned_message = sanitize_commit_message(message) if message is not None else ""
        if len(cleaned_message) > MAX_COMMIT_MESSAGE_CHARS:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"message exceeds {MAX_COMMIT_MESSAGE_CHARS} characters ({len(cleaned_message)}).",
            )
        if not cleaned_message and not auto_message:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="message must be a non-empty string (or pass auto_message=true).",
            )

        try:
            if amend and not _ref_exists("HEAD", repo_root, self.project_root):
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message="Cannot amend: the repository has no commits yet.",
                )

            if repo_rel_paths:
                add_args = ["add", "-A", "--"] + [self._pathspec(p) for p in repo_rel_paths]
                rc, stdout, stderr = self._git(add_args, cwd=repo_root)
                if rc != 0:
                    return ToolResponse.error(
                        code=ToolErrorCode.EXECUTION_ERROR,
                        message=f"git add failed (exit {rc}): {stderr.strip() or stdout.strip() or '[no output]'}",
                    )

            rc, _, stderr = self._git(["diff", "--cached", "--quiet"], cwd=repo_root)
            if rc == 0 and not amend:
                return ToolResponse.error(
                    code=ToolErrorCode.CONFLICT,
                    message=(
                        "Nothing staged to commit. Stage changes first by passing `paths` "
                        "to this tool or by staging them yourself."
                    ),
                )
            if rc not in (0, 1):
                return ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"Failed to inspect the staged changes: {stderr.strip() or '[no output]'}",
                )

            message_source = "user"
            final_message = cleaned_message
            if not final_message:
                staged_files = self._diff_files(repo_root, ["diff", "--no-color", "-M", "--cached"])
                final_message = generate_commit_message(staged_files)
                message_source = "auto"

            commit_args = ["commit", "-m", final_message]
            if amend:
                commit_args.append("--amend")
            if no_verify:
                commit_args.append("--no-verify")
            rc, stdout, stderr = self._git(commit_args, cwd=repo_root)
        except _GitFailure as exc:
            return self._failure_response(exc)

        if rc != 0:
            combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=(
                    f"git commit failed (exit {rc}). Do NOT retry with --amend; fix the "
                    f"problem and create a NEW commit.\n{combined or '[no output]'}"
                ),
            )

        commit_info = self._head_commit(repo_root)
        try:
            files_committed = self._diff_files(repo_root, ["diff-tree", "--root", "--no-commit-id", "-M", "-r", "HEAD"])
        except _GitFailure as exc:
            return self._failure_response(exc)
        status_info = self._status_after_commit(repo_root)

        warnings: List[str] = []
        subject = final_message.splitlines()[0] if final_message else ""
        if len(subject) > COMMIT_SUBJECT_SOFT_LIMIT:
            warnings.append(
                f"subject line is {len(subject)} chars (recommended <= {COMMIT_SUBJECT_SOFT_LIMIT})."
            )
        if amend:
            branch = status_info.get("branch", {})
            if branch.get("upstream") and branch.get("ahead", 0) == 0:
                warnings.append(
                    "amend may have rewritten an already-pushed commit (upstream exists and "
                    "branch is not ahead). Never force-push to shared branches."
                )
        if no_verify:
            warnings.append("commit hooks were skipped (--no-verify).")

        data = {
            "commit": commit_info,
            "message": final_message,
            "message_source": message_source,
            "amend": amend,
            "no_verify": no_verify,
            "staged_paths": repo_rel_paths,
            "files_committed": files_committed,
            "remaining_status": status_info["summary"],
            "warnings": warnings,
        }

        lines = [f"Commit {commit_info.get('short_hash', '?')}: {commit_info.get('subject', final_message)}"]
        lines.append(f"Message source: {message_source}")
        if files_committed:
            total_add = sum(item["additions"] for item in files_committed)
            total_del = sum(item["deletions"] for item in files_committed)
            lines.append(f"Files: {len(files_committed)} (+{total_add} -{total_del})")
            for item in files_committed[:_TEXT_PREVIEW_ENTRIES]:
                lines.append(f"  {item['status']} {item['path']} (+{item['additions']} -{item['deletions']})")
            if len(files_committed) > _TEXT_PREVIEW_ENTRIES:
                lines.append(f"  ... and {len(files_committed) - _TEXT_PREVIEW_ENTRIES} more files")
        summary = status_info["summary"]
        if summary:
            lines.append(
                "After commit: "
                + ("working tree clean" if summary["is_clean"] else
                   f"staged={summary['staged']} unstaged={summary['unstaged']} "
                   f"untracked={summary['untracked']} conflicted={summary['conflicted']}")
            )
        for warning in warnings:
            lines.append(f"WARNING: {warning}")
        return ToolResponse.success(text="\n".join(lines), data=data)

    def _head_commit(self, repo_root: Path) -> Dict[str, Any]:
        try:
            rc, stdout, _ = self._git(["log", "-1", f"--pretty=format:{_HEAD_FORMAT}"], cwd=repo_root)
        except _GitFailure:
            return {}
        if rc != 0:
            return {}
        fields = stdout.strip().split(_LOG_FIELD_SEP)
        if len(fields) < 3:
            return {}
        return {"hash": fields[0], "short_hash": fields[1], "subject": fields[2].strip()}

    def _status_after_commit(self, repo_root: Path) -> Dict[str, Any]:
        try:
            rc, stdout, _ = self._git(
                ["-c", "core.quotePath=false", "status", "--porcelain=v2", "--branch"],
                cwd=repo_root,
            )
        except _GitFailure:
            return {"summary": {}, "branch": {}}
        if rc != 0:
            return {"summary": {}, "branch": {}}
        parsed = parse_status_porcelain_v2(stdout)
        return {
            "branch": parsed["branch"],
            "summary": {
                "is_clean": parsed["is_clean"],
                "staged": len(parsed["staged"]),
                "unstaged": len(parsed["unstaged"]),
                "untracked": len(parsed["untracked"]),
                "conflicted": len(parsed["conflicted"]),
            },
        }


__all__ = [
    "GitStatusTool",
    "GitDiffTool",
    "GitLogTool",
    "GitBlameTool",
    "GitCommitTool",
    "parse_status_porcelain_v2",
    "parse_numstat_z",
    "parse_name_status_z",
    "parse_log",
    "parse_blame",
    "merge_diff_files",
    "generate_commit_message",
    "sanitize_commit_message",
]
