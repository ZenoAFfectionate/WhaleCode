"""Tests for the Git-native tools (code/tools/builtin/git_tools.py).

Layout:
    1. Pure-parser unit tests (no git process required)
    2. Tool integration tests against real temporary repositories
    3. Framework integration (registry / filters / env switch / circuit breaker)

All tests run offline; they only require a ``git`` binary on PATH.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolStatus
from hello_agents.tools.builtin.git_tools import (
    GitBlameTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    generate_commit_message,
    merge_diff_files,
    parse_blame,
    parse_log,
    parse_name_status_z,
    parse_numstat_z,
    parse_status_porcelain_v2,
    sanitize_commit_message,
    _unquote_path,
)

git_available = pytest.mark.skipif(shutil.which("git") is None, reason="git binary not found")
pytestmark = git_available


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def _commit_count(repo: Path) -> int:
    result = _git(repo, "rev-list", "--count", "HEAD")
    return int(result.stdout.strip())


def _head_subject(repo: Path) -> str:
    return _git(repo, "log", "-1", "--pretty=format:%s").stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    """Repo with two known commits on branch `main`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Tester")
    _git(repo, "config", "user.email", "tester@example.com")

    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    src = repo / "src"
    src.mkdir()
    (src / "a.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "first commit: add readme and a.py")

    (src / "a.py").write_text("line1-mod\nline2\nline3\nline4\n", encoding="utf-8")
    (src / "b.py").write_text("b1\nb2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "second commit: modify a.py and add b.py")
    return repo


@pytest.fixture
def empty_repo(tmp_path):
    """Initialized repo without any commit."""
    repo = tmp_path / "empty"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Tester")
    _git(repo, "config", "user.email", "tester@example.com")
    return repo


def _make_conflict(repo: Path) -> None:
    """Create a merge conflict in src/a.py between main and side."""
    _git(repo, "checkout", "-b", "side")
    (repo / "src" / "a.py").write_text("side change\nline2\nline3\nline4\n", encoding="utf-8")
    _git(repo, "commit", "-am", "side change")
    _git(repo, "checkout", "main")
    (repo / "src" / "a.py").write_text("main change\nline2\nline3\nline4\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main change")
    _git(repo, "merge", "side", check=False)


# ===========================================================================
# 1. Pure-parser unit tests
# ===========================================================================


class TestUnquotePath:
    def test_plain_path_unchanged(self):
        assert _unquote_path("src/a.py") == "src/a.py"

    def test_octal_quoted_utf8_path(self):
        assert _unquote_path('"\\344\\270\\255\\346\\226\\207.py"') == "中文.py"

    def test_quoted_with_escaped_quote(self):
        assert _unquote_path('"a\\"b.py"') == 'a"b.py'


class TestParseStatusPorcelainV2:
    def test_branch_headers_and_clean(self):
        out = (
            "# branch.oid 57a1c540391b816b2b9896be37f1edf0a5d5ce46\n"
            "# branch.head main\n"
            "# branch.upstream origin/main\n"
            "# branch.ab +2 -1\n"
            "# stash 3\n"
        )
        parsed = parse_status_porcelain_v2(out)
        assert parsed["branch"]["name"] == "main"
        assert parsed["branch"]["oid"].startswith("57a1c54")
        assert parsed["branch"]["upstream"] == "origin/main"
        assert parsed["branch"]["ahead"] == 2
        assert parsed["branch"]["behind"] == 1
        assert parsed["stash_count"] == 3
        assert parsed["is_clean"] is True

    def test_initial_repo(self):
        out = (
            "# branch.oid (initial)\n"
            "# branch.head main\n"
            "1 A. N... 000000 100644 100644 "
            "0000000000000000000000000000000000000000 "
            "587be6b4c3f93f93c489c0111bba5596147a26cb f.txt\n"
        )
        parsed = parse_status_porcelain_v2(out)
        assert parsed["branch"]["initial"] is True
        assert parsed["branch"]["oid"] == ""
        assert parsed["staged"][0]["path"] == "f.txt"
        assert parsed["is_clean"] is False

    def test_detached_head(self):
        out = (
            "# branch.oid 83db48f84ec878fbfb30b46d16630e944e34f205\n"
            "# branch.head (detached)\n"
        )
        parsed = parse_status_porcelain_v2(out)
        assert parsed["branch"]["detached"] is True
        assert parsed["branch"]["name"] == ""

    def test_staged_unstaged_rename_untracked_conflict(self):
        out = (
            "# branch.oid 57a1c540391b816b2b9896be37f1edf0a5d5ce46\n"
            "# branch.head main\n"
            "2 R. N... 100644 100644 100644 "
            "f8e8a05adfd8322a67d53cf79eae360562e6f03b "
            "f8e8a05adfd8322a67d53cf79eae360562e6f03b R100 src/d.py\tsrc/c.py\n"
            "1 .M N... 100644 100644 100644 "
            "83db48f84ec878fbfb30b46d16630e944e34f205 "
            "83db48f84ec878fbfb30b46d16630e944e34f205 src/e.py\n"
            "? \"\\344\\270\\255\\346\\226\\207.py\"\n"
            "u UU N... 100644 100644 100644 100644 h1 h2 h3 conflict.py\n"
        )
        parsed = parse_status_porcelain_v2(out)
        assert parsed["staged"][0]["path"] == "src/d.py"
        assert parsed["staged"][0]["orig_path"] == "src/c.py"
        assert parsed["unstaged"][0]["path"] == "src/e.py"
        assert parsed["untracked"] == ["中文.py"]
        assert parsed["conflicted"] == [{"path": "conflict.py", "xy": "UU"}]
        assert parsed["is_clean"] is False

    def test_path_with_spaces(self):
        out = (
            "1 .M N... 100644 100644 100644 "
            "83db48f84ec878fbfb30b46d16630e944e34f205 "
            "83db48f84ec878fbfb30b46d16630e944e34f205 src/my file.py\n"
        )
        parsed = parse_status_porcelain_v2(out)
        assert parsed["unstaged"][0]["path"] == "src/my file.py"


class TestParseNumstatZ:
    def test_normal_binary_rename(self):
        out = "\x00".join(["2\t1\tsrc/b.py", "-\t-\tbin.png", "0\t0\t", "old.py", "new.py", ""])
        files = parse_numstat_z(out)
        assert files[0] == {"path": "src/b.py", "additions": 2, "deletions": 1, "binary": False, "orig_path": None}
        assert files[1]["binary"] is True
        assert files[1]["path"] == "bin.png"
        assert files[2]["orig_path"] == "old.py"
        assert files[2]["path"] == "new.py"

    def test_empty(self):
        assert parse_numstat_z("") == []


class TestParseNameStatusZ:
    def test_mixed_entries(self):
        out = "M\0a.py\0R100\0old.py\0new.py\0A\0c.py\0D\0gone.py\0"
        parsed = parse_name_status_z(out)
        assert parsed["a.py"]["status"] == "M"
        assert parsed["new.py"]["status"] == "R"
        assert parsed["new.py"]["orig_path"] == "old.py"
        assert parsed["c.py"]["status"] == "A"
        assert parsed["gone.py"]["status"] == "D"


class TestMergeDiffFiles:
    def test_joins_on_path(self):
        numstat = [
            {"path": "a.py", "additions": 1, "deletions": 2, "binary": False, "orig_path": None},
            {"path": "new.py", "additions": 0, "deletions": 0, "binary": False, "orig_path": "old.py"},
        ]
        name_status = {
            "a.py": {"status": "M", "score": None, "orig_path": None},
            "new.py": {"status": "R", "score": "100", "orig_path": "old.py"},
        }
        merged = merge_diff_files(numstat, name_status)
        assert merged[0]["status"] == "M"
        assert merged[1]["status"] == "R"
        assert merged[1]["orig_path"] == "old.py"


class TestParseLog:
    def test_fields_and_tricky_subject(self):
        out = (
            "a" * 40 + "\x1faaaaaaa\x1fTester\x1ft@e.com\x1f2026-07-31T10:00:00+08:00"
            "\x1ffix: handle, commas: and 中文\x1e\n"
            + "b" * 40 + "\x1fbbbbbbb\x1fOther\x1fo@e.com\x1f2026-07-30T09:00:00+08:00"
            "\x1fsecond subject\x1e"
        )
        commits = parse_log(out)
        assert len(commits) == 2
        assert commits[0]["hash"] == "a" * 40
        assert commits[0]["short_hash"] == "aaaaaaa"
        assert commits[0]["subject"] == "fix: handle, commas: and 中文"
        assert commits[1]["author"] == "Other"

    def test_empty(self):
        assert parse_log("") == []


class TestParseBlame:
    SAMPLE = (
        "5fac2bf78cfd2ee7e2db4b4564f20329a7f7bc04 1 1 1\n"
        "author Alice\n"
        "author-mail <alice@example.com>\n"
        "author-time 1785483827\n"
        "author-tz +0800\n"
        "committer Alice\n"
        "committer-mail <alice@example.com>\n"
        "committer-time 1785483827\n"
        "committer-tz +0800\n"
        "summary changes\n"
        "previous b713e12db9c6d19a05b1e5556667f40b4414619d src/b.py\n"
        "filename src/b.py\n"
        "\tline1-mod\n"
        "08fda5ace1b128d05561e9898a52857a2eeb2d4b 2 2 1\n"
        "author Bob\n"
        "author-mail <bob@example.com>\n"
        "author-time 1785483794\n"
        "author-tz +0000\n"
        "committer Bob\n"
        "committer-mail <bob@example.com>\n"
        "committer-time 1785483794\n"
        "committer-tz +0000\n"
        "summary first commit\n"
        "boundary\n"
        "filename src/a.py\n"
        "\tline2\n"
    )

    def test_per_line_attribution(self):
        lines = parse_blame(self.SAMPLE)
        assert len(lines) == 2
        first, second = lines
        assert first["line"] == 1
        assert first["author"] == "Alice"
        assert first["email"] == "alice@example.com"
        assert first["short_hash"] == "5fac2bf"
        assert first["summary"] == "changes"
        assert first["content"] == "line1-mod"
        assert "+08:00" in first["date"]
        assert second["line"] == 2
        assert second["author"] == "Bob"
        assert second["content"] == "line2"

    def test_group_with_multiple_content_lines(self):
        out = (
            "a" * 40 + " 5 10 2\n"
            "author Carol\n"
            "author-mail <c@e.com>\n"
            "author-time 1785483827\n"
            "author-tz +0000\n"
            "summary grouped\n"
            "filename f.py\n"
            "\tcontent-a\n"
            "\tcontent-b\n"
        )
        lines = parse_blame(out)
        assert [line["line"] for line in lines] == [10, 11]
        assert [line["content"] for line in lines] == ["content-a", "content-b"]
        assert all(line["author"] == "Carol" for line in lines)


class TestSanitizeCommitMessage:
    def test_strips_control_chars_keeps_newlines(self):
        assert sanitize_commit_message("hello\x00\x07 world\n\nbody\x1f") == "hello world\n\nbody"

    def test_strips_surrounding_whitespace(self):
        assert sanitize_commit_message("  padded \n") == "padded"


class TestGenerateCommitMessage:
    def _file(self, path, status="M", orig_path=None):
        return {"path": path, "status": status, "additions": 1, "deletions": 1, "binary": False, "orig_path": orig_path}

    def test_empty(self):
        assert generate_commit_message([]) == "chore: update repository"

    def test_single_added_file(self):
        assert generate_commit_message([self._file("code/git_tools.py", "A")]) == "feat: add git_tools.py"

    def test_single_deleted_file(self):
        assert generate_commit_message([self._file("old.py", "D")]) == "chore: remove old.py"

    def test_single_test_file(self):
        assert generate_commit_message([self._file("tests/test_x.py")]) == "test: update test_x.py"

    def test_single_doc_file(self):
        assert generate_commit_message([self._file("README.md")]) == "docs: update README.md"

    def test_single_plain_file(self):
        assert generate_commit_message([self._file("src/main.py")]) == "update main.py"

    def test_multiple_same_dir(self):
        files = [self._file("code/tools/a.py"), self._file("code/tools/b.py")]
        assert generate_commit_message(files) == "update 2 files in code/tools"

    def test_multiple_across_repo(self):
        files = [self._file("a.py"), self._file("code/tools/b.py")]
        assert generate_commit_message(files) == "update 2 files across the repository"

    def test_multiple_added(self):
        files = [self._file("src/a.py", "A"), self._file("src/b.py", "A")]
        assert generate_commit_message(files) == "feat: add 2 files in src"

    def test_multiple_deleted(self):
        files = [self._file("src/a.py", "D"), self._file("src/b.py", "D")]
        assert generate_commit_message(files) == "chore: remove 2 files in src"

    def test_all_tests(self):
        files = [self._file("tests/test_a.py"), self._file("pkg/test_b.py")]
        assert generate_commit_message(files) == "test: update 2 files across the repository"

    def test_all_docs(self):
        files = [self._file("docs/a.md"), self._file("docs/b.md")]
        assert generate_commit_message(files) == "docs: update 2 files in docs"


# ===========================================================================
# 2. Tool integration tests (real temporary repositories)
# ===========================================================================


class TestGitStatusTool:
    def test_clean_repo(self, git_repo):
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["is_clean"] is True
        assert resp.data["branch"]["name"] == "main"
        assert "Working tree clean." in resp.text

    def test_untracked_and_modified(self, git_repo):
        (git_repo / "new.py").write_text("new\n", encoding="utf-8")
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert "new.py" in resp.data["untracked"]
        assert [e["path"] for e in resp.data["unstaged"]] == ["src/a.py"]
        assert resp.data["is_clean"] is False

    def test_staged_after_add(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        _git(git_repo, "add", "src/a.py")
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert [e["path"] for e in resp.data["staged"]] == ["src/a.py"]
        assert resp.data["unstaged"] == []

    def test_rename_detection(self, git_repo):
        _git(git_repo, "mv", "src/b.py", "src/renamed.py")
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        staged = resp.data["staged"]
        assert staged[0]["path"] == "src/renamed.py"
        assert staged[0]["orig_path"] == "src/b.py"

    def test_merge_conflict(self, git_repo):
        _make_conflict(git_repo)
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert resp.data["conflicted"], "expected conflicted entries"
        assert resp.data["conflicted"][0]["path"] == "src/a.py"
        assert resp.data["is_clean"] is False

    def test_chinese_filename(self, git_repo):
        (git_repo / "中文文件.py").write_text("x\n", encoding="utf-8")
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert "中文文件.py" in resp.data["untracked"]

    def test_path_scope_filter(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        (git_repo / "README.md").write_text("changed\n", encoding="utf-8")
        resp = GitStatusTool(project_root=str(git_repo)).run({"path": "src"})
        assert resp.data["scope"] == "src"
        paths = [e["path"] for e in resp.data["unstaged"]]
        assert paths == ["src/a.py"]

    def test_initial_repo(self, empty_repo):
        (empty_repo / "f.txt").write_text("x\n", encoding="utf-8")
        _git(empty_repo, "add", "f.txt")
        resp = GitStatusTool(project_root=str(empty_repo)).run({})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["branch"]["initial"] is True
        assert resp.data["staged"][0]["path"] == "f.txt"

    def test_not_a_repo(self, tmp_path):
        resp = GitStatusTool(project_root=str(tmp_path)).run({})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.NOT_FOUND

    def test_path_escape(self, git_repo):
        resp = GitStatusTool(project_root=str(git_repo)).run({"path": "../outside"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.ACCESS_DENIED

    def test_invalid_param_type(self, git_repo):
        resp = GitStatusTool(project_root=str(git_repo)).run({"path": 123})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM


class TestGitDiffTool:
    def test_worktree_changes(self, git_repo):
        (git_repo / "src" / "a.py").write_text("line1-mod\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        resp = GitDiffTool(project_root=str(git_repo)).run({})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["mode"] == "worktree"
        assert resp.data["total_files"] == 1
        entry = resp.data["files"][0]
        assert entry["path"] == "src/a.py"
        assert entry["status"] == "M"
        assert entry["additions"] == 1
        assert entry["deletions"] == 0
        assert "+line5" in resp.data["patch"]
        assert resp.data["patch_truncated"] is False

    def test_staged_mode(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        _git(git_repo, "add", "src/a.py")
        resp = GitDiffTool(project_root=str(git_repo)).run({"staged": True})
        assert resp.data["mode"] == "staged"
        assert resp.data["ref"] == "HEAD"
        assert resp.data["files"][0]["path"] == "src/a.py"

    def test_commit_mode(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({"commit": "HEAD~1"})
        assert resp.data["mode"] == "commit"
        paths = {f["path"] for f in resp.data["files"]}
        assert paths == {"src/a.py", "src/b.py"}
        statuses = {f["path"]: f["status"] for f in resp.data["files"]}
        assert statuses["src/b.py"] == "A"

    def test_no_changes(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({})
        assert resp.data["files"] == []
        assert "No changes." in resp.text

    def test_path_filter(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        (git_repo / "README.md").write_text("changed\n", encoding="utf-8")
        resp = GitDiffTool(project_root=str(git_repo)).run({"path": "README.md"})
        assert resp.data["total_files"] == 1
        assert resp.data["files"][0]["path"] == "README.md"

    def test_stat_only(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        resp = GitDiffTool(project_root=str(git_repo)).run({"stat_only": True})
        assert resp.data["patch"] == ""
        assert "--- patch ---" not in resp.text

    def test_binary_file(self, git_repo):
        (git_repo / "blob.bin").write_bytes(b"\x00\x01\x02\x00" * 64)
        _git(git_repo, "add", "blob.bin")
        resp = GitDiffTool(project_root=str(git_repo)).run({"staged": True})
        entry = next(f for f in resp.data["files"] if f["path"] == "blob.bin")
        assert entry["binary"] is True
        assert entry["status"] == "A"

    def test_rename_detection(self, git_repo):
        _git(git_repo, "mv", "src/b.py", "src/renamed.py")
        resp = GitDiffTool(project_root=str(git_repo)).run({"staged": True})
        entry = resp.data["files"][0]
        assert entry["status"] == "R"
        assert entry["orig_path"] == "src/b.py"

    def test_invalid_ref(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({"commit": "no-such-ref"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_staged_and_commit_mutually_exclusive(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({"staged": True, "commit": "HEAD"})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_context_lines_validation(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({"context_lines": 99})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM
        resp = GitDiffTool(project_root=str(git_repo)).run({"context_lines": True})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_unborn_repo_staged_diff(self, empty_repo):
        (empty_repo / "f.txt").write_text("x\n", encoding="utf-8")
        _git(empty_repo, "add", "f.txt")
        resp = GitDiffTool(project_root=str(empty_repo)).run({"staged": True})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["files"][0]["status"] == "A"

    def test_patch_truncation(self, git_repo, tmp_path):
        from hello_agents.context.truncator import ObservationTruncator

        big = "".join(f"line-{i}\n" for i in range(200))
        (git_repo / "src" / "a.py").write_text(big, encoding="utf-8")
        truncator = ObservationTruncator(
            max_lines=5, max_bytes=400, output_dir=str(tmp_path / "tool-output")
        )
        tool = GitDiffTool(project_root=str(git_repo), output_truncator=truncator)
        resp = tool.run({})
        assert resp.status == ToolStatus.PARTIAL
        assert resp.data["patch_truncated"] is True
        full_path = Path(resp.data["full_output_path"])
        assert full_path.exists()
        assert "+line-199" in full_path.read_text(encoding="utf-8")

    def test_path_escape(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({"path": "../x"})
        assert resp.error_info["code"] == ToolErrorCode.ACCESS_DENIED

    def test_path_outside_repo(self, git_repo, tmp_path):
        outside = tmp_path / "outside.py"
        outside.write_text("x\n", encoding="utf-8")
        tool = GitDiffTool(project_root=str(tmp_path), working_dir=str(git_repo))
        resp = tool.run({"path": str(outside)})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM


class TestGitLogTool:
    def test_default_returns_all_fields(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["count"] == 2
        first = resp.data["commits"][0]
        assert first["subject"] == "second commit: modify a.py and add b.py"
        assert first["author"] == "Tester"
        assert first["email"] == "tester@example.com"
        assert len(first["hash"]) == 40
        assert first["date"]

    def test_count_and_has_more(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"count": 1})
        assert resp.data["count"] == 1
        assert resp.data["has_more"] is True
        resp = GitLogTool(project_root=str(git_repo)).run({"count": 2})
        assert resp.data["has_more"] is False

    def test_path_filter(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"path": "src/b.py"})
        assert resp.data["count"] == 1
        assert "second commit" in resp.data["commits"][0]["subject"]

    def test_grep_filter(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"grep": "second"})
        assert resp.data["count"] == 1
        resp = GitLogTool(project_root=str(git_repo)).run({"grep": "no-match-xyz"})
        assert resp.data["count"] == 0
        assert "No matching commits." in resp.text

    def test_author_filter(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"author": "Tester"})
        assert resp.data["count"] == 2
        resp = GitLogTool(project_root=str(git_repo)).run({"author": "Nobody"})
        assert resp.data["count"] == 0

    def test_ref_selection(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"ref": "HEAD~1"})
        assert resp.data["count"] == 1
        assert "first commit" in resp.data["commits"][0]["subject"]

    def test_invalid_ref(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"ref": "no-such-ref"})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_count_validation(self, git_repo):
        assert GitLogTool(project_root=str(git_repo)).run({"count": 0}).error_info["code"] == ToolErrorCode.INVALID_PARAM
        assert GitLogTool(project_root=str(git_repo)).run({"count": 101}).error_info["code"] == ToolErrorCode.INVALID_PARAM
        assert GitLogTool(project_root=str(git_repo)).run({"count": "5"}).error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_empty_repo(self, empty_repo):
        resp = GitLogTool(project_root=str(empty_repo)).run({})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["commits"] == []
        assert "No commits yet" in resp.text

    def test_not_a_repo(self, tmp_path):
        resp = GitLogTool(project_root=str(tmp_path)).run({})
        assert resp.error_info["code"] == ToolErrorCode.NOT_FOUND


class TestGitBlameTool:
    def test_line_attribution(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "src/a.py"})
        assert resp.status == ToolStatus.SUCCESS
        entries = {line["line"]: line for line in resp.data["lines"]}
        assert len(entries) == 4
        # line 1 was rewritten by the second commit; line 2 dates back to the first.
        assert entries[1]["summary"].startswith("second commit")
        assert entries[2]["summary"].startswith("first commit")
        assert entries[1]["author"] == "Tester"
        assert entries[1]["email"] == "tester@example.com"
        assert entries[1]["content"] == "line1-mod"

    def test_line_range(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run(
            {"path": "src/a.py", "start_line": 2, "end_line": 3}
        )
        assert [line["line"] for line in resp.data["lines"]] == [2, 3]
        assert resp.data["end_line"] == 3
        assert resp.data["total_lines"] == 4

    def test_end_line_clamped_to_file_length(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run(
            {"path": "src/b.py", "start_line": 1, "end_line": 500}
        )
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["end_line"] == 2

    def test_start_beyond_file(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run(
            {"path": "src/b.py", "start_line": 99}
        )
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_range_exceeds_limit(self, git_repo):
        big = "".join(f"line {i}\n" for i in range(600))
        (git_repo / "big.py").write_text(big, encoding="utf-8")
        _git(git_repo, "add", "big.py")
        _git(git_repo, "commit", "-m", "add big file")
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "big.py"})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM
        resp = GitBlameTool(project_root=str(git_repo)).run(
            {"path": "big.py", "start_line": 1, "end_line": 500}
        )
        assert resp.status == ToolStatus.SUCCESS
        assert len(resp.data["lines"]) == 500

    def test_end_before_start(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run(
            {"path": "src/a.py", "start_line": 3, "end_line": 1}
        )
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_untracked_file(self, git_repo):
        (git_repo / "untracked.py").write_text("x\n", encoding="utf-8")
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "untracked.py"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.NOT_FOUND
        assert "not tracked" in resp.error_info["message"]

    def test_missing_file(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "does/not_exist.py"})
        assert resp.error_info["code"] == ToolErrorCode.NOT_FOUND

    def test_missing_path_param(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run({})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_path_escape(self, git_repo):
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "../escape.py"})
        assert resp.error_info["code"] == ToolErrorCode.ACCESS_DENIED

    def test_empty_file(self, git_repo):
        (git_repo / "empty.py").write_text("", encoding="utf-8")
        _git(git_repo, "add", "empty.py")
        _git(git_repo, "commit", "-m", "add empty file")
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "empty.py"})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["lines"] == []


class TestGitCommitTool:
    def test_commit_with_paths(self, git_repo):
        (git_repo / "src" / "c.py").write_text("c1\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "add c.py", "paths": ["src/c.py"]}
        )
        assert resp.status == ToolStatus.SUCCESS, resp.text
        assert resp.data["message_source"] == "user"
        assert _head_subject(git_repo) == "add c.py"
        assert _commit_count(git_repo) == 3
        committed_paths = {f["path"] for f in resp.data["files_committed"]}
        assert "src/c.py" in committed_paths
        assert resp.data["remaining_status"]["is_clean"] is True

    def test_commit_requires_message(self, git_repo):
        resp = GitCommitTool(project_root=str(git_repo)).run({})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_empty_message_rejected(self, git_repo):
        resp = GitCommitTool(project_root=str(git_repo)).run({"message": "   "})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_overlong_message_rejected(self, git_repo):
        resp = GitCommitTool(project_root=str(git_repo)).run({"message": "x" * 2001})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_control_chars_sanitized(self, git_repo):
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "ok\x00message\x07", "paths": ["f.py"]}
        )
        assert resp.status == ToolStatus.SUCCESS
        assert _head_subject(git_repo) == "okmessage"

    def test_nothing_staged_conflict(self, git_repo):
        resp = GitCommitTool(project_root=str(git_repo)).run({"message": "noop"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.CONFLICT
        assert _commit_count(git_repo) == 2

    def test_auto_message_feat(self, git_repo):
        (git_repo / "src" / "new_feature.py").write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"auto_message": True, "paths": ["src/new_feature.py"]}
        )
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["message_source"] == "auto"
        assert _head_subject(git_repo) == "feat: add new_feature.py"

    def test_auto_message_test_prefix(self, git_repo):
        tests_dir = git_repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_demo.py").write_text("x\n", encoding="utf-8")
        _git(git_repo, "add", "tests/test_demo.py")
        _git(git_repo, "commit", "-m", "add test file")
        (tests_dir / "test_demo.py").write_text("y\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"auto_message": True, "paths": ["tests/test_demo.py"]}
        )
        assert resp.data["message"].startswith("test:")

    def test_explicit_message_wins_over_auto(self, git_repo):
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "explicit message", "auto_message": True, "paths": ["f.py"]}
        )
        assert resp.data["message_source"] == "user"
        assert _head_subject(git_repo) == "explicit message"

    def test_amend_rewrites_head(self, git_repo):
        count_before = _commit_count(git_repo)
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "amended second commit", "amend": True}
        )
        assert resp.status == ToolStatus.SUCCESS
        assert _head_subject(git_repo) == "amended second commit"
        assert _commit_count(git_repo) == count_before
        assert resp.data["amend"] is True

    def test_default_does_not_amend(self, git_repo):
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        GitCommitTool(project_root=str(git_repo)).run({"message": "c3", "paths": ["f.py"]})
        assert _commit_count(git_repo) == 3

    def test_failing_hook_blocks_commit(self, git_repo):
        hook = git_repo / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\necho hook says no >&2\nexit 1\n", encoding="utf-8")
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run({"message": "c3", "paths": ["f.py"]})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.EXECUTION_ERROR
        assert "hook says no" in resp.error_info["message"]
        assert _commit_count(git_repo) == 2

        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "c3", "paths": ["f.py"], "no_verify": True}
        )
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["no_verify"] is True
        assert any("--no-verify" in w for w in resp.data["warnings"])
        assert _commit_count(git_repo) == 3

    def test_invalid_path_aborts_before_staging(self, git_repo):
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "c3", "paths": ["f.py", "../outside.py"]}
        )
        assert resp.error_info["code"] == ToolErrorCode.ACCESS_DENIED
        # nothing must have been staged
        status = GitStatusTool(project_root=str(git_repo)).run({})
        assert status.data["staged"] == []
        assert _commit_count(git_repo) == 2

    def test_paths_type_validation(self, git_repo):
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "c3", "paths": [123]}
        )
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_flag_type_validation(self, git_repo):
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "c3", "amend": "yes"}
        )
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_not_a_repo(self, tmp_path):
        resp = GitCommitTool(project_root=str(tmp_path)).run({"message": "c1"})
        assert resp.error_info["code"] == ToolErrorCode.NOT_FOUND

    def test_initial_commit(self, empty_repo):
        (empty_repo / "f.txt").write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(empty_repo)).run(
            {"message": "initial commit", "paths": ["f.txt"]}
        )
        assert resp.status == ToolStatus.SUCCESS, resp.text
        assert _commit_count(empty_repo) == 1
        assert resp.data["commit"]["subject"] == "initial commit"
        assert any(f["path"] == "f.txt" for f in resp.data["files_committed"])

    def test_long_subject_soft_warning(self, git_repo):
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        long_subject = "x" * 90
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": long_subject, "paths": ["f.py"]}
        )
        assert resp.status == ToolStatus.SUCCESS
        assert any("72" in w for w in resp.data["warnings"])


# ===========================================================================
# 3. Framework integration
# ===========================================================================


class TestRegistryIntegration:
    def test_execute_tool_full_pipeline(self, git_repo):
        from hello_agents.tools.registry import ToolRegistry

        registry = ToolRegistry(verbose=False)
        registry.register_tool(GitStatusTool(project_root=str(git_repo)))
        resp = registry.execute_tool("GitStatus", json.dumps({}))
        assert resp.status == ToolStatus.SUCCESS
        assert resp.context["tool_name"] == "GitStatus"
        assert "time_ms" in resp.stats

    def test_schema_validation_via_registry(self, git_repo):
        from hello_agents.tools.registry import ToolRegistry

        registry = ToolRegistry(verbose=False)
        registry.register_tool(GitBlameTool(project_root=str(git_repo)))
        resp = registry.execute_tool("GitBlame", json.dumps({}))
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_non_fault_errors_do_not_trip_circuit_breaker(self, tmp_path):
        from hello_agents.tools.registry import ToolRegistry

        registry = ToolRegistry(verbose=False)
        registry.register_tool(GitStatusTool(project_root=str(tmp_path)))
        for _ in range(5):
            resp = registry.execute_tool("GitStatus", json.dumps({}))
            assert resp.error_info["code"] == ToolErrorCode.NOT_FOUND
        assert registry.circuit_breaker.is_open("GitStatus") is False


class TestToolCategories:
    def test_categories(self, git_repo):
        root = str(git_repo)
        assert GitStatusTool(project_root=root).category == "readonly"
        assert GitDiffTool(project_root=root).category == "readonly"
        assert GitLogTool(project_root=root).category == "readonly"
        assert GitBlameTool(project_root=root).category == "readonly"
        assert GitCommitTool(project_root=root).category == "write"


class TestAgentRegistration:
    GIT_TOOL_NAMES = {"GitStatus", "GitDiff", "GitLog", "GitBlame", "GitCommit"}

    def test_git_tools_registered_by_default(self, git_repo, mock_llm):
        from hello_agents.agents.code_agent import CodeAgent

        agent = CodeAgent(
            "c",
            mock_llm,
            project_root=str(git_repo),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
        )
        assert self.GIT_TOOL_NAMES <= set(agent.tool_registry.list_tools())

    def test_git_tools_disabled_by_env(self, git_repo, mock_llm, monkeypatch):
        from hello_agents.agents.code_agent import CodeAgent

        monkeypatch.setenv("GIT_TOOLS_ENABLED", "0")
        agent = CodeAgent(
            "c",
            mock_llm,
            project_root=str(git_repo),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
        )
        assert self.GIT_TOOL_NAMES.isdisjoint(set(agent.tool_registry.list_tools()))

    def test_git_tools_follow_set_working_dir(self, git_repo, mock_llm):
        from hello_agents.agents.code_agent import CodeAgent

        agent = CodeAgent(
            "c",
            mock_llm,
            project_root=str(git_repo),
            register_default_tools=True,
            enable_task_tool=False,
            interactive=False,
        )
        sub = git_repo / "src"
        agent.set_working_dir(str(sub))
        for name in self.GIT_TOOL_NAMES:
            tool = agent.tool_registry.get_tool(name)
            assert tool is not None
            assert tool.working_dir == sub.resolve()


class TestEnableSwitch:
    def test_is_enabled_by_default_env_parsing(self, monkeypatch):
        monkeypatch.delenv("GIT_TOOLS_ENABLED", raising=False)
        assert GitStatusTool.is_enabled_by_default() is True
        monkeypatch.setenv("GIT_TOOLS_ENABLED", "0")
        assert GitStatusTool.is_enabled_by_default() is False
        monkeypatch.setenv("GIT_TOOLS_ENABLED", "false")
        assert GitStatusTool.is_enabled_by_default() is False
        monkeypatch.setenv("GIT_TOOLS_ENABLED", "1")
        assert GitStatusTool.is_enabled_by_default() is True

    def test_config_field(self, monkeypatch):
        from hello_agents.core.config import Config

        assert Config().git_tools_enabled is True
        monkeypatch.setenv("GIT_TOOLS_ENABLED", "0")
        assert Config.from_env().git_tools_enabled is False
        monkeypatch.setenv("GIT_TOOLS_ENABLED", "1")
        assert Config.from_env().git_tools_enabled is True


# ===========================================================================
# 4. Hardening & edge-case tests (audit-driven)
# ===========================================================================


class TestRefInjectionGuard:
    """Refs are positional argv entries — option-like values must be rejected."""

    def test_diff_commit_option_injection_blocked(self, git_repo, tmp_path):
        target = tmp_path / "pwned.txt"
        resp = GitDiffTool(project_root=str(git_repo)).run(
            {"commit": f"--output={target}"}
        )
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM
        assert not target.exists()

    def test_diff_commit_dash_value_blocked(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({"commit": "--cached"})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_diff_commit_whitespace_blocked(self, git_repo):
        resp = GitDiffTool(project_root=str(git_repo)).run({"commit": "HEAD main"})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_log_ref_option_injection_blocked(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"ref": "--oneline"})
        assert resp.error_info["code"] == ToolErrorCode.INVALID_PARAM

    def test_legit_refs_still_accepted(self, git_repo):
        assert GitDiffTool(project_root=str(git_repo)).run({"commit": "HEAD~1"}).status == ToolStatus.SUCCESS
        assert GitLogTool(project_root=str(git_repo)).run({"ref": "HEAD~1"}).status == ToolStatus.SUCCESS


class TestRunGitInfrastructure:
    def test_missing_git_binary_reported(self, git_repo, monkeypatch):
        import hello_agents.tools.builtin.git_tools as git_tools_module

        def _raise_missing(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(git_tools_module.subprocess, "run", _raise_missing)
        monkeypatch.setattr(git_tools_module.shutil, "which", lambda _: None)
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.EXECUTION_ERROR
        assert "not found on PATH" in resp.error_info["message"]

    def test_missing_working_dir_distinguished(self, git_repo, monkeypatch):
        import hello_agents.tools.builtin.git_tools as git_tools_module

        def _raise_missing(*args, **kwargs):
            raise FileNotFoundError("cwd")

        monkeypatch.setattr(git_tools_module.subprocess, "run", _raise_missing)
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert resp.error_info["code"] == ToolErrorCode.EXECUTION_ERROR
        assert "does not exist" in resp.error_info["message"]

    def test_git_timeout_reported(self, git_repo, monkeypatch):
        import hello_agents.tools.builtin.git_tools as git_tools_module

        def _raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=["git"], timeout=30)

        monkeypatch.setattr(git_tools_module.subprocess, "run", _raise_timeout)
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert resp.error_info["code"] == ToolErrorCode.TIMEOUT


class TestPathspecHardening:
    def test_colon_prefixed_filename_commits(self, git_repo):
        weird = git_repo / ":!magic.py"
        weird.write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "add weird file", "paths": [":!magic.py"]}
        )
        assert resp.status == ToolStatus.SUCCESS, resp.text
        assert any(f["path"] == ":!magic.py" for f in resp.data["files_committed"])

    def test_unquote_leaves_literal_quotes(self):
        assert _unquote_path('"foo"') == '"foo"'
        assert _unquote_path('"a\\tb"') == "a\tb"


class TestGitCommitEdgeCases:
    def test_amend_without_commits(self, empty_repo):
        resp = GitCommitTool(project_root=str(empty_repo)).run(
            {"message": "amend me", "amend": True}
        )
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.CONFLICT
        assert "no commits yet" in resp.error_info["message"]

    def test_commit_pre_staged_changes(self, git_repo):
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        _git(git_repo, "add", "f.py")
        resp = GitCommitTool(project_root=str(git_repo)).run({"message": "pre-staged"})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["staged_paths"] == []
        assert _head_subject(git_repo) == "pre-staged"

    def test_multiline_message_preserved(self, git_repo):
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")
        resp = GitCommitTool(project_root=str(git_repo)).run(
            {"message": "subject line\n\nbody paragraph", "paths": ["f.py"]}
        )
        assert resp.status == ToolStatus.SUCCESS
        body = _git(git_repo, "log", "-1", "--pretty=%B").stdout
        assert "subject line" in body
        assert "body paragraph" in body

    def test_auto_message_rename(self, git_repo):
        _git(git_repo, "mv", "src/b.py", "src/renamed.py")
        resp = GitCommitTool(project_root=str(git_repo)).run({"auto_message": True})
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["message"] == "refactor: rename b.py to renamed.py"

    def test_generate_message_rename_unit(self):
        files = [{"path": "src/new.py", "status": "R", "additions": 0, "deletions": 0, "binary": False, "orig_path": "src/old.py"}]
        assert generate_commit_message(files) == "refactor: rename old.py to new.py"
        multi = files + [{"path": "src/n2.py", "status": "R", "additions": 0, "deletions": 0, "binary": False, "orig_path": "src/o2.py"}]
        assert generate_commit_message(multi) == "refactor: rename 2 files in src"

    def test_circuit_breaker_trips_after_repeated_hook_failures(self, git_repo):
        from hello_agents.tools.registry import ToolRegistry

        hook = git_repo / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        (git_repo / "f.py").write_text("x\n", encoding="utf-8")

        registry = ToolRegistry(verbose=False)
        registry.register_tool(GitCommitTool(project_root=str(git_repo)))
        payload = json.dumps({"message": "x", "paths": ["f.py"]})
        for _ in range(3):
            resp = registry.execute_tool("GitCommit", payload)
            assert resp.error_info["code"] == ToolErrorCode.EXECUTION_ERROR
        assert registry.circuit_breaker.is_open("GitCommit") is True
        resp = registry.execute_tool("GitCommit", payload)
        assert resp.error_info["code"] == ToolErrorCode.CIRCUIT_OPEN


class TestGitBlameEdgeCases:
    def test_binary_file_rejected(self, git_repo):
        (git_repo / "blob.bin").write_bytes(b"\x00\x01\x02\x00" * 32)
        _git(git_repo, "add", "blob.bin")
        _git(git_repo, "commit", "-m", "add binary")
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "blob.bin"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.BINARY_FILE

    def test_uncommitted_lines_attribution(self, git_repo):
        (git_repo / "src" / "a.py").write_text(
            "brand new line\nline2\nline3\nline4\n", encoding="utf-8"
        )
        resp = GitBlameTool(project_root=str(git_repo)).run({"path": "src/a.py"})
        assert resp.status == ToolStatus.SUCCESS
        authors = {line["line"]: line["author"] for line in resp.data["lines"]}
        assert authors[1] == "Not Committed Yet"
        assert authors[2] == "Tester"


class TestGitStatusEdgeCases:
    def test_detached_head(self, git_repo):
        _git(git_repo, "checkout", "HEAD~1")
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert resp.data["branch"]["detached"] is True
        assert "detached" in resp.text

    def test_stash_count(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        _git(git_repo, "stash")
        resp = GitStatusTool(project_root=str(git_repo)).run({})
        assert resp.data["stash_count"] == 1
        assert "Stash entries: 1" in resp.text

    def test_scope_shown_in_text(self, git_repo):
        resp = GitStatusTool(project_root=str(git_repo)).run({"path": "src"})
        assert resp.text.startswith("Scope: src")

    def test_file_scope(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        resp = GitStatusTool(project_root=str(git_repo)).run({"path": "src/a.py"})
        assert resp.data["scope"] == "src/a.py"
        assert [e["path"] for e in resp.data["unstaged"]] == ["src/a.py"]


class TestGitDiffEdgeCases:
    def test_context_lines_zero(self, git_repo):
        original = (git_repo / "src" / "a.py").read_text(encoding="utf-8")
        (git_repo / "src" / "a.py").write_text(original + "line5\n", encoding="utf-8")
        resp = GitDiffTool(project_root=str(git_repo)).run({"context_lines": 0})
        assert resp.status == ToolStatus.SUCCESS
        assert "+line5" in resp.data["patch"]
        assert " line2\n" not in resp.data["patch"]

    def test_directory_path_filter(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        (git_repo / "README.md").write_text("changed\n", encoding="utf-8")
        resp = GitDiffTool(project_root=str(git_repo)).run({"path": "src"})
        assert resp.data["total_files"] == 1
        assert resp.data["files"][0]["path"] == "src/a.py"
        assert resp.data["scope"] == "src"

    def test_totals_aggregation(self, git_repo):
        (git_repo / "src" / "a.py").write_text("one\ntwo\n", encoding="utf-8")
        (git_repo / "src" / "b.py").write_text("b1\nb2\nb3\n", encoding="utf-8")
        resp = GitDiffTool(project_root=str(git_repo)).run({})
        assert resp.data["total_files"] == 2
        # a.py: 4 lines -> 2 lines (+2 -4); b.py: 2 lines -> 3 lines (+1 -0)
        assert resp.data["total_additions"] == 3
        assert resp.data["total_deletions"] == 4

    def test_stat_only_consistent_with_patch_mode(self, git_repo):
        (git_repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")
        tool = GitDiffTool(project_root=str(git_repo))
        stat_resp = tool.run({"stat_only": True})
        patch_resp = tool.run({})
        assert stat_resp.data["files"] == patch_resp.data["files"]
        assert stat_resp.data["total_files"] == patch_resp.data["total_files"]


class TestGitLogEdgeCases:
    def test_combined_filters(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run(
            {"author": "Tester", "grep": "first"}
        )
        assert resp.data["count"] == 1
        assert "first commit" in resp.data["commits"][0]["subject"]
        resp = GitLogTool(project_root=str(git_repo)).run(
            {"author": "Nobody", "grep": "first"}
        )
        assert resp.data["count"] == 0

    def test_scope_in_data(self, git_repo):
        resp = GitLogTool(project_root=str(git_repo)).run({"path": "src/b.py"})
        assert resp.data["scope"] == "src/b.py"
