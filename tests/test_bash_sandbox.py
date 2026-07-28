"""严重-1 / 严重-2: BashTool sandbox hardening."""

import shlex

import pytest

from hello_agents.tools.builtin.bash import BashTool, build_sandbox_env, _is_sensitive_env_key


def _tool(tmp_path):
    return BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))


def test_sensitive_env_keys_detected():
    assert _is_sensitive_env_key("LLM_API_KEY")
    assert _is_sensitive_env_key("HF_TOKEN")
    assert _is_sensitive_env_key("QDRANT_API_KEY")
    assert _is_sensitive_env_key("NEO4J_PASSWORD")
    assert _is_sensitive_env_key("AWS_SECRET_ACCESS_KEY")
    assert _is_sensitive_env_key("GITHUB_PERSONAL_ACCESS_TOKEN")
    assert not _is_sensitive_env_key("PATH")
    assert not _is_sensitive_env_key("HOME")
    assert not _is_sensitive_env_key("LANG")


def test_child_env_strips_secrets_keeps_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-value")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("MY_PASSWORD", "pw")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/tester")

    env = build_sandbox_env(tmp_path)

    assert "LLM_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "MY_PASSWORD" not in env
    assert env.get("PATH") == "/usr/bin:/bin"
    assert env.get("HOME") == "/home/tester"
    assert env.get("PROJECT_ROOT") == str(tmp_path)


def test_child_env_method_matches_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "leak-me")
    tool = _tool(tmp_path)
    env = tool._build_child_env()
    assert "OPENAI_API_KEY" not in env


def test_deeply_nested_shell_is_blocked(tmp_path):
    tool = _tool(tmp_path)
    cmd = "echo hi"
    for _ in range(4):  # 4 levels of bash -c nesting exceeds MAX_POLICY_PARSE_DEPTH
        cmd = f"bash -c {shlex.quote(cmd)}"
    reason = tool.validate_command_policy(cmd)
    assert reason is not None
    assert "too deep" in reason.lower()


def test_plain_command_is_allowed(tmp_path):
    tool = _tool(tmp_path)
    assert tool.validate_command_policy("echo hello") is None
    assert tool.validate_command_policy("git status") is None


def test_destructive_still_blocked(tmp_path):
    tool = _tool(tmp_path)
    assert tool.validate_command_policy("rm -rf /") is not None
    assert tool.validate_command_policy("sudo rm x") is not None


def test_preexec_builder_is_callable_or_none(tmp_path):
    tool = _tool(tmp_path)
    preexec = tool._build_preexec()
    assert preexec is None or callable(preexec)


def test_execution_stays_in_env_without_secret(tmp_path, monkeypatch):
    """End-to-end: a command cannot read a stripped secret from its env."""
    monkeypatch.setenv("LLM_API_KEY", "TOPSECRET123")
    tool = _tool(tmp_path)
    resp = tool.run({"command": "echo key=[$LLM_API_KEY]", "block_until_ms": 15000})
    text = resp.text + str(resp.data.get("output", ""))
    assert "TOPSECRET123" not in text
    assert "key=[]" in text
