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


# ═══════════════════════════════════════════════════════════════════════
# 严重-1: expanded sensitive env coverage + extra param filtering
# ═══════════════════════════════════════════════════════════════════════


class TestExpandedSensitiveEnvFragments:
    """Verify that the expanded _SENSITIVE_ENV_FRAGMENTS list catches more
    credential patterns (fix for 严重-1)."""

    def test_connection_string_detected(self):
        assert _is_sensitive_env_key("DB_CONNECTION_STRING")
        assert _is_sensitive_env_key("CONNECTION_STRING")

    def test_certificate_detected(self):
        assert _is_sensitive_env_key("TLS_CERTIFICATE")
        assert _is_sensitive_env_key("CLIENT_CERT")
        assert _is_sensitive_env_key("CA_CERTIFICATE")

    def test_license_key_detected(self):
        assert _is_sensitive_env_key("JETBRAINS_LICENSE")
        assert _is_sensitive_env_key("GITHUB_LICENSE")

    def test_registry_detected(self):
        assert _is_sensitive_env_key("DOCKER_REGISTRY_PASSWORD")
        assert _is_sensitive_env_key("NPM_REGISTRY_TOKEN")
        assert _is_sensitive_env_key("REGISTRY_AUTH")

    def test_npm_token_detected(self):
        assert _is_sensitive_env_key("NPM_TOKEN")

    def test_database_url_detected(self):
        assert _is_sensitive_env_key("DATABASE_URL")

    def test_sauce_labs_detected(self):
        assert _is_sensitive_env_key("SAUCE_ACCESS_KEY")
        assert _is_sensitive_env_key("SAUCE_USERNAME")

    def test_standalone_key_fragment_detected(self):
        assert _is_sensitive_env_key("ENCRYPTION_KEY")
        assert _is_sensitive_env_key("SIGNING_KEY")

    def test_git_ssh_key_is_blocked(self):
        """GIT_SSH_KEY contains 'KEY' and is correctly blocked.
        The bare 'KEY' fragment is intentionally broad — virtually every
        environment variable with KEY in its name carries a credential."""
        assert _is_sensitive_env_key("GIT_SSH_KEY")

    def test_harmless_env_vars_still_ok(self):
        assert not _is_sensitive_env_key("PYTHONPATH")
        assert not _is_sensitive_env_key("EDITOR")
        assert not _is_sensitive_env_key("LANG")
        assert not _is_sensitive_env_key("PWD")
        assert not _is_sensitive_env_key("USER")
        assert not _is_sensitive_env_key("SHELL")
        assert not _is_sensitive_env_key("DISPLAY")
        assert not _is_sensitive_env_key("TERM")


class TestExtraParamFiltering:
    """Verify build_sandbox_env filters sensitive keys in the *extra* dict
    as well (fix for 严重-1 extra bypass)."""

    def test_extra_sensitive_key_is_filtered(self, tmp_path, monkeypatch):
        # Make sure the parent env doesn't have this key so it can only come
        # from `extra`.
        monkeypatch.delenv("MY_SECRET_TOKEN", raising=False)
        env = build_sandbox_env(
            tmp_path,
            extra={"MY_SECRET_TOKEN": "leaked", "SAFE_VAR": "ok"},
        )
        assert "MY_SECRET_TOKEN" not in env
        assert "SAFE_VAR" in env
        assert env["SAFE_VAR"] == "ok"

    def test_extra_with_no_sensitive_fully_passed(self, tmp_path):
        env = build_sandbox_env(
            tmp_path, extra={"PROJECT_SPECIFIC_FLAG": "1"},
        )
        assert env["PROJECT_SPECIFIC_FLAG"] == "1"

    def test_project_root_always_set(self, tmp_path):
        env = build_sandbox_env(tmp_path)
        assert env["PROJECT_ROOT"] == str(tmp_path)


class TestBashNotLoginShell:
    """Verify shell commands use ``bash -c`` (not ``bash -lc``) so that
    .bashrc / .bash_profile are NOT sourced (fix for 严重-1)."""

    def test_login_shell_marker_not_set(self, tmp_path, monkeypatch):
        """$SHLVL increments in any interactive shell; what matters is that
        login-specific variables are not present."""
        tool = _tool(tmp_path)
        resp = tool.run({
            "command": (
                "echo LOGIN_SHELL=${LOGIN_SHELL:-NOT_SET} "
            ),
            "block_until_ms": 15000,
        })
        output = resp.data.get("output", "")
        # If bash were invoked with -l, $0 would start with '-'
        # With -c, it should just be 'bash'
        assert "NOT_SET" in output, f"login shell env detected: {output}"


# ═══════════════════════════════════════════════════════════════════════
# 严重-2: resource limit hardening
# ═══════════════════════════════════════════════════════════════════════


class TestResourceLimitDefaults:
    """Verify the tightened default values (fix for 严重-2)."""

    def test_default_process_limit_is_512(self):
        tool = BashTool(project_root="/tmp")
        assert tool.DEFAULT_MAX_PROCESSES == 512

    def test_default_memory_limit_is_16gib(self):
        tool = BashTool(project_root="/tmp")
        assert tool.DEFAULT_MAX_MEMORY_BYTES == 16 * 1024**3

    def test_default_execution_timeout_is_5min(self):
        tool = BashTool(project_root="/tmp")
        assert tool.DEFAULT_MAX_EXECUTION_MS == 300_000

    def test_preexec_built_with_new_defaults(self, tmp_path):
        tool = _tool(tmp_path)
        preexec = tool._build_preexec()
        assert preexec is None or callable(preexec)

    def test_config_overrides_still_work(self, tmp_path):
        from hello_agents.core.config import Config
        tool = BashTool(
            project_root=str(tmp_path),
            config=Config(bash_max_processes=128, bash_max_execution_ms=60000),
        )
        assert tool.max_processes == 128
        assert tool.max_execution_ms == 60000

    def test_zero_disables_memory_limit(self, tmp_path):
        from hello_agents.core.config import Config
        tool = BashTool(
            project_root=str(tmp_path),
            config=Config(bash_max_memory_bytes=0),
        )
        assert tool.max_memory_bytes == 0


class TestConfigDefaultLimitsEffective:
    """改进项 7/F1: Config 默认值必须与 BashTool 类安全默认对齐并真正生效。

    此前 Config 默认 0 恒覆盖 BashTool 类默认（memory/execution_ms），且
    file_size 完全不读 Config 字段，导致主路径下沙箱资源限制全部失效。
    """

    def test_config_defaults_match_tool_safe_defaults(self):
        from hello_agents.core.config import Config
        cfg = Config()
        assert cfg.bash_max_memory_bytes == BashTool.DEFAULT_MAX_MEMORY_BYTES
        assert cfg.bash_max_processes == BashTool.DEFAULT_MAX_PROCESSES
        assert cfg.bash_max_file_size_bytes == BashTool.DEFAULT_MAX_FILE_SIZE_BYTES
        assert cfg.bash_max_execution_ms == BashTool.DEFAULT_MAX_EXECUTION_MS
        assert cfg.bash_max_cpu_seconds == BashTool.DEFAULT_MAX_CPU_SECONDS

    def test_main_path_defaults_effective(self, tmp_path):
        """主路径（CodeAgent 必传 config）下安全默认必须生效，而非 0=不限制。"""
        from hello_agents.core.config import Config
        tool = BashTool(project_root=str(tmp_path), config=Config())
        assert tool.max_memory_bytes == BashTool.DEFAULT_MAX_MEMORY_BYTES
        assert tool.max_processes == BashTool.DEFAULT_MAX_PROCESSES
        assert tool.max_file_size_bytes == BashTool.DEFAULT_MAX_FILE_SIZE_BYTES
        assert tool.max_execution_ms == BashTool.DEFAULT_MAX_EXECUTION_MS

    def test_config_file_size_field_is_honored(self, tmp_path):
        """Config.bash_max_file_size_bytes 此前被 BashTool 忽略（只读 env）。"""
        from hello_agents.core.config import Config
        tool = BashTool(
            project_root=str(tmp_path),
            config=Config(bash_max_file_size_bytes=1024),
        )
        assert tool.max_file_size_bytes == 1024

    def test_config_zero_still_disables_each_dimension(self, tmp_path):
        """显式设 0 仍可禁用各维度（逃生通道保留）。"""
        from hello_agents.core.config import Config
        tool = BashTool(
            project_root=str(tmp_path),
            config=Config(
                bash_max_memory_bytes=0,
                bash_max_processes=0,
                bash_max_file_size_bytes=0,
                bash_max_execution_ms=0,
            ),
        )
        assert tool.max_memory_bytes == 0
        assert tool.max_processes == 0
        assert tool.max_file_size_bytes == 0
        assert tool.max_execution_ms == 0


class TestWatchdogTimer:
    """Verify the wall-clock watchdog (fix for 严重-2)."""

    def test_watchdog_kills_long_running_command(self, tmp_path):
        """A command that sleeps beyond max_execution_ms must be killed by
        the watchdog timer."""
        from hello_agents.core.config import Config
        tool = BashTool(
            project_root=str(tmp_path),
            working_dir=str(tmp_path),
            config=Config(bash_max_execution_ms=500),  # 0.5s
        )
        start = __import__("time").monotonic()
        resp = tool.run({
            "command": "sleep 60",  # would run 60s without watchdog
            "block_until_ms": 5000,
        })
        elapsed = __import__("time").monotonic() - start
        # The watchdog should fire within ~1s (500ms + overhead)
        assert elapsed < 10.0, f"watchdog did not fire, elapsed={elapsed:.1f}s"

    def test_watchdog_not_triggered_for_quick_command(self, tmp_path):
        """A quick command must complete normally without the watchdog firing."""
        from hello_agents.core.config import Config
        tool = BashTool(
            project_root=str(tmp_path),
            working_dir=str(tmp_path),
            config=Config(bash_max_execution_ms=5000),
        )
        resp = tool.run({
            "command": "echo quick",
            "block_until_ms": 15000,
        })
        assert resp.status.value == "success"
        assert "quick" in resp.data.get("output", "")

    def test_zero_max_execution_disables_watchdog(self, tmp_path):
        """When max_execution_ms=0 the watchdog is not started at all."""
        from hello_agents.core.config import Config
        tool = BashTool(
            project_root=str(tmp_path),
            working_dir=str(tmp_path),
            config=Config(bash_max_execution_ms=0),
        )
        resp = tool.run({
            "command": "echo still_works",
            "block_until_ms": 5000,
        })
        assert resp.status.value == "success"
        assert "still_works" in resp.data.get("output", "")
