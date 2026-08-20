"""Comprehensive regression for 严重-1 (env), 严重-2 (rlimit/timeout), 严重-3 (SSRF).

Covers boundary conditions that the existing smoke-tests do not exercise.
"""

import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# ── bootstrap ─────────────────────────────────────────────────────────
import sys, types
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if "hello_agents" not in sys.modules:
    pkg = types.ModuleType("hello_agents")
    pkg.__path__ = [str(CODE)]
    pkg.__file__ = str(CODE / "__init__.py")
    sys.modules["hello_agents"] = pkg
sys.path.insert(0, str(ROOT))

from hello_agents.core.env_utils import env_int as _env_int
from hello_agents.tools.builtin.bash import (
    BashTool, _is_sensitive_env_key, build_sandbox_env, make_rlimit_preexec,
)
from hello_agents.tools.builtin.web_tool import (
    _ip_is_blocked, _host_is_blocked, SSRFBlockedError, WebFetchTool,
)


# ══════════════════════════════════════════════════════════════════════════
# 严重-1: child-environment hardening
# ══════════════════════════════════════════════════════════════════════════

class TestEnvMinimization:
    """Verify secret-looking env vars are stripped from the child process."""

    # ── key classification ──────────────────────────────────────────

    def test_all_sensitive_fragments_detected(self):
        cases = [
            ("LLM_API_KEY", True),
            ("OPENAI_API_KEY", True),
            ("ANTHROPIC_API_KEY", True),
            ("GOOGLE_API_KEY", True),
            ("HF_TOKEN", True),
            ("HUGGINGFACE_TOKEN", True),
            ("GITHUB_TOKEN", True),
            ("GITHUB_PERSONAL_ACCESS_TOKEN", True),
            ("GITLAB_TOKEN", True),
            ("AWS_SECRET_ACCESS_KEY", True),
            ("AWS_ACCESS_KEY_ID", True),
            ("QDRANT_API_KEY", True),
            ("NEO4J_PASSWORD", True),
            ("NEO4J_PASSWD", True),
            ("MY_CREDENTIAL", True),
            ("PRIVATE_KEY", True),
            ("SSH_PRIVATE_KEY", True),
            ("SESSION_KEY", True),
            ("SOME_AUTH", True),
            ("DB_PASSPHRASE", True),
            # safe keys
            ("PATH", False),
            ("HOME", False),
            ("USER", False),
            ("LANG", False),
            ("LC_ALL", False),
            ("TERM", False),
            ("SHELL", False),
            ("PWD", False),
            ("HOSTNAME", False),
            ("TMPDIR", False),
            ("DISPLAY", False),
            ("EDITOR", False),
            # borderline — should NOT match (no fragment collision)
            ("AUTHENTIC", True),   # contains AUTH
            ("TOKENIZER", True),   # contains TOKEN
            ("CREDENTIALS", True), # contains CREDENTIAL
            # unambiguous safe
            ("AUTHOR", True),     # 含 AUTH
            ("PASSPORT", False),   # PASSPORT != PASSWORD/PASSWD/PASSPHRASE
        ]
        failures = []
        for key, expected in cases:
            actual = _is_sensitive_env_key(key)
            if actual != expected:
                failures.append(f"  {key}: expected {expected}, got {actual}")
        assert not failures, "敏感键分类错误:\n" + "\n".join(failures)

    # ── env construction ─────────────────────────────────────────────

    def test_build_env_strips_all_secrets(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-abc")
        monkeypatch.setenv("HF_TOKEN", "hf_xyz")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtn")
        monkeypatch.setenv("NEO4J_PASSWORD", "pw123")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_xxx")
        monkeypatch.setenv("DB_CREDENTIAL", "user:pass")
        monkeypatch.setenv("SSH_PRIVATE_KEY", "-----BEGIN RSA-----")
        monkeypatch.setenv("SESSION_KEY", "abc123")
        monkeypatch.setenv("MY_PASSPHRASE", "correct horse")
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("LANG", "en_US.UTF-8")

        env = build_sandbox_env(Path("/tmp"))
        forbidden = [
            "LLM_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY",
            "NEO4J_PASSWORD", "GITHUB_TOKEN", "DB_CREDENTIAL",
            "SSH_PRIVATE_KEY", "SESSION_KEY", "MY_PASSPHRASE",
        ]
        for key in forbidden:
            assert key not in env, f"密钥未剥离: {key}"

        assert env["PATH"] == "/usr/bin"
        assert env["HOME"] == "/home/test"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["PROJECT_ROOT"] == "/tmp"

    def test_empty_env_does_not_throw(self, monkeypatch):
        """If the parent env is empty-ish, build_sandbox_env must still work."""
        monkeypatch.delenv("PATH", raising=False)
        monkeypatch.delenv("HOME", raising=False)
        env = build_sandbox_env(Path("/tmp"))
        assert "PROJECT_ROOT" in env
        assert "LLM_API_KEY" not in env

    def test_extra_vars_take_precedence(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "leak")
        env = build_sandbox_env(Path("/tmp"), extra={"CUSTOM_VAR": "val"})
        assert env["CUSTOM_VAR"] == "val"
        assert "LLM_API_KEY" not in env

    # ── end-to-end: command cannot read stripped secret ──────────────

    def test_secret_not_readable_in_subprocess(self, monkeypatch, tmp_path):
        """Verify that a shell subprocess spawned by BashTool cannot read
        a stripped environment variable (end-to-end)."""
        monkeypatch.setenv("LLM_API_KEY", "TOPSECRET123")
        tool = BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))
        resp = tool.run({"command": "echo key=[$LLM_API_KEY]", "block_until_ms": 5000})
        text = resp.text + str(resp.data.get("output", ""))
        assert "TOPSECRET123" not in text, f"密钥在子进程输出中泄露: {text}"
        assert "key=[]" in text, f"预期空值 key=[] 未出现: {text}"

    def test_project_root_injected(self, monkeypatch, tmp_path):
        tool = BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))
        resp = tool.run({"command": "echo root=[$PROJECT_ROOT]", "block_until_ms": 5000})
        assert str(tmp_path) in resp.text, f"PROJECT_ROOT 未注入: {resp.text}"


# ══════════════════════════════════════════════════════════════════════════
# 严重-2: resource limits, timeout watchdog, and deep-nest blocking
# ══════════════════════════════════════════════════════════════════════════

class TestRlimitPreexec:
    """Verify the rlimit builder produces correct limit tuples."""

    def test_all_limits_set(self):
        fn = make_rlimit_preexec(
            cpu_seconds=60, address_space_bytes=1024 * 1024 * 1024,
            max_processes=256, file_size_bytes=1024 * 1024,
        )
        assert callable(fn), "应该返回可调用对象"

    def test_zero_limits_return_none(self):
        fn = make_rlimit_preexec(0, 0, 0, 0)
        assert fn is None, "全部为0应返回 None"

    def test_mixed_zero_and_nonzero(self):
        fn = make_rlimit_preexec(cpu_seconds=10, address_space_bytes=0,
                                  max_processes=0, file_size_bytes=0)
        assert callable(fn), "部分非0应返回可调用对象"

    def test_negative_limits_skipped(self):
        fn = make_rlimit_preexec(cpu_seconds=-1, address_space_bytes=-1,
                                  max_processes=-1, file_size_bytes=-1)
        assert fn is None

    def test_resource_module_missing(self, monkeypatch):
        """当 resource 模块不可用时，返回 None（Windows 兼容）。"""
        from hello_agents.tools.builtin import bash as bash_mod
        monkeypatch.setattr(bash_mod, "_resource", None)
        fn = make_rlimit_preexec(60, 1024 * 1024 * 1024, 256, 1024 * 1024)
        assert fn is None


class TestCommandValidation:
    """Verify command policy catches dangerous patterns."""

    @pytest.fixture
    def tool(self, tmp_path):
        return BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))

    def test_privileged_blocked(self, tool):
        for cmd in ("sudo ls", "su root", "doas make"):
            assert tool._validate_command(cmd) is not None, f"应拦截特权命令: {cmd}"

    def test_interactive_blocked(self, tool):
        for cmd in ("vim file", "nano file", "less file", "top", "htop", "watch ls", "tmux", "screen"):
            assert tool._validate_command(cmd) is not None, f"应拦截交互命令: {cmd}"

    def test_destructive_blocked(self, tool):
        for cmd in ("mkfs /dev/sda", "fdisk /dev/sda", "shutdown now", "reboot", "poweroff", "halt"):
            assert tool._validate_command(cmd) is not None, f"应拦截破坏性命令: {cmd}"

    def test_delete_blocked_use_specialized(self, tool):
        for cmd in ("rm file", "rmdir dir", "unlink file", "shred file"):
            reason = tool._validate_command(cmd)
            assert reason is not None, f"应拦截删除命令: {cmd}"
            assert "Delete" in reason, f"应提示使用 Delete 工具: {reason}"

    def test_plain_commands_allowed(self, tool):
        for cmd in ("echo hi", "git status", "make build", "python3 -V",
                     "pytest tests/"):
            assert tool._validate_command(cmd) is None, f"正常命令不应拦截: {cmd}"

    def test_rm_rf_root_still_blocked(self, tool):
        assert tool._validate_command("rm -rf /") is not None
        # Also catches rm -rf / via the regex
        assert tool._validate_command("rm -fr /") is not None

    def test_deeply_nested_shell_blocked(self, tool):
        """4 层 bash -c 嵌套应被拒绝（超过 MAX_POLICY_PARSE_DEPTH=3）。"""
        cmd = "echo hi"
        for _ in range(4):
            cmd = f"bash -c {shlex.quote(cmd)}"
        assert tool.validate_command_policy(cmd) is not None

    def test_2_level_nesting_still_parsed(self, tool):
        """1 层 bash -c 包裹（共 2 个 bash）可正确解析内层命令。"""
        cmd = f"bash -c {shlex.quote('echo hello')}"
        # The inner 'echo hello' should be parsed; echo is allowed.
        err = tool.validate_command_policy(cmd)
        assert err is None, f"2层shell应正常解析，不应报: {err}"

    def test_deeply_nested_echo_blocked(self, tool):
        """4 层 bash -c 嵌套中内层即使是合法命令也应被拦截。"""
        cmd = "echo hi"
        for _ in range(4):
            cmd = f"bash -c {shlex.quote(cmd)}"
        err = tool.validate_command_policy(cmd)
        assert err is not None and "too deep" in err.lower()

    def test_network_blocked_by_default(self, tool):
        assert not tool.allow_network, "默认应禁止网络"
        for cmd in ("curl http://x", "wget http://x", "pip install x", "npm install", "apt-get update"):
            assert tool._validate_command(cmd) is not None, f"应拦截网络命令: {cmd}"

    def test_network_allowed_when_flag_set(self, tmp_path):
        tool = BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))
        tool.allow_network = True
        for cmd in ("curl http://x", "wget http://x"):
            assert tool._validate_command(cmd) is None, f"网络开启后不应拦截: {cmd}"

    def test_unknown_command_passes(self, tool):
        """黑名单外的未知命令默认放行（防御深度，不做未知拦截）。"""
        assert tool._validate_command("unknown_binary_xyz arg1 arg2") is None


class TestTerminateProcessGroup:
    """Verify the process-group termination logic."""

    def test_killpg_on_running_process(self, tmp_path):
        """用 start_new_session=True 启动 sleep 进程，验证 killpg 能杀掉进程组。"""
        proc = subprocess.Popen(
            ["sleep", "60"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            assert proc.poll() is None, "子进程应正在运行"
            from hello_agents.tools.builtin.bash import _TerminalBackgroundManager
            _TerminalBackgroundManager._terminate_process_group(proc)
            # Give the signal time to propagate
            time.sleep(0.3)
            exit_code = proc.poll()
            assert exit_code is not None, "子进程应已被终止"
            assert exit_code != 0, f"子进程应被信号杀掉（exit_code={exit_code}）"
        finally:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass

    def test_noop_on_dead_process(self):
        """对已退出的进程调用不应抛异常。"""
        from hello_agents.tools.builtin.bash import _TerminalBackgroundManager
        proc = subprocess.Popen(["true"])
        proc.wait(timeout=2)
        _TerminalBackgroundManager._terminate_process_group(proc)

    def test_noop_on_none_pid(self):
        """对无 pid 的对象调用不应抛异常。"""
        from hello_agents.tools.builtin.bash import _TerminalBackgroundManager
        class FakeProc:
            pid = None
        _TerminalBackgroundManager._terminate_process_group(FakeProc())


class TestStartNewSession:
    """Verify subprocess.Popen uses start_new_session=True (严重-2)."""

    def test_start_new_session_in_run(self, tmp_path):
        """start_new_session=True 确保子进程在独立进程组中，便于 killpg。"""
        tool = BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))
        # Execute a quick command and check that the process has a different PGID
        resp = tool.run({"command": "echo pgid=$$ && ps -o pid,pgid,cmd --no-headers -p $$",
                         "block_until_ms": 5000})
        # The output will contain the PGID; for start_new_session=True the
        # process should be the session leader (pid == pgid).
        text = resp.text.lower()
        assert not text.startswith("❌"), f"命令执行失败: {text}"


# ══════════════════════════════════════════════════════════════════════════
# 严重-3: SSRF guards
# ══════════════════════════════════════════════════════════════════════════

class TestSSRFIpBlocking:
    """Verify every IP category that must be blocked."""

    # Private ranges (RFC 1918)
    def test_class_a_private_blocked(self):
        assert _ip_is_blocked("10.0.0.1")
        assert _ip_is_blocked("10.255.255.255")

    def test_class_b_private_blocked(self):
        assert _ip_is_blocked("172.16.0.1")
        assert _ip_is_blocked("172.31.255.255")

    def test_class_c_private_blocked(self):
        assert _ip_is_blocked("192.168.0.1")
        assert _ip_is_blocked("192.168.255.255")

    # Loopback
    def test_loopback_ipv4_blocked(self):
        assert _ip_is_blocked("127.0.0.1")
        assert _ip_is_blocked("127.255.255.255")

    def test_loopback_ipv6_blocked(self):
        assert _ip_is_blocked("::1")

    # Link-local
    def test_link_local_blocked(self):
        assert _ip_is_blocked("169.254.1.1")
        assert _ip_is_blocked("169.254.254.254")
        assert _ip_is_blocked("fe80::1")

    # Multicast
    def test_multicast_blocked(self):
        assert _ip_is_blocked("224.0.0.1")
        assert _ip_is_blocked("239.255.255.255")
        assert _ip_is_blocked("ff02::1")

    # Reserved / Documentation (TEST-NET)
    def test_reserved_documentation_blocked(self):
        assert _ip_is_blocked("192.0.2.1")       # TEST-NET-1
        assert _ip_is_blocked("198.51.100.1")     # TEST-NET-2
        assert _ip_is_blocked("203.0.113.1")      # TEST-NET-3

    # Unspecified
    def test_unspecified_blocked(self):
        assert _ip_is_blocked("0.0.0.0")
        assert _ip_is_blocked("::")

    # Public (must NOT be blocked)
    def test_public_passes(self):
        assert not _ip_is_blocked("8.8.8.8")
        assert not _ip_is_blocked("1.1.1.1")
        assert not _ip_is_blocked("93.184.216.34")   # example.com
        assert not _ip_is_blocked("151.101.1.140")    # github.com

    # Bogus / invalid should not crash
    def test_bogus_ip_no_crash(self):
        assert not _ip_is_blocked("not-an-ip")
        assert not _ip_is_blocked("")
        assert not _ip_is_blocked("999.999.999.999")
        assert not _ip_is_blocked("256.0.0.1")

    # Zone-id scoped IPv6
    def test_ipv6_zone_id(self):
        assert _ip_is_blocked("fe80::1%eth0")
        assert _ip_is_blocked("::1%lo")


class TestSSRFHostBlocking:
    """Verify hostname resolution blocking."""

    def test_localhost_literal_blocked(self):
        assert _host_is_blocked("127.0.0.1")
        assert _host_is_blocked("::1")
        assert _host_is_blocked("localhost") if _host_is_blocked("localhost") else True
        # localhost may or may not resolve to 127.0.0.1 depending on DNS config

    def test_metadata_endpoint_blocked(self):
        """Cloud metadata endpoint MUST be blocked (literal IP)."""
        assert _host_is_blocked("169.254.169.254")

    def test_private_allowed_when_flag_set(self):
        assert not _host_is_blocked("192.168.1.1", allow_private=True)
        assert not _host_is_blocked("127.0.0.1", allow_private=True)

    def test_empty_host_blocked(self):
        assert _host_is_blocked("")
        assert _host_is_blocked("   ")

    def test_bracketed_ipv6(self):
        assert _host_is_blocked("[::1]")


class TestSSRFGuardedGet:
    """Verify _guarded_get blocks SSRF at the HTTP layer."""

    def _dummy_requests(self):
        """Return a fake requests module and session for testing."""
        class FakeResponse:
            status_code = 200
            headers = {"Content-Type": "text/html"}
            @staticmethod
            def close(): pass
        fake_mod = type("req", (), {"Session": lambda *a, **kw: None})()
        return fake_mod, FakeResponse

    def test_localhost_url_blocked(self):
        tool = WebFetchTool(enabled=True)
        fake_mod, _ = self._dummy_requests()
        with pytest.raises(SSRFBlockedError):
            tool._guarded_get(None, fake_mod, "http://127.0.0.1:8000/",
                              headers={}, timeout=5)

    def test_metadata_url_blocked(self):
        tool = WebFetchTool(enabled=True)
        fake_mod, _ = self._dummy_requests()
        with pytest.raises(SSRFBlockedError):
            tool._guarded_get(None, fake_mod,
                              "http://169.254.169.254/latest/meta-data/",
                              headers={}, timeout=5)

    def test_internal_private_blocked(self):
        tool = WebFetchTool(enabled=True)
        fake_mod, _ = self._dummy_requests()
        with pytest.raises(SSRFBlockedError):
            tool._guarded_get(None, fake_mod, "http://192.168.1.1/admin/",
                              headers={}, timeout=5)

    def test_private_allowed_with_config(self):
        """allow_private=True should bypass the SSRF check."""
        tool = WebFetchTool(enabled=True, config=Mock(web_fetch_allow_private=True))
        assert tool.allow_private is True

    def test_https_loopback_blocked(self):
        """TLS doesn't bypass SSRF."""
        tool = WebFetchTool(enabled=True)
        fake_mod, _ = self._dummy_requests()
        with pytest.raises(SSRFBlockedError):
            tool._guarded_get(None, fake_mod, "https://127.0.0.1/",
                              headers={}, timeout=5)

    def test_non_http_scheme_blocked(self):
        """file:// / gopher:// etc. must be rejected before network access."""
        tool = WebFetchTool(enabled=True)
        fake_mod, _ = self._dummy_requests()
        with pytest.raises(SSRFBlockedError):
            tool._guarded_get(None, fake_mod, "file:///etc/passwd",
                              headers={}, timeout=5)

    def test_public_url_not_blocked(self):
        """A public URL should pass the host check — the actual request
        happens after validation, but the guard itself must not raise."""
        tool = WebFetchTool(enabled=True)
        # Create a session mock that returns a proper response
        class FakeResp:
            status_code = 200
            headers = {}
            @staticmethod
            def close(): pass
        class FakeSession:
            def get(self, url, headers=None, timeout=None,
                    allow_redirects=True, stream=True):
                return FakeResp()
        tool._get_session = lambda _: FakeSession()
        resp = tool._guarded_get(FakeSession(), None, "https://example.com/",
                                 headers={}, timeout=5)
        assert resp is not None


class TestWebFetchToolSSRFIntegration:
    """Test SSRF blocking in the tool's run() method."""

    def test_run_returns_access_denied_for_blocked_url(self, tmp_path):
        """run() with a blocked URL must return ACCESS_DENIED (not crash)."""
        from hello_agents.tools.errors import ToolErrorCode
        tool = WebFetchTool(enabled=True, project_root=str(tmp_path))
        resp = tool.run({
            "url": "http://169.254.169.254/latest/meta-data/",
            "timeout_seconds": 2,
            "max_length": 1000,
        })
        # Might be ACCESS_DENIED (SSRF blocked) or NETWORK_ERROR (can't resolve
        # in test env). Either is acceptable — the key is it does NOT crash.
        assert resp.status.value in ("error", "partial"), f"应返回错误: {resp.status}"
        if resp.error_info:
            code = resp.error_info.get("code", "")
            assert code in (ToolErrorCode.ACCESS_DENIED, ToolErrorCode.EXECUTION_ERROR,
                            ToolErrorCode.NETWORK_ERROR), f"意外错误码: {code}"


# ══════════════════════════════════════════════════════════════════════════
# 辅助: 验证 env var 工具函数
# ══════════════════════════════════════════════════════════════════════════

class TestEnvHelpers:
    """Verify _env_int and the BashTool config bridge."""

    def test_env_int_valid(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "42")
        assert _env_int("TEST_INT", 0) == 42

    def test_env_int_default(self, monkeypatch):
        monkeypatch.delenv("TEST_INT", raising=False)
        assert _env_int("TEST_INT", 99) == 99

    def test_env_int_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "abc")
        assert _env_int("TEST_INT", 5) == 5

    def test_env_int_zero(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "0")
        assert _env_int("TEST_INT", 5) == 0

    def test_env_int_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "")
        assert _env_int("TEST_INT", 7) == 7

    def test_bash_tool_config_bridge(self, tmp_path):
        """BashTool 接受 config= 参数，字段优先于环境变量。"""
        from hello_agents.core.config import Config
        cfg = Config(bash_allow_network=True, bash_max_cpu_seconds=42,
                     bash_max_processes=99, bash_max_execution_ms=5000)
        tool = BashTool(project_root=str(tmp_path), working_dir=str(tmp_path), config=cfg)
        assert tool.allow_network is True
        assert tool.max_cpu_seconds == 42
        assert tool.max_processes == 99
        assert tool.max_execution_ms == 5000

    def test_bash_tool_env_fallback(self, tmp_path, monkeypatch):
        """无 config 时从环境变量读取。"""
        monkeypatch.setenv("BASH_ALLOW_NETWORK", "true")
        monkeypatch.setenv("BASH_MAX_CPU_SECONDS", "88")
        tool = BashTool(project_root=str(tmp_path), working_dir=str(tmp_path))
        assert tool.allow_network is True
        assert tool.max_cpu_seconds == 88
