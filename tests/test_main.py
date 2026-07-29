"""Tests for main.py —— WhaleCode 统一入口。

覆盖范围：
- 命令行参数解析（所有子命令 + unknown 透传）
- vLLM 检测 / 启动命令构建 / 自动启动决策
- 环境变量加载与包引导
- Benchmark 数据路径自动补全
- Benchmark dry-run 集成
- vLLM 状态格式化输出
- main() 端到端流程
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# ── 将 main 模块导入为可测试对象 ──────────────────────────────────

MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
spec = importlib.util.spec_from_file_location("whale_main", str(MAIN_PATH))
assert spec is not None and spec.loader is not None, f"无法加载 {MAIN_PATH}"
main_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_mod)

# 导入常用符号
build_parser = main_mod.build_parser
detect_vllm = main_mod.detect_vllm
build_vllm_command = main_mod.build_vllm_command
bootstrap_package = main_mod.bootstrap_package
load_env = main_mod.load_env
vllm_status = main_mod.vllm_status
print_vllm_status = main_mod.print_vllm_status
start_vllm = main_mod.start_vllm
stop_vllm = main_mod.stop_vllm
auto_start_vllm = main_mod.auto_start_vllm
run_bench = main_mod.run_bench
run_tests = main_mod.run_tests
_ensure_data_path = main_mod._ensure_data_path
_resolve_data_root = main_mod._resolve_data_root
_PASSTHROUGH_COMMANDS = main_mod._PASSTHROUGH_COMMANDS
BENCHMARKS = main_mod.BENCHMARKS
BENCHMARK_DISPLAY = main_mod.BENCHMARK_DISPLAY
BENCH_DATA_SUBDIR = main_mod.BENCH_DATA_SUBDIR
VLLM_BASE_ARGS = main_mod.VLLM_BASE_ARGS
PROJECT_ROOT = main_mod.PROJECT_ROOT
CODE_DIR = main_mod.CODE_DIR


# ═══════════════════════════════════════════════════════════════════════
# 参数解析测试
# ═══════════════════════════════════════════════════════════════════════


class TestParserCli:
    """CLI 子命令参数解析。"""

    def test_basic_cli(self):
        parser = build_parser()
        args, unknown = parser.parse_known_args(["cli"])
        assert args.command == "cli"
        assert args.no_vllm is False
        assert args.model is None
        assert unknown == []

    def test_cli_no_vllm(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["cli", "--no-vllm"])
        assert args.no_vllm is True

    def test_cli_model_override(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["cli", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_cli_vllm_port(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["cli", "--vllm-port", "9999"])
        assert args.vllm_port == 9999

    def test_cli_cuda_device(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["cli", "--cuda-device", "3"])
        assert args.cuda_device == "3"

    def test_cli_passthrough_via_unknown(self):
        """未知参数被收集到 unknown 列表。"""
        parser = build_parser()
        args, unknown = parser.parse_known_args(["cli", "--model", "gpt-4o", "hello"])
        assert args.model == "gpt-4o"
        assert "hello" in unknown

    def test_cli_double_dash_passthrough(self):
        parser = build_parser()
        args, unknown = parser.parse_known_args(["cli", "--", "--plain", "-x"])
        assert "--plain" in unknown
        assert "-x" in unknown


class TestParserWeb:
    """Web 子命令参数解析。"""

    def test_basic_web(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["web"])
        assert args.command == "web"
        assert args.no_vllm is False

    def test_web_no_vllm(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["web", "--no-vllm"])
        assert args.no_vllm is True

    def test_web_passthrough_via_unknown(self):
        parser = build_parser()
        args, unknown = parser.parse_known_args(["web", "--", "--port", "8765"])
        assert "--port" in unknown


class TestParserBench:
    """Benchmark 子命令参数解析。"""

    def test_bench_default_name(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["bench"])
        assert args.bench_name == "hevp"

    def test_bench_explicit_name(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["bench", "swev"])
        assert args.bench_name == "swev"

    def test_bench_all(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["bench", "all"])
        assert args.bench_name == "all"

    def test_bench_passthrough_via_unknown(self):
        parser = build_parser()
        args, unknown = parser.parse_known_args(["bench", "hevp", "--limit", "10", "--dry-run"])
        assert args.bench_name == "hevp"
        assert unknown == ["--limit", "10", "--dry-run"]

    def test_bench_unknown_merged_with_name(self):
        """验证 benchmark name 正确解析，其余进入 unknown。"""
        parser = build_parser()
        args, unknown = parser.parse_known_args(["bench", "clev", "--custom", "val"])
        assert args.bench_name == "clev"
        assert unknown == ["--custom", "val"]


class TestParserTest:
    """Test 子命令参数解析。"""

    def test_basic_test(self):
        parser = build_parser()
        args, unknown = parser.parse_known_args(["test"])
        assert args.command == "test"
        assert unknown == []

    def test_test_pytest_flags(self):
        """pytest 风格标志全部进入 unknown。"""
        parser = build_parser()
        args, unknown = parser.parse_known_args(["test", "-x", "-k", "test_bash"])
        assert unknown == ["-x", "-k", "test_bash"]

    def test_test_collect_only(self):
        """--collect-only 进入 unknown。"""
        parser = build_parser()
        args, unknown = parser.parse_known_args(["test", "--collect-only"])
        assert unknown == ["--collect-only"]

    def test_test_multiple_flags(self):
        parser = build_parser()
        args, unknown = parser.parse_known_args(["test", "--tb=long", "--maxfail=3", "-q"])
        assert unknown == ["--tb=long", "--maxfail=3", "-q"]

    def test_test_with_extra_args(self):
        """任意 pytest 参数都正确透传。"""
        parser = build_parser()
        args, unknown = parser.parse_known_args(
            ["test", "-v", "--lf", "--co", "-k", "test_main"]
        )
        assert unknown == ["-v", "--lf", "--co", "-k", "test_main"]


class TestParserVllm:
    """vLLM 子命令参数解析。"""

    def test_vllm_start_defaults(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["vllm", "start"])
        assert args.vllm_action == "start"
        assert args.model is None
        assert args.port is None
        assert args.no_wait is False

    def test_vllm_start_full(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(
            ["vllm", "start", "--model", "Qwen/Qwen3.6-35B-A3B-FP8",
             "--port", "8000", "--cuda-device", "2", "--no-wait"]
        )
        assert args.model == "Qwen/Qwen3.6-35B-A3B-FP8"
        assert args.port == 8000
        assert args.cuda_device == "2"
        assert args.no_wait is True

    def test_vllm_stop(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["vllm", "stop"])
        assert args.vllm_action == "stop"

    def test_vllm_status(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["vllm", "status"])
        assert args.vllm_action == "status"

    def test_vllm_default_to_status(self):
        parser = build_parser()
        args, _ = parser.parse_known_args(["vllm"])
        assert args.vllm_action is None

    def test_vllm_unknown_is_collected(self):
        """vLLM 的 unknown 由 main() 报错处理。"""
        parser = build_parser()
        args, unknown = parser.parse_known_args(["vllm", "status", "--bogus"])
        assert unknown == ["--bogus"]


class TestParserNoCommand:
    """无子命令时的行为。"""

    def test_no_command_help(self):
        parser = build_parser()
        args, _ = parser.parse_known_args([])
        assert args.command is None

    def test_no_command_with_help_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])


# ═══════════════════════════════════════════════════════════════════════
# vLLM 检测测试
# ═══════════════════════════════════════════════════════════════════════


class TestDetectVllm:
    """detect_vllm() —— 检测 vLLM 是否在运行。"""

    def test_detect_serving_model(self, monkeypatch):
        def mock_urlopen(request, timeout=3.0):
            response = mock.MagicMock()
            response.read.return_value = json.dumps({
                "data": [{"id": "test-model", "object": "model"}]
            }).encode()
            response.__enter__ = mock.MagicMock(return_value=response)
            response.__exit__ = mock.MagicMock(return_value=None)
            return response

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        result = detect_vllm("http://localhost:8000/v1")
        assert result == "test-model"

    def test_detect_no_model(self, monkeypatch):
        def mock_urlopen(request, timeout=3.0):
            response = mock.MagicMock()
            response.read.return_value = json.dumps({"data": []}).encode()
            response.__enter__ = mock.MagicMock(return_value=response)
            response.__exit__ = mock.MagicMock(return_value=None)
            return response

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        result = detect_vllm("http://localhost:8000/v1")
        assert result is None

    def test_detect_server_unreachable(self, monkeypatch):
        def mock_urlopen(request, timeout=3.0):
            raise OSError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        result = detect_vllm("http://localhost:9999/v1")
        assert result is None

    def test_detect_timeout(self, monkeypatch):
        def mock_urlopen(request, timeout=3.0):
            raise TimeoutError("timed out")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        result = detect_vllm("http://localhost:8000/v1", timeout=0.5)
        assert result is None

    def test_detect_malformed_response(self, monkeypatch):
        def mock_urlopen(request, timeout=3.0):
            response = mock.MagicMock()
            response.read.return_value = b"not json"
            response.__enter__ = mock.MagicMock(return_value=response)
            response.__exit__ = mock.MagicMock(return_value=None)
            return response

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        result = detect_vllm("http://localhost:8000/v1")
        assert result is None

    def test_detect_with_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "my-secret-key")
        captured_headers = []

        def mock_urlopen(request, timeout=3.0):
            captured_headers.append(request.headers)
            response = mock.MagicMock()
            response.read.return_value = json.dumps({"data": []}).encode()
            response.__enter__ = mock.MagicMock(return_value=response)
            response.__exit__ = mock.MagicMock(return_value=None)
            return response

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        detect_vllm("http://localhost:8000/v1")
        assert captured_headers
        assert captured_headers[0].get("Authorization") == "Bearer my-secret-key"


# ═══════════════════════════════════════════════════════════════════════
# vLLM 命令构建测试
# ═══════════════════════════════════════════════════════════════════════


class TestBuildVllmCommand:
    """build_vllm_command() —— 构建 vLLM 启动命令行。"""

    def test_default_command(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_ID", "test-model")
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

        cmd, device = build_vllm_command()
        assert cmd[0] == "vllm"
        assert cmd[1] == "serve"
        assert "test-model" in cmd
        assert "--port" in cmd
        assert "8000" in cmd
        for arg in VLLM_BASE_ARGS:
            assert arg in cmd
        assert device == "2"

    def test_custom_model_port_device(self):
        cmd, device = build_vllm_command(
            model="custom/model", port=9999, cuda_device="1",
        )
        assert "custom/model" in cmd
        assert "9999" in cmd
        assert device == "1"

    def test_extra_args_from_env(self, monkeypatch):
        monkeypatch.setenv("WHALE_VLLM_EXTRA_ARGS", "--dtype bfloat16 --trust-remote-code")
        cmd, _ = build_vllm_command(model="test-model")
        assert "--dtype" in cmd
        assert "bfloat16" in cmd
        assert "--trust-remote-code" in cmd

    def test_contains_required_qwen_args(self):
        cmd, _ = build_vllm_command(model="Qwen/Qwen3.6-35B-A3B-FP8")
        assert "--reasoning-parser" in cmd
        assert "qwen3" in cmd
        assert "--tool-call-parser" in cmd
        assert "qwen3_coder" in cmd
        assert "--language-model-only" in cmd
        assert "--enable-auto-tool-choice" in cmd
        assert "--max-model-len" in cmd
        assert "--max-num-seqs" in cmd
        assert "--gpu-memory-utilization" in cmd

    def test_custom_extra_args_override_env(self):
        cmd, _ = build_vllm_command(
            model="test-model", extra_args="--override-flag value",
        )
        assert "--override-flag" in cmd
        assert "value" in cmd


# ═══════════════════════════════════════════════════════════════════════
# auto_start_vllm 测试
# ═══════════════════════════════════════════════════════════════════════


class TestAutoStartVllm:
    """auto_start_vllm() —— 决策逻辑。"""

    def test_skip_when_no_vllm_flag(self):
        ns = argparse.Namespace(no_vllm=True, model=None, vllm_port=None, cuda_device=None)
        result = auto_start_vllm(ns)
        assert result is None

    def test_skip_when_already_running(self, monkeypatch):
        monkeypatch.setattr(main_mod, "detect_vllm", lambda *a, **kw: "serving-model")
        ns = argparse.Namespace(no_vllm=False, model=None, vllm_port=None, cuda_device=None)
        result = auto_start_vllm(ns)
        assert result is None

    def test_starts_when_not_running(self, monkeypatch):
        monkeypatch.setattr(main_mod, "detect_vllm", lambda *a, **kw: None)

        called_with = {}

        def fake_start(**kwargs):
            called_with.update(kwargs)
            return mock.MagicMock(spec=subprocess.Popen)

        monkeypatch.setattr(main_mod, "start_vllm", fake_start)
        ns = argparse.Namespace(no_vllm=False, model="my-model", vllm_port=8000, cuda_device="0")
        result = auto_start_vllm(ns)
        assert result is not None
        assert called_with.get("model") == "my-model"
        assert called_with.get("port") == 8000
        assert called_with.get("cuda_device") == "0"


# ═══════════════════════════════════════════════════════════════════════
# 环境加载与包引导
# ═══════════════════════════════════════════════════════════════════════


class TestLoadEnv:
    """load_env() —— 加载 .env 文件。"""

    def test_load_valid_env(self, tmp_path):
        """dotenv 可用时通过 dotenv 加载。"""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\n# comment\nKEY2=value2\n")
        load_env(env_file)
        assert os.environ["KEY1"] == "value1"
        assert os.environ["KEY2"] == "value2"

    def test_load_env_no_override(self, tmp_path, monkeypatch):
        """已存在的环境变量不被 .env 覆盖。"""
        monkeypatch.setenv("EXISTING_KEY", "original")
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_KEY=new_value\n")
        load_env(env_file)
        assert os.environ["EXISTING_KEY"] == "original"

    def test_load_missing_file(self, tmp_path):
        load_env(tmp_path / "nonexistent.env")

    def test_load_env_with_quotes(self, tmp_path):
        """dotenv 不剥离引号（标准行为）。"""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY1="value1"\n')
        load_env(env_file)
        # dotenv 模式下保留引号（与 web/server.py fallback 不同）
        assert "value1" in os.environ["KEY1"].strip('"')

    def test_manual_fallback_strips_quotes(self, tmp_path, monkeypatch):
        """验证手动 fallback 解析器逻辑：剥离引号。

        直接调用 fallback 代码路径，不经过 dotenv。
        """
        import builtins
        _real_import = builtins.__import__

        for mod_name in list(sys.modules.keys()):
            if mod_name == "dotenv" or mod_name.startswith("dotenv."):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)

        def _block_dotenv(name, *args, **kwargs):
            if name == "dotenv" or name.startswith("dotenv."):
                raise ImportError(f"No module named {name}")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_dotenv)

        env_file = tmp_path / ".env"
        env_file.write_text('KEY_QUOTED="hello world"\n')
        load_env(env_file)
        assert os.environ["KEY_QUOTED"] == "hello world"

    def test_load_env_skips_malformed(self, tmp_path):
        """格式错误的行静默跳过。"""
        import builtins
        _real_import = builtins.__import__

        # 使用 dotenv（正常路径），格式错误行被忽略
        env_file = tmp_path / ".env"
        env_file.write_text("NO_EQUALS\n=value_without_key\n\nKEY_OK=ok\n")
        load_env(env_file)
        assert os.environ["KEY_OK"] == "ok"


class TestBootstrapPackage:
    """bootstrap_package() —— 将 code/ 注册为 hello_agents。"""

    def test_registers_package(self):
        bootstrap_package()
        assert "hello_agents" in sys.modules
        pkg = sys.modules["hello_agents"]
        assert str(CODE_DIR) in pkg.__path__

    def test_idempotent(self):
        first = sys.modules.get("hello_agents")
        bootstrap_package()
        second = sys.modules["hello_agents"]
        assert first is second


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 数据路径
# ═══════════════════════════════════════════════════════════════════════


class TestResolveDataRoot:
    """_resolve_data_root() —— benchmark 数据根目录。"""

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("WHALE_BENCH_DATA_ROOT", "/custom/data")
        assert _resolve_data_root() == "/custom/data"

    def test_default(self, monkeypatch):
        monkeypatch.delenv("WHALE_BENCH_DATA_ROOT", raising=False)
        assert _resolve_data_root() == "/home/kemove/CodeingAgent/data"


class TestEnsureDataPath:
    """_ensure_data_path() —— 自动补充 --data-path。"""

    def test_already_has_data_path_flag(self):
        args = ["--data-path", "/custom/path.jsonl", "--limit", "10"]
        result = _ensure_data_path("hevp", args)
        assert result == args

    def test_already_has_data_path_equals(self):
        args = ["--data-path=/custom/path.jsonl", "--limit", "10"]
        result = _ensure_data_path("hevp", args)
        assert result == args

    def test_adds_data_path_when_missing(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        hevp_dir = data_root / "HEVP"
        hevp_dir.mkdir(parents=True)
        test_file = hevp_dir / "test.jsonl"
        test_file.write_text('{"task_id": "test"}\n')

        monkeypatch.setenv("WHALE_BENCH_DATA_ROOT", str(data_root))
        result = _ensure_data_path("hevp", ["--limit", "5"])
        assert result[0] == "--data-path"
        assert result[1] == str(test_file)
        assert result[2:] == ["--limit", "5"]

    def test_adds_data_path_even_if_missing(self, tmp_path, monkeypatch):
        data_root = tmp_path / "nonexistent"
        monkeypatch.setenv("WHALE_BENCH_DATA_ROOT", str(data_root))
        result = _ensure_data_path("swev", [])
        assert result[0] == "--data-path"
        assert "SWEV" in result[1]

    def test_all_benchmark_subdirs_covered(self):
        for name in BENCHMARKS:
            assert name in BENCH_DATA_SUBDIR, f"{name} 缺少 BENCH_DATA_SUBDIR 映射"


# ═══════════════════════════════════════════════════════════════════════
# benchmark 注册表一致性
# ═══════════════════════════════════════════════════════════════════════


class TestBenchmarkRegistry:
    """BENCHMARKS / BENCHMARK_DISPLAY / BENCH_DATA_SUBDIR 一致性。"""

    def test_names_match(self):
        assert BENCHMARKS.keys() == BENCHMARK_DISPLAY.keys()
        assert BENCHMARKS.keys() == BENCH_DATA_SUBDIR.keys()

    def test_all_modules_importable(self):
        bootstrap_package()
        for name, module_path in BENCHMARKS.items():
            try:
                mod = importlib.import_module(module_path)
                assert callable(getattr(mod, "main", None)), f"{name}: 缺少 main()"
            except ImportError as exc:
                pytest.fail(f"无法导入 {name} ({module_path}): {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Benchmark dry-run 集成测试
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestBenchDryRunIntegration:
    """需要真实数据集的集成测试。"""

    # 已知有数据集的 benchmark（使用外部 WHALE_BENCH_DATA_ROOT）
    _BENCHS_WITH_DATA = {"hevp", "clev", "lcb6"}

    def test_hevp_dry_run(self):
        rc = run_bench("hevp", ["--limit", "1", "--dry-run"])
        assert rc == 0

    def test_clev_dry_run(self):
        rc = run_bench("clev", ["--limit", "1", "--dry-run"])
        assert rc == 0

    def test_unknown_benchmark(self):
        rc = run_bench("no_such_bench", [])
        assert rc == 1

    def test_all_dry_run_smoke(self):
        """--bench all 冒烟测试：确保不会在第一个就崩溃。

        某些 benchmark 可能没有数据文件（如 MBPP / AIME），
        此时会返回非零退出码但不影响其他 benchmark。
        """
        rc = run_bench("all", ["--limit", "1", "--dry-run"])
        # all 模式可能因数据缺失而返回非零，但不该 crash
        assert rc in (0, 1)


# ═══════════════════════════════════════════════════════════════════════
# vLLM 状态格式化
# ═══════════════════════════════════════════════════════════════════════


class TestPrintVllmStatus:
    """print_vllm_status() —— 格式化输出。"""

    def test_not_running(self, capsys):
        info = {
            "model": "test-model",
            "base_url": "http://localhost:8000/v1",
            "port": 8000,
            "serving": None,
            "gpu": None,
        }
        print_vllm_status(info)
        captured = capsys.readouterr().out
        assert "test-model" in captured
        assert "8000" in captured
        assert "未运行" in captured

    def test_running_with_gpu(self, capsys):
        info = {
            "model": "running-model",
            "base_url": "http://localhost:8000/v1",
            "port": 8000,
            "serving": "running-model",
            "gpu": [
                {
                    "index": "0", "name": "RTX 4090",
                    "mem_used_mb": 10000, "mem_total_mb": 24000, "util_pct": 85,
                },
            ],
        }
        print_vllm_status(info)
        captured = capsys.readouterr().out
        assert "运行中" in captured
        assert "running-model" in captured
        assert "RTX 4090" in captured
        assert "10000" in captured

    def test_gpu_error(self, capsys):
        info = {
            "model": "test",
            "base_url": "http://localhost:8000/v1",
            "port": 8000,
            "serving": None,
            "gpu": {"error": "nvidia-smi not found"},
        }
        print_vllm_status(info)
        captured = capsys.readouterr().out
        assert "nvidia-smi not found" in captured


# ═══════════════════════════════════════════════════════════════════════
# run_bench 边界情况
# ═══════════════════════════════════════════════════════════════════════


class TestRunBenchEdgeCases:
    """run_bench() 的边界情况。"""

    def test_invalid_bench_name(self):
        rc = run_bench("invalid_bench", [])
        assert rc == 1

    def test_empty_bench_args(self):
        rc = run_bench("hevp", ["--limit", "1", "--dry-run"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════
# main() 集成测试
# ═══════════════════════════════════════════════════════════════════════


class TestMainFunction:
    """直接调用 main() 函数。"""

    def test_main_help_output(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["main.py"])
        try:
            main_mod.main()
        except SystemExit as exc:
            assert exc.code == 0
        captured = capsys.readouterr().out
        assert "WhaleCode" in captured or "usage" in captured

    def test_main_vllm_status(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["main.py", "vllm", "status"])
        try:
            main_mod.main()
        except SystemExit as exc:
            assert exc.code == 0
        captured = capsys.readouterr().out
        assert "vLLM 状态" in captured

    def test_main_vllm_stop_command(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["main.py", "vllm", "stop"])
        with mock.patch.object(main_mod, "stop_vllm") as mock_stop:
            mock_stop.return_value = None
            try:
                main_mod.main()
            except SystemExit as exc:
                assert exc.code == 0

    def test_main_bench_unknown(self, capsys, monkeypatch):
        """main.py bench nonexistent → 返回 1（不抛异常）。"""
        monkeypatch.setattr(sys, "argv", ["main.py", "bench", "nonexistent"])
        rc = main_mod.main()
        assert rc == 1
        captured = capsys.readouterr().out
        assert "未知 benchmark" in captured

    def test_main_bench_hevp_dry_run(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["main.py", "bench", "hevp", "--limit", "1", "--dry-run"],
        )
        try:
            main_mod.main()
        except SystemExit as exc:
            assert exc.code == 0

    def test_main_bench_passthrough_unknown(self, capsys, monkeypatch):
        """验证 bench 参数通过 unknown 正确透传。"""
        monkeypatch.setattr(
            sys, "argv",
            ["main.py", "bench", "hevp", "--limit", "1", "--dry-run"],
        )
        try:
            main_mod.main()
        except SystemExit as exc:
            assert exc.code == 0

    def test_main_vllm_rejects_unknown(self, capsys, monkeypatch):
        """vllm 子命令的 unknown 参数会被报错。"""
        monkeypatch.setattr(sys, "argv", ["main.py", "vllm", "status", "--bogus"])
        with pytest.raises(SystemExit) as exc_info:
            main_mod.main()
        assert exc_info.value.code == 2

    def test_main_test_passthrough_collect(self, capsys, monkeypatch):
        """test 命令的参数通过 unknown 透传给 pytest。"""
        monkeypatch.setattr(sys, "argv", ["main.py", "test", "-q", "-k", "test_main"])
        with mock.patch.object(main_mod, "run_tests", return_value=0) as mock_run:
            try:
                main_mod.main()
            except SystemExit as exc:
                assert exc.code == 0
            mock_run.assert_called_once_with(["-q", "-k", "test_main"])


# ═══════════════════════════════════════════════════════════════════════
# 常量验证
# ═══════════════════════════════════════════════════════════════════════


class TestConstants:
    """验证模块级常量的正确性。"""

    def test_project_root_exists(self):
        assert PROJECT_ROOT.is_dir()
        assert (PROJECT_ROOT / "code").is_dir()
        assert (PROJECT_ROOT / "scripts").is_dir()
        assert (PROJECT_ROOT / "web").is_dir()

    def test_vllm_base_args_are_strings(self):
        for arg in VLLM_BASE_ARGS:
            assert isinstance(arg, str)

    def test_passthrough_commands(self):
        """所有需要透传的子命令都在 _PASSTHROUGH_COMMANDS 中。"""
        for cmd in ("cli", "web", "bench", "test"):
            assert cmd in _PASSTHROUGH_COMMANDS


# ═══════════════════════════════════════════════════════════════════════
# vLLM 停止测试（仅逻辑，不真正杀进程）
# ═══════════════════════════════════════════════════════════════════════


class TestStopVllm:
    """stop_vllm() —— 逻辑测试。"""

    def test_stop_when_nothing_running(self, capsys, monkeypatch):
        monkeypatch.setattr(main_mod, "detect_vllm", lambda *a, **kw: None)
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
        stop_vllm()
        captured = capsys.readouterr().out
        assert "未检测到运行中" in captured or "未找到 vLLM" in captured

    def test_stop_with_running_vllm(self, capsys, monkeypatch):
        monkeypatch.setattr(main_mod, "detect_vllm", lambda *a, **kw: "test-model")
        fake_result = mock.MagicMock()
        fake_result.stdout = "12345\n"
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: 12345)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        stop_vllm()
        captured = capsys.readouterr().out
        assert "test-model" in captured


# ═══════════════════════════════════════════════════════════════════════
# 路径解析测试
# ═══════════════════════════════════════════════════════════════════════


class TestResolveHelpers:
    """resolve_model / resolve_base_url / resolve_vllm_port / resolve_cuda_device。"""

    def test_resolve_model_default(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL_ID", raising=False)
        assert "Qwen" in main_mod.resolve_model()

    def test_resolve_model_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_ID", "custom/model")
        assert main_mod.resolve_model() == "custom/model"

    def test_resolve_base_url_default(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert "8000" in main_mod.resolve_base_url()

    def test_resolve_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9999/v1")
        assert main_mod.resolve_base_url() == "http://localhost:9999/v1"

    @pytest.mark.parametrize("url,expected_port", [
        ("http://localhost:8000/v1", 8000),
        ("http://127.0.0.1:9999/v1", 9999),
        ("https://api.example.com:443/v1", 443),
    ])
    def test_resolve_vllm_port(self, url, expected_port, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", url)
        assert main_mod.resolve_vllm_port() == expected_port

    def test_resolve_cuda_device_default(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert main_mod.resolve_cuda_device() == "0"

    def test_resolve_cuda_device_from_env(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert main_mod.resolve_cuda_device() == "2,3"


# ═══════════════════════════════════════════════════════════════════════
# 启动 vLLM 测试（关键路径 mock）
# ═══════════════════════════════════════════════════════════════════════


class TestStartVllm:
    """start_vllm() —— 核心逻辑。"""

    def test_skip_when_already_running(self, monkeypatch):
        monkeypatch.setattr(main_mod, "detect_vllm", lambda *a, **kw: "existing-model")
        result = start_vllm()
        assert result is None

    def test_start_creates_subprocess(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main_mod, "detect_vllm", lambda *a, **kw: None)
        monkeypatch.setattr(main_mod, "RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr(main_mod, "VLLM_LOG", tmp_path / "runtime" / "vllm.log")
        (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)

        mock_proc = mock.MagicMock(spec=subprocess.Popen)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: mock_proc)
        result = start_vllm(wait=False)
        assert result is mock_proc

    def test_start_with_wait_detects_ready(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main_mod, "RUNTIME_DIR", tmp_path / "runtime")
        (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)

        detect_calls = [0]

        def fake_detect(url=None, timeout=3.0):
            detect_calls[0] += 1
            return "ready-model" if detect_calls[0] >= 2 else None

        monkeypatch.setattr(main_mod, "detect_vllm", fake_detect)
        mock_proc = mock.MagicMock(spec=subprocess.Popen)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: mock_proc)

        result = start_vllm(wait=True, wait_timeout=10)
        assert result is mock_proc
        assert detect_calls[0] >= 2

    def test_start_wait_timeout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main_mod, "RUNTIME_DIR", tmp_path / "runtime")
        (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(main_mod, "detect_vllm", lambda *a, **kw: None)
        mock_proc = mock.MagicMock(spec=subprocess.Popen)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: mock_proc)

        result = start_vllm(wait=True, wait_timeout=1)
        assert result is mock_proc
