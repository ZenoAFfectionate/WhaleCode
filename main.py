#!/usr/bin/env python3
"""
WhaleCode 统一入口 —— 一个命令管理 CLI / Web / Benchmark / 测试 / vLLM。

用法::

    python main.py cli [--no-vllm] [-- cli-args ...]
    python main.py web [--no-vllm] [-- web-args ...]
    python main.py bench <name> [bench-args ...]
    python main.py test [pytest-args ...]
    python main.py vllm start|stop|status

示例::

    python main.py cli
    python main.py cli --no-vllm -- --model gpt-4o "帮我写一个函数"
    python main.py web -- --port 8765
    python main.py bench hevp --limit 10 --dry-run
    python main.py bench all
    python main.py test -k test_bash -x
    python main.py vllm start
    python main.py vllm start --cuda-device 1
    python main.py vllm status

vLLM 自动检测:
    CLI 和 Web 模式启动前会自动检测 vLLM 是否在运行；未运行则自动启动。
    使用 ``--no-vllm`` 跳过自动启动（连接远程 API 时）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
# 路径常量
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WEB_DIR = PROJECT_ROOT / "web"
RUNTIME_DIR = WEB_DIR / "runtime"
VLLM_LOG = RUNTIME_DIR / "vllm.log"
ENV_FILE = PROJECT_ROOT / ".env"

# ═══════════════════════════════════════════════════════════════════════
# Benchmark 注册表
# ═══════════════════════════════════════════════════════════════════════

BENCHMARKS: dict[str, str] = {
    "hevp": "hello_agents.benchmark.hevp_bench",
    "clev": "hello_agents.benchmark.clev_bench",
    "aime": "hello_agents.benchmark.aime_bench",
    "lcb6": "hello_agents.benchmark.lcb6_bench",
    "mbpp": "hello_agents.benchmark.mbpp_bench",
    "swev": "hello_agents.benchmark.swev_bench",
}

BENCHMARK_DISPLAY: dict[str, str] = {
    "hevp": "HumanEval+",
    "clev": "ClassEval",
    "aime": "AIME",
    "lcb6": "LiveCodeBench v6",
    "mbpp": "MBPP+",
    "swev": "SWE-bench Verified",
}

# Benchmark → 数据集子目录名（与 WHALE_BENCH_DATA_ROOT 默认的 PROJECT_ROOT/data/ 下一致）
BENCH_DATA_SUBDIR: dict[str, str] = {
    "hevp": "HEVP",
    "clev": "CLEV",
    "aime": "AIME",
    "lcb6": "LCB6",
    "mbpp": "MBPP",
    "swev": "SWEV",
}

# ═══════════════════════════════════════════════════════════════════════
# vLLM 默认启动参数（与 README 中 Qwen3.6-35B 推荐配置一致）
# ═══════════════════════════════════════════════════════════════════════

VLLM_BASE_ARGS = [
    "--max-model-len", "262144",
    "--max-num-seqs", "2",
    "--gpu-memory-utilization", "0.95",
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_coder",
    "--language-model-only",
    "--enable-auto-tool-choice",
]


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════


def load_env(env_path: Path = ENV_FILE) -> None:
    """加载项目 .env 文件。

    优先使用 python-dotenv；不可用时退化为手动逐行解析
    （与 ``web/server.py`` 的 load_project_env 行为一致）。
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv(env_path, override=False)
    except ImportError:
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def bootstrap_package() -> None:
    """将 ``code/`` 目录注册为 ``hello_agents`` 包。

    与 ``scripts/cli.py`` 及 ``web/server.py`` 中的同名函数逻辑一致。
    """
    if "hello_agents" in sys.modules:
        return
    pkg = types.ModuleType("hello_agents")
    pkg.__path__ = [str(CODE_DIR)]
    pkg.__file__ = str(CODE_DIR / "__init__.py")
    sys.modules["hello_agents"] = pkg


def resolve_model() -> str:
    """返回当前配置的模型名。"""
    return os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.6-35B-A3B-FP8")


def resolve_base_url() -> str:
    """返回当前 LLM base URL。"""
    return os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")


def resolve_vllm_port() -> int:
    """从 LLM_BASE_URL 解析端口号。"""
    url = resolve_base_url()
    try:
        # 形如 http://localhost:8000/v1
        port_str = url.rsplit(":", 1)[-1]  # "8000/v1"
        port_str = port_str.split("/")[0]  # "8000"
        return int(port_str)
    except (ValueError, IndexError):
        return 8000


def resolve_cuda_device() -> str:
    """返回 CUDA 设备号（从环境变量或默认 0）。"""
    return os.getenv("CUDA_VISIBLE_DEVICES", "0")


# ═══════════════════════════════════════════════════════════════════════
# vLLM 检测
# ═══════════════════════════════════════════════════════════════════════


def _models_url(base_url: str | None = None) -> str | None:
    """拼接 OpenAI 兼容的 ``/v1/models`` 地址。"""
    url = (base_url or resolve_base_url()).rstrip("/")
    if url.endswith("/v1"):
        return url + "/models"
    # 处理非标准 URL（如 /v1 已在路径中；或没有 v1 前缀）
    if "/v1" in url:
        return url + "/models"
    return url + "/v1/models"


def detect_vllm(base_url: str | None = None, timeout: float = 3.0) -> str | None:
    """检测 vLLM 是否在运行并返回模型 ID，不可达时返回 None。

    通过调用 OpenAI-compatible ``/v1/models`` 端点实现，
    与 ``web/server.py`` 中 ``detect_served_model`` 一致。
    """
    url = _models_url(base_url)
    if not url:
        return None
    try:
        req = urllib.request.Request(url)
        api_key = os.getenv("LLM_API_KEY", "vllm")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    return None


# ═══════════════════════════════════════════════════════════════════════
# vLLM 启动 / 停止 / 状态
# ═══════════════════════════════════════════════════════════════════════


def build_vllm_command(
    model: str | None = None,
    port: int | None = None,
    cuda_device: str | None = None,
    extra_args: str | None = None,
) -> tuple[list[str], str]:
    """构建 vLLM 命令行。

    Returns:
        (cmd_list, cuda_device) —— cmd_list 可直接传给 Popen，
        cuda_device 用于设置 ``CUDA_VISIBLE_DEVICES`` 环境变量。
    """
    model = model or resolve_model()
    port = port or resolve_vllm_port()
    device = cuda_device or resolve_cuda_device()

    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
    ]
    # 合并基础参数
    cmd.extend(VLLM_BASE_ARGS)

    # 追加环境变量中的额外参数
    extra = extra_args or os.getenv("WHALE_VLLM_EXTRA_ARGS", "")
    if extra:
        cmd.extend(extra.split())

    return cmd, device


def start_vllm(
    model: str | None = None,
    port: int | None = None,
    cuda_device: str | None = None,
    *,
    wait: bool = True,
    wait_timeout: int = 300,
) -> subprocess.Popen | None:
    """启动 vLLM 服务。

    Args:
        model: 模型名（默认从 .env 读取 LLM_MODEL_ID）。
        port: 服务端口（默认从 LLM_BASE_URL 解析）。
        cuda_device: CUDA 设备号（默认 CUDA_VISIBLE_DEVICES 或 "0"）。
        wait: 是否轮询等待 vLLM 就绪。
        wait_timeout: 最大等待秒数（默认 300s，大模型加载可能较慢）。

    Returns:
        新启动的 Popen 对象；如果已有 vLLM 在运行则返回 None。
    """
    serving = detect_vllm()
    if serving:
        print(f"✓ vLLM 已在运行，模型: {serving}")
        return None

    cmd, device = build_vllm_command(model, port, cuda_device)
    effective_port = port or resolve_vllm_port()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log_fp = open(str(VLLM_LOG), "a", encoding="utf-8")  # noqa: SIM115

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device

    print(f"⚡ 启动 vLLM (CUDA:{device} port:{effective_port})")
    print(f"   模型: {model or resolve_model()}")
    print(f"   命令: {' '.join(cmd)}")
    print(f"   日志: {VLLM_LOG}")
    print()

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    if wait:
        base_url = f"http://localhost:{effective_port}/v1"
        print("⏳ 等待 vLLM 就绪 ...", end="", flush=True)
        deadline = time.monotonic() + wait_timeout
        ready = False
        while time.monotonic() < deadline:
            if detect_vllm(base_url, timeout=2.0):
                ready = True
                break
            print(".", end="", flush=True)
            time.sleep(3)
        if ready:
            print(" ✓")
        else:
            print(f"\n⚠ 等待超时 ({wait_timeout}s)，vLLM 可能仍在加载。")
            print(f"   检查日志: tail -f {VLLM_LOG}")

    return proc


def stop_vllm() -> None:
    """停止所有 vLLM 进程。"""
    serving = detect_vllm()
    if serving:
        print(f"检测到 vLLM 模型: {serving}")
    else:
        print("未检测到运行中的 vLLM 服务。")

    found = False
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm serve"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(pid) for pid in result.stdout.strip().splitlines() if pid.strip()]
    except (FileNotFoundError, ValueError):
        pids = []

    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"  SIGTERM → PID {pid}")
            found = True
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"  SIGTERM → PID {pid}")
                found = True
            except (ProcessLookupError, PermissionError):
                pass

    if found:
        time.sleep(2)
        for pid in pids:
            try:
                os.kill(pid, 0)  # 检查是否还活着
            except (ProcessLookupError, PermissionError):
                continue
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                print(f"  SIGKILL → PID {pid}")
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    if not found:
        print("未找到 vLLM 进程 (pgrep 不可用或无匹配)。")


def vllm_status() -> dict:
    """返回 vLLM + GPU 状态快照。"""
    info: dict = {
        "model": resolve_model(),
        "base_url": resolve_base_url(),
        "port": resolve_vllm_port(),
        "serving": None,
        "gpu": None,
    }

    serving = detect_vllm()
    info["serving"] = serving

    # GPU 状态（通过 nvidia-smi）
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True, timeout=5,
            )
            gpus = []
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": parts[0],
                        "name": parts[1],
                        "mem_used_mb": int(parts[2]),
                        "mem_total_mb": int(parts[3]),
                        "util_pct": int(parts[4]),
                    })
            info["gpu"] = gpus
        except Exception as exc:
            info["gpu"] = {"error": str(exc)}

    return info


def print_vllm_status(info: dict) -> None:
    """格式化打印 vLLM 状态。"""
    print()
    print("═══ vLLM 状态 ═══")
    print(f"  模型:      {info['model']}")
    print(f"  Base URL:  {info['base_url']}")
    if info["serving"]:
        print(f"  状态:      ✓ 运行中 ({info['serving']})")
    else:
        print(f"  状态:      ✗ 未运行")

    gpu = info.get("gpu")
    if isinstance(gpu, list) and gpu:
        print("─── GPU ───")
        for g in gpu:
            mem_pct = (g["mem_used_mb"] / g["mem_total_mb"] * 100) if g["mem_total_mb"] else 0
            print(
                f"  GPU {g['index']}: {g['name']}  "
                f"显存 {g['mem_used_mb']}/{g['mem_total_mb']} MB ({mem_pct:.0f}%)  "
                f"利用率 {g['util_pct']}%"
            )
    elif isinstance(gpu, dict) and "error" in gpu:
        print(f"  GPU:       nvidia-smi 错误: {gpu['error']}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# vLLM 自动启动（CLI / Web 模式共用）
# ═══════════════════════════════════════════════════════════════════════


def auto_start_vllm(args: argparse.Namespace) -> subprocess.Popen | None:
    """根据命令行参数决定是否自动启动 vLLM。

    ``--no-vllm`` 时跳过；已运行时打印状态并跳过。
    """
    if getattr(args, "no_vllm", False):
        print("ℹ --no-vllm: 跳过 vLLM 自动启动")
        return None

    serving = detect_vllm()
    if serving:
        print(f"✓ vLLM 已在运行: {serving}")
        return None

    print("⚡ vLLM 未运行，自动启动 ...")
    return start_vllm(
        model=getattr(args, "model", None),
        port=getattr(args, "vllm_port", None),
        cuda_device=getattr(args, "cuda_device", None),
        wait=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# 模式：CLI
# ═══════════════════════════════════════════════════════════════════════


def run_cli(cli_args: list[str]) -> int:
    """启动交互式 CLI。

    通过 subprocess 运行 ``scripts/cli.py``，stdin/stdout 透传，
    保证交互式体验不受影响。
    """
    cli_script = SCRIPTS_DIR / "cli.py"
    cmd = [sys.executable, str(cli_script)] + cli_args
    print(f"▶ 启动 CLI: {' '.join(cmd)}")
    print()
    try:
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\nCLI 已退出。")
        return 0


# ═══════════════════════════════════════════════════════════════════════
# 模式：Web
# ═══════════════════════════════════════════════════════════════════════


def run_web(web_args: list[str]) -> int:
    """启动 Web 控制台。

    通过 subprocess 运行 ``web/server.py``，stdin/stdout 透传。
    """
    web_script = WEB_DIR / "server.py"
    cmd = [sys.executable, str(web_script)] + web_args
    print(f"▶ 启动 Web: {' '.join(cmd)}")
    print()
    try:
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\nWeb 服务已停止。")
        return 0


# ═══════════════════════════════════════════════════════════════════════
# 模式：Benchmark
# ═══════════════════════════════════════════════════════════════════════


def _resolve_data_root() -> str:
    """返回 benchmark 数据集根目录。

    优先级：环境变量 ``WHALE_BENCH_DATA_ROOT`` → 默认路径
    ``PROJECT_ROOT / "data"``（与所有 shell 脚本的默认值一致）。
    """
    return os.getenv("WHALE_BENCH_DATA_ROOT", str(PROJECT_ROOT / "data"))


def _ensure_data_path(bench_name: str, bench_args: list[str]) -> list[str]:
    """当用户未显式提供 ``--data-path`` 时，自动从数据集根目录补充。"""
    has_data_path = any(
        arg == "--data-path" or arg.startswith("--data-path=")
        for arg in bench_args
    )
    if has_data_path:
        return bench_args

    data_root = _resolve_data_root()
    subdir = BENCH_DATA_SUBDIR.get(bench_name, bench_name.upper())
    data_file = Path(data_root) / subdir / "test.jsonl"
    if data_file.exists():
        return ["--data-path", str(data_file)] + bench_args

    # 文件不存在 —— 仍然传参，让 benchmark 自己报错
    return ["--data-path", str(data_file)] + bench_args


def run_bench(bench_name: str, bench_args: list[str]) -> int:
    """运行指定 benchmark。

    Args:
        bench_name: benchmark 标识（如 hevp / clev / all）。
        bench_args: 透传给 benchmark 模块的额外参数。
    """
    if bench_name == "all":
        return _run_all_benchmarks(bench_args)

    if bench_name not in BENCHMARKS:
        print(f"✗ 未知 benchmark: {bench_name}")
        print(f"  可用: {', '.join(BENCHMARKS.keys())}, all")
        return 1

    module_path = BENCHMARKS[bench_name]
    display_name = BENCHMARK_DISPLAY.get(bench_name, bench_name)

    print(f"═══ {display_name} Benchmark ═══")
    print()

    # 自动补充 data-path
    effective_args = _ensure_data_path(bench_name, bench_args)

    bootstrap_package()

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        print(f"✗ 无法导入 {module_path}: {exc}")
        return 1

    if not callable(getattr(mod, "main", None)):
        print(f"✗ {module_path} 没有 main() 入口")
        return 1

    saved_argv = sys.argv.copy()
    try:
        sys.argv = [f"{bench_name}_bench"] + effective_args
        mod.main()
        return 0
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1
    except Exception as exc:
        print(f"✗ {display_name} benchmark 异常: {exc}")
        return 1
    except KeyboardInterrupt:
        print(f"\n{display_name} benchmark 已中断。")
        return 130
    finally:
        sys.argv = saved_argv


def _run_all_benchmarks(bench_args: list[str]) -> int:
    """按顺序运行所有 benchmark。"""
    all_names = list(BENCHMARKS.keys())
    print(f"═══ 全部 Benchmark: {', '.join(all_names)} ═══")
    print()

    overall = 0
    for i, name in enumerate(all_names):
        display = BENCHMARK_DISPLAY.get(name, name)
        print(f"\n{'─' * 60}")
        print(f"  [{i + 1}/{len(all_names)}] {display}")
        print(f"{'─' * 60}\n")
        rc = run_bench(name, bench_args)
        if rc != 0:
            print(f"\n⚠ {name} 返回非零退出码: {rc}")
            overall = 1

    if overall == 0:
        print("\n✓ 全部 benchmark 完成。")
    return overall


# ═══════════════════════════════════════════════════════════════════════
# 模式：测试
# ═══════════════════════════════════════════════════════════════════════


def run_tests(test_args: list[str]) -> int:
    """运行 pytest。

    透传所有参数给 pytest，如 ``-k``, ``-x``, ``--lf`` 等。
    """
    cmd = [sys.executable, "-m", "pytest"] + test_args
    print(f"▶ 运行测试: {' '.join(cmd)}")
    print()
    try:
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n测试已中断。")
        return 130


# ═══════════════════════════════════════════════════════════════════════
# 主入口与参数解析
# ═══════════════════════════════════════════════════════════════════════


_PASSTHROUGH_COMMANDS = frozenset({"cli", "web", "bench", "test"})


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="WhaleCode 统一入口 —— CLI / Web / Benchmark / 测试 / vLLM 管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python main.py cli                             交互式 CLI
  python main.py cli --no-vllm                   跳过 vLLM 自动启动
  python main.py web -- --port 8765              Web 控制台（自定义端口）
  python main.py bench hevp --limit 10           HumanEval+ 前 10 题
  python main.py bench all --dry-run             全部 benchmark (dry run)
  python main.py test -k test_bash -x            运行名称包含 test_bash 的测试
  python main.py vllm status                     查看 vLLM 和 GPU 状态
        """,
    )

    sub = parser.add_subparsers(dest="command", help="可用命令")

    # ---- cli ----
    cli_p = sub.add_parser(
        "cli", help="启动交互式 CLI（自动检测/启动 vLLM）",
        description="启动后，所有未识别的参数将透传给 scripts/cli.py。",
    )
    cli_p.add_argument("--no-vllm", action="store_true", help="跳过 vLLM 自动启动")
    cli_p.add_argument("--model", default=None, help="覆盖 LLM_MODEL_ID")
    cli_p.add_argument("--vllm-port", type=int, default=None, help="vLLM 端口（默认从 LLM_BASE_URL 解析）")
    cli_p.add_argument("--cuda-device", default=None, help="CUDA 设备号（默认 CUDA_VISIBLE_DEVICES 或 0）")

    # ---- web ----
    web_p = sub.add_parser(
        "web", help="启动 Web 控制台（自动检测/启动 vLLM）",
        description="启动后，所有未识别的参数将透传给 web/server.py。",
    )
    web_p.add_argument("--no-vllm", action="store_true", help="跳过 vLLM 自动启动")
    web_p.add_argument("--model", default=None, help="覆盖 LLM_MODEL_ID")
    web_p.add_argument("--vllm-port", type=int, default=None, help="vLLM 端口（默认从 LLM_BASE_URL 解析）")
    web_p.add_argument("--cuda-device", default=None, help="CUDA 设备号（默认 CUDA_VISIBLE_DEVICES 或 0）")

    # ---- bench ----
    bench_p = sub.add_parser(
        "bench", help="运行 benchmark",
        description="运行指定的 benchmark。所有未识别的参数将透传给 benchmark 模块。",
    )
    bench_p.add_argument(
        "bench_name", nargs="?", default="hevp",
        help=f"Benchmark 名称: {', '.join(BENCHMARKS.keys())}, all（默认: hevp）",
    )

    # ---- test ----
    test_p = sub.add_parser(
        "test", help="运行测试 (pytest)",
        description="所有未识别的参数将透传给 pytest。",
    )

    # ---- vllm ----
    vllm_p = sub.add_parser("vllm", help="管理 vLLM 服务")
    vllm_sp = vllm_p.add_subparsers(dest="vllm_action", help="操作")

    vllm_start = vllm_sp.add_parser("start", help="启动 vLLM")
    vllm_start.add_argument("--model", default=None, help="模型名（覆盖 .env 中的 LLM_MODEL_ID）")
    vllm_start.add_argument("--port", type=int, default=None, help="服务端口（默认 8000）")
    vllm_start.add_argument("--cuda-device", default=None, help="CUDA 设备号")
    vllm_start.add_argument("--no-wait", action="store_true", help="不等待 vLLM 就绪即返回")

    vllm_sp.add_parser("stop", help="停止 vLLM")
    vllm_sp.add_parser("status", help="查看 vLLM 与 GPU 状态")

    return parser


def main() -> int:
    """WhaleCode 统一入口。"""
    load_env()
    parser = build_parser()

    # 使用 parse_known_args：所有子命令中不认识的参数（如 pytest 的 -x -k）
    # 都被收集到 unknown 中，由各模式函数作为 "passthrough args" 接收。
    args, unknown = parser.parse_known_args()

    # argparse 会把用户写的 `--` 分隔符原样留在 unknown 中，但子进程/子模块
    # 自己的 argparse 无法处理它（会把后面的选项误当位置参数），这里剥掉。
    if "--" in unknown:
        unknown.remove("--")

    # vllm 命令不接收透传参数
    if unknown and args.command == "vllm":
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    # ── vllm ──
    if args.command == "vllm":
        if args.vllm_action == "start":
            proc = start_vllm(
                model=getattr(args, "model", None),
                port=getattr(args, "port", None),
                cuda_device=getattr(args, "cuda_device", None),
                wait=not getattr(args, "no_wait", False),
            )
            if proc is not None and not getattr(args, "no_wait", False):
                print("✓ vLLM 启动成功并可响应请求。")
            return 0
        elif args.vllm_action == "stop":
            stop_vllm()
            return 0
        elif args.vllm_action == "status":
            print_vllm_status(vllm_status())
            return 0
        else:
            # 无子命令 → 显示状态
            print_vllm_status(vllm_status())
            return 0

    # ── cli ──
    elif args.command == "cli":
        auto_start_vllm(args)
        return run_cli(unknown)

    # ── web ──
    elif args.command == "web":
        auto_start_vllm(args)
        return run_web(unknown)

    # ── bench ──
    elif args.command == "bench":
        return run_bench(args.bench_name, unknown)

    # ── test ──
    elif args.command == "test":
        return run_tests(unknown)

    # ── 无子命令 → 帮助 ──
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
