"""Performance benchmarks (IMPROVEMENT.md B5) — pytest-benchmark 基准.

设计偏离说明 (相对原计划): 原计划为"包装 test_stress_*.py 的场景函数",
但 stress 场景依赖完整 agent fixture, 多次重复执行会显著拖慢测试套件;
本文件改为对**关键纯函数热点**建立独立基准 (数据构造一次性完成,
被测代码零改动), 与 stress 测试的"压不坏"目标互补, 回答"快不快".

运行方式:
    pytest tests/test_benchmarks.py                 # 单次运行, 输出统计
    pytest tests/test_benchmarks.py --benchmark-autosave  \
        --benchmark-compare  # 与上次结果对比, 捕捉性能回归

未安装 pytest-benchmark 时整个文件优雅跳过 (importorskip).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed (B5 optional dep)")

from hello_agents.tools.builtin._code_utils import (
    detect_line_ending,
    normalize_line_endings,
    replace_with_flexible_match,
    resolve_path,
)
from hello_agents.tools.builtin.todowrite_tool import TodoSessionStore

_ROOT = "/srv/bench-proj"
_DIR = "/srv/bench-proj/pkg/mod"


def _make_text(size_kb: int, seed: int = 42) -> str:
    """确定性伪随机文本 (避免随机抖动干扰基准)."""
    import random

    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "function", "return", "class"]
    lines = []
    total = 0
    while total < size_kb * 1024:
        line = " ".join(rng.choice(words) for _ in range(rng.randint(4, 10)))
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _make_todos(count: int) -> list[dict]:
    priorities = ["high", "medium", "low"]
    # 仅第 1 项 in_progress (规则: 同时至多一个); 其余在合法状态间轮换
    non_progress = ["pending", "completed", "cancelled"]
    return [
        {
            "content": f"task-{i}: implement feature module {i}",
            "status": "in_progress" if i == 1 else non_progress[i % 3],
            "priority": priorities[i % 3],
        }
        for i in range(count)
    ]


class TestCoreUtilsBenchmarks:
    def test_replace_exact_unique(self, benchmark):
        """Edit 热路径: 大文本中唯一匹配的 exact 替换."""
        content = _make_text(200) + "\nUNIQUE-BENCHMARK-MARKER-LINE\n"
        result = benchmark(
            replace_with_flexible_match, content, "UNIQUE-BENCHMARK-MARKER-LINE", "replaced"
        )
        assert result.replacements == 1
        assert result.strategy == "exact"

    def test_replace_exact_all(self, benchmark):
        """replace_all 全量替换 + 非重叠计数."""
        content = _make_text(200)
        old = "alpha"
        if content.count(old) == 0:
            content = "alpha " + content
        result = benchmark(
            lambda: replace_with_flexible_match(content, old, "x-alpha", replace_all=True)
        )
        assert result.replacements == content.count(old)

    def test_resolve_path_throughput(self, benchmark):
        """路径解析吞吐: 500 次/轮 (单次 ~µs 级, 需放大才有统计意义)."""
        def _run_batch():
            for i in range(500):
                resolve_path(_ROOT, _DIR, f"../mod/file_{i % 7}.py")

        benchmark(_run_batch)

    def test_normalize_todos_large_list(self, benchmark):
        """TodoWrite 清洗: 200 项列表."""
        todos = _make_todos(200)
        result = benchmark(TodoSessionStore._normalize_todos, todos)
        assert len(result) == 200

    def test_line_ending_detection(self, benchmark):
        """行尾检测 + 归一化: 1MB 混合行尾文本."""
        text = _make_text(1024)
        mixed = text.replace("\n", "\r\n", 100)
        assert benchmark(detect_line_ending, mixed) == "\r\n"
