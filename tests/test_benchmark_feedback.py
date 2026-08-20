"""Benchmark evaluator-feedback bounding tests (统一反馈截断防线).

背景: 评测器输出 (stdout+stderr 合并, 无天然上限) 曾直接经
``_run_controlled_submission_rounds`` 注入下一轮 retry prompt —
hevp/mbpp 的失败诊断 (大 repr / 完整 traceback) 可达数十 KB~MB 级,
直接撑爆 LLM 上下文; lcb6 有自己的 ``truncate_feedback`` 防线,
其余 bench 没有, 存在系统性防护缺口与跨 bench 不公平。

修复 (第 15 轮): ``base.py`` 的受控提交循环在注入 feedback 前统一
调用 ``truncate_feedback`` (默认 80 行 / 12000 字符, 与 lcb6 一致,
可经参数覆盖)。本文件锁定该契约。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from hello_agents.benchmark.base import BenchmarkRunner
from hello_agents.benchmark._utils import truncate_feedback


class _LoopHarness:
    """最小 runner 桩: 只实现受控提交循环依赖的 ``_run_agent_prompt``."""

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["resp"] * 10
        self.seen_prompts: List[str] = []

    def _run_agent_prompt(self, *, agent, task_id, prompt_text, start_time, run_kwargs=None, error_extra=None):
        self.seen_prompts.append(prompt_text)
        return self.responses[len(self.seen_prompts) - 1], None

    # 借用基类实现 (未走 __init__, 但方法是普通方法可直接调用)
    _run_controlled_submission_rounds = BenchmarkRunner._run_controlled_submission_rounds


def _run_loop(harness: _LoopHarness, *, evaluate, retry_builder, max_rounds: int = 3, **overrides):
    return harness._run_controlled_submission_rounds(
        task_id="T1",
        agent=None,
        start_time=0.0,
        initial_prompt="initial",
        max_rounds=max_rounds,
        prompt_history=[],
        evaluate_submission=evaluate,
        retry_prompt_builder=retry_builder,
        **overrides,
    )


def _embedding_retry_builder(round_idx: int, feedback: str) -> str:
    """hevp/mbpp 风格: 反馈原样嵌入 prompt."""
    return f"ROUND {round_idx} FEEDBACK:\n{feedback}\nREVISE."


class TestControlledFeedbackBounding:
    def test_huge_output_is_bounded_before_retry_prompt(self):
        """超大评测输出 → 注入 retry prompt 前被截到上限内 (hevp/mbpp 场景)."""
        huge = "\n".join(f"FAILED test #{i}  Input: {['x' * 200] * 50}" for i in range(500))
        assert len(huge) > 1_000_000  # 确认构造出 MB 级反馈

        harness = _LoopHarness()
        captured: Dict[str, Any] = {}

        def _capture_retry(round_idx: int, feedback: str) -> str:
            captured[f"round{round_idx}"] = feedback
            return _embedding_retry_builder(round_idx, feedback)

        _run_loop(
            harness,
            evaluate=lambda r, resp: {"passed": False, "output": huge},
            retry_builder=_capture_retry,
        )

        bounded = captured["round2"]
        assert len(bounded) <= 12000 + len("[feedback truncated]") + 2
        assert bounded.count("\n") <= 80
        assert bounded.endswith("[feedback truncated]")

    def test_many_lines_bounded_even_if_chars_small(self):
        """行数超限 (每行很短) → 80 行上限生效."""
        wide = "\n".join(f"L{i}" for i in range(500))  # ~3KB, 字符不超限
        assert len(wide) < 12000

        harness = _LoopHarness()
        captured: Dict[str, Any] = {}

        def _capture(round_idx: int, feedback: str) -> str:
            captured["fb"] = feedback
            return "retry"

        _run_loop(
            harness,
            evaluate=lambda r, resp: {"passed": False, "output": wide},
            retry_builder=_capture,
        )

        assert captured["fb"].count("\n") <= 80
        assert captured["fb"].endswith("[feedback truncated]")
        # 保留头部诊断上下文 (前 80 行)
        assert captured["fb"].splitlines()[0] == "L0"

    def test_short_feedback_passthrough(self):
        """短反馈 (aime 场景: 固定格式提示) → 原样传递, 零改动."""
        short = "Format invalid. End with one boxed integer."
        harness = _LoopHarness()
        captured: Dict[str, Any] = {}

        def _capture(round_idx: int, feedback: str) -> str:
            captured["fb"] = feedback
            return "retry"

        _run_loop(
            harness,
            evaluate=lambda r, resp: {"passed": False, "output": "", "feedback": short},
            retry_builder=_capture,
        )

        assert captured["fb"] == short

    def test_explicit_feedback_key_takes_priority_and_is_bounded(self):
        """evaluation['feedback'] 优先于 output, 且同样受统一截断."""
        explicit = "E" * 50_000
        harness = _LoopHarness()
        captured: Dict[str, Any] = {}

        def _capture(round_idx: int, feedback: str) -> str:
            captured["fb"] = feedback
            return "retry"

        _run_loop(
            harness,
            evaluate=lambda r, resp: {"passed": False, "output": "unused-output", "feedback": explicit},
            retry_builder=_capture,
        )

        assert len(captured["fb"]) <= 12000 + 64
        assert captured["fb"].endswith("[feedback truncated]")

    def test_custom_limits_overridable(self):
        """上限可经参数覆盖 (bench 级差异化配置能力)."""
        harness = _LoopHarness()
        captured: Dict[str, Any] = {}

        def _capture(round_idx: int, feedback: str) -> str:
            captured["fb"] = feedback
            return "retry"

        _run_loop(
            harness,
            evaluate=lambda r, resp: {"passed": False, "output": "x" * 5000},
            retry_builder=_capture,
            feedback_max_lines=10,
            feedback_max_chars=100,
        )

        assert len(captured["fb"]) <= 100 + 64

    def test_feedback_not_retruncated_across_rounds(self):
        """已截断的反馈再次进入循环 → 幂等 (不会再叠加 marker 噪声)."""
        once = truncate_feedback("y" * 50_000, max_lines=80, max_chars=12000)
        harness = _LoopHarness()
        captured: Dict[str, Any] = {}

        def _capture(round_idx: int, feedback: str) -> str:
            captured[f"round{round_idx}"] = feedback
            return "retry"

        _run_loop(
            harness,
            evaluate=lambda r, resp: {"passed": False, "output": once},
            retry_builder=_capture,
            max_rounds=3,
        )

        assert captured["round2"] == once  # 已达界的文本原样通过
        assert captured["round3"] == once


class TestTruncateFeedbackUnit:
    """_utils.truncate_feedback 的单元契约 (防线本体)."""

    def test_empty_passthrough(self):
        assert truncate_feedback("", max_lines=10, max_chars=100) == ""

    def test_within_bounds_untouched(self):
        text = "line1\nline2\nline3"
        assert truncate_feedback(text, max_lines=10, max_chars=100) == text

    def test_lines_marker_appended(self):
        text = "\n".join(f"l{i}" for i in range(20))
        result = truncate_feedback(text, max_lines=5, max_chars=1000)
        assert result.splitlines() == ["l0", "l1", "l2", "l3", "l4", "[feedback truncated]"]

    def test_chars_marker_appended(self):
        result = truncate_feedback("a" * 500, max_lines=10, max_chars=100)
        assert len(result) <= 100 + len("\n[feedback truncated]")
        assert result.endswith("[feedback truncated]")
        assert result.startswith("a")

    def test_head_preserved(self):
        """截断保头部: 前缀诊断信息 (SUMMARY 块) 必须存活."""
        text = "\n".join(f"HEAD-{i}" for i in range(3)) + "\n" + "\n".join("tail" for _ in range(200))
        result = truncate_feedback(text, max_lines=10, max_chars=10000)
        assert "HEAD-0" in result
        assert "HEAD-2" in result
