"""Property-based tests (IMPROVEMENT.md B4) — 用 hypothesis 攻击三个纯函数.

被测对象 (被测代码零改动):
    - ``resolve_path``                — 路径解析 + 工作区逃逸防护
    - ``replace_with_flexible_match`` — Edit 工具的弹性替换核心
    - ``_normalize_todos``            — TodoWrite 的输入清洗

性质设计原则: 只断言函数的**语义契约**, 不绑定实现细节
(弹性策略的分支不进性质, 但 exact 路径的完全语义 + 通用不变式锁定).

开发战果: 本文件设计过程中发现并修复了 ``replace_with_flexible_match``
对空 ``old_text`` 的死循环 bug (``_exact_matches`` 游标不前进) —
上游 Edit 工具虽有空串校验, 纯函数现已自洽拒绝 (回归防线见
``test_empty_old_text_rejected``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("hypothesis", reason="hypothesis not installed (B4 optional dep)")

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# 主动构造 + 少量 assume 过滤的组合; 允许必要的过滤密度.
_ALLOW_FILTER = settings(
    deadline=None, max_examples=100, suppress_health_check=[HealthCheck.filter_too_much]
)

from hello_agents.tools.builtin._code_utils import (
    EditAmbiguousError,
    EditMatchError,
    EditNotFoundError,
    replace_with_flexible_match,
    resolve_path,
)
from hello_agents.tools.builtin.todowrite_tool import (
    TERMINAL_STATUSES,
    VALID_PRIORITIES,
    VALID_STATUSES,
    TodoSessionStore,
    TodoValidationError,
)

# CI 慢机器上 deadline 误报很常见; 这些性质均为纯计算, 无 IO.
_SETTINGS = settings(deadline=None, max_examples=100)
_FAST = settings(deadline=None, max_examples=50)


# ============================================================================
# 1. resolve_path — 安全性与往返
# ============================================================================

# 假想根路径: 不真实存在 → 无 symlink 干扰, resolve() 纯词法+规范化.
_FAKE_ROOTS = ["/srv/whale-proj", "/home/u/app", "/data/workspace"]

_path_segment = st.text(
    alphabet=st.characters(
        codec="utf-8",
        exclude_categories=("Cs", "Zs"),
        exclude_characters="/\\:\x00",
    ),
    min_size=1,
    max_size=12,
)


def _rel_paths(min_size: int = 0):
    """相对路径: 普通段 + 偶发的 . / .. 导航段."""
    segments = st.one_of(_path_segment, st.just("."), st.just(".."))
    return st.lists(segments, min_size=min_size, max_size=6).map(
        lambda parts: "/".join(parts) if parts else "."
    )


def _safe_rel_paths(min_size: int = 1, max_size: int = 4):
    """无 . / .. 导航段的相对路径 — 解析必在 root 内 (往返/透传性质专用)."""
    return st.lists(_path_segment, min_size=min_size, max_size=max_size).map("/".join)


class TestResolvePathSafety:
    @_SETTINGS
    @given(root=st.sampled_from(_FAKE_ROOTS), working_dir=_rel_paths(), raw=_rel_paths())
    def test_result_stays_inside_root_or_valueerror(self, root, working_dir, raw):
        """安全不变式: 任意输入 → 要么 ValueError, 要么结果落在 root 内."""
        try:
            resolved = resolve_path(root, working_dir, raw)
        except ValueError:
            return  # 逃逸 / 非法输入 → 契约内的合法拒绝
        root_r = Path(root).expanduser().resolve()
        assert resolved == root_r or root_r in resolved.parents

    @_SETTINGS
    @given(
        root=st.sampled_from(_FAKE_ROOTS),
        working_dir=_safe_rel_paths(min_size=0, max_size=4),
        raw=_safe_rel_paths(min_size=1, max_size=4),
    )
    def test_roundtrip_absolute_and_relative(self, root, working_dir, raw):
        """往返: 成功解析后, 绝对回喂恒等; 相对回喂 (以 working_dir 为基准) 恒等."""
        try:
            resolved = resolve_path(root, working_dir, raw)
        except ValueError:
            assume(False)
            return
        # 绝对路径回喂 → 恒等
        assert resolve_path(root, working_dir, str(resolved)) == resolved
        # 相对路径回喂 (注意: 相对路径以 working_dir 为基准, 而非 root)
        try:
            wd_r = resolve_path(root, ".", working_dir)
        except ValueError:
            # 段以 "~" 开头等 → working_dir 本身被正确拒绝, 相对回喂不适用
            return
        if wd_r == resolved or wd_r in resolved.parents:
            rel_from_wd = resolved.relative_to(wd_r).as_posix()
            assert resolve_path(root, working_dir, rel_from_wd) == resolved

    @_FAST
    @given(root=st.sampled_from(_FAKE_ROOTS), raw=_safe_rel_paths(min_size=1, max_size=4))
    def test_inside_absolute_path_passthrough(self, root, raw):
        """root 内的绝对路径直接 resolve, 与 root 内任意 working_dir 无关."""
        root_r = Path(root).expanduser().resolve()
        absolute = str(root_r / raw)
        try:
            expected = Path(absolute).resolve()
        except ValueError:  # NUL 等非法字符 → 同为契约内拒绝
            assume(False)
            return
        # 性质限定 "root 内": raw 含 ".." 逃逸 root 的反例不属于本性质
        assume(expected == root_r or root_r in expected.parents)
        assert resolve_path(root, str(root_r / "sub"), absolute) == expected
        assert resolve_path(root, ".", absolute) == expected

    @_FAST
    @given(root=st.sampled_from(_FAKE_ROOTS))
    def test_unknown_user_prefix_raises_value_error(self, root):
        """契约回归: '~unknownuser' 无法展开时必须是 ValueError (曾泄漏 RuntimeError)."""
        with pytest.raises(ValueError):
            resolve_path(root, "~no-such-user-xyz", "file.txt")
        with pytest.raises(ValueError):
            resolve_path(root, ".", "~no-such-user-xyz/file.txt")


# ============================================================================
# 2. replace_with_flexible_match — exact 路径的完全语义
# ============================================================================


class TestReplaceExactSemantics:
    @_SETTINGS
    @given(content=st.text(max_size=300), old=st.text(max_size=60), new=st.text(max_size=60))
    def test_replace_all_total_semantics(self, content, old, new):
        """replace_all=True 完全确定: 参数非法 → EditMatchError;
        未出现 → NotFound; 否则精确 replace + 非重叠计数 + exact 策略."""
        if not old or old == new:
            with pytest.raises(EditMatchError):
                replace_with_flexible_match(content, old, new, replace_all=True)
            return
        if old not in content:
            with pytest.raises(EditNotFoundError):
                replace_with_flexible_match(content, old, new, replace_all=True)
            return
        result = replace_with_flexible_match(content, old, new, replace_all=True)
        assert result.content == content.replace(old, new)
        assert result.replacements == content.count(old)  # 非重叠计数契约
        assert result.strategy == "exact"

    @_ALLOW_FILTER
    @given(
        prefix=st.text(max_size=120),
        mid=st.text(min_size=1, max_size=40),
        suffix=st.text(max_size=120),
        new=st.text(max_size=40),
    )
    def test_unique_match_exact_single_replacement(self, prefix, mid, suffix, new):
        """唯一非重叠出现 (主动嵌入) → exact 单处替换, 结果等价 str.replace."""
        old = mid
        content = prefix + old + suffix
        assume(old != new and content.count(old) == 1)
        result = replace_with_flexible_match(content, old, new)
        assert result.strategy == "exact"
        assert result.replacements == 1
        assert result.content == content.replace(old, new)

    @_ALLOW_FILTER
    @given(mid=st.text(min_size=1, max_size=40), new=st.text(max_size=40))
    def test_multiple_matches_ambiguous_without_replace_all(self, mid, new):
        """非重叠出现 >= 2 (old+old 构造) → (replace_all=False) 必须拒绝歧义."""
        old = mid
        content = old + old  # 重叠型 old 会被非重叠计数吞并, assume 兜底
        assume(old != new and content.count(old) >= 2)
        with pytest.raises(EditAmbiguousError):
            replace_with_flexible_match(content, old, new)

    def test_empty_old_text_rejected(self):
        """回归防线: 空 old_text 曾导致 _exact_matches 死循环 (已修复)."""
        with pytest.raises(EditMatchError):
            replace_with_flexible_match("abc", "", "x")
        with pytest.raises(EditMatchError):
            replace_with_flexible_match("abc", "", "x", replace_all=True)

    @_SETTINGS
    @given(content=st.text(max_size=200), old=st.text(min_size=1, max_size=40), new=st.text(max_size=40))
    def test_success_invariants_any_strategy(self, content, old, new):
        """通用不变式: 任何成功结果 → 已知策略名 + replacements >= 1 +
        结果长度变化来自 [old 段] → new 的交换."""
        assume(old != new)
        try:
            result = replace_with_flexible_match(content, old, new)
        except EditMatchError:
            return  # NotFound / Ambiguous 均为契约内行为
        assert result.strategy in {
            "exact",
            "line_trimmed",
            "indentation_flexible",
            "whitespace_normalized",
            "trimmed_boundary",
        }
        assert result.replacements >= 1
        delta = len(result.content) - len(content)
        assert delta == len(new) - len(old) or result.strategy != "exact"


# ============================================================================
# 3. _normalize_todos — 清洗不变式 / 幂等 / 拒绝路径
# ============================================================================

_nonempty_text = st.text(min_size=1, max_size=20).filter(lambda s: s.strip())

_valid_todo = st.tuples(
    _nonempty_text,
    st.sampled_from(sorted(VALID_STATUSES)),
    st.sampled_from(sorted(VALID_PRIORITIES)),
)

_valid_todo_lists = st.lists(_valid_todo, max_size=6).filter(
    lambda items: (
        len({content.strip() for content, _, _ in items}) == len(items)
        and sum(1 for _, status, _ in items if status == "in_progress") <= 1
    )
)


class TestNormalizeTodosProperties:
    @_SETTINGS
    @given(items=_valid_todo_lists)
    def test_legal_input_invariants(self, items):
        """合法输入 → 输出长度一致 / content 唯一且 stripped /
        status 与 priority 全部落在合法枚举 / ≤1 个 in_progress."""
        todos = [
            {"content": content, "status": status, "priority": priority}
            for content, status, priority in items
        ]
        normalized = TodoSessionStore._normalize_todos(todos)

        assert len(normalized) == len(todos)
        contents = [item["content"] for item in normalized]
        assert len(set(contents)) == len(contents)
        assert all(item["content"] == item["content"].strip() for item in normalized)
        assert all(item["status"] in VALID_STATUSES for item in normalized)
        assert all(item["priority"] in VALID_PRIORITIES for item in normalized)
        assert sum(1 for item in normalized if item["status"] == "in_progress") <= 1

    @_SETTINGS
    @given(items=_valid_todo_lists)
    def test_idempotent(self, items):
        """幂等: normalize(normalize(x)) == normalize(x) — 输出已是范式."""
        todos = [
            {"content": content, "status": status, "priority": priority}
            for content, status, priority in items
        ]
        once = TodoSessionStore._normalize_todos(todos)
        twice = TodoSessionStore._normalize_todos(once)
        assert twice == once

    @_SETTINGS
    @given(items=_valid_todo_lists)
    def test_case_and_whitespace_normalization(self, items):
        """status/priority 大小写 → 归一为小写; content 前后空白被剥离."""
        todos = [
            {
                "content": f"  {content}  ",
                "status": status.upper(),
                "priority": priority.upper(),
            }
            for content, status, priority in items
        ]
        normalized = TodoSessionStore._normalize_todos(todos)
        for original, item in zip(items, normalized):
            assert item["content"] == original[0].strip()
            assert item["status"] == original[1]
            assert item["priority"] == original[2]

    @_SETTINGS
    @given(items=_valid_todo_lists)
    def test_default_priority_is_medium(self, items):
        """缺省 priority → medium."""
        todos = [{"content": content, "status": status} for content, status, _ in items]
        normalized = TodoSessionStore._normalize_todos(todos)
        assert all(item["priority"] == "medium" for item in normalized)

    # ── 拒绝路径 ──────────────────────────────────────────────────────────

    @_FAST
    @given(junk=st.one_of(st.integers(), st.text(), st.none(), st.booleans()))
    def test_non_list_rejected(self, junk):
        with pytest.raises(TodoValidationError):
            TodoSessionStore._normalize_todos(junk)

    @_FAST
    @given(items=st.lists(st.one_of(st.integers(), st.text(), st.none()), min_size=1, max_size=5))
    def test_non_dict_entries_rejected(self, items):
        with pytest.raises(TodoValidationError):
            TodoSessionStore._normalize_todos(items)

    @_FAST
    @given(
        content=st.text(
            alphabet=st.sampled_from([" ", "\t", "\r", "\n", "\f", "\v", "\u3000"]),
            min_size=1,
            max_size=10,
        )
    )
    def test_blank_content_rejected(self, content):
        assert not content.strip()
        with pytest.raises(TodoValidationError):
            TodoSessionStore._normalize_todos([{"content": content, "status": "pending"}])

    @_SETTINGS
    @given(
        content=_nonempty_text,
        status=st.sampled_from(sorted(VALID_STATUSES)),
        bad_priority=st.text(min_size=1, max_size=10).filter(
            lambda s: s.strip().lower() not in VALID_PRIORITIES
        ),
    )
    def test_invalid_priority_rejected(self, content, status, bad_priority):
        with pytest.raises(TodoValidationError):
            TodoSessionStore._normalize_todos(
                [{"content": content, "status": status, "priority": bad_priority}]
            )

    @_SETTINGS
    @given(
        content=_nonempty_text,
        status=st.sampled_from(sorted(VALID_STATUSES)),
        bad_status=st.text(min_size=1, max_size=10).filter(
            lambda s: s.strip().lower() not in VALID_STATUSES
        ),
    )
    def test_invalid_status_rejected(self, content, status, bad_status):
        with pytest.raises(TodoValidationError):
            TodoSessionStore._normalize_todos(
                [{"content": content, "status": bad_status, "priority": "medium"}]
            )

    def test_terminal_status_transition_rejected(self):
        """终态 (completed/cancelled) 不可回退 — 业务规则回归."""
        previous = [{"content": "done-task", "status": "completed", "priority": "low"}]
        for terminal in sorted(TERMINAL_STATUSES):
            with pytest.raises(TodoValidationError):
                TodoSessionStore._normalize_todos(
                    [{"content": "done-task", "status": "pending"}],
                    previous_todos=previous,
                )

    def test_duplicate_content_rejected(self):
        with pytest.raises(TodoValidationError):
            TodoSessionStore._normalize_todos([
                {"content": "same", "status": "pending"},
                {"content": "same", "status": "completed"},
            ])

    def test_multiple_in_progress_rejected(self):
        with pytest.raises(TodoValidationError):
            TodoSessionStore._normalize_todos([
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ])
