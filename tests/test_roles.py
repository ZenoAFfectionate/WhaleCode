"""角色系统测试: 配置 / 注册表 / 子Agent创建 / 工具过滤 / 隔离性."""

from __future__ import annotations

from pathlib import Path

import pytest

from hello_agents.agents.code_agent import CodeAgent
from hello_agents.agents.roles import (
    ExplorerRole,
    ReviewerRole,
    Role,
    RoleConfig,
    TesterRole,
    get_role,
    list_roles,
)


@pytest.fixture
def tmp_workspace(tmp_path):
    (tmp_path / "memory" / "tool-output").mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


def _make_main_agent(mock_llm, workspace):
    return CodeAgent(
        "main",
        mock_llm,
        project_root=workspace,
        register_default_tools=False,
    )


class TestRoleConfig:
    def test_explorer_config_readonly(self):
        cfg = ExplorerRole.get_config()
        assert "Write" in cfg.denied_tools
        assert "Edit" in cfg.denied_tools
        assert "Delete" in cfg.denied_tools
        assert "Bash" in cfg.denied_tools
        assert "readonly" in cfg.allowed_categories

    def test_reviewer_config_has_bash(self):
        cfg = ReviewerRole.get_config()
        assert "Bash" in cfg.allowed_tools
        assert "Write" in cfg.denied_tools
        assert "Edit" in cfg.denied_tools
        # denied_categories 不得含 dangerous, 否则 Bash 被黑名单误删
        assert "dangerous" not in cfg.denied_categories

    def test_tester_config(self):
        cfg = TesterRole.get_config()
        assert "Bash" in cfg.allowed_tools
        assert "Delete" in cfg.denied_tools
        assert "write" in cfg.allowed_categories
        assert "readonly" in cfg.allowed_categories
        assert "dangerous" not in cfg.denied_categories


class TestRoleRegistry:
    def test_get_role_returns_correct_class(self):
        assert get_role("explorer") is ExplorerRole
        assert get_role("reviewer") is ReviewerRole
        assert get_role("tester") is TesterRole

    def test_get_role_case_insensitive(self):
        assert get_role("Explorer") is ExplorerRole
        assert get_role("TESTER") is TesterRole

    def test_get_role_invalid_raises(self):
        with pytest.raises(ValueError):
            get_role("nonexistent")

    def test_list_roles(self):
        roles = list_roles()
        assert "explorer" in roles
        assert "reviewer" in roles
        assert "tester" in roles


class TestRoleSubagentCreation:
    def test_subagent_has_independent_history(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.history_manager is not main.history_manager

    def test_subagent_has_independent_registry(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry is not main.tool_registry

    def test_subagent_trace_disabled(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.config.trace_enabled is False

    def test_subagent_no_session(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.config.session_enabled is False

    def test_explorer_subagent_cannot_write(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Write") is None
        assert sub.tool_registry.get_tool("Edit") is None
        assert sub.tool_registry.get_tool("Bash") is None
        assert sub.tool_registry.get_tool("Delete") is None
        # 但仍可读取
        assert sub.tool_registry.get_tool("Read") is not None
        assert sub.tool_registry.get_tool("Grep") is not None
        assert sub.tool_registry.get_tool("Glob") is not None
        assert sub.tool_registry.get_tool("LS") is not None

    def test_reviewer_subagent_can_use_bash(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ReviewerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Bash") is not None
        assert sub.tool_registry.get_tool("Write") is None
        assert sub.tool_registry.get_tool("Edit") is None
        assert sub.tool_registry.get_tool("Delete") is None
        assert sub.tool_registry.get_tool("Read") is not None

    def test_tester_subagent_tools(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = TesterRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Write") is not None
        assert sub.tool_registry.get_tool("Edit") is not None
        assert sub.tool_registry.get_tool("Bash") is not None
        assert sub.tool_registry.get_tool("Read") is not None
        assert sub.tool_registry.get_tool("Delete") is None
        assert sub.tool_registry.get_tool("AskUser") is None
        assert sub.tool_registry.get_tool("WebSearch") is None
        assert sub.tool_registry.get_tool("WebFetch") is None

    def test_explorer_system_prompt_present(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        prompt = sub._get_context_system_prompt()
        assert "code exploration" in prompt.lower()
        assert "Workspace root:" in prompt  # CodeAgent 自动追加

    def test_reviewer_system_prompt_present(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ReviewerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        prompt = sub._get_context_system_prompt()
        assert "code review" in prompt.lower()
        assert "security" in prompt.lower()

    def test_tester_system_prompt_present(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = TesterRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        prompt = sub._get_context_system_prompt()
        assert "testing specialist" in prompt.lower()
        assert "Workspace root:" in prompt

    def test_config_isolation_deep_copy(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        sub.config.subagent_timeout_seconds = 99
        assert main.config.subagent_timeout_seconds != 99

    def test_parent_registry_untouched_by_subagent_policy(self, mock_llm, tmp_workspace):
        """子 Agent 的工具过滤不得影响父 Agent 的注册表."""
        main = _make_main_agent(mock_llm, tmp_workspace)
        before = set(main.tool_registry.list_tools())
        ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert set(main.tool_registry.list_tools()) == before

    def test_subagent_max_steps_from_role_config(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        cfg = TesterRole.get_config()
        sub = TesterRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.max_steps == cfg.max_steps == 25
        cfg = ExplorerRole.get_config()
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.max_steps == cfg.max_steps == 20

    def test_subagent_name_and_dirs(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.name == "subagent-explorer"
        assert str(sub.project_root) == tmp_workspace
        assert str(sub.working_dir) == tmp_workspace

    def test_subagent_skills_disabled(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.config.skills_enabled is False
        assert sub.config.todowrite_enabled is False

    @pytest.mark.parametrize("role_cls", [ExplorerRole, ReviewerRole, TesterRole])
    def test_subagent_task_tool_disabled(self, role_cls, mock_llm, tmp_workspace):
        """防递归契约: 子代理不得持有 Task 工具 (不得再派生子代理)."""
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = role_cls.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.config.subagent_task_enabled is False
        assert sub.tool_registry.get_tool("Task") is None

    @pytest.mark.parametrize("role_cls", [ExplorerRole, ReviewerRole, TesterRole])
    def test_control_tools_always_preserved(self, role_cls, mock_llm, tmp_workspace):
        """回归: Thought/Finish 控制工具不得被白名单过滤 (否则 ReAct loop 无法结束)."""
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = role_cls.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Thought") is not None
        assert sub.tool_registry.get_tool("Finish") is not None

    def test_multiple_subagents_have_independent_registries(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub_a = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        sub_b = ExplorerRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub_a.tool_registry is not sub_b.tool_registry
        # 对 sub_a 再卸载工具不影响 sub_b
        sub_a.tool_registry.unregister("Read")
        assert sub_a.tool_registry.get_tool("Read") is None
        assert sub_b.tool_registry.get_tool("Read") is not None


class _PermissiveRole(Role):
    """无任何白名单/黑名单 — 所有工具应保留."""

    @staticmethod
    def get_config() -> RoleConfig:
        return RoleConfig(name="permissive", description="keep all")


class _ReadOnlyNamesRole(Role):
    """仅 allowed_tools 白名单 (无类别) — 只保留点名工具."""

    @staticmethod
    def get_config() -> RoleConfig:
        return RoleConfig(
            name="names-only",
            description="only Read",
            allowed_tools=["Read"],
        )


class _DenyReadonlyRole(Role):
    """黑名单 denied_categories={"readonly"} — 只读工具全移除."""

    @staticmethod
    def get_config() -> RoleConfig:
        return RoleConfig(
            name="deny-readonly",
            description="drop readonly",
            denied_categories={"readonly"},
        )


class TestToolPolicyEdgeCases:
    def test_no_whitelist_keeps_everything(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = _PermissiveRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Write") is not None
        assert sub.tool_registry.get_tool("Bash") is not None
        assert sub.tool_registry.get_tool("Read") is not None

    def test_allowed_tools_alone_acts_as_whitelist(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = _ReadOnlyNamesRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Read") is not None
        # 其余均不在白名单 → 移除
        assert sub.tool_registry.get_tool("Write") is None
        assert sub.tool_registry.get_tool("Grep") is None
        assert sub.tool_registry.get_tool("Bash") is None

    def test_denied_categories_removes_category(self, mock_llm, tmp_workspace):
        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = _DenyReadonlyRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Read") is None
        assert sub.tool_registry.get_tool("Grep") is None
        # 非 readonly 工具保留 (无白名单限制)
        assert sub.tool_registry.get_tool("Write") is not None
        assert sub.tool_registry.get_tool("Bash") is not None

    def test_blacklist_beats_explicit_allow(self, mock_llm, tmp_workspace):
        """同一工具同时被 deny 与 allow 点名时, 黑名单优先."""

        class _ConflictRole(Role):
            @staticmethod
            def get_config() -> RoleConfig:
                return RoleConfig(
                    name="conflict",
                    description="d",
                    allowed_tools=["Write"],
                    denied_tools=["Write"],
                )

        main = _make_main_agent(mock_llm, tmp_workspace)
        sub = _ConflictRole.create_subagent(main.llm, main.config, tmp_workspace, tmp_workspace)
        assert sub.tool_registry.get_tool("Write") is None


class TestRoleConfigShape:
    def test_role_config_no_dead_fields(self):
        """temperature/agent_type 已移除 (共享 LLM 无法按角色生效), 防止回归."""
        cfg = ExplorerRole.get_config()
        assert not hasattr(cfg, "temperature")
        assert not hasattr(cfg, "agent_type")

    def test_role_config_defaults(self):
        cfg = RoleConfig(name="x", description="d")
        assert cfg.allowed_tools == []
        assert cfg.denied_tools == []
        assert cfg.allowed_categories == set()
        assert cfg.max_steps == 15

    def test_get_role_rejects_blank(self):
        with pytest.raises(ValueError):
            get_role("")
        with pytest.raises(ValueError):
            get_role("   ")


class TestPromptFiles:
    """SubAgent 提示词集中管理于 code/prompts/ 的保障性测试."""

    _PROMPTS = Path(__file__).resolve().parents[1] / "code" / "prompts"

    def test_role_prompt_files_exist(self):
        for name in ("explorer", "reviewer", "tester"):
            assert (self._PROMPTS / "roles" / f"{name}.md").is_file(), name

    def test_constants_loaded_from_files(self):
        from hello_agents.agents.roles.explorer import EXPLORER_SYSTEM_PROMPT
        from hello_agents.agents.roles.reviewer import REVIEWER_SYSTEM_PROMPT
        from hello_agents.agents.roles.tester import TESTER_SYSTEM_PROMPT

        assert EXPLORER_SYSTEM_PROMPT == (self._PROMPTS / "roles" / "explorer.md").read_text(encoding="utf-8")
        assert REVIEWER_SYSTEM_PROMPT == (self._PROMPTS / "roles" / "reviewer.md").read_text(encoding="utf-8")
        assert TESTER_SYSTEM_PROMPT == (self._PROMPTS / "roles" / "tester.md").read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", ["explorer", "reviewer", "tester"])
    def test_role_prompts_carry_output_contract(self, name):
        """蒸馏性契约: 每个角色提示词都必须声明子智能体输出契约
        (编排器只看得到最终响应, 结果必须自包含)."""
        text = (self._PROMPTS / "roles" / f"{name}.md").read_text(encoding="utf-8").lower()
        assert "sub-agent" in text
        assert "only see your final" in text
        assert "self-contained" in text


class TestSubagentTaskConfigEnv:
    def test_from_env_reads_subagent_task_vars(self, monkeypatch):
        from hello_agents.core.config import Config

        monkeypatch.setenv("SUBAGENT_TASK_ENABLED", "false")
        monkeypatch.setenv("SUBAGENT_TIMEOUT_SECONDS", "120.5")
        monkeypatch.setenv("REVIEW_MAX_FILES", "20")
        monkeypatch.setenv("REVIEW_MAX_FINDINGS", "10")
        monkeypatch.setenv("REVIEW_GH_CLI_ENABLED", "false")
        cfg = Config.from_env()
        assert cfg.subagent_task_enabled is False
        assert cfg.subagent_timeout_seconds == 120.5
        assert cfg.review_max_files == 20
        assert cfg.review_max_findings == 10
        assert cfg.review_gh_cli_enabled is False

    def test_from_env_defaults(self, monkeypatch):
        from hello_agents.core.config import Config

        for var in (
            "SUBAGENT_TASK_ENABLED", "SUBAGENT_TIMEOUT_SECONDS",
            "REVIEW_MAX_FILES", "REVIEW_MAX_FINDINGS", "REVIEW_GH_CLI_ENABLED",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = Config.from_env()
        assert cfg.subagent_task_enabled is True
        assert cfg.subagent_timeout_seconds == 300.0
