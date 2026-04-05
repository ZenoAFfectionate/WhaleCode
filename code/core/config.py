import os
from typing import Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents配置类"""

    debug: bool = False

    # Context Engineering Config
    context_window: int = 262144
    compact_enabled: bool = True
    compression_threshold: float = 0.8
    compact_output_buffer: int = 16384
    
    compact_preserve_recent_rounds: int = 3
    compact_keep_recent_tool_results: int = 3

    compact_transcript_dir: str = "memory/transcripts"
    summary_max_tokens: int = 4096
    summary_temperature: float = 0.3

    # 可观测性配置
    trace_enabled: bool = True  # 是否启用 Trace 记录
    trace_dir: str = "memory/traces"  # Trace 文件保存目录
    trace_sanitize: bool = True  # 是否脱敏敏感信息
    trace_html_include_raw_response: bool = False  # HTML 是否包含原始响应

    # Skills 知识外化配置
    skills_enabled: bool = True  # 是否启用 Skills 系统
    skills_dir: str = "skills"   # Skills 目录路径
    skills_auto_register: bool = True  # 是否自动注册 SkillTool

    # 熔断器配置
    circuit_enabled: bool = True
    circuit_failure_threshold: int = 3
    circuit_recovery_timeout: int = 300

    # 会话持久化配置
    session_enabled: bool = True  # 是否启用会话持久化
    session_dir: str = "memory/sessions"  # 会话文件保存目录
    auto_save_enabled: bool = False  # 是否启用自动保存
    auto_save_interval: int = 10  # 自动保存间隔（每N条消息）

    # 子代理机制配置
    subagent_max_steps: int = 15   # 子代理默认最大步数

    # TodoWrite 进度管理配置
    todowrite_enabled: bool = True  # 是否启用 TodoWrite 工具
    todowrite_persistence_dir: str = "memory/todos"  # session 级 todo 快照目录

    # 执行与生命周期配置
    max_concurrent_tools: int = 3  # 最大并发工具数
    hook_timeout_seconds: float = 5.0  # 生命周期钩子超时时间（秒）

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置

        支持的环境变量（均可选，未设置时使用字段默认值）：
            DEBUG, CONTEXT_WINDOW, COMPRESSION_THRESHOLD,
            COMPACT_OUTPUT_BUFFER,
            CIRCUIT_ENABLED, CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_TIMEOUT
        """
        kwargs: Dict[str, Any] = {
            "debug": os.getenv("DEBUG", "false").lower() == "true",
        }

        if os.getenv("CONTEXT_WINDOW"):
            kwargs["context_window"] = int(os.getenv("CONTEXT_WINDOW"))
        if os.getenv("COMPRESSION_THRESHOLD"):
            kwargs["compression_threshold"] = float(os.getenv("COMPRESSION_THRESHOLD"))
        if os.getenv("COMPACT_OUTPUT_BUFFER"):
            kwargs["compact_output_buffer"] = int(os.getenv("COMPACT_OUTPUT_BUFFER"))
        if os.getenv("CIRCUIT_ENABLED"):
            kwargs["circuit_enabled"] = os.getenv("CIRCUIT_ENABLED", "true").lower() == "true"
        if os.getenv("CIRCUIT_FAILURE_THRESHOLD"):
            kwargs["circuit_failure_threshold"] = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD"))
        if os.getenv("CIRCUIT_RECOVERY_TIMEOUT"):
            kwargs["circuit_recovery_timeout"] = int(os.getenv("CIRCUIT_RECOVERY_TIMEOUT"))

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()
