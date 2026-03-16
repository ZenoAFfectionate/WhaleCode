import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents配置类"""

    # LLM Config
    default_model: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    default_provider: str = "vllm"
    temperature: float = 0.3
    max_tokens: Optional[int] = 8192

    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100

    # Context Engineering Config
    context_window: int = 262144
    compression_threshold: float = 0.85
    min_retain_rounds: int = 3
    
    # Context Compact Config
    compact_enabled: bool = True
    compact_token_threshold: int = 225000
    compact_keep_recent_tool_results: int = 3
    compact_transcript_dir: str = "memory/transcripts"
    summary_max_tokens: int = 8192
    summary_temperature: float = 0.3

    # 工具输出截断配置
    tool_output_max_lines: int = 1024   # 工具输出最大行数
    tool_output_max_bytes: int = 25600  # 工具输出最大字节数
    tool_output_dir: str = "memory/tool-output"   # 完整输出保存目录
    tool_output_truncate_direction: str = "head"  # 截断方向：head/tail/head_tail

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
    circuit_enabled: bool = True         # 是否启用熔断器
    circuit_failure_threshold: int = 3   # 连续失败多少次后熔断
    circuit_recovery_timeout: int = 300  # 熔断后恢复时间（秒）

    # 会话持久化配置
    session_enabled: bool = True  # 是否启用会话持久化
    session_dir: str = "memory/sessions"  # 会话文件保存目录
    auto_save_enabled: bool = False  # 是否启用自动保存
    auto_save_interval: int = 10  # 自动保存间隔（每N条消息）

    # 子代理机制配置
    subagent_enabled: bool = True  # 是否启用子代理机制
    subagent_max_steps: int = 15   # 子代理默认最大步数
    subagent_use_light_llm: bool = False  # 是否使用轻量模型
    subagent_light_llm_provider: str = "deepseek"    # 轻量模型提供商
    subagent_light_llm_model: str = "deepseek-chat"  # 轻量模型名称

    # TodoWrite 进度管理配置
    todowrite_enabled: bool = True  # 是否启用 TodoWrite 工具
    todowrite_persistence_dir: str = "memory/todos"  # 任务列表持久化目录

    # 异步生命周期配置
    async_enabled: bool = True     # 是否启用异步执行
    max_concurrent_tools: int = 3  # 最大并发工具数
    hook_timeout_seconds: float = 5.0  # 生命周期钩子超时时间（秒）
    llm_async_timeout: int = 600  # LLM 异步调用超时时间（秒），与 .env LLM_TIMEOUT 保持一致
    tool_async_timeout: int = 30  # 工具异步调用超时时间（秒）

    # 流式输出配置
    stream_enabled: bool = True    # 是否启用流式输出
    stream_buffer_size: int = 100  # 流式缓冲区大小
    stream_include_thinking: bool = True    # 是否包含思考过程
    stream_include_tool_calls: bool = True  # 是否包含工具调用

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置

        支持的环境变量（均可选，未设置时使用字段默认值）：
            DEBUG, LOG_LEVEL, TEMPERATURE, MAX_TOKENS,
            LLM_MODEL_ID, LLM_PROVIDER, LLM_TIMEOUT,
            CONTEXT_WINDOW, COMPACT_TOKEN_THRESHOLD
        """
        kwargs: Dict[str, Any] = {
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "temperature": float(os.getenv("TEMPERATURE", "0.3")),
        }

        if os.getenv("MAX_TOKENS"):
            kwargs["max_tokens"] = int(os.getenv("MAX_TOKENS"))
        if os.getenv("LLM_MODEL_ID"):
            kwargs["default_model"] = os.getenv("LLM_MODEL_ID")
        if os.getenv("LLM_PROVIDER"):
            kwargs["default_provider"] = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_TIMEOUT"):
            kwargs["llm_async_timeout"] = int(os.getenv("LLM_TIMEOUT"))
        if os.getenv("CONTEXT_WINDOW"):
            kwargs["context_window"] = int(os.getenv("CONTEXT_WINDOW"))
        if os.getenv("COMPACT_TOKEN_THRESHOLD"):
            kwargs["compact_token_threshold"] = int(os.getenv("COMPACT_TOKEN_THRESHOLD"))

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()
