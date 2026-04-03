import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents配置类"""

    # LLM Config
    default_model: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    default_provider: str = "vllm"
    temperature: float = 1.0
    max_tokens: Optional[int] = 128000

    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100

    # Context Engineering Config
    context_window: int = 262144
    compression_threshold: float = 0.8
    min_retain_rounds: int = 3
    max_rounds_before_compression: int = 8

    # Context Compact Config
    compact_enabled: bool = True
    compact_token_threshold: int = 225000
    compact_output_buffer: int = 16000
    compact_preserve_recent_rounds: int = 5
    compact_keep_recent_tool_results: int = 5
    compact_transcript_dir: str = "memory/transcripts"
    summary_max_tokens: int = 8192
    summary_temperature: float = 0.3
    smart_summary_enabled: bool = True
    # Backward-compatibility alias used by older tests/configs.
    enable_smart_compression: Optional[bool] = None
    # Reserved for future dedicated summary model routing.
    summary_llm_provider: Optional[str] = None
    summary_llm_model: Optional[str] = None

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
    todowrite_persistence_dir: str = "memory/todos"  # session 级 todo 快照目录

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

    def __init__(self, **data):
        super().__init__(**data)
        if self.enable_smart_compression is not None:
            self.smart_summary_enabled = bool(self.enable_smart_compression)

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
            "temperature": float(os.getenv("TEMPERATURE", "1.0")),
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
        if os.getenv("COMPRESSION_THRESHOLD"):
            kwargs["compression_threshold"] = float(os.getenv("COMPRESSION_THRESHOLD"))
        if os.getenv("COMPACT_TOKEN_THRESHOLD"):
            kwargs["compact_token_threshold"] = int(os.getenv("COMPACT_TOKEN_THRESHOLD"))
        if os.getenv("COMPACT_OUTPUT_BUFFER"):
            kwargs["compact_output_buffer"] = int(os.getenv("COMPACT_OUTPUT_BUFFER"))
        if os.getenv("ENABLE_SMART_COMPRESSION"):
            kwargs["enable_smart_compression"] = (
                os.getenv("ENABLE_SMART_COMPRESSION", "true").lower() == "true"
            )

        return cls(**kwargs)

    def get_compact_trigger_limit(self) -> int:
        """Return the effective prompt-token limit that should trigger compact.

        Compaction starts once prompt usage approaches the configured fraction
        of the model context window. ``compact_output_buffer`` still reserves
        room for the next completion, and ``compact_token_threshold`` remains an
        explicit upper bound for compatibility.
        """
        context_window = max(1, int(self.context_window))
        compression_threshold = float(self.compression_threshold or 0.8)
        compression_threshold = min(max(compression_threshold, 0.01), 1.0)

        percentage_limit = max(1, int(context_window * compression_threshold))
        buffer_limit = max(1, context_window - max(0, int(self.compact_output_buffer)))
        effective_limit = min(percentage_limit, buffer_limit)

        threshold = int(self.compact_token_threshold)
        if threshold > 0:
            effective_limit = min(effective_limit, threshold)
        return max(1, effective_limit)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()
