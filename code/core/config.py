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

    # 沙箱 / 网络 / 工具安全（建议-6：集中登记；工具/适配器目前从同名环境变量读取，
    # 这里作为单一事实来源与文档，便于多环境切换与审计）
    bash_allow_network: bool = False            # BASH_ALLOW_NETWORK
    bash_max_cpu_seconds: int = 3600            # BASH_MAX_CPU_SECONDS（严重-2）
    bash_max_memory_bytes: int = 0              # BASH_MAX_MEMORY_BYTES（0=不限，严重-2 修复后默认由 BashTool 控制）
    bash_max_processes: int = 4096              # BASH_MAX_PROCESSES（fork bomb 防护）
    bash_max_file_size_bytes: int = 0           # BASH_MAX_FILE_SIZE_BYTES（0=不限）
    bash_max_execution_ms: int = 0              # BASH_MAX_EXECUTION_MS（硬超时，0=不强杀）
    web_tools_enabled: bool = True              # WEB_TOOLS_ENABLED
    web_fetch_allow_private: bool = False       # WEBFETCH_ALLOW_PRIVATE（严重-3 SSRF 放行）

    # LLM 调用重试（重要-12）
    llm_max_retries: int = 2                    # LLM_MAX_RETRIES
    llm_retry_base_delay: float = 0.5           # LLM_RETRY_BASE_DELAY
    llm_retry_max_delay: float = 8.0            # LLM_RETRY_MAX_DELAY

    # ReAct 步数上限（重要-4；0=无限，仅显式高级用法）
    code_agent_max_steps: int = 100             # 对应 CLI --max-steps

    # 文件工具备份与锁（建议-12）
    backup_enabled: bool = True                 # 是否在 Write/Edit 前自动备份
    backup_dir: str = "memory/.backups"         # 统一备份根目录（取代 per-file .backups/）
    backup_max_per_file: int = 5                # 每个文件最多保留的旧版本数
    backup_retention_days: int = 7              # 备份保留天数

    # Benchmark runtime profiles（Benchmark-P5）
    bench_eval_cpu_seconds: int = 0
    bench_eval_memory_bytes: int = 0
    bench_eval_max_processes: int = 128
    bench_eval_file_size_bytes: int = 256 * 1024 * 1024
    bench_eval_network: bool = False
    bench_eval_artifact_retention: int = 200

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置

        支持的环境变量（均可选，未设置时使用字段默认值）：
            DEBUG, CONTEXT_WINDOW, COMPRESSION_THRESHOLD, COMPACT_OUTPUT_BUFFER,
            CIRCUIT_ENABLED, CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_TIMEOUT,
            BASH_ALLOW_NETWORK, BASH_MAX_CPU_SECONDS, BASH_MAX_PROCESSES,
            BASH_MAX_EXECUTION_MS, WEB_TOOLS_ENABLED, WEBFETCH_ALLOW_PRIVATE,
            LLM_MAX_RETRIES, LLM_RETRY_BASE_DELAY, LLM_RETRY_MAX_DELAY,
            CODE_AGENT_MAX_STEPS, WHALE_BENCH_EVAL_CPU_SECONDS,
            WHALE_BENCH_EVAL_MEMORY_BYTES, WHALE_BENCH_EVAL_MAX_PROCESSES,
            WHALE_BENCH_EVAL_FILE_SIZE_BYTES, WHALE_BENCH_EVAL_NETWORK,
            WHALE_BENCH_EVAL_ARTIFACT_RETENTION
        """
        def _bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or not str(raw).strip():
                return default
            try:
                return int(str(raw).strip())
            except (TypeError, ValueError):
                return default

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or not str(raw).strip():
                return default
            try:
                return float(str(raw).strip())
            except (TypeError, ValueError):
                return default

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

        # 建议-6: sandbox / network / retry / step-limit switches.
        kwargs["bash_allow_network"] = _bool("BASH_ALLOW_NETWORK", cls.model_fields["bash_allow_network"].default)
        kwargs["bash_max_cpu_seconds"] = _int("BASH_MAX_CPU_SECONDS", cls.model_fields["bash_max_cpu_seconds"].default)
        kwargs["bash_max_memory_bytes"] = _int("BASH_MAX_MEMORY_BYTES", cls.model_fields["bash_max_memory_bytes"].default)
        kwargs["bash_max_processes"] = _int("BASH_MAX_PROCESSES", cls.model_fields["bash_max_processes"].default)
        kwargs["bash_max_file_size_bytes"] = _int("BASH_MAX_FILE_SIZE_BYTES", cls.model_fields["bash_max_file_size_bytes"].default)
        kwargs["bash_max_execution_ms"] = _int("BASH_MAX_EXECUTION_MS", cls.model_fields["bash_max_execution_ms"].default)
        kwargs["web_tools_enabled"] = _bool("WEB_TOOLS_ENABLED", cls.model_fields["web_tools_enabled"].default)
        kwargs["web_fetch_allow_private"] = _bool("WEBFETCH_ALLOW_PRIVATE", cls.model_fields["web_fetch_allow_private"].default)
        kwargs["llm_max_retries"] = _int("LLM_MAX_RETRIES", cls.model_fields["llm_max_retries"].default)
        kwargs["llm_retry_base_delay"] = _float("LLM_RETRY_BASE_DELAY", cls.model_fields["llm_retry_base_delay"].default)
        kwargs["llm_retry_max_delay"] = _float("LLM_RETRY_MAX_DELAY", cls.model_fields["llm_retry_max_delay"].default)
        kwargs["code_agent_max_steps"] = _int("CODE_AGENT_MAX_STEPS", cls.model_fields["code_agent_max_steps"].default)
        kwargs["bench_eval_cpu_seconds"] = _int("WHALE_BENCH_EVAL_CPU_SECONDS", cls.model_fields["bench_eval_cpu_seconds"].default)
        kwargs["bench_eval_memory_bytes"] = _int("WHALE_BENCH_EVAL_MEMORY_BYTES", cls.model_fields["bench_eval_memory_bytes"].default)
        kwargs["bench_eval_max_processes"] = _int("WHALE_BENCH_EVAL_MAX_PROCESSES", cls.model_fields["bench_eval_max_processes"].default)
        kwargs["bench_eval_file_size_bytes"] = _int("WHALE_BENCH_EVAL_FILE_SIZE_BYTES", cls.model_fields["bench_eval_file_size_bytes"].default)
        kwargs["bench_eval_network"] = _bool("WHALE_BENCH_EVAL_NETWORK", cls.model_fields["bench_eval_network"].default)
        kwargs["bench_eval_artifact_retention"] = _int("WHALE_BENCH_EVAL_ARTIFACT_RETENTION", cls.model_fields["bench_eval_artifact_retention"].default)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()
