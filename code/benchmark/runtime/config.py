"""Runtime configuration profiles for benchmark evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

try:
    from .._utils import _env_bool, _env_int
except ImportError:  # pragma: no cover - direct script execution
    from _utils import _env_bool, _env_int  # type: ignore


BenchmarkRuntimeProfile = Literal["python_strict", "repo_docker", "terminal_docker"]


@dataclass(frozen=True)
class BenchmarkRuntimeConfig:
    """Auditable resource and retention settings for benchmark runtimes."""

    profile: BenchmarkRuntimeProfile = "python_strict"
    cpu_seconds: int = 0
    memory_bytes: int = 0
    max_processes: int = 128
    file_size_bytes: int = 256 * 1024 * 1024
    network: bool = False
    artifact_retention: int = 200

    @classmethod
    def for_profile(cls, profile: BenchmarkRuntimeProfile) -> "BenchmarkRuntimeConfig":
        if profile in ("repo_docker", "terminal_docker"):
            return cls(
                profile=profile,
                max_processes=4096,
                file_size_bytes=1024 * 1024 * 1024,
            )
        return cls(profile="python_strict")

    @classmethod
    def from_env(
        cls,
        *,
        profile: BenchmarkRuntimeProfile = "python_strict",
    ) -> "BenchmarkRuntimeConfig":
        base = cls.for_profile(profile)
        return cls(
            profile=base.profile,
            cpu_seconds=_env_int("WHALE_BENCH_EVAL_CPU_SECONDS", base.cpu_seconds),
            memory_bytes=_env_int("WHALE_BENCH_EVAL_MEMORY_BYTES", base.memory_bytes),
            max_processes=_env_int("WHALE_BENCH_EVAL_MAX_PROCESSES", base.max_processes),
            file_size_bytes=_env_int(
                "WHALE_BENCH_EVAL_FILE_SIZE_BYTES",
                base.file_size_bytes,
            ),
            network=_env_bool("WHALE_BENCH_EVAL_NETWORK", base.network),
            artifact_retention=_env_int(
                "WHALE_BENCH_EVAL_ARTIFACT_RETENTION",
                base.artifact_retention,
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "cpu_seconds": self.cpu_seconds,
            "memory_bytes": self.memory_bytes,
            "max_processes": self.max_processes,
            "file_size_bytes": self.file_size_bytes,
            "network": self.network,
            "artifact_retention": self.artifact_retention,
        }

    def to_sandbox_limits(self, timeout_s: int):
        from ..safe_exec import BenchmarkSandboxLimits

        cpu_seconds = self.cpu_seconds if self.cpu_seconds > 0 else max(1, int(timeout_s) + 2)
        return BenchmarkSandboxLimits(
            cpu_seconds=cpu_seconds,
            address_space_bytes=self.memory_bytes,
            max_processes=self.max_processes,
            file_size_bytes=self.file_size_bytes,
        )


__all__ = ["BenchmarkRuntimeConfig", "BenchmarkRuntimeProfile"]
