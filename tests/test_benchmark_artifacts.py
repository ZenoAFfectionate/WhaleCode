"""Tests for benchmark runtime artifacts, feedback, and profiles."""

from __future__ import annotations

import json
from pathlib import Path

from hello_agents.benchmark.runtime import (
    BenchmarkArtifactStore,
    BenchmarkRuntimeConfig,
    EvalRequest,
    PythonSubprocessEnvironment,
    artifact_hint,
)
from hello_agents.benchmark.safe_exec import run_python_code
from hello_agents.core.config import Config


def _read_artifact(store_root: Path, relative_path: str) -> str:
    return (store_root.parent / relative_path).read_text(encoding="utf-8")


def test_python_runtime_records_success_artifacts(tmp_path):
    store = BenchmarkArtifactStore(tmp_path / "artifacts")
    env = PythonSubprocessEnvironment(artifact_store=store)

    result = env.evaluate(
        EvalRequest(
            code="print('artifact-ok')\n",
            cwd=tmp_path,
            timeout_s=5,
            artifact_stem="success-case",
            metadata={"task_id": "success"},
            visibility="public",
        )
    )

    assert result.passed is True
    assert result.status == "passed"
    assert result.artifacts.keys() >= {"script", "stdout", "stderr", "metadata"}
    assert "artifact-ok" in _read_artifact(store.root, result.artifacts["stdout"])
    assert "Full evaluator artifacts:" in result.feedback

    metadata = json.loads(_read_artifact(store.root, result.artifacts["metadata"]))
    assert metadata["status"] == "passed"
    assert metadata["request"]["metadata"] == {"task_id": "success"}
    assert metadata["runtime_config"]["profile"] == "python_strict"


def test_safe_exec_facade_records_default_artifact(monkeypatch, tmp_path):
    artifact_root = tmp_path / "default-artifacts"
    monkeypatch.setenv("WHALE_BENCH_EVAL_ARTIFACT_DIR", str(artifact_root))

    result = run_python_code(
        "print('facade-artifact')\n",
        cwd=tmp_path,
        timeout=5,
        artifact_stem="facade-case",
    )

    assert result.returncode == 0
    assert result.artifacts
    assert "metadata" in result.artifacts
    assert "Full evaluator artifacts:" in result.output
    metadata = json.loads(_read_artifact(artifact_root, result.artifacts["metadata"]))
    assert metadata["status"] == "passed"


def test_python_runtime_records_failure_and_timeout_artifacts(tmp_path):
    store = BenchmarkArtifactStore(tmp_path / "artifacts")
    env = PythonSubprocessEnvironment(artifact_store=store)

    failed = env.evaluate(
        EvalRequest(
            code="raise SystemExit(7)\n",
            cwd=tmp_path,
            timeout_s=5,
            artifact_stem="failure-case",
        )
    )
    timed_out = env.evaluate(
        EvalRequest(
            code="while True:\n    pass\n",
            cwd=tmp_path,
            timeout_s=1,
            artifact_stem="timeout-case",
        )
    )

    assert failed.passed is False
    assert failed.status == "failed"
    assert failed.artifacts["metadata"]
    assert timed_out.passed is False
    assert timed_out.status == "timeout"
    assert "TIMEOUT" in timed_out.feedback

    timeout_metadata = json.loads(_read_artifact(store.root, timed_out.artifacts["metadata"]))
    assert timeout_metadata["status"] == "timeout"
    assert timeout_metadata["returncode"] == -9


def test_artifact_store_enforces_retention(tmp_path):
    store = BenchmarkArtifactStore(tmp_path / "artifacts", retention=2)

    store.record_eval(stem="one", code="print(1)", stdout="1", stderr="", metadata={})
    store.record_eval(stem="two", code="print(2)", stdout="2", stderr="", metadata={})
    store.record_eval(stem="three", code="print(3)", stdout="3", stderr="", metadata={})

    assert len([path for path in store.root.iterdir() if path.is_dir()]) == 2


def test_benchmark_runtime_config_profiles_and_env(monkeypatch):
    monkeypatch.setenv("WHALE_BENCH_EVAL_CPU_SECONDS", "11")
    monkeypatch.setenv("WHALE_BENCH_EVAL_MEMORY_BYTES", "123456")
    monkeypatch.setenv("WHALE_BENCH_EVAL_MAX_PROCESSES", "9")
    monkeypatch.setenv("WHALE_BENCH_EVAL_FILE_SIZE_BYTES", "789")
    monkeypatch.setenv("WHALE_BENCH_EVAL_NETWORK", "true")
    monkeypatch.setenv("WHALE_BENCH_EVAL_ARTIFACT_RETENTION", "17")

    runtime_config = BenchmarkRuntimeConfig.from_env(profile="python_strict")
    app_config = Config.from_env()

    assert runtime_config.to_metadata() == {
        "profile": "python_strict",
        "cpu_seconds": 11,
        "memory_bytes": 123456,
        "max_processes": 9,
        "file_size_bytes": 789,
        "network": True,
        "artifact_retention": 17,
    }
    assert app_config.bench_eval_cpu_seconds == 11
    assert app_config.bench_eval_memory_bytes == 123456
    assert app_config.bench_eval_max_processes == 9
    assert app_config.bench_eval_file_size_bytes == 789
    assert app_config.bench_eval_network is True
    assert app_config.bench_eval_artifact_retention == 17


def test_artifact_hint_prefers_metadata_path():
    assert artifact_hint({"stdout": "a/stdout.txt", "metadata": "a/metadata.json"}) == (
        "Full evaluator artifacts: a/metadata.json"
    )
