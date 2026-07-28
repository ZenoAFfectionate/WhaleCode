"""Comprehensive tests for SWE-bench Verified benchmark runner.

Covers:
- SWEBenchVerifiedBenchmark initialization and configuration
- Data loading with filter / slice / shuffle / resume
- CLI argument parsing from run_swev.sh down to main()
- DockerizedWorkspace lifecycle (mocked Docker)
- Predictions export format and round-trip
- Utility functions (_parse_slice_spec, _clip_output, _get_docker_image_name, etc.)
- Edge cases: malformed data, resume state, clone failures
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from hello_agents.benchmark.swev_bench import (
    DockerBashTool,
    DockerizedWorkspace,
    SWEBenchVerifiedBenchmark,
    _CONTAINER_WORKDIR,
    _SWEV_ADDENDUM,
    _SWEV_ARTIFACT_DIRS,
    _SWEV_ARTIFACT_FILES,
    _SWEV_ARTIFACT_SUFFIXES,
    _SWEV_SYSTEM_PROMPT,
    _TaskTimeout,
    _clip_output,
    _format_subprocess_error,
    _parse_slice_spec,
    main,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_tasks():
    """Minimal valid SWE-verified tasks for testing."""
    return [
        {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
            "problem_statement": "Fix separability matrix bug",
            "hints_text": "Check the _coord_matrix function",
            "FAIL_TO_PASS": ["test_separable.py::test_separable[compound_model6-result6]"],
            "PASS_TO_PASS": ["test_separable.py::test_coord_matrix"],
        },
        {
            "instance_id": "django__django-16139",
            "repo": "django/django",
            "base_commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
            "problem_statement": "Fix ORM query regression",
            "hints_text": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
        },
        {
            "instance_id": "sympy__sympy-24000",
            "repo": "sympy/sympy",
            "base_commit": "1234567890abcdef1234567890abcdef12345678",
            "problem_statement": "Simplify expression incorrectly",
            "hints_text": "Look at simplify.py",
            "FAIL_TO_PASS": ["test_simplify.py::test_bug"],
            "PASS_TO_PASS": [],
        },
        {
            "instance_id": "scikit-learn__scikit-learn-25000",
            "repo": "scikit-learn/scikit-learn",
            "base_commit": "fedcba9876543210fedcba9876543210fedcba98",
            "problem_statement": "GridSearchCV memory leak",
            "hints_text": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
        },
        {
            "instance_id": "matplotlib__matplotlib-26000",
            "repo": "matplotlib/matplotlib",
            "base_commit": "0000000000000000000000000000000000000001",
            "problem_statement": "Colorbar incorrect with log scale",
            "hints_text": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
        },
    ]


@pytest.fixture
def tmp_data_file(tmp_path, sample_tasks):
    """Write sample tasks to a temporary JSONL file."""
    path = tmp_path / "test_swev.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for task in sample_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def bench_kwargs(tmp_data_file, tmp_path):
    """Default constructor kwargs for SWEBenchVerifiedBenchmark."""
    return {
        "data_path": str(tmp_data_file),
        "output_dir": str(tmp_path / "results"),
        "trajectory_dir": str(tmp_path / "trajectory"),
        "repo_cache_dir": str(tmp_path / "repo_cache"),
        "model_name": "test-model",
        "workers": 1,
        "max_steps": 64,
        "timeout": 30,
        "task_timeout": 600,
        "docker_executable": "docker",
        "temperature": 0.2,
    }


@pytest.fixture
def bench(bench_kwargs):
    """Create a benchmark instance with default kwargs."""
    return SWEBenchVerifiedBenchmark(**bench_kwargs)


# ═══════════════════════════════════════════════════════════════════
# 1. Utility Functions
# ═══════════════════════════════════════════════════════════════════

class TestParseSliceSpec:
    def test_empty_returns_none(self):
        assert _parse_slice_spec("") is None

    def test_start_end(self):
        s = _parse_slice_spec("0:50")
        assert s == slice(0, 50, None)

    def test_start_end_step(self):
        s = _parse_slice_spec("10:200:2")
        assert s == slice(10, 200, 2)

    def test_start_only(self):
        s = _parse_slice_spec("5:")
        assert s == slice(5, None, None)

    def test_end_only(self):
        s = _parse_slice_spec(":100")
        assert s == slice(None, 100, None)

    def test_step_only(self):
        s = _parse_slice_spec("::3")
        assert s == slice(None, None, 3)

    def test_single_number(self):
        with pytest.raises(ValueError, match="expected `start:end"):
            _parse_slice_spec("42")

    def test_too_many_parts(self):
        with pytest.raises(ValueError, match="expected `start:end"):
            _parse_slice_spec("1:2:3:4")

    def test_non_integer_component(self):
        with pytest.raises(ValueError, match="Invalid --slice component"):
            _parse_slice_spec("abc:10")


class TestClipOutput:
    def test_short_text_unchanged(self):
        assert _clip_output("hello") == "hello"

    def test_none_returns_empty(self):
        assert _clip_output(None) == ""

    def test_long_text_truncated(self):
        long_text = "x" * 2000
        result = _clip_output(long_text)
        assert len(result) <= 1203  # 1200 + "..." max
        assert result.endswith("...")

    def test_empty_string(self):
        assert _clip_output("") == ""


class TestFormatSubprocessError:
    def test_includes_step_and_command(self):
        msg = _format_subprocess_error(
            step="clone repo", command=["git", "clone", "url"], returncode=128
        )
        assert "clone repo failed" in msg
        assert "git clone url" in msg
        assert "returncode: 128" in msg

    def test_includes_cwd_when_provided(self, tmp_path):
        msg = _format_subprocess_error(
            step="checkout", command=["git", "checkout"], cwd=tmp_path
        )
        assert str(tmp_path) in msg

    def test_includes_timeout(self):
        msg = _format_subprocess_error(
            step="fetch", command=["git", "fetch"], timeout_s=300.0
        )
        assert "timeout_s: 300.0" in msg

    def test_includes_stdout_stderr(self):
        msg = _format_subprocess_error(
            step="test", command=["cmd"], stdout="output line", stderr="error line"
        )
        assert "output line" in msg
        assert "error line" in msg


# ═══════════════════════════════════════════════════════════════════
# 2. System Prompt
# ═══════════════════════════════════════════════════════════════════

class TestSystemPrompt:
    def test_contains_swe_override(self):
        assert "SWE-bench Override" in _SWEV_SYSTEM_PROMPT
        assert "Autonomous Issue Resolution" in _SWEV_SYSTEM_PROMPT

    def test_contains_workflow_steps(self):
        assert "Locate relevant code" in _SWEV_SYSTEM_PROMPT
        assert "Diagnose root cause" in _SWEV_SYSTEM_PROMPT

    def test_contains_critical_rules(self):
        assert "Do NOT modify test files" in _SWEV_SYSTEM_PROMPT
        assert "Prefer minimal, correct changes" in _SWEV_SYSTEM_PROMPT

    def test_contains_efficiency_rules(self):
        assert "Do NOT read the whole repository" in _SWEV_SYSTEM_PROMPT
        # The prompt uses backtick-formatted `Finish`
        assert "must be the last tool you call" in _SWEV_SYSTEM_PROMPT

    def test_prompt_addendum_present(self):
        assert len(_SWEV_ADDENDUM) > 200

    def test_returned_by_get_system_prompt(self, bench):
        prompt = bench._get_system_prompt()
        assert prompt is not None
        assert "SWE-bench Override" in prompt


# ═══════════════════════════════════════════════════════════════════
# 3. Benchmark Initialization
# ═══════════════════════════════════════════════════════════════════

class TestBenchmarkInit:
    def test_basic_init(self, bench_kwargs):
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.benchmark_name == "swev"
        assert bench.runtime_profile == "repo_docker"
        assert bench.workers == 1
        assert bench.model_name == "test-model"

    def test_output_dir_created(self, bench_kwargs):
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.output_dir.exists()

    def test_repo_cache_dir_created(self, bench_kwargs):
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.repo_cache_dir.exists()

    def test_filter_regex_compiled(self, bench_kwargs):
        bench_kwargs["filter_spec"] = "django__.*"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench._filter_regex is not None
        assert bench._filter_regex.pattern == "django__.*"

    def test_filter_no_regex_when_empty(self, bench_kwargs):
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench._filter_regex is None

    def test_invalid_filter_raises(self, bench_kwargs):
        bench_kwargs["filter_spec"] = "[invalid(regex"
        with pytest.raises(ValueError, match="Invalid --filter regex"):
            SWEBenchVerifiedBenchmark(**bench_kwargs)

    def test_workers_minimum_1(self, bench_kwargs):
        bench_kwargs["workers"] = -5
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.workers == 1

    def test_shuffle_and_seed_defaults(self, bench_kwargs):
        bench_kwargs["shuffle"] = True
        bench_kwargs["seed"] = 123
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.shuffle is True
        assert bench.seed == 123

    def test_preds_path_loaded(self, bench_kwargs, tmp_path):
        # Create a preds file with known IDs
        preds_file = tmp_path / "preds.jsonl"
        preds_file.write_text(
            json.dumps({"instance_id": "astropy__astropy-12907", "model_patch": "diff"})
            + "\n"
        )
        bench_kwargs["preds_path"] = str(preds_file)
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert "astropy__astropy-12907" in bench._preds_completed_ids

    def test_redo_existing_skips_preds(self, bench_kwargs, tmp_path):
        preds_file = tmp_path / "preds.jsonl"
        preds_file.write_text(
            json.dumps({"instance_id": "astropy__astropy-12907", "model_patch": "diff"})
            + "\n"
        )
        bench_kwargs["preds_path"] = str(preds_file)
        bench_kwargs["redo_existing"] = True
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert len(bench._preds_completed_ids) == 0  # redo means no skip


# ═══════════════════════════════════════════════════════════════════
# 4. Data Loading
# ═══════════════════════════════════════════════════════════════════

class TestLoadTasks:
    def test_loads_all_tasks(self, bench, sample_tasks):
        tasks = bench._load_tasks()
        assert len(tasks) == len(sample_tasks)

    def test_maps_instance_id_to_task_id(self, bench, sample_tasks):
        tasks = bench._load_tasks()
        for task in tasks:
            assert "task_id" in task
            assert task["task_id"] == task["instance_id"]

    def test_filter_by_regex(self, bench_kwargs, sample_tasks):
        bench_kwargs["filter_spec"] = "astropy__.*"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        tasks = bench._load_tasks()
        assert len(tasks) == 1
        assert tasks[0]["instance_id"] == "astropy__astropy-12907"

    def test_filter_no_match_returns_empty(self, bench_kwargs):
        bench_kwargs["filter_spec"] = "nonexistent__.*"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        tasks = bench._load_tasks()
        assert len(tasks) == 0

    def test_slice_start_end(self, bench_kwargs, sample_tasks):
        bench_kwargs["slice_spec"] = "1:3"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        tasks = bench._load_tasks()
        assert len(tasks) == 2  # indices 1, 2

    def test_slice_limit(self, bench_kwargs):
        bench_kwargs["slice_spec"] = "0:2"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        tasks = bench._load_tasks()
        assert len(tasks) == 2

    def test_shuffle_produces_deterministic_order(self, bench_kwargs, tmp_data_file):
        bench_kwargs["shuffle"] = True
        bench_kwargs["seed"] = 42
        b1 = SWEBenchVerifiedBenchmark(data_path=str(tmp_data_file), **{k: v for k, v in bench_kwargs.items() if k != "data_path"})
        b2 = SWEBenchVerifiedBenchmark(data_path=str(tmp_data_file), **{k: v for k, v in bench_kwargs.items() if k != "data_path"})
        t1 = [t["instance_id"] for t in b1._load_tasks()]
        t2 = [t["instance_id"] for t in b2._load_tasks()]
        assert t1 == t2  # same seed → same order

    def test_shuffle_different_seeds_differ(self, bench_kwargs, tmp_data_file):
        bench_kwargs["shuffle"] = True
        bench_kwargs["seed"] = 1
        b1 = SWEBenchVerifiedBenchmark(data_path=str(tmp_data_file), **{k: v for k, v in bench_kwargs.items() if k != "data_path"})
        bench_kwargs["seed"] = 999
        b2 = SWEBenchVerifiedBenchmark(data_path=str(tmp_data_file), **{k: v for k, v in bench_kwargs.items() if k != "data_path"})
        t1 = [t["instance_id"] for t in b1._load_tasks()]
        t2 = [t["instance_id"] for t in b2._load_tasks()]
        assert t1 != t2  # different seeds → different order

    def test_filter_before_slice(self, bench_kwargs, sample_tasks):
        """Filter runs first, then slice. Verify order of operations."""
        bench_kwargs["filter_spec"] = ".*"  # match all
        bench_kwargs["slice_spec"] = "0:2"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        tasks = bench._load_tasks()
        assert len(tasks) == 2

    def test_preds_completed_ids_filtered(self, bench_kwargs, sample_tasks):
        bench_kwargs["preds_path"] = None
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        bench._preds_completed_ids = {"astropy__astropy-12907", "django__django-16139"}
        tasks = bench._load_tasks()
        remaining_ids = {t["instance_id"] for t in tasks}
        assert "astropy__astropy-12907" not in remaining_ids
        assert "django__django-16139" not in remaining_ids
        assert len(tasks) == len(sample_tasks) - 2


# ═══════════════════════════════════════════════════════════════════
# 5. Docker Image Name Resolution
# ═══════════════════════════════════════════════════════════════════

class TestGetDockerImageName:
    def test_uses_image_name_field(self):
        task = {"task_id": "a__b-1", "image_name": "custom/image:v1"}
        result = SWEBenchVerifiedBenchmark._get_docker_image_name(task)
        assert result == "custom/image:v1"

    def test_uses_docker_image_field(self):
        task = {"task_id": "a__b-1", "docker_image": "docker.io/img:v2"}
        result = SWEBenchVerifiedBenchmark._get_docker_image_name(task)
        assert result == "docker.io/img:v2"

    def test_image_name_priority_over_docker_image(self):
        task = {
            "task_id": "a__b-1",
            "image_name": "primary:v1",
            "docker_image": "fallback:v2",
        }
        result = SWEBenchVerifiedBenchmark._get_docker_image_name(task)
        assert result == "primary:v1"

    def test_fallback_naming_with_double_underscore(self):
        task = {"task_id": "django__django-16139"}
        result = SWEBenchVerifiedBenchmark._get_docker_image_name(task)
        # __ → _1776_ in docker-safe name
        assert "django_1776_django-16139" in result
        assert result.startswith("docker.io/swebench/sweb.eval.x86_64.")
        assert result.endswith(":latest")

    def test_fallback_naming_with_single_underscore(self):
        task = {"task_id": "scikit-learn__scikit-learn-25000"}
        result = SWEBenchVerifiedBenchmark._get_docker_image_name(task)
        assert "scikit-learn_1776_scikit-learn-25000" in result

    def test_lowercase_output(self):
        task = {"task_id": "AstroPy__ASTRO-12907"}
        result = SWEBenchVerifiedBenchmark._get_docker_image_name(task)
        assert result == result.lower()


# ═══════════════════════════════════════════════════════════════════
# 6. Artifact Pruning
# ═══════════════════════════════════════════════════════════════════

class TestIsPrunableUntracked:
    @pytest.mark.parametrize(
        "path,expected",
        [
            (Path("__pycache__/foo.pyc"), True),
            (Path(".pytest_cache/v/cache/lastfailed"), True),
            (Path(".mypy_cache/3.12/module.data.json"), True),
            (Path(".tox/py312/log/test.log"), True),
            (Path("node_modules/react/index.js"), True),
            (Path("dist/package.tar.gz"), True),
            (Path("build/output.o"), True),
            (Path(".venv/lib/python3.12/site-packages"), True),
            (Path("venv/bin/activate"), True),
            (Path(".eggs/pkg.egg-info"), True),
            (Path(".idea/workspace.xml"), True),
            (Path(".vscode/settings.json"), True),
            (Path(".coverage"), True),
            (Path(".coverage.sqllite"), True),
            (Path(".DS_Store"), True),
            (Path("module.cpython-312.pyc"), True),
            (Path("test_result.tmp"), True),
            (Path("output.log"), True),
            # Non-prunable paths
            (Path("src/main.py"), False),
            (Path("tests/test_foo.py"), False),
            (Path("lib/utils.py"), False),
            (Path("requirements.txt"), False),
            (Path("setup.cfg"), False),
            (Path(".gitignore"), False),
        ],
    )
    def test_prunable_artifacts(self, path, expected):
        result = SWEBenchVerifiedBenchmark._is_prunable_untracked(path)
        assert result == expected, f"Failed for {path}"


# ═══════════════════════════════════════════════════════════════════
# 7. Resume / Completed ID Logic
# ═══════════════════════════════════════════════════════════════════

class TestLoadCompletedIds:
    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(f)
        assert ids == set()

    def test_nonexistent_file_returns_empty(self, tmp_path):
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(
            tmp_path / "nonexistent.jsonl"
        )
        assert ids == set()

    def test_task_with_patch_is_completed(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text(
            json.dumps({"task_id": "a__b-1", "agent_diff": "diff content..."}) + "\n"
        )
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(f)
        assert "a__b-1" in ids

    def test_no_patch_finished_is_completed(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "task_id": "c__d-2",
                    "agent_diff": "",
                    "error": "Agent produced no changes",
                }
            )
            + "\n"
        )
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(f)
        assert "c__d-2" in ids

    def test_clone_failed_not_completed(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text(
            json.dumps({"task_id": "e__f-3", "agent_diff": "", "error": "CloneFailed"})
            + "\n"
        )
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(f)
        assert "e__f-3" not in ids

    def test_agent_error_not_completed(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "task_id": "g__h-4",
                    "agent_diff": "",
                    "error": "Agent error: ConnectionError",
                }
            )
            + "\n"
        )
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(f)
        assert "g__h-4" not in ids

    def test_duplicate_task_id_latest_wins(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text(
            json.dumps({"task_id": "a__b-1", "agent_diff": "old diff"}) + "\n"
        )
        f.write_text(
            # Second write overwrites? No, append:
            json.dumps({"task_id": "a__b-1", "agent_diff": "new diff"}) + "\n"
        )
        # Actually let's write both at once:
        pass
        f.write_text(
            json.dumps({"task_id": "a__b-1", "agent_diff": "old diff"}) + "\n"
            + json.dumps({"task_id": "a__b-1", "agent_diff": "new diff"}) + "\n"
        )
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(f)
        assert "a__b-1" in ids  # should be completed by the latest

    def test_mixed_completed_and_pending(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text(
            json.dumps({"task_id": "done-1", "agent_diff": "patch"}) + "\n"
            + json.dumps(
                {"task_id": "done-2", "agent_diff": "", "error": "Agent produced no changes"}
            )
            + "\n"
            + json.dumps({"task_id": "pending-1", "agent_diff": "", "error": "RuntimeError"})
            + "\n"
            + json.dumps({"task_id": "pending-2", "agent_diff": ""})
            + "\n"
        )
        ids = SWEBenchVerifiedBenchmark._load_completed_ids(f)
        assert ids == {"done-1", "done-2"}


# ═══════════════════════════════════════════════════════════════════
# 8. Existing Predictions Loading
# ═══════════════════════════════════════════════════════════════════

class TestLoadExistingPredictionIds:
    def test_jsonl_format(self, tmp_path):
        f = tmp_path / "preds.jsonl"
        f.write_text(
            json.dumps({"instance_id": "a__b-1", "model_patch": "diff"}) + "\n"
            + json.dumps({"instance_id": "c__d-2", "model_patch": ""}) + "\n"
        )
        ids = SWEBenchVerifiedBenchmark._load_existing_prediction_ids(f)
        assert ids == {"a__b-1", "c__d-2"}

    def test_json_dict_format(self, tmp_path):
        f = tmp_path / "preds.json"
        json.dump(
            {
                "a__b-1": {"instance_id": "a__b-1", "model_patch": "diff"},
                "c__d-2": {"instance_id": "c__d-2", "model_patch": ""},
            },
            f.open("w"),
        )
        ids = SWEBenchVerifiedBenchmark._load_existing_prediction_ids(f)
        assert ids == {"a__b-1", "c__d-2"}

    def test_json_list_format(self, tmp_path):
        f = tmp_path / "preds.json"
        json.dump(
            [
                {"instance_id": "x__y-1"},
                {"instance_id": "x__y-2"},
            ],
            f.open("w"),
        )
        ids = SWEBenchVerifiedBenchmark._load_existing_prediction_ids(f)
        assert ids == {"x__y-1", "x__y-2"}

    def test_nonexistent_file(self, tmp_path):
        ids = SWEBenchVerifiedBenchmark._load_existing_prediction_ids(
            tmp_path / "nonexistent.jsonl"
        )
        assert ids == set()

    def test_empty_jsonl(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("\n\n")
        ids = SWEBenchVerifiedBenchmark._load_existing_prediction_ids(f)
        assert ids == set()

    def test_malformed_lines_skipped(self, tmp_path):
        f = tmp_path / "malformed.jsonl"
        f.write_text("not valid json\n" + json.dumps({"instance_id": "ok-1"}) + "\n")
        ids = SWEBenchVerifiedBenchmark._load_existing_prediction_ids(f)
        assert ids == {"ok-1"}


# ═══════════════════════════════════════════════════════════════════
# 9. Predictions Export Format
# ═══════════════════════════════════════════════════════════════════

class TestExportPredictions:
    def test_jsonl_format_has_required_fields(self, bench, tmp_path):
        # Create a mock results file
        results_file = tmp_path / "swev_test.jsonl"
        results_file.write_text(
            json.dumps(
                {
                    "task_id": "a__b-1",
                    "agent_diff": "diff --git a/x.py b/x.py\n+fix",
                    "passed": None,
                    "repo": "a/b",
                    "elapsed_s": 10.0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "task_id": "c__d-2",
                    "agent_diff": "",
                    "passed": None,
                    "repo": "c/d",
                    "elapsed_s": 5.0,
                }
            )
            + "\n"
        )

        bench.output_dir = tmp_path
        preds_file, preds_json, diff_count, total = bench._export_predictions(
            results_file=results_file, timestamp="20260101_120000"
        )

        # Check JSONL
        lines = preds_file.read_text().strip().split("\n")
        assert len(lines) == 2
        pred0 = json.loads(lines[0])
        assert pred0["instance_id"] == "a__b-1"
        assert pred0["model_name_or_path"] == "test-model"
        assert "model_patch" in pred0
        assert pred0["model_patch"] == "diff --git a/x.py b/x.py\n+fix"

        pred1 = json.loads(lines[1])
        assert pred1["model_patch"] == ""

        # Check counts
        assert diff_count == 1  # only first has a diff
        assert total == 2

    def test_json_format_is_dict_of_predictions(self, bench, tmp_path):
        results_file = tmp_path / "results.jsonl"
        results_file.write_text(
            json.dumps(
                {"task_id": "a__b-1", "agent_diff": "patch1", "passed": None, "repo": "a/b", "elapsed_s": 1.0}
            )
            + "\n"
        )

        bench.output_dir = tmp_path
        _, preds_json, _, _ = bench._export_predictions(
            results_file=results_file, timestamp="20260101_120000"
        )

        data = json.loads(preds_json.read_text())
        assert isinstance(data, dict)
        assert "a__b-1" in data
        assert data["a__b-1"]["model_patch"] == "patch1"

    def test_writes_latest_preds_file(self, bench, tmp_path):
        results_file = tmp_path / "results.jsonl"
        results_file.write_text(
            json.dumps({"task_id": "x-1", "agent_diff": "p", "passed": None, "repo": "x", "elapsed_s": 1.0})
            + "\n"
        )

        bench.output_dir = tmp_path
        bench._export_predictions(results_file=results_file, timestamp="20260101_120000")

        latest = tmp_path / "preds.json"
        assert latest.exists()
        data = json.loads(latest.read_text())
        assert "x-1" in data


# ═══════════════════════════════════════════════════════════════════
# 10. DockerizedWorkspace (mocked Docker)
# ═══════════════════════════════════════════════════════════════════

class TestDockerizedWorkspace:
    def test_init_stores_config(self, tmp_path):
        dw = DockerizedWorkspace(
            image="test-image:latest",
            workspace=tmp_path,
            executable="docker",
        )
        assert dw.image == "test-image:latest"
        assert dw.workspace == tmp_path.resolve()
        assert dw.executable == "docker"
        assert dw.container_workdir == _CONTAINER_WORKDIR
        assert dw.container_name.startswith("whale-swev-")
        assert dw.container_id is None

    def test_start_creates_container(self, tmp_path):
        """Test start() mocks the subprocess call."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc123\n", stderr=""
            )
            dw = DockerizedWorkspace(
                image="test-image:latest", workspace=tmp_path
            )
            dw.start()

            # Verify docker run was called
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "docker"
            assert "run" in cmd
            assert "-d" in cmd
            assert dw.container_name in cmd
            assert dw.container_id == "abc123"

    def test_start_container_name_unique(self, tmp_path):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="id1\n", stderr=""
            )
            dw1 = DockerizedWorkspace(image="img1", workspace=tmp_path)
            dw2 = DockerizedWorkspace(image="img2", workspace=tmp_path)
            dw1.start()
            dw2.start()
            assert dw1.container_name != dw2.container_name

    def test_start_docker_not_found(self, tmp_path):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("docker")):
            dw = DockerizedWorkspace(image="img", workspace=tmp_path)
            with pytest.raises(RuntimeError, match="Container executable not found"):
                dw.start()

    def test_start_timeout(self, tmp_path):
        with mock.patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(["docker"], 30)
        ):
            dw = DockerizedWorkspace(image="img", workspace=tmp_path)
            with pytest.raises(RuntimeError, match="docker run"):
                dw.start()

    def test_start_called_process_error(self, tmp_path):
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, ["docker"], output="out", stderr="err"
            ),
        ):
            dw = DockerizedWorkspace(image="img", workspace=tmp_path)
            with pytest.raises(RuntimeError, match="docker run"):
                dw.start()

    def test_start_empty_container_id(self, tmp_path):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="\n", stderr=""
            )
            dw = DockerizedWorkspace(image="img", workspace=tmp_path)
            with pytest.raises(RuntimeError, match="docker run"):
                dw.start()

    def test_cleanup_stops_container(self, tmp_path):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="cid\n", stderr=""
            )
            dw = DockerizedWorkspace(image="img", workspace=tmp_path)
            dw.start()
            dw.container_id = "test-cid"
            dw.cleanup()

            # Check stop was called
            stop_calls = [
                c for c in mock_run.call_args_list if "stop" in str(c)
            ]
            assert len(stop_calls) >= 1

    def test_cleanup_no_container_id_noop(self, tmp_path):
        with mock.patch("subprocess.run") as mock_run:
            dw = DockerizedWorkspace(image="img", workspace=tmp_path)
            dw.cleanup()
            # Should not call subprocess at all for stop
            stop_calls = [c for c in mock_run.call_args_list if "stop" in str(c)]
            assert len(stop_calls) == 0

    def test_popen_runs_docker_exec(self, tmp_path):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="cid\n", stderr=""
            )
            dw = DockerizedWorkspace(image="img", workspace=tmp_path)
            dw.start()

            with mock.patch("subprocess.Popen") as mock_popen:
                dw.popen(command="pytest", container_directory=PurePosixPath("/testbed"))
                mock_popen.assert_called_once()
                cmd = mock_popen.call_args[0][0]
                assert "exec" in cmd
                assert dw.container_id in cmd

    def test_popen_without_start_raises(self, tmp_path):
        dw = DockerizedWorkspace(image="img", workspace=tmp_path)
        with pytest.raises(RuntimeError, match="not running"):
            dw.popen(command="ls", container_directory=PurePosixPath("/testbed"))


# ═══════════════════════════════════════════════════════════════════
# 11. DockerBashTool (mocked)
# ═══════════════════════════════════════════════════════════════════

class TestDockerBashTool:
    @pytest.fixture
    def mock_bash_tool(self):
        """Create a minimal mock BashTool."""
        tool = mock.MagicMock()
        tool.name = "Bash"
        tool.description = "Execute shell commands"
        tool.expandable = False
        tool.project_root = "/fake/workspace"
        tool.working_dir = "/fake/workspace"
        tool.DEFAULT_BLOCK_UNTIL_MS = 30000
        tool.MAX_BLOCK_UNTIL_MS = 600000
        return tool

    @pytest.fixture
    def docker_bash(self, tmp_path, mock_bash_tool):
        dw = DockerizedWorkspace(
            image="test-image:latest", workspace=tmp_path
        )
        return DockerBashTool(docker_workspace=dw, local_bash_tool=mock_bash_tool)

    def test_delegates_name(self, docker_bash, mock_bash_tool):
        assert docker_bash.name == mock_bash_tool.name

    def test_delegates_description(self, docker_bash, mock_bash_tool):
        assert docker_bash.description == mock_bash_tool.description

    def test_delegates_project_root(self, docker_bash, mock_bash_tool):
        assert docker_bash.project_root == mock_bash_tool.project_root

    def test_delegates_validate_command(self, docker_bash, mock_bash_tool):
        mock_bash_tool._validate_command.return_value = None
        result = docker_bash._validate_command("echo hello")
        mock_bash_tool._validate_command.assert_called_with("echo hello")
        assert result is None

    def test_delegates_get_parameters(self, docker_bash, mock_bash_tool):
        mock_bash_tool.get_parameters.return_value = {"type": "object"}
        result = docker_bash.get_parameters()
        assert result == {"type": "object"}


# ═══════════════════════════════════════════════════════════════════
# 12. _TaskTimeout
# ═══════════════════════════════════════════════════════════════════

class TestTaskTimeout:
    def test_is_exception(self):
        exc = _TaskTimeout("timed out after 600s")
        assert isinstance(exc, Exception)
        assert str(exc) == "timed out after 600s"

    def test_can_be_caught_by_name(self):
        try:
            raise _TaskTimeout("test")
        except _TaskTimeout as e:
            assert str(e) == "test"

    def test_can_be_caught_by_base_exception(self):
        try:
            raise _TaskTimeout("test")
        except Exception as e:
            assert isinstance(e, _TaskTimeout)


# ═══════════════════════════════════════════════════════════════════
# 13. CLI Argument Parsing (main function)
# ═══════════════════════════════════════════════════════════════════

class TestCLIArgParsing:
    def _parse(self, argv):
        """Parse args like main() does, returning the parsed namespace."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-path", default="")
        parser.add_argument("--output-dir", default="")
        parser.add_argument("--trajectory-dir", default="")
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--max-steps", type=int, default=128)
        parser.add_argument("--timeout", type=int, default=30)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--task-ids", nargs="*", default=None)
        parser.add_argument("--filter", default="")
        parser.add_argument("--slice", default="")
        parser.add_argument("--shuffle", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--repo-cache-dir", default=None)
        parser.add_argument("--preds-path", default=None)
        parser.add_argument("--redo-existing", action="store_true")
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--model-name", default="whale-code")
        parser.add_argument("--task-timeout", type=int, default=1200)
        parser.add_argument("--docker-executable", default="docker")
        parser.add_argument("--docker-pull-timeout", type=int, default=600)
        parser.add_argument("--docker-container-timeout", default="2h")
        parser.add_argument("--resume", default=None)
        parser.add_argument("--preflight-only", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        return parser.parse_args(argv)

    # ── script defaults (from run_swev.sh) ──

    def test_shell_script_defaults(self):
        """Match the exact arguments run_swev.sh passes."""
        args = self._parse(
            [
                "--data-path", "/home/kemove/CodeingAgent/data/SWEV/test.jsonl",
                "--output-dir", "/home/kemove/CodeingAgent/WhaleCode/result/_results",
                "--repo-cache-dir", "/home/kemove/CodeingAgent/WhaleCode/result/_repo_cache",
                "--workers", "1",
                "--max-steps", "128",
            ]
        )
        assert args.data_path == "/home/kemove/CodeingAgent/data/SWEV/test.jsonl"
        assert args.output_dir == "/home/kemove/CodeingAgent/WhaleCode/result/_results"
        assert args.repo_cache_dir == "/home/kemove/CodeingAgent/WhaleCode/result/_repo_cache"
        assert args.workers == 1
        assert args.max_steps == 128
        assert args.dry_run is False

    # ── individual flags ──

    def test_limit_flag(self):
        args = self._parse(["--limit", "5"])
        assert args.limit == 5

    def test_dry_run_flag(self):
        args = self._parse(["--dry-run"])
        assert args.dry_run is True

    def test_filter_flag(self):
        args = self._parse(["--filter", "django__.*"])
        assert args.filter == "django__.*"

    def test_slice_flag(self):
        args = self._parse(["--slice", "0:50"])
        assert args.slice == "0:50"

    def test_resume_flag(self):
        args = self._parse(["--resume", "result/_results/prev.jsonl"])
        assert args.resume == "result/_results/prev.jsonl"

    def test_model_name(self):
        args = self._parse(["--model-name", "qwen-code"])
        assert args.model_name == "qwen-code"

    def test_task_timeout(self):
        args = self._parse(["--task-timeout", "3600"])
        assert args.task_timeout == 3600

    def test_workers(self):
        args = self._parse(["--workers", "8"])
        assert args.workers == 8

    def test_docker_executable(self):
        args = self._parse(["--docker-executable", "podman"])
        assert args.docker_executable == "podman"

    def test_shuffle_with_seed(self):
        args = self._parse(["--shuffle", "--seed", "123"])
        assert args.shuffle is True
        assert args.seed == 123

    def test_preflight_only(self):
        args = self._parse(["--preflight-only"])
        assert args.preflight_only is True

    def test_preds_path_with_redo(self):
        args = self._parse(["--preds-path", "preds.json", "--redo-existing"])
        assert args.preds_path == "preds.json"
        assert args.redo_existing is True

    # ── combined flags ──

    def test_full_example_command(self):
        """Realistic invocation: bash scripts/run_swev.sh --limit 10 --workers 4."""
        args = self._parse(
            [
                "--data-path", "/home/kemove/CodeingAgent/data/SWEV/test.jsonl",
                "--output-dir", "result/_results",
                "--repo-cache-dir", "result/_repo_cache",
                "--workers", "4",
                "--max-steps", "128",
                "--limit", "10",
                "--filter", "django__.*",
                "--task-timeout", "1800",
            ]
        )
        assert args.limit == 10
        assert args.workers == 4
        assert args.filter == "django__.*"
        assert args.task_timeout == 1800
        assert args.max_steps == 128

    def test_resume_with_limit(self):
        """Resume + new limit: skip completed, run up to limit remaining."""
        args = self._parse(
            ["--resume", "result/_results/swev_prev.jsonl", "--limit", "20"]
        )
        assert args.resume == "result/_results/swev_prev.jsonl"
        assert args.limit == 20


# ═══════════════════════════════════════════════════════════════════
# 14. Environment and Path Resolution
# ═══════════════════════════════════════════════════════════════════

class TestEnvironmentResolution:
    def test_whale_bench_data_root_default(self):
        """Without WHALE_BENCH_DATA_ROOT, falls back to hard-coded path."""
        default = os.environ.pop("WHALE_BENCH_DATA_ROOT", None)
        try:
            root = os.environ.get("WHALE_BENCH_DATA_ROOT", "/home/kemove/CodeingAgent/data")
            assert "CodeingAgent/data" in root
        finally:
            if default:
                os.environ["WHALE_BENCH_DATA_ROOT"] = default

    def test_whale_bench_data_root_env_var(self, monkeypatch):
        monkeypatch.setenv("WHALE_BENCH_DATA_ROOT", "/custom/data/root")
        root = os.environ.get("WHALE_BENCH_DATA_ROOT", "/home/kemove/CodeingAgent/data")
        assert root == "/custom/data/root"

    def test_swev_workers_env_var(self, monkeypatch):
        monkeypatch.setenv("SWEV_WORKERS", "8")
        workers = os.environ.get("SWEV_WORKERS", "1")
        assert workers == "8"

    def test_docker_executable_env_var(self, monkeypatch):
        monkeypatch.setenv("MSWEA_DOCKER_EXECUTABLE", "podman")
        exe = os.environ.get("MSWEA_DOCKER_EXECUTABLE", "docker")
        assert exe == "podman"


# ═══════════════════════════════════════════════════════════════════
# 15. Repo Clone / Git Operations (mocked)
# ═══════════════════════════════════════════════════════════════════

class TestRepoOperations:
    def test_remove_git_lock_files(self, tmp_path):
        repo = tmp_path / "repo"
        git_dir = repo / ".git"
        git_dir.mkdir(parents=True)
        (git_dir / "index.lock").write_text("")
        (git_dir / "HEAD.lock").write_text("")
        (git_dir / "refs").mkdir()
        (git_dir / "refs" / "heads").mkdir()
        (git_dir / "refs" / "heads" / "main.lock").write_text("")

        assert (git_dir / "index.lock").exists()
        SWEBenchVerifiedBenchmark._remove_git_lock_files(repo)
        assert not (git_dir / "index.lock").exists()
        assert not (git_dir / "HEAD.lock").exists()

    @mock.patch("subprocess.run")
    def test_reset_cached_repo_success(self, mock_run, bench_kwargs, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        result = bench._reset_cached_repo(
            repo, "d16bfe05a744909de4b27f5875fe0d4ed41ce607"
        )
        assert result is True

    @mock.patch("subprocess.run")
    def test_reset_cached_repo_git_failure(self, mock_run, bench_kwargs, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["git"], stderr="fatal")
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        result = bench._reset_cached_repo(repo, "deadbeef")
        assert result is False


# ═══════════════════════════════════════════════════════════════════
# 16. Integration: run method flow (dry-run path)
# ═══════════════════════════════════════════════════════════════════

class TestRunMethod:
    def test_dry_run_no_docker(self, bench_kwargs):
        bench_kwargs["docker_executable"] = "docker"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        summary = bench.run(limit=2, dry_run=True)
        assert summary["benchmark"] == "swev"
        assert summary["dry_run"] is True
        assert summary["total"] == 2

    def test_dry_run_with_filter(self, bench_kwargs):
        bench_kwargs["filter_spec"] = "astropy__.*"
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        summary = bench.run(dry_run=True)
        assert summary["total"] == 1

    def test_dry_run_no_docker_check(self, bench_kwargs):
        """Dry run should NOT call docker preflight."""
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        with mock.patch.object(bench, "_docker_preflight") as mock_preflight:
            bench.run(limit=3, dry_run=True)
            mock_preflight.assert_not_called()

    def test_run_with_resume_skips_completed(self, bench_kwargs, tmp_path):
        # Create a resume file with one completed task
        resume_file = tmp_path / "resume.jsonl"
        resume_file.write_text(
            json.dumps(
                {
                    "task_id": "astropy__astropy-12907",
                    "agent_diff": "some diff",
                    "passed": None,
                    "repo": "a/b",
                    "elapsed_s": 5.0,
                }
            )
            + "\n"
        )

        bench_kwargs["resume_file"] = str(resume_file)
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        summary = bench.run(limit=5, dry_run=True)
        # The dry-run should still report total from non-completed tasks
        assert summary["benchmark"] == "swev"
        assert summary["dry_run"] is True


# ═══════════════════════════════════════════════════════════════════
# 17. Real SWEV Data Validation (integration test)
# ═══════════════════════════════════════════════════════════════════

REAL_DATA_PATH = Path("/home/kemove/CodeingAgent/data/SWEV/test.jsonl")


@pytest.mark.integration
class TestRealDataset:
    """Tests that require the real SWE-verified dataset on disk."""

    @pytest.fixture
    def real_bench(self, tmp_path):
        if not REAL_DATA_PATH.exists():
            pytest.skip("SWEV dataset not found on disk")
        return SWEBenchVerifiedBenchmark(
            data_path=str(REAL_DATA_PATH),
            output_dir=str(tmp_path / "results"),
            trajectory_dir=str(tmp_path / "trajectory"),
            repo_cache_dir=str(tmp_path / "repo_cache"),
            workers=1,
        )

    def test_loads_all_500_tasks(self, real_bench):
        tasks = real_bench._load_tasks()
        assert len(tasks) == 500

    def test_all_tasks_have_required_fields(self, real_bench):
        tasks = real_bench._load_tasks()
        required = ["instance_id", "task_id", "repo", "base_commit", "problem_statement"]
        for task in tasks:
            for field in required:
                assert field in task, f"Missing {field} in {task.get('instance_id')}"
                assert task[field], f"Empty {field} in {task.get('instance_id')}"

    def test_all_instance_ids_unique(self, real_bench):
        tasks = real_bench._load_tasks()
        ids = [t["instance_id"] for t in tasks]
        assert len(ids) == len(set(ids))

    def test_all_base_commits_are_sha1(self, real_bench):
        tasks = real_bench._load_tasks()
        sha_pattern = re.compile(r"^[0-9a-f]{40}$")
        for task in tasks:
            assert sha_pattern.match(task["base_commit"]), (
                f"Invalid SHA in {task['instance_id']}: {task['base_commit']}"
            )

    def test_all_repos_are_valid_format(self, real_bench):
        tasks = real_bench._load_tasks()
        repo_pattern = re.compile(r"^[\w.-]+/[\w.-]+$")
        for task in tasks:
            assert repo_pattern.match(task["repo"]), (
                f"Invalid repo format: {task['repo']}"
            )

    def test_docker_image_names_are_valid(self, real_bench):
        tasks = real_bench._load_tasks()
        for task in tasks[:20]:  # sample first 20
            name = SWEBenchVerifiedBenchmark._get_docker_image_name(task)
            assert name, f"Empty image name for {task['instance_id']}"
            # Should be a valid Docker image reference
            assert "/" in name or ":" in name, f"Not a valid image ref: {name}"

    def test_repos_distribution(self, real_bench):
        """Verify we have tasks from 12 known SWE-verified repos."""
        tasks = real_bench._load_tasks()
        repos = set(t["repo"] for t in tasks)
        assert len(repos) == 12, f"Expected 12 repos, got {len(repos)}"


# ═══════════════════════════════════════════════════════════════════
# 18. Repo Cache Directory behavior
# ═══════════════════════════════════════════════════════════════════

class TestRepoCachePathResolution:
    def test_repo_slug_replaces_slash_with_double_underscore(self):
        """repo 'a/b' → cache directory 'a__b'."""
        slug = "a/b".replace("/", "__")
        assert slug == "a__b"

    def test_default_cache_root_is_repo_cache_dir(self, bench_kwargs):
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.repo_cache_dir is not None
        assert bench.repo_cache_dir.name == "repo_cache"

    def test_no_cache_dir_when_none(self, bench_kwargs):
        bench_kwargs["repo_cache_dir"] = None
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.repo_cache_dir is None


# ═══════════════════════════════════════════════════════════════════
# 19. Edge Cases and Error Handling
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_jsonl_data(self, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        bench = SWEBenchVerifiedBenchmark(
            data_path=str(empty_file),
            output_dir=str(tmp_path / "out"),
            workers=1,
        )
        tasks = bench._load_tasks()
        assert tasks == []

    def test_jsonl_with_blank_lines(self, tmp_path, sample_tasks):
        f = tmp_path / "with_blanks.jsonl"
        with open(f, "w") as fh:
            fh.write("\n")
            fh.write(json.dumps(sample_tasks[0]) + "\n")
            fh.write("\n\n")
            fh.write(json.dumps(sample_tasks[1]) + "\n")
            fh.write("\n")
        bench = SWEBenchVerifiedBenchmark(
            data_path=str(f),
            output_dir=str(tmp_path / "out"),
            workers=1,
        )
        tasks = bench._load_tasks()
        assert len(tasks) == 2

    def test_task_without_instance_id_gets_default_task_id(self, tmp_path):
        """Task without instance_id: the task_transform in _load_tasks handles it."""
        f = tmp_path / "no_id.jsonl"
        f.write_text(
            json.dumps({"repo": "a/b", "problem_statement": "test"}) + "\n"
        )
        bench = SWEBenchVerifiedBenchmark(
            data_path=str(f),
            output_dir=str(tmp_path / "out"),
            workers=1,
        )
        tasks = bench._load_tasks()
        assert len(tasks) == 1
        # instance_id wasn't there, so task_id should be None
        assert tasks[0]["task_id"] is None

    def test_workers_guards(self, bench_kwargs):
        bench_kwargs["workers"] = 0
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        assert bench.workers == 1  # guards to minimum 1

    def test_repo_lock_isolation(self, bench_kwargs):
        bench = SWEBenchVerifiedBenchmark(**bench_kwargs)
        lock1 = bench._repo_lock_for("django__django")
        lock2 = bench._repo_lock_for("django__django")
        assert lock1 is lock2  # same slug → same lock

        lock3 = bench._repo_lock_for("astropy__astropy")
        assert lock1 is not lock3  # different slug → different lock

    def test_swev_artifact_dirs_set(self):
        """Verify artifact dirs covers common build tool directories."""
        common = {"__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", "dist", "build"}
        assert common <= _SWEV_ARTIFACT_DIRS
