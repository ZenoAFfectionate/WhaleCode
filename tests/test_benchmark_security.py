"""Regression tests for benchmark evaluator hardening."""

from __future__ import annotations

import base64
import os
import pickle
import zlib
from pathlib import Path

import pytest

from hello_agents.benchmark import clev_bench, hevp_bench, lcb6_bench, mbpp_bench
from hello_agents.benchmark.safe_exec import (
    UnsafeBenchmarkCodeError,
    benchmark_child_env,
    run_python_code,
    run_python_code_with_stdin,
    run_python_command,
    validate_python_source_safe,
)
from hello_agents.benchmark.runtime import EvalRequest, PythonSubprocessEnvironment
from hello_agents.benchmark.runtime.python_adapters import (
    PythonAssertionAdapter,
    PythonFunctionalAdapter,
    PythonStdinAdapter,
    PythonVerifierAdapter,
)


def _encode_lcb6_private_payload(obj) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(obj))).decode("ascii")


def test_lcb6_private_cases_allow_plain_data():
    payload = _encode_lcb6_private_payload(
        [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
    )

    decoded = lcb6_bench._decode_private_cases(payload)

    assert decoded == [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]


def test_lcb6_private_cases_reject_pickle_gadgets():
    class _Evil:
        def __reduce__(self):
            return (os.system, ("echo SHOULD_NOT_RUN",))

    payload = _encode_lcb6_private_payload([_Evil()])

    with pytest.raises(pickle.UnpicklingError):
        lcb6_bench._decode_private_cases(payload)


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\n",
        "from urllib import request\n",
        "eval('1 + 1')\n",
        "__import__('os')\n",
        "import os\nos.system('echo bad')\n",
        "import os\nprint(os.environ)\n",
        "from pathlib import Path\nPath('/tmp/secret').read_text()\n",
    ],
)
def test_validate_python_source_safe_rejects_dangerous_code(source):
    with pytest.raises(UnsafeBenchmarkCodeError):
        validate_python_source_safe(source)


def test_run_python_code_strips_secret_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "secret-value")
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    result = run_python_code(
        "import os\nprint(os.getenv('LLM_API_KEY'))\nprint(os.getenv('HF_TOKEN'))\n",
        cwd=tmp_path,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["None", "None"]


def test_benchmark_child_env_caps_numeric_library_threads():
    env = benchmark_child_env()

    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == "1"


def test_benchmark_child_env_allows_explicit_thread_override():
    env = benchmark_child_env({"OPENBLAS_NUM_THREADS": "2"})

    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["OMP_NUM_THREADS"] == "1"


def test_run_python_code_times_out_and_reports_timeout(tmp_path):
    result = run_python_code("while True:\n    pass\n", cwd=tmp_path, timeout=1)

    assert result.timed_out is True
    assert result.returncode == -9


def test_eval_request_requires_code_or_command(tmp_path):
    with pytest.raises(ValueError, match="requires non-empty code or command"):
        EvalRequest(cwd=tmp_path, timeout_s=5)


def test_python_subprocess_environment_runs_code_and_reports_status(tmp_path):
    env = PythonSubprocessEnvironment()
    result = env.evaluate(
        EvalRequest(
            code="print('runtime-ok')",
            cwd=tmp_path,
            timeout_s=5,
            visibility="public",
        )
    )

    assert result.passed is True
    assert result.status == "passed"
    assert result.stdout.strip() == "runtime-ok"
    assert result.timeout_s == 5
    assert result.metrics["visibility"] == "public"


def test_python_subprocess_environment_supports_stdin_file_mode(tmp_path):
    env = PythonSubprocessEnvironment()
    result = env.evaluate(
        EvalRequest(
            code="import sys\nprint(sys.stdin.read().strip().upper())\n",
            command=(),
            stdin="hello runtime\n",
            cwd=tmp_path,
            timeout_s=5,
        )
    )

    assert result.passed is True
    assert result.status == "passed"
    assert result.stdout.strip() == "HELLO RUNTIME"
    assert not list(tmp_path.glob("._bench_eval_*.py"))


def test_safe_exec_facade_uses_runtime_without_behavior_regression(tmp_path):
    command_result = run_python_command(
        ["-c", "print('command-ok')"],
        cwd=tmp_path,
        timeout=5,
    )
    stdin_result = run_python_code_with_stdin(
        "import sys\nprint(sys.stdin.read().strip())\n",
        "stdin-ok\n",
        cwd=tmp_path,
        timeout=5,
    )

    assert command_result.returncode == 0
    assert command_result.stdout.strip() == "command-ok"
    assert command_result.timed_out is False
    assert stdin_result.returncode == 0
    assert stdin_result.stdout.strip() == "stdin-ok"
    assert stdin_result.timed_out is False


def test_python_assertion_adapter_runs_assertion_checks(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text("def inc(x):\n    return x + 1\n", encoding="utf-8")

    def _build(assertions: str) -> str:
        return (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from solution import *\n"
            f"{assertions}\n"
            "print('adapter assertions passed')\n"
        )

    passed, output = PythonAssertionAdapter(_build).evaluate(
        workspace=tmp_path,
        solution_file=solution,
        assertion_code="assert inc(2) == 3",
        timeout=5,
    )

    assert passed is True
    assert "adapter assertions passed" in output


def test_python_verifier_adapter_blocks_unsafe_source(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text("import subprocess\n", encoding="utf-8")

    passed, output = PythonVerifierAdapter(
        lambda source: source + "\nprint('should not run')\n"
    ).evaluate(
        workspace=tmp_path,
        solution_file=solution,
        fallback_solution="",
        timeout=5,
    )

    assert passed is False
    assert "[SECURITY]" in output
    assert "subprocess" in output


def test_python_stdin_adapter_reports_public_failure(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text("import sys\nprint(sys.stdin.read().strip())\n", encoding="utf-8")

    result = PythonStdinAdapter(
        source_wrapper=lambda source: source,
        output_matcher=lambda actual, expected: actual == expected,
        public_context_formatter=lambda mode, case: f"context {mode}: {case['input']!r}",
    ).evaluate(
        solution_file=solution,
        cases=[{"input": "abc\n", "output": "xyz\n"}],
        public_count=1,
    )

    assert result["passed"] is False
    assert result["public_passed"] == 0
    assert "[FAIL] public case 1" in result["output"]
    assert "context stdin" in result["output"]


def test_python_functional_adapter_evaluates_helper_json(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    helper = (
        "import json, sys\n"
        "from solution import add\n"
        "args = json.loads(sys.stdin.read())\n"
        "print(json.dumps({'status': 'ok', 'value_repr': repr(add(*args))}))\n"
    )

    result = PythonFunctionalAdapter(
        helper_code=helper,
        call_spec="{}",
        expected_parser=lambda text: int(text),
        public_context_formatter=lambda mode, case: f"context {mode}: {case['input']}",
    ).evaluate(
        solution_file=solution,
        cases=[{"input": "[2, 5]", "output": "7"}],
        public_count=1,
    )

    assert result["passed"] is True
    assert result["public_passed"] == 1
    assert "All benchmark cases passed!" in result["output"]


def test_mbpp_evaluation_passes_safe_solution(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    passed, output = mbpp_bench._evaluate_solution(
        workspace=tmp_path,
        solution_file=solution,
        assertion_code="assert add(1, 2) == 3\nassert add(-1, 1) == 0",
        timeout=5,
    )

    assert passed is True
    assert "All tests passed" in output


def test_mbpp_evaluation_blocks_unsafe_solution(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text(
        "import subprocess\n"
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    passed, output = mbpp_bench._evaluate_solution(
        workspace=tmp_path,
        solution_file=solution,
        assertion_code="assert add(1, 2) == 3",
        timeout=5,
    )

    assert passed is False
    assert "[SECURITY]" in output
    assert "subprocess" in output


def test_lcb6_stdin_evaluation_passes_safe_solution(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text(
        "import sys\n"
        "nums = list(map(int, sys.stdin.read().split()))\n"
        "print(sum(nums))\n",
        encoding="utf-8",
    )

    result = lcb6_bench._evaluate_stdin_solution(
        solution,
        [{"input": "1 2 3\n", "output": "6\n"}],
        public_count=1,
    )

    assert result["passed"] is True
    assert result["public_passed"] == 1


def test_lcb6_functional_evaluation_blocks_unsafe_solution(tmp_path):
    solution = tmp_path / "solution.py"
    solution.write_text(
        "import os\n"
        "def add(a, b):\n"
        "    return os.system('echo bad')\n",
        encoding="utf-8",
    )

    result = lcb6_bench._evaluate_functional_solution(
        solution,
        [{"input": "[1, 2]", "output": "3"}],
        public_count=1,
        starter_code="def add(a, b):\n    pass\n",
        metadata={"func_name": "add"},
    )

    assert result["passed"] is False
    assert "[SECURITY]" in result["output"]


def test_humaneval_and_classeval_block_unsafe_solution(tmp_path):
    unsafe = tmp_path / "solution.py"
    unsafe.write_text("import subprocess\n", encoding="utf-8")

    hevp_passed, hevp_output = hevp_bench._evaluate_solution(
        workspace=tmp_path,
        solution_file=unsafe,
        fallback_solution="",
        entry_point="candidate",
        test_code="def check(candidate):\n    pass\n",
        timeout=5,
    )
    clev_passed, clev_output = clev_bench._evaluate_solution(
        workspace=tmp_path,
        solution_file=unsafe,
        fallback_solution="",
        test_code="",
        timeout=5,
    )

    assert hevp_passed is False
    assert clev_passed is False
    assert "[SECURITY]" in hevp_output
    assert "[SECURITY]" in clev_output
