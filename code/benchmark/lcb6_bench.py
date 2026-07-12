"""LiveCodeBench v6 benchmark runner for Whale Code agent."""

from __future__ import annotations

import argparse
import ast
import base64
import decimal
import io
import json
import pickle
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from .base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
        build_minimal_child_env,
        truncate_feedback,
    )
except ImportError:
    from base import (
        BenchmarkRunner,
        BENCHMARK_BASE_SYSTEM_PROMPT,
        _PROJECT_ROOT,
        build_minimal_child_env,
        truncate_feedback,
    )


_LCB6_ADDENDUM = """\
You are solving LiveCodeBench v6 code-generation tasks.

The workspace contains:
- `problem.txt`: full problem statement and metadata
- `solution.py`: your answer file

# Core Principle: Analyze first, then code

Before editing code, identify the exact I/O contract, constraints, edge cases,
and target complexity. For math-heavy tasks, derive the key invariant/formula
before implementation.

Workflow:
1. Read `problem.txt` carefully.
2. Extract interface, constraints, and tricky cases before touching code.
3. Read `solution.py`; preserve required starter interfaces.
4. Implement after you have a clear algorithmic plan.
5. Submit with `Finish`; runner executes public+hidden tests and returns bounded feedback.

Rules:
- For stdin/stdout tasks, `solution.py` must be a complete Python program reading from stdin and writing to stdout.
- For call-based tasks, preserve the provided function/class structure exactly.
- Benchmark test data is not stored in the workspace.
- Do not create uncontrolled submission loops; submit via `Finish`.
- Do not try to access hidden tests, hidden directories, env vars, or files outside workspace.
- Prefer clean, correct code over clever shortcuts.
- `Finish` must be the last tool you call for that submission.
- Do not treat sample outputs, a few hand-picked checks, or numerical experiments as proof of correctness.
- Use local experiments only to validate an existing hypothesis.
- If feedback contradicts your code, revisit reasoning before patching.

Functional-task rules:
- Keep the starter-code interface exactly.
- For functional public examples, treat each non-empty input line as one positional argument.
- If feedback says `missing positional argument` or `KeyError: 'Solution'`, fix the interface first, not the algorithm.
"""

_LCB6_SYSTEM_PROMPT = (
    BENCHMARK_BASE_SYSTEM_PROMPT
    + "\n\n---\n\n## LiveCodeBench v6 Benchmark Override\n\n"
    + _LCB6_ADDENDUM
)


_OFFICIAL_IMPORT_STRING = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""


_FUNCTIONAL_EVAL_HELPER = """\
import ast
import decimal
import json
import os
import sys
import types
from pathlib import Path

SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP")
OFFICIAL_IMPORT_STRING = __OFFICIAL_IMPORT_STRING__


def sanitize_runtime_env():
    safe_env = {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}
    safe_env["PYTHONIOENCODING"] = "utf-8"
    safe_env["PYTHONUNBUFFERED"] = "1"
    os.environ.clear()
    os.environ.update(safe_env)


CALL_SPEC = json.loads(sys.argv[1])
SOLUTION = Path("solution.py").resolve()


def load_namespace():
    source = SOLUTION.read_text(encoding="utf-8")
    module = types.ModuleType("solution_mod")
    exec(merge_import_string(source), module.__dict__)
    return vars(module).copy()


def get_callable(ns):
    fn_name = CALL_SPEC.get("function_name")
    class_name = CALL_SPEC.get("class_name") or "Solution"
    cls = ns.get(class_name)
    if isinstance(cls, type):
        instance = cls()
        if fn_name and hasattr(instance, fn_name):
            return getattr(instance, fn_name)
    if fn_name and fn_name in ns and callable(ns[fn_name]):
        return ns[fn_name]
    raise KeyError(class_name if class_name not in ns else fn_name)


def merge_import_string(source):
    lines = source.splitlines()
    insertion_idx = 0

    while insertion_idx < len(lines) and (
        not lines[insertion_idx].strip()
        or lines[insertion_idx].lstrip().startswith("#")
        or lines[insertion_idx].startswith("#!")
    ):
        insertion_idx += 1

    if insertion_idx < len(lines) and lines[insertion_idx].lstrip().startswith(("\\\"\\\"\\\"", "'''")):
        quote = "\\\"\\\"\\\"" if lines[insertion_idx].lstrip().startswith("\\\"\\\"\\\"") else "'''"
        insertion_idx += 1
        while insertion_idx < len(lines) and quote not in lines[insertion_idx]:
            insertion_idx += 1
        if insertion_idx < len(lines):
            insertion_idx += 1

    while insertion_idx < len(lines) and lines[insertion_idx].lstrip().startswith("from __future__ import"):
        insertion_idx += 1

    merged = lines[:insertion_idx] + [OFFICIAL_IMPORT_STRING.rstrip(), ""] + lines[insertion_idx:]
    return "\\n".join(merged)


def parse_single_value(raw):
    stripped = raw.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped), False
        except Exception:
            pass
    return stripped, True


def parse_input_payload(raw):
    stripped = raw.strip()
    parsed, is_raw = parse_single_value(stripped)
    if not is_raw:
        return parsed, False

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1:
        values = []
        any_raw = False
        for line in lines:
            value, line_is_raw = parse_single_value(line)
            values.append(value)
            any_raw = any_raw or line_is_raw
        return values, any_raw

    return stripped, True


def invoke_callable(fn, ns, raw):
    stripped = raw.strip()
    if stripped.startswith(CALL_SPEC["function_name"] + "(") or stripped.startswith("Solution()."):
        return eval(stripped, ns)

    payload, is_raw = parse_input_payload(raw)
    param_count = CALL_SPEC["param_count"]
    param_names = CALL_SPEC["param_names"]

    if isinstance(payload, dict) and set(payload.keys()).issubset(set(param_names)):
        return fn(**payload)
    if param_count == 1:
        return fn(payload)
    if isinstance(payload, (list, tuple)) and len(payload) == param_count:
        return fn(*payload)
    if is_raw:
        return fn(raw)
    return fn(payload)


def main():
    sanitize_runtime_env()
    raw = sys.stdin.read()
    try:
        ns = load_namespace()
        fn = get_callable(ns)
        actual = invoke_callable(fn, ns, raw)
        print(json.dumps({"status": "ok", "value_repr": repr(actual)}, ensure_ascii=False))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
""".replace("__OFFICIAL_IMPORT_STRING__", repr(_OFFICIAL_IMPORT_STRING))


def _parse_public_cases(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return []


# 严重-4: LiveCodeBench ships private test cases as base64+zlib+pickle. Plain
# ``pickle.loads`` on such payloads is arbitrary code execution if the dataset
# is tampered with. Restrict unpickling to inert data types so a malicious
# ``__reduce__`` gadget (os.system / eval / ...) cannot execute.
_SAFE_PICKLE_GLOBALS = {
    "builtins": {
        "list", "dict", "tuple", "set", "frozenset", "str", "bytes", "bytearray",
        "int", "float", "bool", "complex", "NoneType",
    },
    "collections": {"OrderedDict", "defaultdict"},
    "decimal": {"Decimal"},
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D401
        allowed = _SAFE_PICKLE_GLOBALS.get(module)
        if allowed and name in allowed:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to unpickle {module}.{name}: only plain data types are "
            "allowed (严重-4 guard against untrusted-dataset code execution)."
        )


def _safe_pickle_loads(data: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def _decode_private_cases(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return []
    decoded = base64.b64decode(raw)
    inflated = zlib.decompress(decoded)
    obj = _safe_pickle_loads(inflated)
    if isinstance(obj, bytes):
        obj = obj.decode("utf-8")
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj if isinstance(obj, list) else []


def _normalize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(case)
    if "input" in normalized and isinstance(normalized["input"], str):
        normalized["input"] = normalized["input"].replace("\r\n", "\n").replace("\r", "\n")
    if "output" in normalized and isinstance(normalized["output"], str):
        normalized["output"] = normalized["output"].replace("\r\n", "\n").replace("\r", "\n")
    return normalized


def _infer_mode(task: Dict[str, Any], cases: List[Dict[str, Any]]) -> str:
    testtypes = {str(case.get("testtype", "")).lower() for case in cases if case.get("testtype")}
    if "stdin" in testtypes:
        return "stdin"
    starter = str(task.get("starter_code") or "")
    if starter.strip():
        return "functional"
    return "functional"


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

def _initial_solution_contents(task: Dict[str, Any], mode: str) -> str:
    starter = str(task.get("starter_code") or "")
    if starter.strip():
        return starter.rstrip() + "\n"
    if mode == "stdin":
        return (
            "# Implement the full solution program here.\n"
            "# Read from stdin and write to stdout.\n"
        )
    return "# Implement the required function or class here.\n"


def _extract_call_spec_from_starter(starter_code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        class_match = re.search(r"class\s+(\w+)\s*:", starter_code)
        def_match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", starter_code)
        if def_match:
            raw_args = [arg.strip() for arg in def_match.group(2).split(',') if arg.strip()]
            arg_names = [arg.split(':', 1)[0].split('=', 1)[0].strip() for arg in raw_args]
            if class_match:
                arg_names = [arg for arg in arg_names if arg != 'self']
                return {
                    "kind": "method",
                    "class_name": class_match.group(1),
                    "function_name": def_match.group(1),
                    "param_count": len(arg_names),
                    "param_names": arg_names,
                }
            return {
                "kind": "function",
                "function_name": def_match.group(1),
                "param_count": len(arg_names),
                "param_names": arg_names,
            }
        return {"kind": "function", "function_name": "solve", "param_count": 1, "param_names": ["arg"]}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            return {
                "kind": "function",
                "function_name": node.name,
                "param_count": len(args),
                "param_names": args,
            }
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name != "__init__":
                    args = [arg.arg for arg in child.args.args if arg.arg != "self"]
                    return {
                        "kind": "method",
                        "class_name": node.name,
                        "function_name": child.name,
                        "param_count": len(args),
                        "param_names": args,
                    }
    return {"kind": "function", "function_name": "solve", "param_count": 1, "param_names": ["arg"]}


def _resolve_call_spec(starter_code: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    spec = _extract_call_spec_from_starter(starter_code)
    fn_name = metadata.get("func_name")
    if isinstance(fn_name, str) and fn_name.strip():
        spec["function_name"] = fn_name.strip()
    return spec


def _parse_scalar_repr(raw: str) -> Any:
    stripped = raw.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except Exception:
            pass
    return stripped


def _format_public_context(mode: str, case: Dict[str, Any]) -> str:
    field = "stdin" if mode == "stdin" else "expr"
    value = str(case.get("input", ""))
    return f"  {field}: {value[:200]!r}" if mode == "stdin" else f"  {field}: {value}"


def _truncate_feedback(text: str, max_lines: int = 80, max_chars: int = 12000) -> str:
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["[feedback truncated]"]
    truncated = "\n".join(lines)
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars].rstrip() + "\n[feedback truncated]"
    return truncated


def _normalize_output_lines(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    return [line.strip() for line in text.split("\n")]


def _decimal_tokens(line: str) -> tuple[bool, List[decimal.Decimal]]:
    try:
        return True, [decimal.Decimal(part) for part in line.split()]
    except Exception:
        return False, []


def _stdio_outputs_match(actual: str, expected: str) -> bool:
    actual_lines = _normalize_output_lines(actual)
    expected_lines = _normalize_output_lines(expected)
    if len(actual_lines) != len(expected_lines):
        return False

    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if actual_line == expected_line:
            continue
        actual_ok, actual_decimals = _decimal_tokens(actual_line)
        expected_ok, expected_decimals = _decimal_tokens(expected_line)
        if actual_ok and expected_ok and actual_decimals == expected_decimals:
            continue
        return False
    return True


def _merge_official_imports(source: str) -> str:
    lines = source.splitlines()
    insertion_idx = 0

    while insertion_idx < len(lines) and (
        not lines[insertion_idx].strip()
        or lines[insertion_idx].lstrip().startswith("#")
        or lines[insertion_idx].startswith("#!")
    ):
        insertion_idx += 1

    if insertion_idx < len(lines) and lines[insertion_idx].lstrip().startswith(('"""', "'''")):
        quote = '"""' if lines[insertion_idx].lstrip().startswith('"""') else "'''"
        insertion_idx += 1
        while insertion_idx < len(lines) and quote not in lines[insertion_idx]:
            insertion_idx += 1
        if insertion_idx < len(lines):
            insertion_idx += 1

    while insertion_idx < len(lines) and lines[insertion_idx].lstrip().startswith("from __future__ import"):
        insertion_idx += 1

    merged = lines[:insertion_idx] + [_OFFICIAL_IMPORT_STRING.rstrip(), ""] + lines[insertion_idx:]
    return "\n".join(merged)


def _evaluate_stdin_solution(
    solution_file: Path,
    cases: List[Dict[str, Any]],
    public_count: int,
) -> Dict[str, Any]:
    failed = 0
    private_failed = 0
    timed_out = False
    public_passed = 0
    private_passed = 0
    lines: List[str] = []

    for idx, case in enumerate(cases, start=1):
        visibility = "public" if idx <= public_count else "private"
        stdin_text = str(case.get("input", ""))
        if stdin_text and not stdin_text.endswith("\n"):
            stdin_text += "\n"
        expected = str(case.get("output", "")).strip()

        try:
            wrapped_solution = solution_file.parent / "._lcb6_wrapped_stdio.py"
            wrapped_solution.write_text(
                _merge_official_imports(solution_file.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(wrapped_solution)],
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=10,
                cwd=str(solution_file.parent),
                env=build_minimal_child_env(),
            )
        except subprocess.TimeoutExpired:
            failed += 1
            timed_out = True
            if visibility == "public":
                lines.append(f"[TIMEOUT] public case {idx}")
                lines.append(_format_public_context("stdin", case))
            else:
                private_failed += 1
            continue
        finally:
            try:
                wrapped_solution.unlink(missing_ok=True)
            except Exception:
                pass

        if proc.returncode != 0:
            failed += 1
            if visibility == "public":
                lines.append(f"[ERROR] public case {idx}")
                lines.append(_format_public_context("stdin", case))
                stderr_text = proc.stderr.strip()
                if stderr_text:
                    lines.append(stderr_text)
            else:
                private_failed += 1
                lines.append(f"[ERROR] private case {idx}")
            continue

        actual = proc.stdout.strip()
        if not _stdio_outputs_match(actual, expected):
            failed += 1
            if visibility == "public":
                lines.append(f"[FAIL] public case {idx}")
                lines.append(_format_public_context("stdin", case))
                lines.append(f"  actual:   {actual!r}")
                lines.append(f"  expected: {expected!r}")
            else:
                private_failed += 1
                lines.append(f"[FAIL] private case {idx}")
            continue

        if visibility == "public":
            public_passed += 1
        else:
            private_passed += 1

    total = len(cases)
    if timed_out:
        lines.append(
            "Timeout hint: prioritize algorithmic complexity reduction over local patching; "
            "re-check constraints and replace asymptotically mismatched approaches instead of micro-optimizing loops."
        )
    if public_count == 0:
        lines.append("No public tests were provided for this task.")
    lines.append(f"{total - failed}/{total} total cases passed")
    if public_count:
        lines.append(f"{public_passed}/{public_count} public cases passed")
    private_count = max(total - public_count, 0)
    if private_count:
        lines.append(f"{private_passed}/{private_count} private cases passed")
        if private_failed:
            lines.append("Private-case failures detected (details withheld).")
    if failed == 0:
        lines.append("All benchmark cases passed!")

    return {
        "passed": failed == 0,
        "output": "\n".join(lines),
        "public_passed": public_passed,
        "private_passed": private_passed,
    }


def _evaluate_functional_solution(
    solution_file: Path,
    cases: List[Dict[str, Any]],
    public_count: int,
    starter_code: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    failed = 0
    private_failed = 0
    timed_out = False
    public_passed = 0
    private_passed = 0
    lines: List[str] = []
    call_spec = json.dumps(_resolve_call_spec(starter_code, metadata or {}), ensure_ascii=False)

    for idx, case in enumerate(cases, start=1):
        visibility = "public" if idx <= public_count else "private"
        expr = str(case.get("input", ""))
        expected = _parse_scalar_repr(str(case.get("output", "")))

        try:
            proc = subprocess.run(
                [sys.executable, "-c", _FUNCTIONAL_EVAL_HELPER, call_spec],
                input=expr,
                text=True,
                capture_output=True,
                timeout=10,
                cwd=str(solution_file.parent),
                env=build_minimal_child_env(),
            )
        except subprocess.TimeoutExpired:
            failed += 1
            timed_out = True
            if visibility == "public":
                lines.append(f"[TIMEOUT] public case {idx}")
                lines.append(_format_public_context("functional", case))
            else:
                private_failed += 1
            continue

        helper_stdout = proc.stdout.strip()
        try:
            helper_result = json.loads(helper_stdout) if helper_stdout else {}
        except json.JSONDecodeError:
            helper_result = {}

        if proc.returncode != 0 or helper_result.get("status") != "ok":
            failed += 1
            if visibility == "public":
                lines.append(f"[ERROR] public case {idx}")
                lines.append(_format_public_context("functional", case))
                err_type = helper_result.get("type") or "RuntimeError"
                err_message = helper_result.get("message") or proc.stderr.strip() or "helper execution failed"
                lines.append(f"  {err_type}: {err_message}")
            else:
                private_failed += 1
                lines.append(f"[ERROR] private case {idx}")
            continue

        actual = _parse_scalar_repr(str(helper_result.get("value_repr", "")))
        if actual != expected:
            failed += 1
            if visibility == "public":
                lines.append(f"[FAIL] public case {idx}")
                lines.append(_format_public_context("functional", case))
                lines.append(f"  actual:   {actual!r}")
                lines.append(f"  expected: {expected!r}")
            else:
                private_failed += 1
                lines.append(f"[FAIL] private case {idx}")
            continue

        if visibility == "public":
            public_passed += 1
        else:
            private_passed += 1

    total = len(cases)
    if timed_out:
        lines.append(
            "Timeout hint: prioritize algorithmic complexity reduction over local patching; "
            "re-check constraints and replace asymptotically mismatched approaches instead of micro-optimizing loops."
        )
    if public_count == 0:
        lines.append("No public tests were provided for this task.")
    lines.append(f"{total - failed}/{total} total cases passed")
    if public_count:
        lines.append(f"{public_passed}/{public_count} public cases passed")
    private_count = max(total - public_count, 0)
    if private_count:
        lines.append(f"{private_passed}/{private_count} private cases passed")
        if private_failed:
            lines.append("Private-case failures detected (details withheld).")
    if failed == 0:
        lines.append("All benchmark cases passed!")

    return {
        "passed": failed == 0,
        "output": "\n".join(lines),
        "public_passed": public_passed,
        "private_passed": private_passed,
    }


class LCB6Benchmark(BenchmarkRunner):
    """Evaluate Whale Code agent on LiveCodeBench v6 code-generation tasks."""

    benchmark_name = "lcb6"

    def __init__(self, *args, max_submission_rounds: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_submission_rounds = max(1, int(max_submission_rounds))

    def _get_system_prompt(self) -> str:
        return _LCB6_SYSTEM_PROMPT

    def _load_tasks(self) -> List[Dict[str, Any]]:
        counter = {"value": 0}

        def transform(task: Dict[str, Any]) -> Dict[str, Any]:
            task_id = task.get("task_id")
            if not task_id:
                task_id = f"LCB6/{task.get('question_id', counter['value'])}"
            counter["value"] += 1
            return {**task, "task_id": task_id}

        return self._load_jsonl_tasks(task_transform=transform)

    def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        title = task.get("question_title", "")
        question = task.get("question_content", "")
        platform = task.get("platform", "")
        difficulty = task.get("difficulty", "")
        contest_date = task.get("contest_date", "")
        metadata = _parse_metadata(task.get("metadata"))

        public_cases = [_normalize_case(case) for case in _parse_public_cases(task.get("public_test_cases", []))]
        private_cases = [_normalize_case(case) for case in _decode_private_cases(task.get("private_test_cases", ""))]
        all_cases = public_cases + private_cases
        mode = _infer_mode(task, all_cases)

        workspace = self._make_workspace(f"lcb6_{task_id.replace('/', '_')}_")
        agent = None
        agent_response = ""
        prompt_history: List[str] = []
        result: Optional[Dict[str, Any]] = None
        try:
            problem_file = workspace / "problem.txt"
            problem_file.write_text(
                f"Title: {title}\n"
                f"Platform: {platform}\n"
                f"Difficulty: {difficulty}\n"
                f"Contest date: {contest_date}\n"
                f"Mode: {mode}\n"
                f"Benchmark tests available only via controlled evaluation feedback.\n\n"
                f"{question}\n",
                encoding="utf-8",
            )

            solution_file = workspace / "solution.py"
            solution_file.write_text(_initial_solution_contents(task, mode), encoding="utf-8")

            initial_prompt = (
                f"Solve this LiveCodeBench v6 task in `solution.py`.\n\n"
                f"Task ID: {task_id}\n"
                f"Title: {title}\n"
                f"Platform: {platform}\n"
                f"Difficulty: {difficulty}\n"
                f"Mode: {mode}\n\n"
                f"Requirements:\n"
                f"- Controlled submissions only: call `Finish` alone when ready; runner returns bounded feedback.\n"
                f"- Benchmark test data is not present in the workspace.\n"
                f"- Do not run your own benchmark test loop.\n"
                f"- Read `problem.txt` first, derive algorithm + complexity target, then edit `solution.py`.\n"
                f"- Keep starter-code interface unchanged when provided.\n"
                f"- For stdin tasks, write a complete executable program.\n"
                f"- Use local checks only to validate reasoning, not as proof of correctness.\n"
            )

            agent = self._create_agent(workspace)
            start = time.time()
            evaluation = None
            last_feedback = None
            rounds_used = 0
            cumulative_steps = 0
            total_step_budget = max(int(getattr(agent, "max_steps", 0) or 0), 0)

            for round_idx in range(1, self.max_submission_rounds + 1):
                rounds_used = round_idx
                if total_step_budget > 0 and cumulative_steps >= total_step_budget:
                    evaluation = {
                        "passed": False,
                        "output": (
                            f"Step budget exhausted after {cumulative_steps}/{total_step_budget} steps "
                            "before another controlled submission could start."
                        ),
                        "public_passed": 0,
                        "private_passed": 0,
                    }
                    break

                prompt = initial_prompt if round_idx == 1 else (
                    f"Controlled evaluation feedback for submission round {round_idx - 1}:\n\n"
                    f"{last_feedback}\n\n"
                    f"Revise `solution.py` based on this feedback.\n"
                    f"- Public-case details are exact.\n"
                    f"- Private-case feedback is intentionally limited.\n"
                    f"- Do not search for hidden tests; use the feedback above plus the problem statement.\n"
                    f"- Re-check the underlying reasoning, invariants, and complexity before patching the code.\n"
                    f"- Do not overfit to the observed failing cases; fix the general logic, proof, or interface.\n"
                    f"When you are ready for the next controlled submission, call `Finish` alone with a brief summary of the revision."
                )
                prompt_history.append(prompt)

                agent_response, error_result = self._run_agent_prompt(
                    agent=agent,
                    task_id=task_id,
                    prompt_text=prompt,
                    start_time=start,
                    run_kwargs={"start_step": cumulative_steps},
                    error_extra={
                        "mode": mode,
                        "submission_rounds": round_idx,
                        "steps_used": cumulative_steps,
                        "step_budget": total_step_budget,
                    },
                )
                cumulative_steps = max(
                    cumulative_steps,
                    int(getattr(agent, "_current_step", cumulative_steps) or cumulative_steps),
                )
                if error_result is not None:
                    result = error_result
                    return result

                if not solution_file.exists():
                    result = self._build_result(
                        task_id,
                        passed=False,
                        error="solution.py not found after agent run",
                        start_time=start,
                        agent_response=agent_response,
                        extra={
                            "mode": mode,
                            "submission_rounds": round_idx,
                            "steps_used": cumulative_steps,
                            "step_budget": total_step_budget,
                        },
                    )
                    return result

                evaluation = (
                    _evaluate_stdin_solution(solution_file, all_cases, len(public_cases))
                    if mode == "stdin"
                    else _evaluate_functional_solution(
                        solution_file,
                        all_cases,
                        len(public_cases),
                        str(task.get("starter_code") or ""),
                        metadata,
                    )
                )
                if evaluation["passed"]:
                    break
                if total_step_budget > 0 and cumulative_steps >= total_step_budget:
                    evaluation["output"] = (
                        f"{evaluation['output']}\n\n"
                        f"Step budget exhausted after {cumulative_steps}/{total_step_budget} steps."
                    )
                    break

                last_feedback = truncate_feedback(evaluation["output"], max_lines=80, max_chars=12000)

            elapsed = round(time.time() - start, 2)
            if evaluation is None:
                result = self._build_result(
                    task_id,
                    passed=False,
                    error="Evaluation did not run",
                    agent_response=agent_response,
                    elapsed_s=elapsed,
                    extra={
                        "mode": mode,
                        "submission_rounds": 0,
                        "steps_used": cumulative_steps,
                        "step_budget": total_step_budget,
                    },
                )
                return result

            if not evaluation["passed"] and self.max_submission_rounds > 0:
                final_error = (
                    f"Failed after {rounds_used} controlled submission rounds.\n\n"
                    f"{evaluation['output']}"
                )
            else:
                final_error = None

            result = self._build_result(
                task_id,
                passed=evaluation["passed"],
                error=final_error,
                agent_response=agent_response,
                elapsed_s=elapsed,
                extra={
                    "mode": mode,
                    "public_tests": len(public_cases),
                    "private_tests": len(private_cases),
                    "public_passed": evaluation["public_passed"],
                    "private_passed": evaluation["private_passed"],
                    "submission_rounds": rounds_used,
                    "steps_used": cumulative_steps,
                    "step_budget": total_step_budget,
                },
            )
            return result
        finally:
            self._finalize_workspace_task(
                task=task,
                workspace=workspace,
                agent=agent,
                prompt_texts=prompt_history,
                result=result,
                artifact_paths=["problem.txt", "solution.py"],
                extra={"mode": mode},
            )


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run LiveCodeBench v6 benchmark")
    parser.add_argument(
        "--data-path",
        default=str(_PROJECT_ROOT / "data" / "LCB6" / "test.jsonl"),
        help="Path to LiveCodeBench v6 JSONL file",
    )
    BenchmarkRunner.add_shared_run_args(
        parser,
        default_temperature=1.0,
        default_max_steps=96,
        default_timeout=120,
        include_task_timeout=True,
        default_task_timeout=1200,
    )
    parser.add_argument("--max-submission-rounds", type=int, default=3)
    args = parser.parse_args()

    bench = LCB6Benchmark(
        data_path=args.data_path,
        max_submission_rounds=args.max_submission_rounds,
        **BenchmarkRunner.runner_kwargs_from_args(args, include_task_timeout=True),
    )
    bench.run(limit=args.limit, task_ids=args.task_ids, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
