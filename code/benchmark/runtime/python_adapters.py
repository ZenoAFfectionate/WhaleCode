"""Adapters that map Python benchmark checks onto the shared runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from ..safe_exec import (
        UnsafeBenchmarkCodeError,
        format_safe_python_failure,
        run_python_code,
        run_python_code_with_stdin,
        run_python_command,
        validate_python_source_safe,
    )
except ImportError:  # pragma: no cover - direct script execution
    from safe_exec import (  # type: ignore
        UnsafeBenchmarkCodeError,
        format_safe_python_failure,
        run_python_code,
        run_python_code_with_stdin,
        run_python_command,
        validate_python_source_safe,
    )


@dataclass(frozen=True)
class PythonAssertionAdapter:
    """Evaluate assertion-style Python benchmarks such as MBPP+."""

    verify_script_builder: Callable[[str], str]
    timeout_label: str = "benchmark evaluation"
    failure_label: str = "Benchmark evaluation failed."
    success_label: str = "All tests passed!"

    def evaluate(
        self,
        *,
        workspace: Path,
        solution_file: Path,
        assertion_code: str,
        timeout: int,
    ) -> tuple[bool, str]:
        if not solution_file.exists():
            return False, "solution.py not found"

        security_error = _validate_solution_file(solution_file)
        if security_error:
            return False, security_error

        result = run_python_code(
            self.verify_script_builder(assertion_code),
            cwd=workspace,
            timeout=timeout,
            artifact_stem="mbpp-assertion",
        )
        if result.timed_out:
            return False, format_safe_python_failure(self.timeout_label, result)
        output = result.output
        if result.returncode == 0:
            return True, output or self.success_label
        return False, output or self.failure_label


@dataclass(frozen=True)
class PythonVerifierAdapter:
    """Evaluate benchmarks that build one combined verifier script."""

    verify_code_builder: Callable[[str], str]
    timeout_label: str = "hidden evaluation"
    failure_label: str = "Hidden evaluation failed."
    success_label: str = "All hidden tests passed!"

    def evaluate(
        self,
        *,
        workspace: Path,
        solution_file: Path,
        fallback_solution: str,
        timeout: int,
    ) -> tuple[bool, str]:
        solution_code = (
            solution_file.read_text(encoding="utf-8")
            if solution_file.exists()
            else fallback_solution
        )
        security_error = _validate_solution_source(solution_code, solution_file)
        if security_error:
            return False, security_error

        result = run_python_code(
            self.verify_code_builder(solution_code),
            cwd=workspace,
            timeout=timeout,
            artifact_stem="python-verifier",
        )
        if result.timed_out:
            return False, format_safe_python_failure(self.timeout_label, result)
        output = result.output
        return (
            result.returncode == 0,
            output or (self.success_label if result.returncode == 0 else self.failure_label),
        )


@dataclass(frozen=True)
class PythonStdinAdapter:
    """Evaluate stdin/stdout style Python benchmark cases."""

    source_wrapper: Callable[[str], str]
    output_matcher: Callable[[str, str], bool]
    public_context_formatter: Callable[[str, Dict[str, Any]], str]
    timeout: int = 10

    def evaluate(
        self,
        *,
        solution_file: Path,
        cases: List[Dict[str, Any]],
        public_count: int,
    ) -> Dict[str, Any]:
        try:
            solution_source = solution_file.read_text(encoding="utf-8")
            validate_python_source_safe(solution_source, filename=str(solution_file))
            wrapped_source = self.source_wrapper(solution_source)
        except UnsafeBenchmarkCodeError as exc:
            return _security_result(exc)

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

            proc = run_python_code_with_stdin(
                wrapped_source,
                stdin_text,
                cwd=solution_file.parent,
                timeout=self.timeout,
                artifact_stem=f"stdin-case-{idx}",
            )
            if proc.timed_out:
                failed += 1
                timed_out = True
                if visibility == "public":
                    lines.append(f"[TIMEOUT] public case {idx}")
                    lines.append(self.public_context_formatter("stdin", case))
                else:
                    private_failed += 1
                continue

            if proc.returncode != 0:
                failed += 1
                if visibility == "public":
                    lines.append(f"[ERROR] public case {idx}")
                    lines.append(self.public_context_formatter("stdin", case))
                    stderr_text = format_safe_python_failure("public case execution", proc)
                    if stderr_text:
                        lines.append(stderr_text)
                else:
                    private_failed += 1
                    lines.append(f"[ERROR] private case {idx}")
                continue

            actual = proc.stdout.strip()
            if not self.output_matcher(actual, expected):
                failed += 1
                if visibility == "public":
                    lines.append(f"[FAIL] public case {idx}")
                    lines.append(self.public_context_formatter("stdin", case))
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

        return _case_summary(
            cases=cases,
            public_count=public_count,
            failed=failed,
            timed_out=timed_out,
            public_passed=public_passed,
            private_passed=private_passed,
            private_failed=private_failed,
            lines=lines,
        )


@dataclass(frozen=True)
class PythonFunctionalAdapter:
    """Evaluate function-call style Python benchmark cases via helper code."""

    helper_code: str
    call_spec: str
    expected_parser: Callable[[str], Any]
    public_context_formatter: Callable[[str, Dict[str, Any]], str]
    timeout: int = 10

    def evaluate(
        self,
        *,
        solution_file: Path,
        cases: List[Dict[str, Any]],
        public_count: int,
    ) -> Dict[str, Any]:
        security_error = _validate_solution_file(solution_file)
        if security_error:
            return {
                "passed": False,
                "output": security_error,
                "public_passed": 0,
                "private_passed": 0,
            }

        failed = 0
        private_failed = 0
        timed_out = False
        public_passed = 0
        private_passed = 0
        lines: List[str] = []

        for idx, case in enumerate(cases, start=1):
            visibility = "public" if idx <= public_count else "private"
            expr = str(case.get("input", ""))
            expected = self.expected_parser(str(case.get("output", "")))

            proc = run_python_command(
                ["-c", self.helper_code, self.call_spec],
                input_text=expr,
                cwd=solution_file.parent,
                timeout=self.timeout,
                artifact_stem=f"functional-case-{idx}",
            )
            if proc.timed_out:
                failed += 1
                timed_out = True
                if visibility == "public":
                    lines.append(f"[TIMEOUT] public case {idx}")
                    lines.append(self.public_context_formatter("functional", case))
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
                    lines.append(self.public_context_formatter("functional", case))
                    err_type = helper_result.get("type") or "RuntimeError"
                    err_message = (
                        helper_result.get("message")
                        or proc.stderr.strip()
                        or "helper execution failed"
                    )
                    lines.append(f"  {err_type}: {err_message}")
                else:
                    private_failed += 1
                    lines.append(f"[ERROR] private case {idx}")
                continue

            actual = self.expected_parser(str(helper_result.get("value_repr", "")))
            if actual != expected:
                failed += 1
                if visibility == "public":
                    lines.append(f"[FAIL] public case {idx}")
                    lines.append(self.public_context_formatter("functional", case))
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

        return _case_summary(
            cases=cases,
            public_count=public_count,
            failed=failed,
            timed_out=timed_out,
            public_passed=public_passed,
            private_passed=private_passed,
            private_failed=private_failed,
            lines=lines,
        )


def _validate_solution_file(solution_file: Path) -> str:
    try:
        validate_python_source_safe(
            solution_file.read_text(encoding="utf-8"),
            filename=str(solution_file),
        )
    except UnsafeBenchmarkCodeError as exc:
        return f"[SECURITY] {exc}"
    return ""


def _validate_solution_source(solution_code: str, solution_file: Path) -> str:
    try:
        validate_python_source_safe(solution_code, filename=str(solution_file))
    except UnsafeBenchmarkCodeError as exc:
        return f"[SECURITY] {exc}"
    return ""


def _security_result(exc: UnsafeBenchmarkCodeError) -> Dict[str, Any]:
    return {
        "passed": False,
        "output": f"[SECURITY] {exc}",
        "public_passed": 0,
        "private_passed": 0,
    }


def _case_summary(
    *,
    cases: List[Dict[str, Any]],
    public_count: int,
    failed: int,
    timed_out: bool,
    public_passed: int,
    private_passed: int,
    private_failed: int,
    lines: List[str],
) -> Dict[str, Any]:
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


__all__ = [
    "PythonAssertionAdapter",
    "PythonFunctionalAdapter",
    "PythonStdinAdapter",
    "PythonVerifierAdapter",
]
