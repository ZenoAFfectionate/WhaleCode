#!/usr/bin/env python3
"""Run a small SWE-bench Verified before/after comparison on the same sample.

Default behavior:
- before = current git HEAD
- after = current working tree
- sample size = 20
- sample selection = deterministic random sample from data/SWEV/test.jsonl

The script runs both variants on the exact same instance IDs, then compares:
- diff_count
- no_diff_rate
- timeout_rate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "SWEV" / "test.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "_results"
BOOTSTRAP_CODE = """
import sys, types
from pathlib import Path
CODE_DIR = Path('code')
pkg = types.ModuleType('hello_agents')
pkg.__path__ = [str(CODE_DIR)]
pkg.__file__ = str(CODE_DIR / '__init__.py')
sys.modules['hello_agents'] = pkg
from hello_agents.benchmark.swev_bench import main
sys.argv = ['swev_bench'] + sys.argv[1:]
main()
""".strip()


@dataclass
class RunMetrics:
    label: str
    total: int
    diff_count: int
    diff_rate: float
    no_diff_count: int
    no_diff_rate: float
    timeout_count: int
    timeout_rate: float
    other_error_count: int
    results_file: str
    predictions_file: Optional[str]


def _run(cmd: List[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def _latest_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    anonymous: List[Dict[str, Any]] = []
    for record in records:
        task_id = record.get("task_id")
        if task_id is None:
            anonymous.append(record)
        else:
            latest[str(task_id)] = record
    return list(latest.values()) + anonymous


def _load_results(results_file: Path) -> List[Dict[str, Any]]:
    if not results_file.exists():
        return []
    records: List[Dict[str, Any]] = []
    with results_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return _latest_records(records)


def _is_timeout(record: Dict[str, Any]) -> bool:
    if record.get("timeout") is True:
        return True
    error = str(record.get("error") or "").lower()
    return "timed out" in error or error.startswith("timeout:")


def _is_no_diff(record: Dict[str, Any]) -> bool:
    if record.get("agent_diff"):
        return False
    return record.get("error") == "Agent produced no changes"


def _compute_metrics(label: str, records: List[Dict[str, Any]], results_file: Path, predictions_file: Optional[Path]) -> RunMetrics:
    total = len(records)
    diff_count = sum(1 for record in records if bool(record.get("agent_diff")))
    no_diff_count = sum(1 for record in records if _is_no_diff(record))
    timeout_count = sum(1 for record in records if _is_timeout(record))
    other_error_count = sum(
        1
        for record in records
        if not record.get("agent_diff") and not _is_no_diff(record) and not _is_timeout(record)
    )
    return RunMetrics(
        label=label,
        total=total,
        diff_count=diff_count,
        diff_rate=round(diff_count / total * 100, 2) if total else 0.0,
        no_diff_count=no_diff_count,
        no_diff_rate=round(no_diff_count / total * 100, 2) if total else 0.0,
        timeout_count=timeout_count,
        timeout_rate=round(timeout_count / total * 100, 2) if total else 0.0,
        other_error_count=other_error_count,
        results_file=str(results_file),
        predictions_file=str(predictions_file) if predictions_file and predictions_file.exists() else None,
    )


def _load_task_ids(data_path: Path) -> List[str]:
    task_ids: List[str] = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_id = obj.get("instance_id") or obj.get("task_id")
            if task_id:
                task_ids.append(str(task_id))
    if not task_ids:
        raise ValueError(f"No task IDs found in {data_path}")
    return task_ids


def _select_task_ids(data_path: Path, sample_size: int, seed: int, explicit_task_ids: Optional[List[str]]) -> List[str]:
    if explicit_task_ids:
        return [str(task_id) for task_id in explicit_task_ids]
    task_ids = _load_task_ids(data_path)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(task_ids):
        raise ValueError(f"sample_size={sample_size} exceeds dataset size {len(task_ids)}")
    rng = random.Random(seed)
    return sorted(rng.sample(task_ids, sample_size))


def _add_worktree(repo_root: Path, git_ref: str, prefix: str) -> Path:
    worktree_dir = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        _run(
            ["git", "worktree", "add", "--detach", str(worktree_dir), git_ref],
            cwd=repo_root,
            check=True,
        )
    except Exception:
        shutil.rmtree(worktree_dir, ignore_errors=True)
        raise
    return worktree_dir


def _remove_worktree(repo_root: Path, worktree_dir: Path) -> None:
    try:
        _run(
            ["git", "worktree", "remove", "--force", str(worktree_dir)],
            cwd=repo_root,
            check=False,
        )
    finally:
        shutil.rmtree(worktree_dir, ignore_errors=True)


def _find_predictions_file(output_dir: Path, baseline_names: set[str]) -> Optional[Path]:
    current_names = {path.name for path in output_dir.glob("swev_predictions_*.jsonl")}
    new_names = sorted(current_names - baseline_names)
    if not new_names:
        return None
    return output_dir / new_names[-1]


def _run_variant(
    *,
    label: str,
    project_root: Path,
    output_dir: Path,
    repo_cache_dir: Path,
    data_path: Path,
    results_file: Path,
    task_ids: List[str],
    temperature: float,
    max_steps: int,
    timeout: int,
    task_timeout: int,
    model_name: str,
) -> tuple[subprocess.CompletedProcess[str], Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_cache_dir.mkdir(parents=True, exist_ok=True)
    existing_prediction_names = {path.name for path in output_dir.glob("swev_predictions_*.jsonl")}

    cmd = [
        sys.executable,
        "-c",
        BOOTSTRAP_CODE,
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--repo-cache-dir",
        str(repo_cache_dir),
        "--temperature",
        str(temperature),
        "--max-steps",
        str(max_steps),
        "--timeout",
        str(timeout),
        "--task-timeout",
        str(task_timeout),
        "--model-name",
        model_name,
        "--resume",
        str(results_file),
        "--task-ids",
        *task_ids,
    ]

    print(f"[compare] running {label}: {project_root}")
    proc = _run(cmd, cwd=project_root, check=False)
    predictions_file = _find_predictions_file(output_dir, existing_prediction_names)
    return proc, predictions_file


def _build_task_report(task_ids: List[str], before_records: List[Dict[str, Any]], after_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    before_by_id = {str(record.get("task_id")): record for record in before_records if record.get("task_id") is not None}
    after_by_id = {str(record.get("task_id")): record for record in after_records if record.get("task_id") is not None}
    rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        before = before_by_id.get(task_id, {})
        after = after_by_id.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "before_has_diff": bool(before.get("agent_diff")),
                "after_has_diff": bool(after.get("agent_diff")),
                "before_no_diff": _is_no_diff(before),
                "after_no_diff": _is_no_diff(after),
                "before_timeout": _is_timeout(before),
                "after_timeout": _is_timeout(after),
                "before_error": before.get("error"),
                "after_error": after.get("error"),
            }
        )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare before/after SWEV behavior on a 20-sample subset.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--task-ids", nargs="*", default=None, help="Use explicit SWEV task IDs instead of random sampling.")
    parser.add_argument("--before-ref", default="HEAD", help="Git ref for the baseline run. Default: HEAD.")
    parser.add_argument("--after-ref", default="current", help="Git ref for the candidate run, or 'current' for the current worktree.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--task-timeout", type=int, default=1200)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--keep-worktrees", action="store_true", help="Keep temporary comparison worktrees for debugging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    output_root = Path(args.output_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_dir = output_root / f"swev_small_compare_{timestamp}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    task_ids = _select_task_ids(data_path, args.sample_size, args.seed, args.task_ids)
    (compare_dir / "task_ids.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")

    before_root: Optional[Path] = None
    after_root: Optional[Path] = None

    if args.before_ref == "current":
        before_root = PROJECT_ROOT
    else:
        before_root = _add_worktree(PROJECT_ROOT, args.before_ref, "whale_swev_before_")

    if args.after_ref == "current":
        after_root = PROJECT_ROOT
    else:
        after_root = _add_worktree(PROJECT_ROOT, args.after_ref, "whale_swev_after_")

    before_results = compare_dir / "before_results.jsonl"
    after_results = compare_dir / "after_results.jsonl"
    before_cache = compare_dir / "cache_before"
    after_cache = compare_dir / "cache_after"

    try:
        before_proc, before_predictions = _run_variant(
            label="before",
            project_root=before_root,
            output_dir=compare_dir,
            repo_cache_dir=before_cache,
            data_path=data_path,
            results_file=before_results,
            task_ids=task_ids,
            temperature=args.temperature,
            max_steps=args.max_steps,
            timeout=args.timeout,
            task_timeout=args.task_timeout,
            model_name="before",
        )
        after_proc, after_predictions = _run_variant(
            label="after",
            project_root=after_root,
            output_dir=compare_dir,
            repo_cache_dir=after_cache,
            data_path=data_path,
            results_file=after_results,
            task_ids=task_ids,
            temperature=args.temperature,
            max_steps=args.max_steps,
            timeout=args.timeout,
            task_timeout=args.task_timeout,
            model_name="after",
        )
    finally:
        if not args.keep_worktrees:
            if before_root is not None and before_root != PROJECT_ROOT:
                _remove_worktree(PROJECT_ROOT, before_root)
            if after_root is not None and after_root != PROJECT_ROOT:
                _remove_worktree(PROJECT_ROOT, after_root)

    before_records = _load_results(before_results)
    after_records = _load_results(after_results)

    before_metrics = _compute_metrics("before", before_records, before_results, before_predictions)
    after_metrics = _compute_metrics("after", after_records, after_results, after_predictions)
    task_report = _build_task_report(task_ids, before_records, after_records)

    comparison = {
        "meta": {
            "timestamp": timestamp,
            "project_root": str(PROJECT_ROOT),
            "data_path": str(data_path),
            "sample_size": len(task_ids),
            "seed": args.seed,
            "task_ids": task_ids,
            "before_ref": args.before_ref,
            "after_ref": args.after_ref,
            "temperature": args.temperature,
            "max_steps": args.max_steps,
            "timeout": args.timeout,
            "task_timeout": args.task_timeout,
            "compare_dir": str(compare_dir),
        },
        "before": asdict(before_metrics),
        "after": asdict(after_metrics),
        "delta": {
            "diff_count": after_metrics.diff_count - before_metrics.diff_count,
            "no_diff_rate": round(after_metrics.no_diff_rate - before_metrics.no_diff_rate, 2),
            "timeout_rate": round(after_metrics.timeout_rate - before_metrics.timeout_rate, 2),
        },
        "run_exit_codes": {
            "before": before_proc.returncode,
            "after": after_proc.returncode,
        },
    }

    _write_json(compare_dir / "comparison_summary.json", comparison)
    _write_json(compare_dir / "task_report.json", task_report)
    (compare_dir / "before_stdout.log").write_text(before_proc.stdout, encoding="utf-8")
    (compare_dir / "before_stderr.log").write_text(before_proc.stderr, encoding="utf-8")
    (compare_dir / "after_stdout.log").write_text(after_proc.stdout, encoding="utf-8")
    (compare_dir / "after_stderr.log").write_text(after_proc.stderr, encoding="utf-8")

    print("")
    print("=== SWEV Small-Sample Comparison ===")
    print(f"compare_dir:   {compare_dir}")
    print(f"sample_size:   {len(task_ids)}")
    print(f"before_ref:    {args.before_ref}")
    print(f"after_ref:     {args.after_ref}")
    print("")
    print("before:")
    print(f"  diff_count:   {before_metrics.diff_count}/{before_metrics.total}")
    print(f"  no_diff_rate: {before_metrics.no_diff_rate:.2f}%")
    print(f"  timeout_rate: {before_metrics.timeout_rate:.2f}%")
    print("")
    print("after:")
    print(f"  diff_count:   {after_metrics.diff_count}/{after_metrics.total}")
    print(f"  no_diff_rate: {after_metrics.no_diff_rate:.2f}%")
    print(f"  timeout_rate: {after_metrics.timeout_rate:.2f}%")
    print("")
    print("delta (after - before):")
    print(f"  diff_count:   {comparison['delta']['diff_count']:+d}")
    print(f"  no_diff_rate: {comparison['delta']['no_diff_rate']:+.2f}%")
    print(f"  timeout_rate: {comparison['delta']['timeout_rate']:+.2f}%")

    return 0 if before_proc.returncode == 0 and after_proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
