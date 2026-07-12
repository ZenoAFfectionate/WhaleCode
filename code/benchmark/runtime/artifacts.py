"""Artifact storage for benchmark evaluator runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BenchmarkArtifactStore:
    """Save evaluator scripts, outputs, and metadata under a stable root."""

    root: Path
    retention: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        self.root.mkdir(parents=True, exist_ok=True)

    def record_eval(
        self,
        *,
        stem: str,
        code: Optional[str],
        stdout: str,
        stderr: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, str]:
        run_dir = self._new_run_dir(stem)
        paths: Dict[str, str] = {}
        if code is not None:
            paths["script"] = self._write_text(run_dir / "script.py", code)
        paths["stdout"] = self._write_text(run_dir / "stdout.txt", stdout or "")
        paths["stderr"] = self._write_text(run_dir / "stderr.txt", stderr or "")
        paths["metadata"] = self._write_text(
            run_dir / "metadata.json",
            json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False),
        )
        self._enforce_retention()
        return {key: self._relative(value) for key, value in paths.items()}

    def _new_run_dir(self, stem: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
        safe = safe[:48] or "eval"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for index in range(1, 10000):
            candidate = self.root / f"{stamp}-{index:04d}-{safe}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("Unable to allocate benchmark artifact directory")

    @staticmethod
    def _write_text(path: Path, content: str) -> str:
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _relative(self, value: str) -> str:
        path = Path(value)
        try:
            return str(path.relative_to(self.root.parent))
        except ValueError:
            return str(path)

    def _enforce_retention(self) -> None:
        if self.retention <= 0:
            return
        dirs = sorted(
            [path for path in self.root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        for old in dirs[self.retention :]:
            _remove_tree(old)


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink(missing_ok=True)
    path.rmdir()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


__all__ = ["BenchmarkArtifactStore"]
