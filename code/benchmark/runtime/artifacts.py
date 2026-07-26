"""Artifact storage for benchmark evaluator runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .._utils import _json_safe_full as _json_safe
except ImportError:  # pragma: no cover - direct script execution
    from _utils import _json_safe_full as _json_safe  # type: ignore


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
        try:
            if child.is_dir():
                _remove_tree(child)
            else:
                child.chmod(0o644)
                child.unlink(missing_ok=True)
        except (PermissionError, OSError):
            continue
    try:
        path.rmdir()
    except OSError:
        pass


__all__ = ["BenchmarkArtifactStore"]
