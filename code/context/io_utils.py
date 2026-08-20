"""Context-layer atomic IO helpers.

Single implementation of atomic text writes shared by the context layer
(``HistoryManager``, ``ObservationTruncator``) and re-exported by the tool
layer (``tools/builtin/_code_utils.py``) — previously four near-identical
copies existed across modules (Q2-7).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def atomic_write(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write a file atomically via a temporary sibling path.

    - Creates parent directories on demand.
    - Uses a unique temp name (pid + uuid) so concurrent writers never
      collide on the same temp file.
    - Preserves the original file's permission bits when overwriting.
    - Cleans the temp file up on failure.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    original_mode = None
    if target.exists():
        try:
            original_mode = target.stat().st_mode
        except OSError:
            original_mode = None

    try:
        with open(temp_path, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
        if original_mode is not None:
            os.chmod(temp_path, original_mode)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
