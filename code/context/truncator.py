"""Tool output truncation with recoverable full-output persistence."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ObservationTruncator:
    """Truncate oversized tool output while preserving a recoverable full copy."""

    CLEANUP_INTERVAL_SECONDS = 3600
    DEFAULT_RETENTION_DAYS = 7
    DEFAULT_GAP_MARKER = "...(中间省略)..."

    def __init__(
        self,
        max_lines: int = 2000,
        max_bytes: int = 51200,
        truncate_direction: str = "head",
        output_dir: str = "memory/tool-output",
        retention_days: int = DEFAULT_RETENTION_DAYS,
        hint_message: str = "Use Read to inspect specific sections of the saved output.",
        gap_marker: str = DEFAULT_GAP_MARKER,
    ):
        self.max_lines = max(1, int(max_lines))
        self.max_bytes = max(256, int(max_bytes))
        self.truncate_direction = truncate_direction
        self.output_dir = Path(output_dir)
        self.retention_days = max(1, int(retention_days))
        self.hint_message = hint_message
        self.gap_marker = gap_marker
        self._last_cleanup_ts = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_outputs(force=True)

    def truncate(
        self,
        tool_name: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        self._cleanup_old_outputs()

        text = output or ""
        lines = text.splitlines()
        total_bytes = len(text.encode("utf-8"))
        full_output_path = self._existing_output_path(metadata)

        if len(lines) <= self.max_lines and total_bytes <= self.max_bytes:
            stats = self._build_stats(
                original_lines=len(lines),
                original_bytes=total_bytes,
                kept_lines=len(lines),
                kept_bytes=total_bytes,
                omitted_lines=0,
                omitted_bytes=0,
                start_time=start,
            )
            return {
                "truncated": False,
                "preview": text,
                "display_preview": text,
                "notice": None,
                "full_output_path": full_output_path,
                "stats": stats,
            }

        preview_info = self._build_preview(text)
        reused_output = bool(full_output_path)
        if not full_output_path:
            full_output_path = self._save_full_output(tool_name, text, metadata)

        preview = preview_info["preview"].rstrip("\n")
        notice = self._build_notice(preview_info, full_output_path)
        display_preview = "\n\n".join(part for part in (preview, notice) if part)
        stats = self._build_stats(
            original_lines=len(lines),
            original_bytes=total_bytes,
            kept_lines=preview_info["kept_lines"],
            kept_bytes=preview_info["kept_bytes"],
            omitted_lines=preview_info["omitted_lines"],
            omitted_bytes=preview_info["omitted_bytes"],
            start_time=start,
            reused_output_path=reused_output,
        )

        return {
            "truncated": True,
            "preview": preview,
            "display_preview": display_preview,
            "notice": notice,
            "full_output_path": full_output_path,
            "stats": stats,
        }

    def _build_notice(self, preview_info: Dict[str, Any], full_output_path: str) -> str:
        return "\n".join(
            [
                (
                    "[output truncated: omitted approximately "
                    f"{preview_info['omitted_lines']} line(s) / {preview_info['omitted_bytes']} byte(s)]"
                ),
                f"Full output saved to: {full_output_path}",
                self.hint_message,
            ]
        )

    def _build_stats(
        self,
        *,
        original_lines: int,
        original_bytes: int,
        kept_lines: int,
        kept_bytes: int,
        omitted_lines: int,
        omitted_bytes: int,
        start_time: float,
        reused_output_path: bool = False,
    ) -> Dict[str, Any]:
        return {
            "direction": self.truncate_direction,
            "original_lines": original_lines,
            "original_bytes": original_bytes,
            "kept_lines": kept_lines,
            "kept_bytes": kept_bytes,
            "omitted_lines": omitted_lines,
            "omitted_bytes": omitted_bytes,
            "reused_output_path": reused_output_path,
            "time_ms": int((time.time() - start_time) * 1000),
        }

    def _build_preview(self, text: str) -> Dict[str, Any]:
        lines = text.splitlines()
        total_bytes = len(text.encode("utf-8"))

        if self.truncate_direction == "tail":
            preview = self._collect_tail(lines, self.max_lines, self.max_bytes)
        elif self.truncate_direction == "head_tail":
            preview = self._collect_head_tail(lines, self.max_lines, self.max_bytes)
        else:
            preview = self._collect_head(lines, self.max_lines, self.max_bytes)

        if not preview["lines"] and text:
            raw_preview = self._truncate_raw_text(text)
            raw_bytes = len(raw_preview.encode("utf-8"))
            raw_lines = raw_preview.splitlines() or [raw_preview]
            return {
                "preview": raw_preview,
                "kept_lines": len(raw_lines),
                "kept_bytes": raw_bytes,
                "omitted_lines": max(0, len(lines) - len(raw_lines)),
                "omitted_bytes": max(0, total_bytes - raw_bytes),
            }

        preview_text = "\n".join(preview["lines"])
        kept_bytes = len(preview_text.encode("utf-8"))
        kept_lines = len(preview["lines"])
        return {
            "preview": preview_text,
            "kept_lines": kept_lines,
            "kept_bytes": kept_bytes,
            "omitted_lines": max(0, len(lines) - preview["source_line_count"]),
            "omitted_bytes": max(0, total_bytes - preview["source_bytes"]),
        }

    def _collect_head(self, lines: List[str], max_lines: int, max_bytes: int) -> Dict[str, Any]:
        out: List[str] = []
        used_bytes = 0
        source_bytes = 0
        source_line_count = 0

        for line in lines:
            separator = 1 if out else 0
            line_bytes = len(line.encode("utf-8"))
            if len(out) >= max_lines or used_bytes + separator + line_bytes > max_bytes:
                break
            out.append(line)
            used_bytes += separator + line_bytes
            source_bytes += separator + line_bytes
            source_line_count += 1

        return {"lines": out, "source_bytes": source_bytes, "source_line_count": source_line_count}

    def _collect_tail(self, lines: List[str], max_lines: int, max_bytes: int) -> Dict[str, Any]:
        out: List[str] = []
        used_bytes = 0
        source_bytes = 0
        source_line_count = 0

        for line in reversed(lines):
            separator = 1 if out else 0
            line_bytes = len(line.encode("utf-8"))
            if len(out) >= max_lines or used_bytes + separator + line_bytes > max_bytes:
                break
            out.append(line)
            used_bytes += separator + line_bytes
            source_bytes += separator + line_bytes
            source_line_count += 1

        out.reverse()
        return {"lines": out, "source_bytes": source_bytes, "source_line_count": source_line_count}

    def _collect_head_tail(self, lines: List[str], max_lines: int, max_bytes: int) -> Dict[str, Any]:
        if max_lines < 3 or len(lines) <= max_lines:
            return self._collect_head(lines, max_lines, max_bytes)

        gap_bytes = len(self.gap_marker.encode("utf-8"))
        head_line_budget = max(1, max_lines // 2)
        tail_line_budget = max(1, max_lines - head_line_budget - 1)
        head_byte_budget = max(128, max_bytes // 2)

        head = self._collect_head(lines, head_line_budget, head_byte_budget)
        remaining_lines = lines[head["source_line_count"] :]
        tail_byte_budget = max(128, max_bytes - head["source_bytes"] - gap_bytes - 1)
        tail = self._collect_tail(remaining_lines, tail_line_budget, tail_byte_budget)

        if not tail["lines"]:
            return head

        if head["source_line_count"] + tail["source_line_count"] >= len(lines):
            merged_lines = head["lines"] + tail["lines"]
        else:
            merged_lines = head["lines"] + [self.gap_marker] + tail["lines"]

        return {
            "lines": merged_lines,
            "source_bytes": head["source_bytes"] + tail["source_bytes"],
            "source_line_count": head["source_line_count"] + tail["source_line_count"],
        }

    def _truncate_raw_text(self, text: str) -> str:
        payload = text.encode("utf-8")
        if self.truncate_direction == "tail":
            return payload[-self.max_bytes :].decode("utf-8", errors="ignore")
        if self.truncate_direction == "head_tail" and len(payload) > self.max_bytes:
            half = max(1, self.max_bytes // 2)
            head = payload[:half].decode("utf-8", errors="ignore")
            tail = payload[-half:].decode("utf-8", errors="ignore")
            return f"{head}\n{self.gap_marker}\n{tail}"
        return payload[: self.max_bytes].decode("utf-8", errors="ignore")

    def _existing_output_path(self, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        candidate = metadata.get("full_output_path")
        if not isinstance(candidate, str) or not candidate:
            return None
        path = Path(candidate).expanduser()
        return str(path.resolve()) if path.exists() else None

    def _save_full_output(
        self,
        tool_name: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = self.output_dir / f"tool_{timestamp}_{tool_name}.json"
        payload = {
            "tool": tool_name,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        self._atomic_write(filepath, serialized)
        return str(filepath.resolve())

    def _cleanup_old_outputs(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_cleanup_ts < self.CLEANUP_INTERVAL_SECONDS:
            return

        cutoff = now - self.retention_days * 24 * 60 * 60
        for item in self.output_dir.glob("tool_*.json"):
            try:
                if item.stat().st_mtime < cutoff:
                    item.unlink()
            except (FileNotFoundError, OSError):
                continue

        self._last_cleanup_ts = now

    @staticmethod
    def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temp_path.write_text(content, encoding=encoding)
        os.replace(temp_path, path)
