"""Feedback helpers for benchmark runtime outputs."""

from __future__ import annotations

from typing import Dict


def artifact_hint(artifacts: Dict[str, str]) -> str:
    """Return a concise pointer to complete evaluator artifacts."""

    if not artifacts:
        return ""
    metadata = artifacts.get("metadata")
    if metadata:
        return f"Full evaluator artifacts: {metadata}"
    first = next(iter(artifacts.values()), "")
    return f"Full evaluator artifacts: {first}" if first else ""


def append_artifact_hint(feedback: str, artifacts: Dict[str, str]) -> str:
    hint = artifact_hint(artifacts)
    if not hint:
        return feedback
    text = feedback.strip()
    return f"{text}\n{hint}" if text else hint


__all__ = ["append_artifact_hint", "artifact_hint"]
