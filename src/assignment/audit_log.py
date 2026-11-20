"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    @staticmethod
    def _redact(text: str) -> tuple[str, list[str]]:
        result = content_filter(text)
        return result["redacted"], result["issues"]

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store a redacted request and its monotonic start time."""
        key = request_id or uuid4().hex
        safe_user_id, user_issues = self._redact(str(user_id))
        safe_text, text_issues = self._redact(text)
        self._open[key] = {
            "user_id": str(user_id),
            "safe_user_id": safe_user_id,
            "request": safe_text,
            "request_redacted": bool(text_issues),
            "user_id_redacted": bool(user_issues),
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store a redacted response, policy decision, and request latency."""
        key = request_id
        if key is None:
            key = next(
                (
                    candidate
                    for candidate, opened in self._open.items()
                    if opened["user_id"] == str(user_id)
                ),
                None,
            )
        opened = self._open.pop(key, {}) if key is not None else {}

        timestamp = utc_now_iso()
        latency_ms = (
            max(0.0, (time.perf_counter() - opened["started_monotonic"]) * 1000)
            if opened
            else 0.0
        )
        safe_user_id, user_issues = self._redact(str(user_id))
        safe_response, response_issues = self._redact(text)
        status = "error" if layer == "error" else ("blocked" if blocked else "allowed")
        reason = layer or ("policy_block" if blocked else "request_processed")

        self.logs.append(
            {
                "request_id": content_filter(key or "")["redacted"],
                "user_id": safe_user_id if user_issues else opened.get("safe_user_id", safe_user_id),
                "request": opened.get("request", ""),
                "response": safe_response,
                "started_at": opened.get("started_at"),
                "timestamp": timestamp,
                "completed_at": timestamp,
                "latency_ms": round(latency_ms, 3),
                "status": status,
                "blocked": bool(blocked),
                "layer": layer,
                "reason": reason,
                "metadata": {
                    "request_redacted": opened.get("request_redacted", False),
                    "response_redacted": bool(response_issues),
                    "user_id_redacted": bool(user_issues)
                    or opened.get("user_id_redacted", False),
                },
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
