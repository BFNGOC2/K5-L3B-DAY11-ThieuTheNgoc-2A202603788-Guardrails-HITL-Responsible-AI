"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    error_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    total_latency_ms: float = 0.0

    def record_request(
        self,
        *,
        blocked: bool,
        latency_ms: float,
        rate_limited: bool = False,
        error: bool = False,
    ) -> None:
        """Record one completed request exactly once."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if error:
            self.error_requests += 1
        if rate_limited:
            self.rate_limit_hits += 1
        self.total_latency_ms += max(0.0, float(latency_ms))

    def check_metrics(self) -> list[Alert]:
        """Refresh threshold alerts from the current aggregate counters."""
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        alerts = []

        if self.total_requests and block_rate >= self.block_rate_threshold:
            alerts.append(
                Alert(
                    metric="block_rate",
                    value=block_rate,
                    threshold=self.block_rate_threshold,
                    message="Blocked request rate reached the configured threshold.",
                )
            )
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alerts.append(
                Alert(
                    metric="rate_limit_hits",
                    value=float(self.rate_limit_hits),
                    threshold=float(self.rate_limit_hit_threshold),
                    message="Rate-limit hits reached the configured threshold.",
                )
            )
        if self.judge_checks and judge_fail_rate >= self.judge_fail_rate_threshold:
            alerts.append(
                Alert(
                    metric="judge_fail_rate",
                    value=judge_fail_rate,
                    threshold=self.judge_fail_rate_threshold,
                    message="Judge failure rate reached the configured threshold.",
                )
            )

        self.alerts = alerts
        return list(self.alerts)

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        average_latency_ms = (
            self.total_latency_ms / self.total_requests
            if self.total_requests
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "allowed_requests": max(
                0, self.total_requests - self.blocked_requests - self.error_requests
            ),
            "blocked_requests": self.blocked_requests,
            "error_requests": self.error_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "average_latency_ms": round(average_latency_ms, 3),
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
