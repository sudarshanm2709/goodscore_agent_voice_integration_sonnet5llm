"""CloudWatch-compatible structured logging and lightweight in-process metrics.

CloudWatch Logs ingests whatever a container writes to stdout — there is no
special SDK needed, only a consistent, greppable/queryable shape. Each log
line is one JSON object so CloudWatch Logs Insights can filter on fields
like call_id/turn_id directly. Metrics are recorded as EMF (Embedded Metric
Format) lines, which CloudWatch parses into real custom metrics without any
extra agent — appropriate for an ECS Fargate task with no CloudWatch Agent
sidecar.

Safety rule enforced here: callers pass structured fields, never a report,
transcript, or token — see the redaction check in log_event().
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

logger = logging.getLogger("voice_service")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)

# Field names that must never appear in a log/metric payload. This is a
# defence-in-depth guard, not the only control — callers are still
# responsible for not passing sensitive values in the first place.
_FORBIDDEN_FIELDS = {
    "credit_report", "report", "transcript", "audio", "audio_bytes",
    "auth_token", "access_token", "api_key", "openrouter_api_key",
    "score", "account_number", "phone_number", "email",
}


def _redact(fields: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if key.lower() in _FORBIDDEN_FIELDS:
            safe[key] = "[redacted]"
        else:
            safe[key] = value
    return safe


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured JSON log line.

    Always include call_id/turn_id/request_id when available (pass as
    keyword fields) so CloudWatch Logs Insights queries can correlate a
    whole call across STT, chatbot, and TTS stages.
    """
    payload = {
        "ts": time.time(),
        "event": event,
        **_redact(fields),
    }
    logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))


def log_error(event: str, error: BaseException, **fields: Any) -> None:
    log_event(event, level=logging.ERROR, error_type=type(error).__name__, error_message=str(error), **fields)


def log_metric(name: str, value: float, unit: str = "Milliseconds", **dimensions: Any) -> None:
    """Emit one CloudWatch Embedded Metric Format (EMF) line.

    EMF lets CloudWatch extract a real custom metric from a plain stdout
    log line — no extra agent process, which matters for a bare Fargate
    task. Namespace/metric grouping is fixed; dimensions carry safe
    correlation fields only (never financial data).
    """
    safe_dimensions = _redact(dimensions)
    document = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": "GoodScore/VoiceService",
                "Dimensions": [list(safe_dimensions.keys())] if safe_dimensions else [[]],
                "Metrics": [{"Name": name, "Unit": unit}],
            }],
        },
        name: value,
        **safe_dimensions,
    }
    logger.info(json.dumps(document, ensure_ascii=False, default=str))


class StageTimer:
    """Context manager that logs a metric for how long a pipeline stage took.

    Usage:
        with StageTimer("stt_latency", call_id=call_id, turn_id=turn_id):
            transcript = await stt.transcribe(...)
    """

    __slots__ = ("metric_name", "dimensions", "_start")

    def __init__(self, metric_name: str, **dimensions: Any) -> None:
        self.metric_name = metric_name
        self.dimensions = dimensions
        self._start = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        log_metric(self.metric_name, elapsed_ms, unit="Milliseconds", **self.dimensions)
