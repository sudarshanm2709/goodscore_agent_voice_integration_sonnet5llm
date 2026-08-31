"""Token usage export — one human-readable text file per LLM call made
during a voice turn.

This is a voice-layer-only feature: the chatbot's `/invocations` `done`
event carries an additive `token_usage` field (input/output/total tokens)
only when the request was tagged channel="voice" (see
backend/chat/agent.py) — chat requests never get it, so this module never
runs for chat activity. Every turn that reaches the chatbot and gets a
usable answer produces exactly one export file here; turns that never
reach the chatbot (e.g. STT failed) have no usage to report and are
skipped, not written as an empty file.

Kept deliberately simple: synchronous file I/O on the event loop is
acceptable here because writes are small (one short text file), happen at
most once per turn (never in a hot loop), and must never block audio
delivery — see write_token_usage_log()'s caller in turn_controller.py,
invoked only after the turn's response is already fully sent.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .models import CallSession, Turn
from .observability import log_error, log_event


def _anonymize(user_id: str) -> str:
    """Same short, non-reversible reference used in sessions.py's logs —
    kept as a local copy rather than imported, since this module has no
    other reason to depend on sessions.py.
    """
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def write_token_usage_log(log_dir_path: str, session: CallSession, turn: Turn) -> None:
    """Write one text file for this turn's LLM token usage, if any.

    Silently does nothing if the turn has no usage to report (chatbot
    never replied, or didn't include usage — e.g. it failed before
    reaching the model). Never raises — a logging failure must not affect
    the call itself, which has already completed by the time this runs.
    """
    if not turn.token_usage:
        return

    try:
        log_dir = Path(log_dir_path)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_path = log_dir / f"{turn.call_id}__{turn.turn_id}__{timestamp}.txt"

        usage = turn.token_usage
        lines = [
            "GoodScore Voice — Token Usage Log",
            "=" * 34,
            f"call_id:        {turn.call_id}",
            f"turn_id:        {turn.turn_id}",
            f"user_ref:       {_anonymize(session.user_id)}",
            f"timestamp_utc:  {timestamp}",
            f"language:       {turn.detected_language or session.language}",
            "",
            f"input_tokens:   {usage.get('input_tokens')}",
            f"output_tokens:  {usage.get('output_tokens')}",
            f"total_tokens:   {usage.get('total_tokens')}",
            "",
        ]
        file_path.write_text("\n".join(lines), encoding="utf-8")

        log_event(
            "token_usage_logged",
            call_id=session.call_id,
            turn_id=turn.turn_id,
            total_tokens=usage.get("total_tokens"),
            file=str(file_path.name),
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break the call
        log_error("token_usage_log_failed", exc, call_id=session.call_id, turn_id=turn.turn_id)
