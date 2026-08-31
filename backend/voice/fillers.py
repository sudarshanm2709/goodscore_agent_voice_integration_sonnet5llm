"""Deterministic filler messages — no LLM involved (see <llm_usage>).

A filler is selected from a fixed, configuration-driven table keyed by
the backend operation currently in flight (e.g. waiting on the credit
report tool vs. a generic "thinking" delay), and is only spoken if that
operation is still running after a minimum wait. It stops immediately
once the real answer is ready or the turn is interrupted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .config import FillerConfig


class FillerOperation(str, Enum):
    """Backend states a filler message can describe.

    CREDIT_REPORT and BILLS map to the two slowest existing chat tools
    (get_credit_report, get_prefetched_bills hit the live GoodScore
    staging API); GENERIC covers any other in-flight chatbot turn.
    """

    CREDIT_REPORT = "credit_report"
    BILLS = "bills"
    GENERIC = "generic"


# Approved filler copy — Hindi-forward Hinglish per <language_requirements>,
# with English equivalents. Selection language comes from the active turn's
# detected/explicit language; English is the default when unknown.
_MESSAGES: dict[FillerOperation, dict[str, list[str]]] = {
    FillerOperation.CREDIT_REPORT: {
        "hi-en": ["Ek moment, main aapki credit report check kar raha hoon."],
        "hi": ["Ek pal rukiye, main aapki credit report dekh raha hoon."],
        "en": ["One moment, I'm checking your credit report."],
    },
    FillerOperation.BILLS: {
        "hi-en": ["Ek second, main aapke bills check kar raha hoon."],
        "hi": ["Ek pal rukiye, main aapke bills dekh raha hoon."],
        "en": ["One second, I'm checking your bills."],
    },
    FillerOperation.GENERIC: {
        "hi-en": ["Ek moment please."],
        "hi": ["Ek pal rukiye."],
        "en": ["One moment, please."],
    },
}


def _resolve_language(language: str) -> str:
    if language in _MESSAGES[FillerOperation.GENERIC]:
        return language
    if language.startswith("hi"):
        return "hi-en"
    return "en"


@dataclass
class FillerController:
    """Per-turn filler state: when to speak, what to say, when to stop.

    One instance is created per turn by the turn controller. It never
    calls Sonnet — the message text is a lookup, not a generation.
    """

    config: FillerConfig
    operation: FillerOperation
    language: str
    turn_started_at: float = field(default_factory=time.monotonic)
    _spoken: bool = field(default=False, init=False)
    _cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        """Called on interruption or once the real answer starts streaming."""
        self._cancelled = True

    @property
    def already_spoken(self) -> bool:
        return self._spoken

    def ready_to_speak(self, now: float | None = None) -> bool:
        """True exactly once, when the minimum wait has elapsed and the
        turn hasn't already produced a real answer or been cancelled.
        """
        if self._spoken or self._cancelled:
            return False
        elapsed = (now if now is not None else time.monotonic()) - self.turn_started_at
        return elapsed >= self.config.min_wait_before_filler_seconds

    def take_message(self) -> str | None:
        """Return the filler text to speak, marking it spoken (idempotent —
        a second call returns None so the same filler is never repeated).
        """
        if self._spoken or self._cancelled:
            return None
        self._spoken = True
        lang = _resolve_language(self.language)
        options = _MESSAGES[self.operation].get(lang) or _MESSAGES[self.operation]["en"]
        return options[0]
