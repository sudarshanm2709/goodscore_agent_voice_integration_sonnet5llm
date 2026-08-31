"""Authenticated credit-report prefetch — starts right after call authentication.

Design constraints from <credit_prefetch>:
- Runs in parallel with audio/WebSocket setup, never delays the connection.
- Uses the authenticated GoodScore user context (the call's user_id) only.
- Talks to the existing GoodScore backend API, never Aurora directly.
- Result is call-scoped: stored on the CallSession, dropped when the call
  ends (sessions.py owns that cleanup — this module never persists
  anything itself).
- Only a small summary digest is kept, and it is never sent to the mobile
  client and never logged (see observability.py's redaction list).
- Does NOT change chatbot tool behaviour — the chatbot's own
  get_credit_report tool call (unchanged) remains the ground truth the
  model answers from. This prefetch exists to warm the GoodScore API path
  and to drive deterministic filler selection (fillers.py) while that
  tool call is in flight; it is a latency optimisation, not a bypass of
  the chatbot's tool-only-answers rule.
"""
from __future__ import annotations

import asyncio
import time

from .clients.goodscore import GoodScoreApiError, GoodScoreClient
from .models import CreditPrefetchResult
from .observability import StageTimer, log_error, log_event


class CreditPrefetchController:
    def __init__(self, client: GoodScoreClient) -> None:
        self._client = client

    async def prefetch(self, call_id: str, user_id: str) -> CreditPrefetchResult:
        """Fetch and summarise the user's credit report for this call.

        Never raises — a prefetch failure must not affect call setup or
        the user's ability to talk to the bot; the chatbot's own tool
        call is still the authoritative path if this fails.
        """
        try:
            with StageTimer("prefetch_latency", call_id=call_id):
                raw = await self._client.get_credit_report(user_id)
            summary = _summarize(raw)
            log_event("prefetch_complete", call_id=call_id, ok=True)
            return CreditPrefetchResult(fetched_at=time.monotonic(), ok=True, summary=summary)

        except GoodScoreApiError as exc:
            log_error("prefetch_failed", exc, call_id=call_id)
            return CreditPrefetchResult(fetched_at=time.monotonic(), ok=False, summary=None, error=str(exc))

        except Exception as exc:  # noqa: BLE001 - prefetch must never crash call setup
            log_error("prefetch_unexpected_error", exc, call_id=call_id)
            return CreditPrefetchResult(fetched_at=time.monotonic(), ok=False, summary=None, error=str(exc))


def start_prefetch_task(controller: CreditPrefetchController, call_id: str, user_id: str) -> asyncio.Task:
    """Fire the prefetch as a background task so call setup never waits on it.

    The caller (app.py's connection handler) attaches the resulting task
    to the CallSession and awaits it only at the point the result is
    actually needed (first turn), not before.
    """
    return asyncio.ensure_future(controller.prefetch(call_id, user_id))


def _summarize(raw: dict) -> dict:
    """Reduce the raw credit report to a small, non-sensitive digest.

    Deliberately minimal: only fields needed for filler-selection logic
    (fillers.py checks whether a report is available at all) are kept.
    The full report is discarded — it is never cached beyond this call
    and never forwarded to the mobile client per <credit_prefetch>.
    """
    return {
        "available": bool(raw) and raw.get("status", True) is not False,
    }
