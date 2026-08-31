from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone, timedelta

import httpx
from strands import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IST timezone — used for Firestore timestamp conversion in transaction history
# ---------------------------------------------------------------------------
_IST = timezone(timedelta(hours=5, minutes=30))


def _firestore_to_ist(obj: object) -> object:
    """Recursively replace Firestore timestamp dicts with IST date strings.

    Firestore timestamps look like: {"_seconds": 1777921568, "_nanoseconds": 177000000}
    Converted to: "05 May 2026"

    Walks the entire response so nested timestamps are also converted.
    """
    if isinstance(obj, dict):
        # Detect Firestore timestamp — has _seconds or seconds key
        sec_key = "_seconds" if "_seconds" in obj else ("seconds" if "seconds" in obj else None)
        if sec_key:
            try:
                seconds = obj[sec_key]
                dt = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(_IST)
                return dt.strftime("%d %B %Y")
            except Exception:
                return obj
        return {k: _firestore_to_ist(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_firestore_to_ist(i) for i in obj]
    return obj

# ---------------------------------------------------------------------------
# Shared HTTP client — one persistent connection pool for the staging API.
# Created once at import time; reused across all requests and tool calls.
# ---------------------------------------------------------------------------
_STAGE_BASE = "https://subscription.stage.goodscore.io"

# Timeout raised to 30 s — staging can be slow on cold starts.
# connect=5 s prevents hanging forever if the server is unreachable.
_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

_http: httpx.Client = httpx.Client(
    base_url=_STAGE_BASE,
    verify=False,          # staging cert is self-signed
    timeout=_TIMEOUT,
    limits=httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
        keepalive_expiry=30,
    ),
    headers={"Content-Type": "application/json"},
)

# ---------------------------------------------------------------------------
# Retry config — exponential backoff with jitter.
#
# Attempt 1 fails → wait BASE_DELAY * 2^0 + jitter  =  1.0s ± 0.2s
# Attempt 2 fails → wait BASE_DELAY * 2^1 + jitter  =  2.0s ± 0.2s
# Attempt 3 fails → wait BASE_DELAY * 2^2 + jitter  =  4.0s ± 0.2s
# → give up, return error dict
#
# Jitter spreads concurrent retries so they don't all hammer staging at the
# same millisecond after a shared timeout (thundering herd prevention).
# ---------------------------------------------------------------------------
_MAX_RETRIES  = 3
_BASE_DELAY   = 1.0  
_JITTER_MAX   = 0.2   


def _backoff_delay(attempt: int) -> float:
    """Return the wait time for a given attempt number (1-indexed)."""
    delay = _BASE_DELAY * (2 ** (attempt - 1))          # 1s, 2s, 4s …
    jitter = random.uniform(0.0, _JITTER_MAX)           # 0–200ms spread
    return delay + jitter


def _fetch_stage_api(path: str, user_id: str) -> dict:
    """POST {userId} to a staging subscription endpoint, return parsed JSON.

    Retry policy:
    - Retries up to _MAX_RETRIES times on TimeoutException, ConnectError, 5xx.
    - Uses exponential backoff (1s → 2s → 4s) + random jitter on each wait.
    - Fails immediately on 4xx — these are client errors that won't resolve.
    - Uses the module-level persistent httpx client so TLS/TCP connections
      are reused — no per-call SSL handshake overhead.
    """
    last_error: Exception | None = None
    t_start = time.perf_counter()
    logger.info("API CALL START | path=%s user=%s", path, user_id)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = _http.post(path, json={"userId": user_id})
            r.raise_for_status()
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "API CALL OK    | path=%s status=%s attempt=%d elapsed=%.0fms",
                path, r.status_code, attempt, elapsed_ms,
            )
            return r.json()

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            wait = _backoff_delay(attempt)
            logger.warning(
                "API TIMEOUT    | attempt=%d/%d path=%s elapsed=%.0fms error=%s → retry in %.2fs",
                attempt, _MAX_RETRIES, path, elapsed_ms, e, wait,
            )

        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            if e.response.status_code >= 500:
                # 5xx — transient server error, retry with backoff
                last_error = e
                wait = _backoff_delay(attempt)
                logger.warning(
                    "API 5xx        | attempt=%d/%d path=%s status=%s elapsed=%.0fms → retry in %.2fs",
                    attempt, _MAX_RETRIES, path, e.response.status_code, elapsed_ms, wait,
                )
            else:
                # 4xx — client error, retrying won't help, fail immediately
                logger.error(
                    "API 4xx        | path=%s status=%s elapsed=%.0fms (no retry)",
                    path, e.response.status_code, elapsed_ms,
                )
                return {
                    "status": False,
                    "error_code": "CLIENT_ERROR",
                    "message": "There was an issue with the request. Please try again.",
                }

        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.error(
                "API ERROR      | path=%s elapsed=%.0fms error=%s (no retry)",
                path, elapsed_ms, e,
            )
            return {
                "status": False,
                "error_code": "SERVICE_UNAVAILABLE",
                "message": "I'm unable to fetch your data right now. Please try again in a few minutes.",
            }

        # Wait before next attempt (skip wait on last attempt)
        if attempt < _MAX_RETRIES:
            wait = _backoff_delay(attempt)
            time.sleep(wait)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.error(
        "API EXHAUSTED  | path=%s all %d attempts failed elapsed=%.0fms last_error=%s",
        path, _MAX_RETRIES, elapsed_ms, last_error,
    )
    return {
        "status": False,
        "error_code": "SERVICE_UNAVAILABLE",
        "message": "I'm unable to fetch your data right now. Please try again in a few minutes.",
    }


# ---------------------------------------------------------------------------
# Per-request tool factory — user-scoped data tools
# ---------------------------------------------------------------------------

def make_user_tools(user_id: str) -> list:
    """Return a list of Strands @tool functions bound to this user's user_id."""

    @tool
    def get_credit_report() -> dict:
        """
        Fetch the user's latest credit report from getCreditReportV3.

        """
        return _fetch_stage_api("/subscription/getCreditReportV3", user_id)

    @tool
    def get_prefetched_bills() -> dict:
        """Fetch the user's outstanding loan/EMI/card bills from getPreFetchedBillNew.
        Always recalculate overdue status by comparing dueDate with today's date
        — the status field can be stale.
        """
        return _fetch_stage_api("/subscription/getPreFetchedBillNew", user_id)

    @tool
    def get_subscription_details() -> dict:
        """Fetch the user's GoodScore subscription status and details.
        Use ONLY for subscription management queries: plan status, renewal date,
        cancellation requests, payment failures, or autopay issues.
        """
        return _fetch_stage_api("/autopay/action/homepage", user_id)

    @tool
    def get_transaction_history() -> dict:
        """Fetch the user's overall payment and refund transaction history.

        Use this to check for:
        - Recent failed payments or double charges
        - Refund statuses and dates
        - Listing the user's latest transactions / past payments

        All Firestore timestamps in the response are pre-converted to IST
        date strings (e.g. "16 July 2026"). Use these dates directly.
        """
        payload = _fetch_stage_api("/subscription/getOverallTransactionHistory", user_id)
        return _firestore_to_ist(payload)

    return [
        get_credit_report,
        get_prefetched_bills,
        get_subscription_details,
        get_transaction_history,
    ]
