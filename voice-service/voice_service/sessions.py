"""Temporary, in-memory call session state and its cleanup.

Phase 1 scope (per <voice_service> instructions): no Redis, no DynamoDB —
call state lives only as long as the process and the call. It is kept
behind a SessionStore interface purely so tests can swap in a fresh store
per test and so a future phase could add a distributed store without
touching call-handling code; there is exactly one implementation today.
"""
from __future__ import annotations

import abc
import asyncio
import time
from typing import Iterable, Optional

from .models import CallSession, CallState
from .observability import log_event


class SessionStore(abc.ABC):
    """Owns CallSession lifecycle: create, look up, and expire."""

    @abc.abstractmethod
    async def create(self, session: CallSession) -> None: ...

    @abc.abstractmethod
    async def get(self, call_id: str) -> Optional[CallSession]: ...

    @abc.abstractmethod
    async def end(self, call_id: str) -> Optional[CallSession]: ...

    @abc.abstractmethod
    async def sweep_expired(self, ttl_seconds: int) -> list[str]: ...

    @abc.abstractmethod
    async def active_count(self) -> int: ...


class InMemorySessionStore(SessionStore):
    """Dict-backed SessionStore guarded by an asyncio.Lock.

    Call isolation: every lookup is keyed strictly by call_id, and
    call_id is a server-generated UUID (see models.new_call_id) never
    accepted from the client — so one call can never address another
    call's state, satisfying the "calls cannot access each other's
    context" security requirement.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, session: CallSession) -> None:
        async with self._lock:
            self._sessions[session.call_id] = session
        log_event("call_session_created", call_id=session.call_id, user_id=_anonymize(session.user_id))

    async def get(self, call_id: str) -> Optional[CallSession]:
        async with self._lock:
            return self._sessions.get(call_id)

    async def end(self, call_id: str) -> Optional[CallSession]:
        async with self._lock:
            session = self._sessions.pop(call_id, None)
        if session is not None:
            session.state = CallState.ENDED
            log_event(
                "call_session_ended",
                call_id=call_id,
                duration_seconds=round(time.monotonic() - session.created_at, 2),
            )
        return session

    async def sweep_expired(self, ttl_seconds: int) -> list[str]:
        """Remove sessions idle for longer than ttl_seconds. Returns their ids.

        Called on a periodic timer from app.py's lifespan — this is the
        backstop that guarantees prefetched credit data and call context
        are dropped even if a client disconnects without a clean close.
        """
        now = time.monotonic()
        expired_ids: list[str] = []
        async with self._lock:
            for call_id, session in list(self._sessions.items()):
                if now - session.last_activity_at >= ttl_seconds:
                    expired_ids.append(call_id)
                    del self._sessions[call_id]
        for call_id in expired_ids:
            log_event("call_session_expired", call_id=call_id, ttl_seconds=ttl_seconds)
        return expired_ids

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._sessions)

    async def all_ids(self) -> Iterable[str]:
        async with self._lock:
            return list(self._sessions.keys())


def _anonymize(user_id: str) -> str:
    """Return a short, non-reversible reference safe for logs.

    Not cryptographic — this only needs to avoid printing the real
    user_id in plaintext logs while still letting the same user's log
    lines be correlated within a deployment.
    """
    import hashlib

    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


async def run_session_sweeper(store: SessionStore, ttl_seconds: int, interval_seconds: int) -> None:
    """Background task: periodically expire idle call sessions.

    Runs for the lifetime of the app (started in app.py's lifespan,
    cancelled on shutdown). A crash inside one sweep must not kill the
    loop — errors are logged and the loop continues.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await store.sweep_expired(ttl_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_event("session_sweeper_error", error=str(exc))
