import asyncio
import time

import pytest

from voice.models import CallSession, CallState, new_call_id
from voice.sessions import InMemorySessionStore


def _make_session(call_id: str | None = None) -> CallSession:
    return CallSession(
        call_id=call_id or new_call_id(),
        user_id="user-abc",
        chat_session_id="voice-sess-1",
        language="en",
        state=CallState.ACTIVE,
    )


async def test_create_and_get_round_trips():
    store = InMemorySessionStore()
    session = _make_session()
    await store.create(session)

    fetched = await store.get(session.call_id)
    assert fetched is session


async def test_get_unknown_call_id_returns_none():
    store = InMemorySessionStore()
    assert await store.get("call-does-not-exist") is None


async def test_end_removes_and_returns_session():
    store = InMemorySessionStore()
    session = _make_session()
    await store.create(session)

    ended = await store.end(session.call_id)
    assert ended is session
    assert ended.state == CallState.ENDED
    assert await store.get(session.call_id) is None


async def test_end_unknown_call_id_is_a_noop():
    store = InMemorySessionStore()
    assert await store.end("call-missing") is None


async def test_calls_are_isolated_by_call_id():
    """Two concurrent calls must never see each other's session state —
    this is the structural guarantee behind "calls cannot access each
    other's context" (each call_id is a distinct, server-generated key).
    """
    store = InMemorySessionStore()
    session_a = _make_session()
    session_b = _make_session()
    await store.create(session_a)
    await store.create(session_b)

    fetched_a = await store.get(session_a.call_id)
    fetched_b = await store.get(session_b.call_id)
    assert fetched_a.call_id != fetched_b.call_id
    assert fetched_a.user_id == "user-abc"
    assert fetched_a is not fetched_b


async def test_sweep_expired_removes_only_idle_sessions():
    store = InMemorySessionStore()
    fresh = _make_session()
    stale = _make_session()
    stale.last_activity_at = 0.0  # far in the past relative to time.monotonic()

    await store.create(fresh)
    await store.create(stale)

    expired_ids = await store.sweep_expired(ttl_seconds=1)

    assert stale.call_id in expired_ids
    assert fresh.call_id not in expired_ids
    assert await store.get(stale.call_id) is None
    assert await store.get(fresh.call_id) is not None


async def test_active_count_reflects_store_contents():
    store = InMemorySessionStore()
    assert await store.active_count() == 0
    await store.create(_make_session())
    await store.create(_make_session())
    assert await store.active_count() == 2
