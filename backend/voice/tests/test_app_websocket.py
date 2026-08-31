"""WebSocket protocol tests for app.py — the hello/ready handshake, turn_end
routing to the turn controller, and barge-in cancellation.

Uses a fake VoiceServiceState (fake turn controller / prefetch controller)
assigned directly to app.state.voice instead of running the real lifespan
— so these tests need no OpenRouter key, no chatbot server, and no
network access, matching every other test in this suite.

test_barge_in_echoes_the_interrupted_turn_id_not_the_new_one below is a
regression test for a real bug caught on review: the handler was echoing
back whatever turn_id the CLIENT sent in the barge_in message (the id of
the NEW turn starting) instead of the id of the turn actually being
interrupted, contradicting the documented protocol in app.py's docstring.
"""
import asyncio

import pytest
from starlette.testclient import TestClient

from voice.app import app
from voice.models import CallState
from voice.sessions import InMemorySessionStore


class FakeTurnController:
    """Replaces the real STT/chatbot/TTS pipeline with a no-op that just
    records what it was asked to do — these tests exercise app.py's
    WebSocket/session wiring, not the turn pipeline itself (that's
    test_turn_controller.py's job).
    """

    def __init__(self, block_until_cancelled: bool = False):
        self.calls: list[dict] = []
        self.block_until_cancelled = block_until_cancelled

    async def run_turn(self, session, turn, audio_bytes, audio_format, streamer):
        self.calls.append({
            "call_id": session.call_id,
            "turn_id": turn.turn_id,
            "audio_len": len(audio_bytes),
        })
        # Mirrors what the real TurnController populates on the Turn
        # object, so tests can verify app.py forwards it as captions.
        turn.transcript = "what is my credit score"
        turn.answer_text_parts.append("Your score is 742.")
        if self.block_until_cancelled:
            # Simulate a turn that's still "in flight" when barge-in
            # arrives, so app.py's current_task.cancel() actually has
            # something to cancel.
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise


class FakePrefetchController:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def prefetch(self, call_id: str, user_id: str):
        self.calls.append((call_id, user_id))
        return None


class _FakeAudioConfig:
    input_format = "webm"


class _FakeConfig:
    default_language = "hi-en"
    audio = _FakeAudioConfig()


class FakeVoiceServiceState:
    def __init__(self, block_until_cancelled: bool = False):
        self.session_store = InMemorySessionStore()
        self.turn_controller = FakeTurnController(block_until_cancelled)
        self.prefetch_controller = FakePrefetchController()
        self.config = _FakeConfig()


@pytest.fixture
def fake_state():
    state = FakeVoiceServiceState()
    app.state.voice = state
    yield state


@pytest.fixture
def client():
    return TestClient(app)


def test_hello_without_user_id_is_rejected(client, fake_state):
    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "auth_token": "tok"}')
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_valid_hello_receives_ready_with_call_id(client, fake_state):
    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-1", "auth_token": "tok"}')
        msg = ws.receive_json()
        assert msg["type"] == "ready"
        assert msg["call_id"].startswith("call-")
        assert msg["language"] == "hi-en"


def test_hello_starts_credit_prefetch(client, fake_state):
    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-42", "auth_token": "tok"}')
        ws.receive_json()  # ready
    # give the fire-and-forget prefetch task a moment to run
    import time
    time.sleep(0.05)
    assert fake_state.prefetch_controller.calls
    assert fake_state.prefetch_controller.calls[0][1] == "user-42"


def test_turn_end_with_no_buffered_audio_returns_error(client, fake_state):
    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-1", "auth_token": "tok"}')
        ws.receive_json()  # ready
        ws.send_text('{"type": "turn_end", "turn_id": "turn-1"}')
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert fake_state.turn_controller.calls == []


def test_turn_end_with_audio_invokes_turn_controller(client, fake_state):
    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-1", "auth_token": "tok"}')
        ws.receive_json()  # ready
        ws.send_bytes(b"some-audio-bytes")
        ws.send_text('{"type": "turn_end", "turn_id": "turn-1"}')
        msg = ws.receive_json()
        assert msg["type"] == "turn_started"
        assert msg["turn_id"] == "turn-1"

    import time
    time.sleep(0.05)
    assert fake_state.turn_controller.calls == [
        {"call_id": fake_state.turn_controller.calls[0]["call_id"], "turn_id": "turn-1", "audio_len": len(b"some-audio-bytes")}
    ]


def test_turn_end_eventually_receives_turn_done(client, fake_state):
    """Regression test for a protocol gap found while building a test
    client: without an explicit completion signal, a client can't tell
    when a turn's audio is fully sent except by guessing from silence.
    """
    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-1", "auth_token": "tok"}')
        ws.receive_json()  # ready
        ws.send_bytes(b"some-audio-bytes")
        ws.send_text('{"type": "turn_end", "turn_id": "turn-1"}')

        started = ws.receive_json()
        assert started == {"type": "turn_started", "turn_id": "turn-1"}
        done = ws.receive_json()
        assert done["type"] == "turn_done"
        assert done["turn_id"] == "turn-1"


def test_turn_done_carries_transcript_and_answer_text_captions(client, fake_state):
    """turn_done should forward what STT heard and what the chatbot said
    — captions for the UI, populated once the turn completes (see
    FakeTurnController, which mirrors what the real TurnController sets
    on the Turn object).
    """
    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-1", "auth_token": "tok"}')
        ws.receive_json()  # ready
        ws.send_bytes(b"some-audio-bytes")
        ws.send_text('{"type": "turn_end", "turn_id": "turn-1"}')

        ws.receive_json()  # turn_started
        done = ws.receive_json()

        assert done == {
            "type": "turn_done",
            "turn_id": "turn-1",
            "transcript": "what is my credit score",
            "answer_text": "Your score is 742.",
        }


def test_cancelled_turn_never_carries_captions(client):
    """A barge-in's turn_cancelled must never leak partial transcript or
    answer text — that content was discarded, not spoken to the user.
    """
    state = FakeVoiceServiceState(block_until_cancelled=True)
    app.state.voice = state

    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-1", "auth_token": "tok"}')
        ws.receive_json()  # ready
        ws.send_bytes(b"some-audio-bytes")
        ws.send_text('{"type": "turn_end", "turn_id": "turn-1"}')
        ws.receive_json()  # turn_started

        ws.send_text('{"type": "barge_in", "turn_id": "turn-2"}')
        cancelled = ws.receive_json()

        assert cancelled == {"type": "turn_cancelled", "turn_id": "turn-1"}
        assert "transcript" not in cancelled
        assert "answer_text" not in cancelled


def test_barge_in_echoes_the_interrupted_turn_id_not_the_new_one(client):
    """Regression test for the bug found on review: barge_in must reply
    with the id of the turn being interrupted (turn-1), not the id of the
    new turn the client is announcing (turn-2) — see app.py's docstring
    protocol description.
    """
    state = FakeVoiceServiceState(block_until_cancelled=True)
    app.state.voice = state

    with client.websocket_connect("/v1/voice/stream") as ws:
        ws.send_text('{"type": "hello", "user_id": "user-1", "auth_token": "tok"}')
        ws.receive_json()  # ready

        ws.send_bytes(b"first utterance audio")
        ws.send_text('{"type": "turn_end", "turn_id": "turn-1"}')
        started = ws.receive_json()
        assert started == {"type": "turn_started", "turn_id": "turn-1"}

        ws.send_text('{"type": "barge_in", "turn_id": "turn-2"}')
        cancelled = ws.receive_json()
        assert cancelled["type"] == "turn_cancelled"
        assert cancelled["turn_id"] == "turn-1"  # the interrupted turn, not turn-2
