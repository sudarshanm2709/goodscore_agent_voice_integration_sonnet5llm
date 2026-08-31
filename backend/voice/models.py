"""Typed models for calls, turns, and the WebSocket audio protocol.

These are the external boundaries of the voice service: what a mobile
client sends/receives over the WebSocket, and the internal call/turn
state tracked in sessions.py. Kept as plain dataclasses + a small enum
set rather than a framework-heavy state machine — the call lifecycle is
simple enough not to need one.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TurnState(str, Enum):
    LISTENING = "listening"      # capturing user audio, running VAD/turn detection
    TRANSCRIBING = "transcribing"  # STT in flight
    THINKING = "thinking"        # chatbot request in flight
    SPEAKING = "speaking"        # TTS audio is being streamed to the client
    CANCELLED = "cancelled"      # superseded by a barge-in; output must be discarded
    DONE = "done"


class CallState(str, Enum):
    CONNECTING = "connecting"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    ENDED = "ended"


def new_call_id() -> str:
    return "call-" + uuid.uuid4().hex


def new_turn_id() -> str:
    return "turn-" + uuid.uuid4().hex[:12]


@dataclass
class Turn:
    """One user-utterance-to-bot-answer exchange within a call."""

    turn_id: str
    call_id: str
    started_at: float = field(default_factory=time.monotonic)
    state: TurnState = TurnState.LISTENING
    transcript: str | None = None
    detected_language: str | None = None
    answer_text_parts: list[str] = field(default_factory=list)
    cancelled: bool = False
    # Set once the first audio chunk of this turn's spoken response (filler
    # or real answer) has been sent — used to log the time_to_first_audio
    # metric exactly once per turn regardless of how many sentences/fillers
    # are spoken. See turn_controller.py's _speak().
    first_audio_logged: bool = False
    # Token usage for this turn's LLM call, read off the chatbot's `done`
    # SSE event (see backend/chat/agent.py — additive, voice-channel-only
    # field). None if the turn never reached the chatbot (e.g. STT failed)
    # or the chatbot didn't report usage (e.g. agent construction failed).
    token_usage: dict | None = None

    def mark_cancelled(self) -> None:
        self.cancelled = True
        self.state = TurnState.CANCELLED


@dataclass
class CreditPrefetchResult:
    """Call-scoped, in-memory-only credit report snapshot.

    Never persisted, never sent to the mobile client, and dropped when
    the call ends (see sessions.py cleanup and prefetch.py).
    """

    fetched_at: float
    ok: bool
    summary: dict[str, Any] | None   # small digest only (e.g. score, bureau) — not the raw report
    error: str | None = None


@dataclass
class CallSession:
    """All state the voice service keeps for one active call.

    In-memory only per <voice_service> Phase-1 scope — no Redis/DynamoDB.
    Owned exclusively by sessions.py; other modules receive it by
    reference through SessionStore lookups, never construct it directly.
    """

    call_id: str
    user_id: str
    chat_session_id: str            # distinct from any web-chat session_id for this user
    language: str
    state: CallState = CallState.CONNECTING
    created_at: float = field(default_factory=time.monotonic)
    last_activity_at: float = field(default_factory=time.monotonic)
    current_turn: Turn | None = None
    turn_history_ids: list[str] = field(default_factory=list)
    credit_prefetch: CreditPrefetchResult | None = None
    fillers_spoken_for_turn: set[str] = field(default_factory=set)

    def touch(self) -> None:
        self.last_activity_at = time.monotonic()


# ---------------------------------------------------------------------------
# WebSocket protocol envelopes (client <-> server, JSON control messages).
# Binary WebSocket frames carry raw audio bytes and are not modelled here.
# ---------------------------------------------------------------------------

@dataclass
class ClientHello:
    """First control message a mobile client sends after connecting."""

    user_id: str
    auth_token: str
    language: Optional[str] = None          # explicit selection; None => auto_stt
    audio_input_format: Optional[str] = None
    audio_output_format: Optional[str] = None

    @staticmethod
    def from_dict(data: dict) -> "ClientHello":
        if not isinstance(data.get("user_id"), str) or not data["user_id"].strip():
            raise ValueError("hello.user_id is required")
        if not isinstance(data.get("auth_token"), str) or not data["auth_token"].strip():
            raise ValueError("hello.auth_token is required")
        return ClientHello(
            user_id=data["user_id"].strip(),
            auth_token=data["auth_token"].strip(),
            language=(data.get("language") or None),
            audio_input_format=data.get("audio_input_format"),
            audio_output_format=data.get("audio_output_format"),
        )


@dataclass
class ClientTurnEnd:
    """Client signals the user has stopped speaking for this turn."""

    turn_id: str


@dataclass
class ClientBargeIn:
    """Client signals the user started speaking while the bot was talking."""

    turn_id: str  # the new turn the user is starting


def server_event(event_type: str, **fields: Any) -> dict:
    """Build a JSON control event sent to the client over the WebSocket.

    Kept as a plain dict factory (not a dataclass) since the event shape
    genuinely varies by type and this is the serialization boundary —
    callers pass exactly the fields that type needs.
    """
    return {"type": event_type, **fields}
