"""Audio buffering and streaming primitives shared by the turn controller.

Two small, focused pieces:
- InboundAudioBuffer: accumulates raw bytes for the user's current turn
  until turn-end (client-signalled or VAD-driven), then hands the whole
  utterance to the STT adapter (OpenRouter's STT endpoint is a per-request
  transcription call, not a bidirectional stream — see adapters/openrouter_stt.py).
- OutboundAudioStreamer: relays TTS audio chunks to the client WebSocket as
  they arrive, and can be stopped instantly on barge-in so stale audio
  never reaches the speaker.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from starlette.websockets import WebSocket, WebSocketState


class InboundAudioBuffer:
    """Accumulates binary audio frames for one turn."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._byte_count = 0

    def append(self, chunk: bytes) -> None:
        self._chunks.append(chunk)
        self._byte_count += len(chunk)

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def is_empty(self) -> bool:
        return self._byte_count == 0

    def drain(self) -> bytes:
        """Return the accumulated audio and reset the buffer for the next turn."""
        data = b"".join(self._chunks)
        self._chunks.clear()
        self._byte_count = 0
        return data


class OutboundAudioStreamer:
    """Streams TTS audio chunks to a client WebSocket, cancellable mid-turn.

    stop() is called by the turn controller the instant a barge-in is
    detected. Any chunk send already awaited when stop() fires is allowed
    to finish (a WebSocket send can't be half-cancelled mid-frame without
    corrupting the stream); every subsequent chunk is dropped.
    """

    def __init__(self, websocket: WebSocket, turn_id: str) -> None:
        self._ws = websocket
        self._turn_id = turn_id
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    async def send_chunk(self, chunk: bytes) -> bool:
        """Send one audio chunk. Returns False if the stream was stopped
        (either by barge-in or because the socket is no longer connected) —
        callers should stop iterating their audio source when this happens.
        """
        if self._stopped:
            return False
        if self._ws.client_state != WebSocketState.CONNECTED:
            self._stopped = True
            return False
        await self._ws.send_bytes(chunk)
        return True
