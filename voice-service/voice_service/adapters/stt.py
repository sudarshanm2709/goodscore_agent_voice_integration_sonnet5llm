"""Provider-independent speech-to-text interface.

Keeping this as a real interface (not just calling OpenRouter inline from
the turn controller) is what lets the provider be swapped later without
touching call-handling logic, and lets tests substitute a fake adapter.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    text: str
    detected_language: str | None = None
    duration_seconds: float | None = None


class TranscriptionError(RuntimeError):
    """Raised for any STT provider failure: timeout, 4xx/5xx, malformed response."""


class SpeechToTextAdapter(abc.ABC):
    """One-shot transcription of a complete user utterance.

    The voice service performs its own turn/VAD detection (turn_controller.py)
    and hands the adapter one finished utterance at a time — this is not a
    bidirectional streaming interface because the confirmed OpenRouter STT
    contract (POST /audio/transcriptions) is request/response, not a socket.
    """

    @abc.abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language_hint: str | None,
    ) -> TranscriptionResult:
        """Transcribe one utterance.

        Args:
            audio_bytes: the complete utterance audio.
            audio_format: container/codec (e.g. "webm", "wav") — must match
                what the client actually sent; never assumed.
            language_hint: explicit language selection if the mobile app
                supplied one; None means let the provider auto-detect.

        Raises:
            TranscriptionError: on any provider failure. Callers must not
                let this propagate as a generic 500 to the client — the
                turn controller converts it into a spoken error/retry.
        """
        raise NotImplementedError
