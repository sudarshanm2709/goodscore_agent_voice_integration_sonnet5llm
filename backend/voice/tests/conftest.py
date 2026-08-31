import sys
from pathlib import Path

# Make the `voice` package importable when pytest is run from backend/voice/
# (`pytest` or `pytest tests/`) without requiring the project to be
# pip-installed. The package root is backend/, one level above this
# directory's parent (backend/voice/) — i.e. three `.parent`s up from this
# file: tests/ -> voice/ -> backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

import pytest

from voice.adapters.stt import SpeechToTextAdapter, TranscriptionError, TranscriptionResult
from voice.adapters.tts import SynthesisError, TextToSpeechAdapter
from voice.config import FillerConfig
from voice.models import CallSession, CallState, Turn, new_call_id, new_turn_id


class FakeSTTAdapter(SpeechToTextAdapter):
    """Returns a scripted transcript, or raises if configured to fail."""

    def __init__(self, text: str = "what is my credit score", language: str | None = "en", fail: bool = False):
        self.text = text
        self.language = language
        self.fail = fail
        self.calls: list[bytes] = []
        self.language_hints_received: list[str | None] = []

    async def transcribe(self, audio_bytes: bytes, audio_format: str, language_hint):
        self.calls.append(audio_bytes)
        self.language_hints_received.append(language_hint)
        if self.fail:
            raise TranscriptionError("simulated STT failure")
        return TranscriptionResult(text=self.text, detected_language=self.language)


class FakeTTSAdapter(TextToSpeechAdapter):
    """Yields deterministic fake audio chunks per synthesize() call."""

    def __init__(self, fail: bool = False, chunk_delay: float = 0.0):
        self.fail = fail
        self.chunk_delay = chunk_delay
        self.synthesized_texts: list[str] = []

    async def synthesize(self, text: str, language_hint) -> AsyncIterator[bytes]:
        self.synthesized_texts.append(text)
        if self.fail:
            raise SynthesisError("simulated TTS failure")
        for i in range(2):
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield f"audio-chunk-{i}-{text[:8]}".encode("utf-8")


class FakeChatbotClient:
    """Replays a scripted list of SSE-style event dicts for every call.

    `invoke_turn` returns an async generator so callers can .aclose() it
    exactly like the real ChatbotClient, which turn_controller relies on.
    """

    def __init__(
        self,
        events: list[dict] | None = None,
        delay_between_events: float = 0.0,
        delays: list[float] | None = None,
        fail: bool = False,
    ):
        self.events = events if events is not None else [
            {"type": "text_delta", "text": "Your score is 742."},
            {"type": "done"},
        ]
        self.delay_between_events = delay_between_events
        # Per-event delay override (index-aligned with self.events); falls
        # back to delay_between_events for any event past the list's end.
        self.delays = delays
        self.fail = fail
        self.requests: list = []

    def invoke_turn(self, request):
        self.requests.append(request)
        return self._generate()

    async def _generate(self):
        from voice.clients.chatbot import ChatbotClientError

        if self.fail:
            raise ChatbotClientError("simulated chatbot failure")
        for index, event in enumerate(self.events):
            delay = self.delays[index] if self.delays and index < len(self.delays) else self.delay_between_events
            if delay:
                await asyncio.sleep(delay)
            yield event


@pytest.fixture
def filler_config() -> FillerConfig:
    return FillerConfig(min_wait_before_filler_seconds=0.05, repeat_cooldown_seconds=1.0)


@pytest.fixture
def call_session() -> CallSession:
    return CallSession(
        call_id=new_call_id(),
        user_id="user-123",
        chat_session_id="voice-" + new_call_id(),
        language="en",
        state=CallState.ACTIVE,
    )


@pytest.fixture
def turn(call_session: CallSession) -> Turn:
    t = Turn(turn_id=new_turn_id(), call_id=call_session.call_id)
    call_session.current_turn = t
    return t
