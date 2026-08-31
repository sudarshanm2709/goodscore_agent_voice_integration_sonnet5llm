"""OpenRouter TTS adapter — Kokoro 82M.

Request/response shape confirmed against OpenRouter's published Text-to-
Speech API (POST {base_url}/audio/speech, OpenAI-Audio-Speech compatible,
raw audio byte stream response). The exact model slug (OPENROUTER_TTS_MODEL)
and voice (OPENROUTER_TTS_VOICE) are required/optional configuration — see
.env.example.

Streaming behaviour: httpx's streaming response relays bytes to the caller
as they arrive over the wire rather than buffering the whole clip, which is
what gives "time to first audio" its lower bound — regardless of whether
OpenRouter itself streams internally, our client never waits for the full
body before forwarding the first chunk.
"""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from ..config import OpenRouterConfig
from ..observability import log_error
from .tts import SynthesisError, TextToSpeechAdapter


class OpenRouterTTSAdapter(TextToSpeechAdapter):
    def __init__(self, config: OpenRouterConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        # See OpenRouterSTTAdapter for why auth headers are sent per-request
        # (below) instead of baked into the client here — an injected
        # client must still get a correctly authenticated request.
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.request_timeout_seconds),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def synthesize(self, text: str, language_hint: str | None) -> AsyncIterator[bytes]:
        if not text or not text.strip():
            return

        body: dict = {
            "model": self._config.tts_model,
            "input": text,
            "response_format": self._config.tts_response_format,
        }
        if self._config.tts_voice:
            body["voice"] = self._config.tts_voice

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        async with self._client.stream("POST", "/audio/speech", json=body, headers=headers) as response:
            if response.status_code >= 400:
                # Error responses are JSON, not audio — safe to buffer fully.
                error_body = await response.aread()
                log_error(
                    "tts_request_failed",
                    SynthesisError(f"status={response.status_code}"),
                    status=response.status_code,
                )
                raise SynthesisError(
                    f"OpenRouter TTS returned {response.status_code}: {error_body[:300]!r}"
                )

            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                log_error("tts_stream_failed", exc)
                raise SynthesisError(f"OpenRouter TTS stream interrupted: {exc}") from exc
