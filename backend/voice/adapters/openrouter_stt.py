"""OpenRouter STT adapter — NVIDIA Nemotron 3.5 ASR.

Request/response shape confirmed against OpenRouter's published Speech-to-
Text API (POST {base_url}/audio/transcriptions, base64 JSON body,
Bearer-key auth, OpenAI-compatible response). The exact model slug
(OPENROUTER_STT_MODEL) is required configuration — see .env.example.

If the configured model is not actually served by OpenRouter, the API
returns a 4xx which this adapter surfaces as TranscriptionError with the
provider's message rather than silently falling back to a different model.
"""
from __future__ import annotations

import asyncio
import base64
import random

import httpx

from ..config import OpenRouterConfig
from ..observability import log_error, log_event
from .stt import SpeechToTextAdapter, TranscriptionError, TranscriptionResult


class OpenRouterSTTAdapter(SpeechToTextAdapter):
    def __init__(self, config: OpenRouterConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        # Client is injectable (tests, or a shared connection pool in
        # production — matching backend/tools.py's persistent-httpx-client
        # pattern). Auth/content-type headers are sent per-request (see
        # transcribe() below) rather than baked into the client here, so
        # an injected client — which has no reason to know this adapter's
        # API key — still gets a correctly authenticated request.
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.request_timeout_seconds),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language_hint: str | None,
    ) -> TranscriptionResult:
        if not audio_bytes:
            raise TranscriptionError("transcribe() called with empty audio")

        body: dict = {
            "model": self._config.stt_model,
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "format": audio_format,
            },
        }
        if language_hint:
            body["language"] = language_hint

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_retries + 2):
            try:
                response = await self._client.post("/audio/transcriptions", json=body, headers=headers)
                response.raise_for_status()
                data = response.json()
                text = data.get("text")
                if not isinstance(text, str):
                    raise TranscriptionError(f"Malformed OpenRouter STT response: {data!r}")
                return TranscriptionResult(
                    text=text,
                    detected_language=language_hint,
                    duration_seconds=(data.get("usage") or {}).get("seconds"),
                )

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status < 500 or attempt > self._config.max_retries:
                    log_error("stt_request_failed", exc, status=status, attempt=attempt)
                    raise TranscriptionError(
                        f"OpenRouter STT returned {status}: {exc.response.text[:300]}"
                    ) from exc
                last_error = exc

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt > self._config.max_retries:
                    log_error("stt_request_failed", exc, attempt=attempt)
                    raise TranscriptionError(f"OpenRouter STT unreachable: {exc}") from exc

            wait = min(0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2), 4.0)
            log_event("stt_retry", attempt=attempt, wait_seconds=round(wait, 2))
            await asyncio.sleep(wait)

        # Unreachable in practice (loop always returns or raises) — defensive.
        raise TranscriptionError(f"OpenRouter STT failed after retries: {last_error}")
