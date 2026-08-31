import base64
import json

import httpx
import pytest

from voice.adapters.openrouter_stt import OpenRouterSTTAdapter
from voice.adapters.openrouter_tts import OpenRouterTTSAdapter
from voice.adapters.stt import TranscriptionError
from voice.adapters.tts import SynthesisError
from voice.config import OpenRouterConfig


def _config(**overrides) -> OpenRouterConfig:
    defaults = dict(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        stt_model="nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
        tts_model="hexgrad/kokoro-82m",
        tts_voice=None,
        tts_response_format="pcm",
        request_timeout_seconds=5.0,
        max_retries=1,
    )
    defaults.update(overrides)
    return OpenRouterConfig(**defaults)


def _mock_client(handler) -> httpx.AsyncClient:
    """httpx.AsyncClient with a base_url set — matches how the real
    adapters construct their client (config.base_url), which is what
    makes the relative paths used in the adapter code ("/audio/...")
    resolvable at all.
    """
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://openrouter.ai/api/v1")


async def test_stt_sends_base64_audio_and_model_and_parses_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(200, json={"text": "what is my score", "usage": {"seconds": 2.1}})

    adapter = OpenRouterSTTAdapter(_config(), client=_mock_client(handler))
    result = await adapter.transcribe(b"raw-audio-bytes", "webm", language_hint="en")

    assert result.text == "what is my score"
    assert result.duration_seconds == 2.1
    assert captured["body"]["model"] == "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b"
    assert captured["body"]["input_audio"]["format"] == "webm"
    assert base64.b64decode(captured["body"]["input_audio"]["data"]) == b"raw-audio-bytes"
    assert captured["body"]["language"] == "en"


async def test_stt_rejects_empty_audio_without_a_network_call():
    adapter = OpenRouterSTTAdapter(_config(), client=_mock_client(
        lambda r: (_ for _ in ()).throw(AssertionError("should not be called"))
    ))
    with pytest.raises(TranscriptionError):
        await adapter.transcribe(b"", "webm", None)


async def test_stt_4xx_fails_immediately_without_retry():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad model"})

    adapter = OpenRouterSTTAdapter(_config(max_retries=3), client=_mock_client(handler))
    with pytest.raises(TranscriptionError):
        await adapter.transcribe(b"audio", "webm", None)
    assert calls["count"] == 1


async def test_stt_5xx_retries_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json={"text": "ok"})

    adapter = OpenRouterSTTAdapter(_config(max_retries=2), client=_mock_client(handler))
    result = await adapter.transcribe(b"audio", "webm", None)
    assert result.text == "ok"
    assert calls["count"] == 2


async def test_tts_streams_audio_chunks_and_sends_model_and_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/audio/speech"
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"chunk-one-chunk-two", headers={"content-type": "audio/pcm"})

    adapter = OpenRouterTTSAdapter(_config(), client=_mock_client(handler))
    chunks = [c async for c in adapter.synthesize("Your score is 742.", "en")]

    assert b"".join(chunks) == b"chunk-one-chunk-two"
    assert captured["body"]["model"] == "hexgrad/kokoro-82m"
    assert captured["body"]["input"] == "Your score is 742."
    assert captured["body"]["response_format"] == "pcm"


async def test_tts_empty_text_yields_no_chunks_and_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("TTS must not be called for empty text")

    adapter = OpenRouterTTSAdapter(_config(), client=_mock_client(handler))
    chunks = [c async for c in adapter.synthesize("   ", "en")]
    assert chunks == []


async def test_tts_error_response_raises_synthesis_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "provider down"})

    adapter = OpenRouterTTSAdapter(_config(), client=_mock_client(handler))
    with pytest.raises(SynthesisError):
        [c async for c in adapter.synthesize("hello", "en")]
