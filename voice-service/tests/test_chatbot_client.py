import json
import os

import httpx
import pytest

from voice_service.config import ChatbotConfig
from voice_service.clients.chatbot import ChatbotClient, ChatbotClientError, ChatbotTurnRequest


def _sse_body(events: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode("utf-8")


def _local_config(base_url: str = "http://testserver") -> ChatbotConfig:
    return ChatbotConfig(
        mode="local",
        local_base_url=base_url,
        agentcore_invoke_url=None,
        agentcore_region="ap-south-1",
        request_timeout_seconds=5.0,
        connect_timeout_seconds=2.0,
    )


def _request() -> ChatbotTurnRequest:
    return ChatbotTurnRequest(
        user_id="user-1",
        session_id="voice-sess-1",
        message="what is my score",
        call_id="call-1",
        turn_id="turn-1",
        language="en",
    )


def test_turn_request_payload_is_additive_and_marks_voice_channel():
    payload = _request().to_payload()

    assert payload["channel"] == "voice"
    assert payload["call_id"] == "call-1"
    assert payload["turn_id"] == "turn-1"
    assert payload["user_id"] == "user-1"
    assert payload["session_id"] == "voice-sess-1"
    assert payload["prompt"] == "what is my score"
    # response_style always present for voice; credit_context_ref omitted when unset.
    assert payload["response_style"] == "voice"
    assert "credit_context_ref" not in payload


async def test_invoke_turn_parses_sse_events_matching_server_format():
    events = [
        {"type": "tool_call", "name": "get_credit_report", "input": {}},
        {"type": "text_delta", "text": "Your score is 742."},
        {"type": "done"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/invocations"
        body = json.loads(request.content)
        assert body["channel"] == "voice"
        return httpx.Response(200, content=_sse_body(events), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = ChatbotClient(_local_config(), client=httpx.AsyncClient(transport=transport))

    received = [e async for e in client.invoke_turn(_request())]
    assert received == events


async def test_invoke_turn_raises_on_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b'{"error": "boom"}')

    transport = httpx.MockTransport(handler)
    client = ChatbotClient(_local_config(), client=httpx.AsyncClient(transport=transport))

    with pytest.raises(ChatbotClientError):
        async for _ in client.invoke_turn(_request()):
            pass


async def test_invoke_turn_can_be_cancelled_early_without_error():
    events = [{"type": "text_delta", "text": "a"}, {"type": "text_delta", "text": "b"}, {"type": "done"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_body(events), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = ChatbotClient(_local_config(), client=httpx.AsyncClient(transport=transport))

    gen = client.invoke_turn(_request())
    first = await gen.__anext__()
    assert first["type"] == "text_delta"
    await gen.aclose()  # simulates barge-in cancellation mid-stream


async def test_agentcore_mode_signs_request_with_available_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fakefakefakefakefakefakefakefakefakefake")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    config = ChatbotConfig(
        mode="agentcore",
        local_base_url=None,
        agentcore_invoke_url="https://example.amazonaws.com/runtimes/fake/invocations",
        agentcore_region="ap-south-1",
        request_timeout_seconds=5.0,
        connect_timeout_seconds=2.0,
    )

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=_sse_body([{"type": "done"}]), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = ChatbotClient(config, client=httpx.AsyncClient(transport=transport))

    [_ async for _ in client.invoke_turn(_request())]

    assert "authorization" in captured["headers"]
    assert captured["headers"]["authorization"].startswith("AWS4-HMAC-SHA256")
