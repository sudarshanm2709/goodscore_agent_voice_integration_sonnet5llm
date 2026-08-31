"""Regression tests for the additive channel/voice fields on /invocations.

Verifies:
  - Existing chat callers (no channel field, or channel="chat") reach
    run_turn_async() with channel="chat" — the prior, unchanged default.
  - A voice-tagged request reaches run_turn_async() with channel="voice"
    and does not otherwise change how the request is parsed (user_id,
    session_id, message extraction all still work exactly as before).
  - An invalid/unknown channel value falls back to "chat" rather than
    raising or being passed through unvalidated.

`agent.run_turn_async` and `agent._get_bedrock_model` are monkeypatched on
the `server` module before any request is made, so this never constructs
a real Strands Agent, calls Bedrock, or needs AWS credentials/network —
consistent with these being regression tests for the additive parsing
logic in server.py itself, not an end-to-end Bedrock integration test.
"""
import pytest
from fastapi.testclient import TestClient

import server as server_module


@pytest.fixture
def captured_calls(monkeypatch):
    calls = []

    async def fake_run_turn_async(user_id, session_id, message, channel="chat"):
        calls.append({
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "channel": channel,
        })
        yield {"type": "text_delta", "text": "ok"}
        yield {"type": "done", "text": ""}

    monkeypatch.setattr(server_module, "run_turn_async", fake_run_turn_async)
    monkeypatch.setattr(server_module, "_get_bedrock_model", lambda: None)
    return calls


@pytest.fixture
def client():
    # Deliberately NOT using `with TestClient(...)` — that would run the
    # app's lifespan (Bedrock client warm-up, a staging-API health probe),
    # neither of which this test needs or wants to depend on.
    return TestClient(server_module.app)


def _read_sse_events(response) -> list[dict]:
    import json
    events = []
    for line in response.text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_existing_chat_request_without_channel_field_defaults_to_chat(client, captured_calls):
    response = client.post("/invocations", json={"user_id": "user-1", "prompt": "hello"})
    assert response.status_code == 200
    assert captured_calls[0]["channel"] == "chat"
    assert captured_calls[0]["user_id"] == "user-1"
    assert captured_calls[0]["message"] == "hello"


def test_explicit_chat_channel_matches_default_behaviour(client, captured_calls):
    response = client.post(
        "/invocations", json={"user_id": "user-1", "prompt": "hello", "channel": "chat"}
    )
    assert response.status_code == 200
    assert captured_calls[0]["channel"] == "chat"


def test_voice_channel_request_is_recognised_and_forwarded(client, captured_calls):
    response = client.post(
        "/invocations",
        json={
            "user_id": "user-1",
            "prompt": "what is my score",
            "channel": "voice",
            "call_id": "call-1",
            "turn_id": "turn-1",
            "language": "hi-en",
        },
    )
    assert response.status_code == 200
    assert captured_calls[0]["channel"] == "voice"
    # Existing extraction logic (user_id/message) must be unaffected by
    # the presence of the new optional fields.
    assert captured_calls[0]["user_id"] == "user-1"
    assert captured_calls[0]["message"] == "what is my score"


def test_unknown_channel_value_falls_back_to_chat(client, captured_calls):
    response = client.post(
        "/invocations", json={"user_id": "user-1", "prompt": "hello", "channel": "sms"}
    )
    assert response.status_code == 200
    assert captured_calls[0]["channel"] == "chat"


def test_reset_action_is_unaffected_by_channel_field(client, captured_calls):
    """The reset branch returns before run_turn_async is ever called —
    must remain true whether or not a channel field is present.
    """
    response = client.post(
        "/invocations", json={"user_id": "user-1", "action": "reset", "channel": "voice"}
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert captured_calls == []


def test_sse_events_stream_through_unchanged_shape(client, captured_calls):
    response = client.post("/invocations", json={"user_id": "user-1", "prompt": "hello"})
    events = _read_sse_events(response)
    assert events == [{"type": "text_delta", "text": "ok"}, {"type": "done", "text": ""}]
