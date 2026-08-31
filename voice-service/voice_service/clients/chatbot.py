"""Client for the existing GoodScore chatbot's /invocations endpoint.

This is the ONLY way the voice service talks to Bedrock, Strands, or
AgentCore Memory — always through the chatbot's existing HTTP contract,
never directly (see <agentcore_and_tools>). Two transport modes:

- "local"     — plain HTTP to a locally running backend/server.py, used
                for development and the local integration tests in
                tests/test_chatbot_client.py. No signing.
- "agentcore" — SigV4-signed HTTPS straight to the AgentCore Runtime
                invoke URL, signed with the ECS task's own IAM role.
                This mirrors the signing steps in backend/lambda_sts_function.py
                (same SigV4Auth/AWSRequest calls, same "bedrock-agentcore"
                service name) but the voice service signs for itself —
                it's a trusted backend caller, not a browser, so it does
                not need the passcode broker the web chat UI uses.

Cancellation (barge-in): callers iterate the async generator from
invoke_turn() and call `await agen.aclose()` to cancel a turn early. That
closes the underlying HTTP stream, which the existing server.py's
StreamingResponse already treats as a client disconnect and stops
generating — no chatbot-side code change needed (see agent.py review).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

import httpx

from ..config import ChatbotConfig
from ..observability import log_error, log_event


class ChatbotClientError(RuntimeError):
    """Raised when the chatbot endpoint cannot be reached or errors out."""


@dataclass
class ChatbotTurnRequest:
    """One voice turn's request to the existing chatbot.

    Maps to the additive, optional /invocations fields (channel, language,
    call_id, turn_id, response_style, credit_context_ref) documented in
    backend/server.py — every field here besides user_id/session_id/message
    is new and optional on the chatbot side; omitting them reproduces a
    plain chat request exactly.
    """

    user_id: str
    session_id: str
    message: str
    call_id: str
    turn_id: str
    language: Optional[str] = None
    response_style: str = "voice"
    credit_context_ref: Optional[str] = None

    def to_payload(self) -> dict:
        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "prompt": self.message,
            "channel": "voice",
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "response_style": self.response_style,
        }
        if self.language:
            payload["language"] = self.language
        if self.credit_context_ref:
            payload["credit_context_ref"] = self.credit_context_ref
        return payload


class ChatbotClient:
    def __init__(self, config: ChatbotConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.request_timeout_seconds, connect=config.connect_timeout_seconds
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def invoke_turn(self, request: ChatbotTurnRequest) -> AsyncIterator[dict]:
        """Stream parsed SSE events for one chatbot turn.

        Yields the same event dicts the existing chat frontend already
        consumes: text_delta | tool_call | tool_result | chips | done | error.
        """
        payload = request.to_payload()
        url, headers = self._build_request(payload)

        log_event(
            "chatbot_request_start",
            call_id=request.call_id,
            turn_id=request.turn_id,
            mode=self._config.mode,
        )
        try:
            async with self._client.stream(
                "POST", url, content=json.dumps(payload).encode("utf-8"), headers=headers
            ) as response:
                if response.status_code >= 400:
                    error_body = await response.aread()
                    raise ChatbotClientError(
                        f"Chatbot endpoint returned {response.status_code}: {error_body[:300]!r}"
                    )
                async for event in _iter_sse_events(response):
                    yield event
        except httpx.HTTPError as exc:
            log_error("chatbot_request_failed", exc, call_id=request.call_id, turn_id=request.turn_id)
            raise ChatbotClientError(f"Chatbot endpoint unreachable: {exc}") from exc
        finally:
            log_event("chatbot_request_end", call_id=request.call_id, turn_id=request.turn_id)

    def _build_request(self, payload: dict) -> tuple[str, dict[str, str]]:
        if self._config.mode == "local":
            base = (self._config.local_base_url or "").rstrip("/")
            return f"{base}/invocations", {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

        # agentcore mode — SigV4-sign the request with the ECS task role.
        url = self._config.agentcore_invoke_url or ""
        headers = _sign_agentcore_request(
            url=url,
            body=json.dumps(payload).encode("utf-8"),
            region=self._config.agentcore_region,
        )
        return url, headers


def _sign_agentcore_request(url: str, body: bytes, region: str) -> dict[str, str]:
    """SigV4-sign a POST to the AgentCore Runtime invoke URL.

    Same signing steps as backend/lambda_sts_function.py's
    get_sts_credentials()/SigV4Auth flow, but credentials come from the
    ECS task's IAM role via botocore's default credential chain
    (AWS_CONTAINER_CREDENTIALS_RELATIVE_URI, injected automatically by
    ECS Fargate) rather than exchanged AKIA keys — no STS call needed
    because the task role's credentials are already short-lived.
    """
    try:
        import botocore.session
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
    except ImportError as exc:  # pragma: no cover - dependency documented in requirements.txt
        raise ChatbotClientError(
            "botocore is required for CHATBOT_MODE=agentcore signing"
        ) from exc

    session = botocore.session.Session()
    credentials = session.get_credentials()
    if credentials is None:
        raise ChatbotClientError(
            "No AWS credentials available to sign the AgentCore Runtime request "
            "(expected the ECS task IAM role)."
        )

    parsed = urlparse(url)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Host": parsed.netloc,
    }
    aws_request = AWSRequest(method="POST", url=url, data=body, headers=headers)
    SigV4Auth(credentials, "bedrock-agentcore", region).add_auth(aws_request)
    return dict(aws_request.headers)


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse `data: <json>\\n\\n` Server-Sent Events, matching server.py's
    exact emission format (json.dumps(event) per line, blank line separator).
    """
    buffer = ""
    async for text_chunk in response.aiter_text():
        buffer += text_chunk
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            for line in raw_event.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data:
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    log_event("chatbot_sse_parse_error", raw=data[:200])
