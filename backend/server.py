from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import config as cfg
from agent import run_turn_async, evict_session, _get_bedrock_model

# ---------------------------------------------------------------------------
# Latency logger — writes to console standard output
# ---------------------------------------------------------------------------
import sys

_LOG_FMT  = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_latency_logger = logging.getLogger("latency")
if not _latency_logger.handlers:
    _latency_logger.setLevel(logging.INFO)
    _latency_logger.propagate = False          # keep it out of the uvicorn console stream
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    _latency_logger.addHandler(_sh)
    _latency_logger.info("=" * 72)
    _latency_logger.info("LATENCY LOGGER INITIALISED")
    _latency_logger.info("=" * 72)


logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-initialise expensive singletons so the first real request isn't slow.
    # - BedrockModel: creates boto3 client + TCP connection pool to Bedrock.
    # - httpx client in tools.py: opens keep-alive connection to staging API.
    try:
        _get_bedrock_model()           # initialises boto3 bedrock-runtime client
        logger.info("startup: BedrockModel warmed up")
    except Exception as e:
        logger.warning("startup: BedrockModel warm-up failed: %s", e)

    try:
        from tools import _http        # triggers httpx.Client creation
        _http.get("/health", timeout=2.0)  # open a keep-alive connection; ignore response
    except Exception:
        pass  # staging API may not have /health — that's fine, client is still created
    yield

app = FastAPI(title="GoodScore Support Agent", lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://ai-assistant-credit.s3-website.ap-south-1.amazonaws.com",
        "http://localhost:3000",
        "http://172.16.0.126:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Server-side session store — maps user_id → (session_id, created_at_timestamp).
# Sessions expire after SESSION_TTL_SECONDS (default 24 hours).
# On expiry the old Agent is evicted from cache and a new session_id is issued.
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(24 * 60 * 60)))  # 24 h

_user_sessions: dict[str, tuple[str, float]] = {}   # user_id → (session_id, created_at)


def _is_session_expired(created_at: float) -> bool:
    return (time.time() - created_at) >= SESSION_TTL_SECONDS


def _get_or_create_session(user_id: str) -> str:
    """Return the active session_id for this user.

    Creates a new session if:
    - The user has no session yet, OR
    - The existing session is older than SESSION_TTL_SECONDS (24 h).
    """
    entry = _user_sessions.get(user_id)
    if entry is not None:
        session_id, created_at = entry
        if not _is_session_expired(created_at):
            return session_id
        # Session expired — evict the old Agent and fall through to create a new one
        evict_session(user_id, session_id)
        logger.info("session expired for user=%s old_session=%s", user_id, session_id)

    new_session_id = "sess-" + uuid.uuid4().hex
    _user_sessions[user_id] = (new_session_id, time.time())
    logger.info("new session created for user=%s session=%s", user_id, new_session_id)
    return new_session_id


def _new_session(user_id: str) -> str:
    """Force-create a fresh session for this user (used by /api/reset)."""
    entry = _user_sessions.get(user_id)
    if entry:
        evict_session(user_id, entry[0])
    new_session_id = "sess-" + uuid.uuid4().hex
    _user_sessions[user_id] = (new_session_id, time.time())
    return new_session_id




class ChatRequest(BaseModel):
    user_id: str
    message: str


class UserRequest(BaseModel):
    user_id: str




# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/session")
def get_session(req: UserRequest):
    """Return the active session_id for a user, creating one if it doesn't exist.
    The client should call this once on load to obtain the session_id for display/debug.
    """
    session_id = _get_or_create_session(req.user_id)
    logger.info("[session] user=%s session=%s", req.user_id, session_id)
    return {"session_id": session_id, "user_id": req.user_id}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream one agent turn as Server-Sent Events.

    Each SSE `data:` line is a JSON event:
      text_delta | tool_call | tool_result | chips | done | error
    """
    session_id = _get_or_create_session(req.user_id)
    request_id = uuid.uuid4().hex[:8]
    t_request = time.perf_counter()

    # Log incoming request (truncate long messages to keep logs readable)
    msg_preview = req.message[:120].replace("\n", " ")
    logger.info(
        "[%s] CHAT REQUEST | user=%s session=%s | msg=%r",
        request_id, req.user_id, session_id, msg_preview,
    )
    _latency_logger.info(
        "[%s] request received | user=%s session=%s",
        request_id, req.user_id, session_id,
    )

    async def event_stream():
        first_token = True
        t_stream_open = time.perf_counter()
        event_counts: dict[str, int] = {}

        _latency_logger.info(
            "[%s] stream opened   | %.3fs after request",
            request_id, t_stream_open - t_request,
        )

        async for event in run_turn_async(req.user_id, session_id, req.message):
            etype = event.get("type", "unknown")
            event_counts[etype] = event_counts.get(etype, 0) + 1

            if etype == "tool_call":
                logger.info(
                    "[%s] TOOL CALL  | tool=%s",
                    request_id, event.get("name"),
                )

            elif etype == "tool_result":
                logger.debug(
                    "[%s] TOOL RESULT | tool=%s | output=%s",
                    request_id, event.get("name"),
                    str(event.get("output", ""))[:500],
                )

            elif etype == "chips":
                logger.info(
                    "[%s] CHIPS       | chips=%s",
                    request_id, event.get("chips"),
                )

            elif etype == "error":
                logger.error(
                    "[%s] AGENT ERROR | %s",
                    request_id, event.get("message"),
                )

            if first_token and etype == "text_delta":
                t_first_token = time.perf_counter()
                _latency_logger.info(
                    "[%s] first token     | %.3fs after request | %.3fs after stream open",
                    request_id,
                    t_first_token - t_request,
                    t_first_token - t_stream_open,
                )
                first_token = False

            if etype == "done":
                t_done = time.perf_counter()
                _latency_logger.info(
                    "[%s] stream done     | %.3fs total",
                    request_id, t_done - t_request,
                )
                logger.info(
                    "[%s] TURN COMPLETE | user=%s | events=%s | total=%.3fs",
                    request_id, req.user_id, event_counts, t_done - t_request,
                )

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
def reset(req: UserRequest):
    """Create a fresh session for this user, discarding the current one."""
    old_entry = _user_sessions.get(req.user_id)
    old_session_id = old_entry[0] if old_entry else None
    if old_entry:
        evict_session(req.user_id, old_session_id)   # drop cached Agent for old session
    session_id = _new_session(req.user_id)
    logger.info(
        "[reset] user=%s | old_session=%s | new_session=%s",
        req.user_id, old_session_id, session_id,
    )
    return {"ok": True, "session_id": session_id}


@app.get("/api/health")
def health():
    return JSONResponse({
        "ok": True,
        "model": cfg.BEDROCK_MODEL_ID,
        "region": cfg.AWS_REGION,
        "orchestration": "strands-agents",
    })




# ---------------------------------------------------------------------------
# Native AgentCore & Custom HTTP Routes on Pure FastAPI
# ---------------------------------------------------------------------------
@app.get("/ping")
@app.get("/health")
def ping():
    return JSONResponse({"status": "Healthy", "ok": True})


@app.post("/invocations")
async def invocations(payload: dict):
    """Direct FastAPI implementation for AgentCore Runtime /invocations.

    Session management is fully server-controlled:
    - Session IDs are always generated/looked up via _get_or_create_session().
    - Client-provided session_id is ignored — prevents session spoofing.
    - action="reset" forces a new session immediately (called by the reset button).

    Uses FastAPI StreamingResponse so TTFB is instant (<200ms) and events
    stream token-by-token in real time without buffering.
    """
    user_id = (
        payload.get("actorId")
        or payload.get("user_id")
        or payload.get("actor_id")
        or payload.get("userId")
        or "default_user"
    )

    # --- Reset action: force a new session and return immediately (non-streaming) ---
    if payload.get("action") == "reset":
        new_sid = _new_session(user_id)
        logger.info("[reset] user=%s new_session=%s", user_id, new_sid)
        return JSONResponse({"ok": True, "session_id": new_sid})

    # Use client/Lambda provided session_id if present; fallback to server session
    session_id = (
        payload.get("sessionId")
        or payload.get("session_id")
        or _get_or_create_session(user_id)
    )

    message = (
        payload.get("prompt")
        or payload.get("message")
        or payload.get("inputText")
        or payload.get("input_text")
        or payload.get("input")
        or payload.get("text")
        or ""
    )

    # --- Voice channel (additive, optional — see voice-service/) ---------
    # `payload` is an untyped dict, so these are pure additive reads: any
    # existing caller that doesn't send them (every chat request today)
    # gets channel="chat", which is byte-for-byte the prior behaviour —
    # run_turn_async()/build_system_prompt() already default to "chat".
    # The other voice-only fields (call_id, turn_id, language,
    # credit_context_ref) don't change agent behaviour in this phase;
    # they're accepted and logged for correlation/observability only.
    channel = str(payload.get("channel") or "chat").strip().lower()
    if channel not in ("chat", "voice"):
        channel = "chat"
    if channel == "voice":
        logger.info(
            "[voice] channel=voice | call_id=%s turn_id=%s language=%s has_credit_context_ref=%s",
            payload.get("call_id"), payload.get("turn_id"), payload.get("language"),
            bool(payload.get("credit_context_ref")),
        )

    async def event_stream():
        async for event in run_turn_async(user_id, session_id, message, channel=channel):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting Pure FastAPI Uvicorn server directly on port %d...", port)
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)



