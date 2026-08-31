"""FastAPI application and WebSocket entry point for the voice service.

WebSocket protocol (see README.md for the full contract):

  client -> server, JSON text frame, first message on the socket:
    {"type": "hello", "user_id": "...", "auth_token": "...", "language": "hi-en"?}

  server -> client, JSON text frame, once the call is set up:
    {"type": "ready", "call_id": "...", "language": "..."}

  client -> server, binary frames:
    raw audio bytes for the turn currently being captured.

  client -> server, JSON text frame, when the user stops speaking:
    {"type": "turn_end", "turn_id": "..."}

  client -> server, JSON text frame, when the user starts speaking over
  the bot (barge-in):
    {"type": "barge_in", "turn_id": "..."}

  server -> client, binary frames:
    synthesized answer audio for the current turn.

  server -> client, JSON text frames, informational:
    {"type": "turn_started", "turn_id": "..."}
    {"type": "turn_cancelled", "turn_id": "..."}
    {"type": "error", "message": "..."}
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .adapters.openrouter_stt import OpenRouterSTTAdapter
from .adapters.openrouter_tts import OpenRouterTTSAdapter
from .audio import InboundAudioBuffer, OutboundAudioStreamer
from .clients.chatbot import ChatbotClient
from .clients.goodscore import GoodScoreClient
from .config import VoiceServiceConfig, load_config
from .models import CallSession, CallState, ClientHello, Turn, TurnState, new_call_id, new_turn_id, server_event
from .observability import log_error, log_event
from .prefetch import CreditPrefetchController, start_prefetch_task
from .sessions import InMemorySessionStore, SessionStore, run_session_sweeper
from .turn_controller import TurnController


class VoiceServiceState:
    """Process-wide singletons, built once in the lifespan and attached to app.state."""

    def __init__(self, config: VoiceServiceConfig) -> None:
        self.config = config
        self.session_store: SessionStore = InMemorySessionStore()
        self.stt = OpenRouterSTTAdapter(config.openrouter)
        self.tts = OpenRouterTTSAdapter(config.openrouter)
        self.chatbot = ChatbotClient(config.chatbot)
        self.goodscore = GoodScoreClient(config.goodscore)
        self.prefetch_controller = CreditPrefetchController(self.goodscore)
        self.turn_controller = TurnController(self.stt, self.tts, self.chatbot, config.fillers)
        self._sweeper_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._sweeper_task = asyncio.ensure_future(
            run_session_sweeper(
                self.session_store,
                self.config.call_ttl_seconds,
                self.config.session_sweep_interval_seconds,
            )
        )

    async def stop(self) -> None:
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
        for adapter in (self.stt, self.tts, self.chatbot, self.goodscore):
            await adapter.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    state = VoiceServiceState(config)
    await state.start()
    app.state.voice = state
    log_event("voice_service_started", port=config.port, chatbot_mode=config.chatbot.mode)
    try:
        yield
    finally:
        await state.stop()


app = FastAPI(title="GoodScore Voice Service", lifespan=lifespan)


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "service": "goodscore-voice-service"})


@app.websocket("/v1/voice/stream")
async def voice_stream(websocket: WebSocket) -> None:
    state: VoiceServiceState = websocket.app.state.voice
    await websocket.accept()

    session: CallSession | None = None
    inbound = InboundAudioBuffer()
    current_task: asyncio.Task | None = None
    current_streamer: OutboundAudioStreamer | None = None

    try:
        # Step 1: require a valid hello before accepting any audio.
        hello_raw = await websocket.receive_text()
        try:
            hello = ClientHello.from_dict(json.loads(hello_raw))
        except (json.JSONDecodeError, ValueError) as exc:
            await websocket.send_json(server_event("error", message=f"invalid hello: {exc}"))
            await websocket.close(code=1008)
            return

        if not await _validate_call_auth(hello.user_id, hello.auth_token):
            await websocket.send_json(server_event("error", message="authentication failed"))
            await websocket.close(code=1008)
            return

        language = hello.language or state.config.default_language
        session = CallSession(
            call_id=new_call_id(),
            user_id=hello.user_id,
            chat_session_id="voice-" + new_call_id(),
            language=language,
            state=CallState.AUTHENTICATED,
        )
        await state.session_store.create(session)

        # Step 2: start credit prefetch in parallel — never block the
        # "ready" handshake on it (see prefetch.py).
        start_prefetch_task(state.prefetch_controller, session.call_id, session.user_id)
        session.state = CallState.ACTIVE

        await websocket.send_json(server_event("ready", call_id=session.call_id, language=language))
        log_event("call_connected", call_id=session.call_id)

        input_format = hello.audio_input_format or state.config.audio.input_format

        # Step 3: main receive loop — binary audio frames accumulate into
        # the current turn's buffer; JSON control frames drive turn
        # boundaries and barge-in.
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                inbound.append(message["bytes"])
                session.touch()
                continue

            if "text" not in message or message["text"] is None:
                continue

            try:
                control = json.loads(message["text"])
            except json.JSONDecodeError:
                await websocket.send_json(server_event("error", message="invalid control frame"))
                continue

            control_type = control.get("type")

            if control_type == "turn_end":
                turn_id = control.get("turn_id") or new_turn_id()
                audio_bytes = inbound.drain()
                if not audio_bytes:
                    await websocket.send_json(server_event("error", message="turn_end with no audio"))
                    continue

                turn = Turn(turn_id=turn_id, call_id=session.call_id)
                session.current_turn = turn
                session.turn_history_ids.append(turn_id)
                session.touch()

                current_streamer = OutboundAudioStreamer(websocket, turn_id)
                await websocket.send_json(server_event("turn_started", turn_id=turn_id))
                current_task = asyncio.ensure_future(
                    state.turn_controller.run_turn(session, turn, audio_bytes, input_format, current_streamer)
                )

            elif control_type == "barge_in":
                if session.current_turn is not None:
                    session.current_turn.mark_cancelled()
                if current_streamer is not None:
                    current_streamer.stop()
                if current_task is not None and not current_task.done():
                    current_task.cancel()
                inbound.drain()  # discard any audio queued for the interrupted turn
                await websocket.send_json(
                    server_event("turn_cancelled", turn_id=control.get("turn_id"))
                )
                log_event("turn_barge_in", call_id=session.call_id)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - never let one call crash the service
        log_error("voice_stream_unhandled_error", exc, call_id=(session.call_id if session else None))
    finally:
        if current_task is not None and not current_task.done():
            current_task.cancel()
        if session is not None:
            await state.session_store.end(session.call_id)


async def _validate_call_auth(user_id: str, auth_token: str) -> bool:
    """Validate the mobile client's authenticated GoodScore session.

    TBD: the existing chatbot has no bearer-token validation endpoint in
    the reviewed source (chat trusts the caller-supplied user_id; the web
    UI's passcode check happens in backend/lambda_sts_function.py, which
    is specific to that browser-signing flow). Until GoodScore supplies a
    real session/token validation endpoint for the mobile app, this
    performs the same minimal check the existing chat surface makes
    today (a non-empty user_id and token) so the contract is ready to
    plug a real check in without changing the WebSocket protocol.
    """
    return bool(user_id.strip()) and bool(auth_token.strip())


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run("voice_service.app:app", host=cfg.host, port=cfg.port, reload=False)
