# GoodScore Voice Service

A separate, additive voice layer on top of the existing GoodScore chatbot.
This service converts speech to text, sends a voice-mode request to the
existing chatbot's `/invocations` endpoint, and converts the returned
answer to speech — streamed back to the mobile app over a WebSocket.

**It is not a second AI brain.** The existing Strands agent running
through AgentCore Runtime (in `backend/chat/`) remains the only model call in
the system. This service never calls Bedrock, AgentCore Memory, Aurora,
S3, or S3 Vectors directly — see `clients/chatbot.py`.

## 1. Required configuration

Copy `.env.example` to `.env` and fill in the values marked `REPLACE_ME`
or `TBD`. Nothing here has a fabricated default for a provider URL,
credential, or model identifier — see the comments in `.env.example` for
what's confirmed vs. what genuinely needs to come from the team:

| Variable | Status |
|---|---|
| `OPENROUTER_API_KEY` | **Required.** Get one at openrouter.ai. |
| `OPENROUTER_STT_MODEL` / `OPENROUTER_TTS_MODEL` | Pre-filled with the confirmed live OpenRouter slugs for Nemotron 3.5 ASR and Kokoro 82M (verified against OpenRouter's docs during this integration). Update if OpenRouter renames/retires either model. |
| `CHATBOT_MODE` / `CHATBOT_LOCAL_BASE_URL` | Defaults to `local`, pointing at a locally running `backend/chat/server.py`. |
| `AGENTCORE_INVOKE_URL` | **TBD.** Required only for `CHATBOT_MODE=agentcore` (production). This is the same AgentCore Runtime invoke URL already used in `backend/chat/lambda_sts_function.py` — get the current value from that deployment. |
| `BEDROCK_MODEL_ID` (in `backend/chat/.env`, not here) | **TBD.** The chatbot's own config still defaults to `global.anthropic.claude-sonnet-4-6`. The approved Sonnet 5 Bedrock model identifier was not supplied to this integration — update `backend/chat/.env` once you have it. This service never talks to Bedrock directly, so it has no separate model-id setting. |
| `VOICE_LANGUAGE_SELECTION_MODE` | **TBD** — final product decision between explicit mobile-app language selection and automatic STT language detection. Both are implemented; `.env.example` defaults to `auto_stt`. |

## 2. Starting the existing chatbot (unchanged)

```bash
cd backend/chat
pip install -r requirements.txt
python -m server
```

This starts the existing FastAPI chatbot on `http://localhost:8000` exactly
as it runs today — nothing in this step changed. The voice service's
`CHATBOT_MODE=local` talks to it over plain HTTP at `/invocations`.

## 3. Starting the voice service

```bash
cd backend/voice
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY at minimum
# --app-dir .. because `voice` is a package rooted at backend/ (the
# parent of this directory) — app.py's own imports are package-relative
# (`from .config import ...`), so it must be loaded as `voice.app`, not
# as a bare top-level `app` module. --env-file loads .env directly (this
# service's config.py reads plain os.environ, unlike backend/chat/config.py
# which loads its own .env via python-dotenv).
uvicorn voice.app:app --app-dir .. --env-file .env --reload --port 8090
```

`GET http://localhost:8090/health` should return `{"ok": true, ...}`.

## 4. Opening a voice connection

Connect a WebSocket client to `ws://localhost:8090/v1/voice/stream`, then:

1. Send a JSON **hello** as the first text frame:
   ```json
   {"type": "hello", "user_id": "user-123", "auth_token": "<goodscore session token>", "language": "hi-en"}
   ```
   (`language` is optional — omit it to let the language-selection mode in
   config decide; see the TBD above.)
2. Wait for the server's **ready** frame:
   ```json
   {"type": "ready", "call_id": "call-...", "language": "hi-en"}
   ```
3. Stream raw audio as binary WebSocket frames while the user speaks.
4. When the user stops speaking, send:
   ```json
   {"type": "turn_end", "turn_id": "turn-1"}
   ```
   The server replies with `{"type": "turn_started", "turn_id": "turn-1"}`
   and then streams the answer as binary audio frames.
5. If the user starts speaking again while the bot is still talking
   (barge-in), send:
   ```json
   {"type": "barge_in", "turn_id": "turn-2"}
   ```
   The server stops sending audio for the interrupted turn immediately
   (see `turn_controller.py`'s stale-turn guard) and replies with
   `{"type": "turn_cancelled", "turn_id": "turn-1"}`, then accepts new
   audio for `turn-2`.

## 5. WebSocket and audio protocol

- **Transport:** one WebSocket per call. Binary frames carry audio;
  JSON text frames carry control messages (see `models.py`).
- **Client → server audio:** raw bytes in the format declared in `hello`
  (`audio_input_format`, default from `VOICE_AUDIO_INPUT_FORMAT`). The
  service buffers a whole utterance and sends it to OpenRouter's STT
  endpoint as one request when `turn_end` arrives — OpenRouter's
  confirmed STT contract (`POST /audio/transcriptions`, base64 JSON body)
  is request/response, not a duplex stream, so this is not simulated
  streaming STT.
- **Server → client audio:** binary chunks relayed as they arrive from
  OpenRouter's TTS endpoint (`POST /audio/speech`), sentence-by-sentence
  as the chatbot's answer streams in — see `_extract_ready_sentences` in
  `turn_controller.py`.
- **Voice activity / turn detection:** Phase 1 relies on the mobile
  client signalling `turn_end` (client-side VAD) rather than server-side
  VAD on the raw audio stream — the server-side pieces this depends on
  (buffering, cancellation, filler timing) are all in place and would
  slot in a server-side VAD detector later without protocol changes.

## 6. Running tests

```bash
cd backend/voice
pip install -r requirements-dev.txt
pytest
```

Tests use fake STT/TTS/chatbot adapters (`tests/conftest.py`) and
`httpx.MockTransport` for the real OpenRouter/chatbot HTTP clients — no
network access or real API keys are needed to run the suite.

Chat regression tests for the additive `backend/chat/` changes live in
`backend/chat/tests/` — run with `cd backend/chat && pip install -r requirements-dev.txt && pytest`.

## 7. Known TBD / deployment items

- **`AGENTCORE_INVOKE_URL`** — production AgentCore Runtime invoke URL for `CHATBOT_MODE=agentcore`.
- **Approved Sonnet 5 Bedrock model identifier** — `backend/chat/config.py`'s `BEDROCK_MODEL_ID` default is `global.anthropic.claude-sonnet-4-6`, not Sonnet 5. Update `backend/chat/.env` once the approved identifier is confirmed; no code change needed.
- **Mobile client auth token validation** — `_validate_call_auth()` in `app.py` currently performs the same minimal check the existing chat surface makes (non-empty user id/token). Wire in GoodScore's real session/token validation endpoint once one is designated for the mobile voice client.
- **AgentCore Gateway / knowledge base** — the diagrams show `Strands Agent → knowledge-retrieval tool → AgentCore Gateway → S3 Vectors`, but no Gateway URL, OAuth config, S3 bucket, or S3 Vectors index was supplied. `backend/chat/knowledge_gateway.py` implements the adapter boundary and is registered as a disabled-by-default fifth Strands tool (`AGENTCORE_GATEWAY_URL` unset ⇒ inactive, chat and voice behaviour unchanged; `KNOWLEDGE_BASE_MODE=local_dummy` backs it with `backend/chat/dummy_knowledge_base.py` for local testing). Real Gateway config is required to activate the production path.
- **Language selection mode** — explicit mobile selection vs. automatic STT detection; both implemented, final choice TBD (see `VOICE_LANGUAGE_SELECTION_MODE`).
- **Server-side voice activity detection** — Phase 1 uses client-signalled `turn_end`; a server-side VAD/turn-detector could replace or supplement this without changing the WebSocket protocol.
- **ECS/ALB/WAF/ACM deployment (Terraform/CDK, task definitions, target groups)** — out of scope per `<working_rules>` ("do not deploy or modify shared AWS infrastructure"); this repo only contains the application code and Dockerfile.

## Architecture notes

- **No second LLM, no extra database.** Call state is in-memory only
  (`sessions.py`), consistent with Phase 1 scope. The existing Strands
  agent + Claude Sonnet (via Bedrock) is the only model in the system for
  both chat and voice.
- **Credit-report prefetch is a latency optimisation, not a tool bypass.**
  `prefetch.py` warms the same GoodScore staging API the chatbot's own
  `get_credit_report` tool calls, and drives deterministic filler
  selection while that tool call is in flight. It does not change the
  chatbot's tool-only-answers behaviour — the four existing customer-data
  tools in `backend/chat/tools.py` are untouched.
