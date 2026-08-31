# backend

Two independently deployable services:

- **[`chat/`](chat/)** — the existing GoodScore chatbot (FastAPI, Strands Agent, AgentCore Runtime/Memory, Bedrock). Unchanged behaviour; see `chat/` for its own run/test instructions.
- **[`voice/`](voice/)** — the voice layer (FastAPI + WebSocket, OpenRouter STT/TTS). Talks to `chat/` only over its `/invocations` HTTP endpoint — never imports `chat/`'s Python modules directly. See [`voice/README.md`](voice/README.md).

There is currently no code shared between them (they communicate over HTTP by design — see `voice/clients/chatbot.py`). If that changes, put the shared module directly under `backend/` (not inside `chat/` or `voice/`) so both can import it without depending on each other's package.
