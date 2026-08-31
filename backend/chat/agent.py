from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

from botocore.exceptions import ClientError
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent
from strands.models import BedrockModel

if TYPE_CHECKING:
    # Only needed for the type hints below — see the deferred runtime
    # import in _get_anthropic_model(). Safe to reference unimported here
    # because `from __future__ import annotations` (top of file) makes
    # every annotation a lazily-evaluated string, never executed at
    # runtime for a plain function signature.
    from strands.models.anthropic import AnthropicModel

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

import config as cfg
import knowledge_gateway
from memory_tracer import install_global_logging, patch_session_manager
from prompt import build_system_prompt
from tools import make_user_tools

# Activate optional boto3-level debug logging to trace_log.txt
install_global_logging()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Latency logger — shared with server.py, writes to console standard output
# ---------------------------------------------------------------------------
import sys

_latency_logger = logging.getLogger("latency")
if not _latency_logger.handlers:
    _latency_logger.setLevel(logging.INFO)
    _latency_logger.propagate = False
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _latency_logger.addHandler(_sh)

# ---------------------------------------------------------------------------
# Module-level BedrockModel singleton — constructed once, reused everywhere.
# ---------------------------------------------------------------------------
_BEDROCK_MODEL: BedrockModel | None = None


def _get_bedrock_model() -> BedrockModel:
    global _BEDROCK_MODEL
    if _BEDROCK_MODEL is None:
        _BEDROCK_MODEL = BedrockModel(
            model_id=cfg.BEDROCK_MODEL_ID,
            region_name=cfg.AWS_REGION,
            max_tokens=cfg.MODEL_MAX_TOKENS,
        )
    return _BEDROCK_MODEL


# ---------------------------------------------------------------------------
# Local-development model provider (additive — see config.MODEL_PROVIDER).
#
# _ANTHROPIC_MODEL is a separate singleton from _BEDROCK_MODEL so switching
# MODEL_PROVIDER never touches the Bedrock path at all. The import of
# strands.models.anthropic is deliberately deferred to inside this function
# (not at module top) — that package needs the `anthropic` extra
# (`pip install 'strands-agents[anthropic]'`, see requirements-local.txt),
# which is NOT in the base requirements.txt. Importing it eagerly at module
# load would break `import agent` — and therefore the whole chatbot,
# including production Bedrock users — on any machine that only installed
# requirements.txt. Deferring it means the import is only ever attempted if
# MODEL_PROVIDER=anthropic is actually selected.
# ---------------------------------------------------------------------------
_ANTHROPIC_MODEL: AnthropicModel | None = None


def _get_anthropic_model() -> AnthropicModel:
    global _ANTHROPIC_MODEL
    if _ANTHROPIC_MODEL is None:
        if not cfg.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "MODEL_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set."
            )
        try:
            from strands.models.anthropic import AnthropicModel
        except ImportError as exc:
            raise RuntimeError(
                "MODEL_PROVIDER=anthropic requires the 'anthropic' extra: "
                "pip install 'strands-agents[anthropic]' "
                "(see backend/requirements-local.txt)."
            ) from exc
        _ANTHROPIC_MODEL = AnthropicModel(
            client_args={"api_key": cfg.ANTHROPIC_API_KEY},
            model_id=cfg.ANTHROPIC_MODEL_ID,
            max_tokens=cfg.MODEL_MAX_TOKENS,
        )
    return _ANTHROPIC_MODEL


def _get_active_model() -> BedrockModel | AnthropicModel:
    """Return the model for the configured provider.

    Defaults to Bedrock (cfg.MODEL_PROVIDER unset ⇒ "bedrock") — every
    existing chat/voice call site that doesn't set MODEL_PROVIDER gets
    the exact prior behaviour. "anthropic" is a local-testing-only
    alternative (see the block above); it is never selected unless a
    local .env explicitly opts in.
    """
    if cfg.MODEL_PROVIDER == "anthropic":
        return _get_anthropic_model()
    return _get_bedrock_model()


# ---------------------------------------------------------------------------
# Presentation tool name set — invisible in chat, never emitted as tool_call
# ---------------------------------------------------------------------------
_PRESENTATION_TOOLS = {"send_chip_response", "get_deeplinks"}


_BASE = "https://goodscore.rupicard.com/home/"

_DEEPLINKS: dict[str, str] = {
    # Score / report
    "score_refresh":       _BASE + "https%3A%2F%2Fscore.rupicard.com%2Fscore-refresh%3Fsource%3Dai_chat",
    "daily_score_refresh": _BASE + "https%3A%2F%2Fscore.rupicard.com%2F%3Fsource%3Dai_chat",
    "score_predictor":     _BASE + "https%3A%2F%2Fscore.rupicard.com%2Fscore-simulator%3Fsource%3Dai_chat",

    # Bills / payments
    "bill_payment":        _BASE + "https%3A%2F%2Fscore.rupicard.com%2Floan-payment%3Fsource%3Dai_chat",
    "emi_conversion":      _BASE + "https%3A%2F%2Fscore.rupicard.com%2Foverdue-emi-conversion%3Fsource%3Dai_chat",
    "set_reminder":        _BASE + "https%3A%2F%2Fscore.rupicard.com%2Floan-reminders%3Fsource%3Dai_chat",
    "set_autopay":         _BASE + "https%3A%2F%2Fscore.rupicard.com%2Floan-payment%3Fsource%3Dai_chat",
    "emi_calculator":      _BASE + "https%3A%2F%2Fscore.rupicard.com%2Femi-calculator%3Fsource%3Dai_chat",
    "utility_payments":    _BASE + "https%3A%2F%2Fscore.rupicard.com%2Fbill-payments%3Fsource%3Dai_chat",

    # Loans
    "loan_apply":          _BASE + "https%3A%2F%2Fscore.rupicard.com%2Fconsent-form%3Fsource%3Dai_chat",

    # Features
    "spend_analyzer":      _BASE + "https%3A%2F%2Fscore.rupicard.com%2F%3FplaySpendAnalyzerPnVideo%3Dtrue%26source%3Dai_chat",
    "learn":               _BASE + "https%3A%2F%2Fscore.rupicard.com%2Fvideos-for-you%3Fsource%3Dai_chat",
    "refer_earn":          _BASE + "https%3A%2F%2Fscore.rupicard.com%2Freferral-status%3Fsource%3Dai_chat",

    # Subscription / account
    "subscription":        _BASE + "https%3A%2F%2Fscore.rupicard.com%2Fscore-improvement-plan%3Fsource%3Dai_chat",
    "account_settings":    _BASE + "https%3A%2F%2Fscore.rupicard.com%2Faccount-info%3Fsource%3Dai_chat",

    # Support
    "expert_call":         f"tel:{cfg.CUSTOMER_SUPPORT_NUMBER}",
    "bbps_escalate":       f"tel:{cfg.CUSTOMER_SUPPORT_NUMBER}",
}


# ---------------------------------------------------------------------------
# TurnContext — a small mutable holder swapped into the cached Agent each turn.
#
# The presentation tools close over this object's lists rather than directly
# over per-turn locals.  Before each turn we clear these lists; the tools
# then write into them during the turn.  The callback_handler also captures
# this context so it can forward events to the correct SSE queue.
# ---------------------------------------------------------------------------
class TurnContext:
    """Mutable per-turn state injected into the cached Agent before each call."""

    __slots__ = ("chips", "deeplinks", "queue", "final_text_parts", "emitted_tool_ids", "tool_id_to_name", "chips_emitted")

    def __init__(self) -> None:
        self.chips: list[str] = []
        self.deeplinks: dict = {}
        self.queue: asyncio.Queue | None = None
        self.final_text_parts: list[str] = []
        self.emitted_tool_ids: set[str] = set()
        self.tool_id_to_name: dict[str, str] = {}
        self.chips_emitted: bool = False

    def reset(self, queue: asyncio.Queue) -> None:
        self.chips = []
        self.deeplinks = {}
        self.queue = queue
        self.final_text_parts = []
        self.emitted_tool_ids = set()
        self.tool_id_to_name = {}
        self.chips_emitted = False


# ---------------------------------------------------------------------------
# Per-session Agent cache
#
# Key: (user_id, session_id)
# Value: (Agent, TurnContext, threading.Lock)
#
# The Lock ensures only one request runs on a given Agent at a time
# (Strands raises ConcurrencyException otherwise).
# The cache is invalidated when the session resets via evict_session().
# ---------------------------------------------------------------------------
_agent_cache: dict[tuple[str, str], tuple[Agent, TurnContext, threading.Lock]] = {}
_cache_lock = threading.Lock()


def evict_session(user_id: str, session_id: str) -> None:
    """Remove a cached Agent entry when the session is reset."""
    with _cache_lock:
        removed = _agent_cache.pop((user_id, session_id), None)
    if removed:
        logger.info("[session] evicted | user=%s session=%s", user_id, session_id)
    else:
        logger.debug("[session] evict no-op (not in cache) | user=%s session=%s", user_id, session_id)


def _make_presentation_tools(ctx: TurnContext) -> list:
    """Build presentation tools that write into a TurnContext.

    Because ctx is the *same object* across all turns (only its contents are
    reset), the closures remain valid on the cached Agent indefinitely.
    """

    @tool
    def send_chip_response(chips: list[str], deeplinks: dict | None = None) -> dict:
        """MANDATORY: Call this at the end of EVERY response, no exceptions.

        Rules:
        - Always call this. Never skip it.
        - 2-3 chips per response.
        - Chips must answer or follow naturally from the closing question.
        - Never include 'Talk to expert', 'Book a call', or vague chips like 'Tell me more'.
        - Chips that navigate to a screen get a deeplink (use get_deeplinks first).
        - Chips that continue the conversation do NOT get a deeplink.

        Args:
            chips: 2-3 short chip label strings.
            deeplinks: Optional dict mapping chip label → deeplink URL (from get_deeplinks only).
        """
        clean = [str(c) for c in (chips or []) if str(c).strip()][:6]
        resolved_dl = deeplinks or {}

        # Guard: if chips already emitted this turn, ignore subsequent calls.
        # Prevents duplicate chip rows when the model calls send_chip_response
        # more than once in a single turn.
        if ctx.chips_emitted:
            return {"chips": clean, "deeplinks": resolved_dl}

        ctx.chips.extend(clean)
        ctx.deeplinks.update(resolved_dl)

        # Emit chips immediately rather than waiting for turn completion.
        # This means chips appear in the UI as soon as send_chip_response
        # is called — not after AgentCore persists the tool result to memory.
        if ctx.queue is not None:
            pres_tool_ids = {
                name: tuid
                for tuid, name in ctx.tool_id_to_name.items()
                if name in _PRESENTATION_TOOLS
            }
            ctx.queue.put_nowait({
                "type": "chips",
                "chips": list(ctx.chips),
                "deeplinks": dict(ctx.deeplinks),
                "pres_tool_ids": pres_tool_ids,
            })
            ctx.chips_emitted = True

        return {"chips": clean, "deeplinks": resolved_dl}

    @tool
    def get_deeplinks(keys: list[str]) -> dict:
        """Resolve named in-app destinations to deeplink URLs.

        Valid keys (use only these):
          score_refresh, daily_score_refresh, score_predictor
          bill_payment, emi_conversion, set_reminder, set_autopay, emi_calculator, utility_payments
          loan_apply
          spend_analyzer, learn, refer_earn
          subscription, account_settings
          expert_call, bbps_escalate

        Dynamic URLs (bill paymentLink, video URL) — pass directly to send_chip_response, skip get_deeplinks.

        Args:
            keys: List of key names to resolve.
        """
        resolved = {k: _DEEPLINKS[k] for k in (keys or []) if k in _DEEPLINKS}
        unknown = [k for k in (keys or []) if k not in _DEEPLINKS]
        out: dict = {"deeplinks": resolved}
        # For support keys, also expose the display number so the agent can
        # mention it in the response text without hallucinating.
        support_keys = {"expert_call", "bbps_escalate"}
        if any(k in support_keys for k in (keys or [])):
            out["support_number"] = cfg.CUSTOMER_SUPPORT_NUMBER
        if unknown:
            out["unknown_keys"] = unknown
            out["available"] = list(_DEEPLINKS.keys())
        ctx.deeplinks.update(resolved)
        return out

    return [send_chip_response, get_deeplinks]


def _make_knowledge_tools() -> list:
    """Return the knowledge-retrieval tool, or an empty list if it isn't
    configured (see knowledge_gateway.py).

    Additive and inactive by default: AGENTCORE_GATEWAY_URL is unset in
    every environment this project ships config for, so this returns []
    and the Agent's tool list is exactly the four existing customer-data
    tools plus the two presentation tools — unchanged from before this
    integration.
    """
    if not knowledge_gateway.is_active():
        return []

    @tool
    def search_knowledge_base(query: str) -> dict:
        """Search the approved GoodScore knowledge base for general
        information, FAQs, credit-score education, and policy/support
        content that isn't in the user's own account data.

        Args:
            query: the user's question, in their own words.
        """
        return knowledge_gateway.search_knowledge_base(query)

    return [search_knowledge_base]


def _make_session_manager(session_id: str, user_id: str) -> AgentCoreMemorySessionManager:
    config = AgentCoreMemoryConfig(
        memory_id=cfg.AGENTCORE_MEMORY_ID,
        session_id=session_id,
        actor_id=user_id,
        batch_size=4,
        retrieval_config={
            "/summaries/{actorId}/{sessionId}/": RetrievalConfig(
                top_k=2,
                relevance_score=0.4,
            ),
            "/preferences/{actorId}/": RetrievalConfig(
                top_k=3,
                relevance_score=0.3,
            ),
        },
    )
    sm = AgentCoreMemorySessionManager(config, region_name=cfg.AWS_REGION)
    patch_session_manager(sm, user_id=user_id, session_id=session_id)
    return sm


def _build_agent(user_id: str, session_id: str, ctx: TurnContext, channel: str = "chat") -> Agent:
    """Construct a fresh Agent for a new session.

    This runs exactly ONCE per (user_id, session_id) lifetime.
    On subsequent turns the cached Agent is reused — no reconstruction,
    no list_messages() AWS call, no tool re-registration.

    `channel` ("chat" default, or "voice") only affects the system prompt
    baked in at construction time (see prompt.build_system_prompt) — it
    does not change the tool list or any other chat behaviour. Voice
    calls use their own session_id (see server.py's /invocations voice
    branch), so this never rebuilds or mutates an existing chat session's
    cached Agent.
    """
    user_tools = make_user_tools(user_id)
    presentation_tools = _make_presentation_tools(ctx)
    knowledge_tools = _make_knowledge_tools()  # empty list unless AGENTCORE_GATEWAY_URL is configured
    session_manager = _make_session_manager(session_id, user_id)

    agent = Agent(
        model=_get_active_model(),
        system_prompt=build_system_prompt(user_id, channel=channel),
        tools=user_tools + presentation_tools + knowledge_tools,
        callback_handler=None,   # installed dynamically each turn (see _install_turn_wiring)
        session_manager=session_manager,
    )
    logger.info(
        "[agent] built | user=%s session=%s channel=%s | tools=%d",
        user_id, session_id, channel, len(user_tools) + len(presentation_tools) + len(knowledge_tools),
    )

    # Hook: capture tool results into ctx.queue (AfterToolCallEvent is not
    # a callback event so it must be registered via hooks).
    def _after_tool(event: AfterToolCallEvent) -> None:
        name = event.tool_use.get("name", "")
        tuid = event.tool_use.get("toolUseId", name)
        if name in _PRESENTATION_TOOLS:
            return
        result = event.result
        output = result.get("content", [{}])
        if isinstance(output, list) and output:
            output = output[0].get("json", output[0])
        if ctx.queue is not None:
            ctx.queue.put_nowait({
                "type": "tool_result",
                "tool_use_id": tuid,
                "name": name,
                "output": output,
            })

    agent.hooks.add_callback(AfterToolCallEvent, _after_tool)
    return agent


def _get_or_create_agent(
    user_id: str,
    session_id: str,
    channel: str = "chat",
) -> tuple[Agent, TurnContext, threading.Lock]:
    """Return the cached (Agent, TurnContext, Lock) for this session.

    Creates and caches a new Agent if one doesn't exist yet.
    Thread-safe: uses _cache_lock for the lookup/insert only.

    `channel` is only consulted on a cache MISS (new Agent construction —
    see _build_agent). An existing cached Agent (chat or voice) is always
    reused as-is on a cache HIT, so this parameter can never change an
    already-running session's prompt.
    """
    key = (user_id, session_id)
    with _cache_lock:
        if key in _agent_cache:
            logger.debug("[agent] cache HIT  | user=%s session=%s", user_id, session_id)
            return _agent_cache[key]

    logger.info("[agent] cache MISS | user=%s session=%s channel=%s — building new agent", user_id, session_id, channel)

    # Build outside the global lock — Agent.__init__ calls list_messages()
    # (an AWS round-trip) and we don't want to hold _cache_lock during that.
    ctx = TurnContext()
    agent = _build_agent(user_id, session_id, ctx, channel=channel)
    lock = threading.Lock()

    with _cache_lock:
        # Double-check: another thread may have created it while we built ours.
        if key not in _agent_cache:
            _agent_cache[key] = (agent, ctx, lock)
        return _agent_cache[key]


def _install_turn_wiring(
    agent: Agent,
    ctx: TurnContext,
    queue: asyncio.Queue,
) -> None:
    """Reset per-turn state and install the callback_handler for this turn.

    Called at the start of every turn on the cached Agent.
    No object construction — just list/dict clears and a function assignment.
    """
    ctx.reset(queue)

    RESPONSE_DONE = _RESPONSE_DONE_SENTINEL  # module-level object, see below

    def _callback(**kwargs: object) -> None:
        if "data" in kwargs:
            text = str(kwargs["data"])
            # Filter out empty strings and placeholder artifacts that Strands
            # emits after send_chip_response completes its agentic loop turn.
            stripped = text.strip()
            if not stripped:
                return
            # Guard against literal placeholder text the model occasionally
            # emits (e.g. "[blank text]", "[blank]") — drop silently.
            lower = stripped.lower()
            if lower in ("[blank text]", "[blank]", "[blank_text]"):
                return
            ctx.final_text_parts.append(text)
            ctx.queue.put_nowait({"type": "text_delta", "text": text})
            return

        if "current_tool_use" in kwargs:
            tu = kwargs["current_tool_use"]
            if not tu:
                return
            name = tu.get("name", "")
            tuid = tu.get("toolUseId", name)
            if tuid not in ctx.emitted_tool_ids:
                ctx.emitted_tool_ids.add(tuid)
                ctx.tool_id_to_name[tuid] = name
                ctx.queue.put_nowait({
                    "type": "tool_call",
                    "tool_use_id": tuid,
                    "name": name,
                    "input": tu.get("input", {}),
                })
            return

        if kwargs.get("complete"):
            ctx.queue.put_nowait(RESPONSE_DONE)
            return

    agent.callback_handler = _callback


# Sentinel — signals that agent output is fully streamed (before memory writes)
_RESPONSE_DONE_SENTINEL = object()


async def run_turn_async(
    user_id: str,
    session_id: str,
    user_message: str,
    channel: str = "chat",
) -> AsyncGenerator[dict, None]:
    """Async generator yielding SSE event dicts for one user turn.

    SSE event types:
      text_delta  — streaming text token
      tool_call   — data tool invocation
      tool_result — data tool result
      chips       — final chips + deeplinks
      done        — turn complete
      error       — unhandled exception

    Agent caching:
      Turn 1  — Agent is constructed (includes list_messages() AWS call).
      Turn 2+ — Cached Agent is reused. No reconstruction, no list_messages().
                 agent.messages already holds the full conversation in RAM.
                 Long-term memory retrieval still fires via MessageAddedEvent hook.

    `channel` ("chat" default, or "voice" — see server.py's /invocations
    handler) only affects which system prompt a *newly constructed*
    Agent gets; every existing chat call site that doesn't pass it keeps
    today's exact behaviour.
    """
    if not user_message or not user_message.strip():
        logger.warning("[turn] Empty message received for user=%s session=%s", user_id, session_id)
        yield {"type": "error", "message": "Prompt message cannot be empty."}
        yield {"type": "done", "text": ""}
        return

    agent, ctx, agent_lock = _get_or_create_agent(user_id, session_id, channel=channel)

    queue: asyncio.Queue[dict | object] = asyncio.Queue()
    _install_turn_wiring(agent, ctx, queue)

    loop = asyncio.get_event_loop()
    RESPONSE_DONE = _RESPONSE_DONE_SENTINEL

    msg_preview = user_message[:80].replace("\n", " ")
    logger.info(
        "[turn] START | user=%s session=%s | msg=%r",
        user_id, session_id, msg_preview,
    )
    t_turn_start = time.perf_counter()

    async def _run() -> None:
        try:
            # agent_lock ensures only one concurrent call per cached Agent.
            # Strands raises ConcurrencyException on re-entrant calls.
            await loop.run_in_executor(None, lambda: _invoke_with_lock(agent, agent_lock, user_message))
        except ClientError as e:
            # Bedrock throttle/service error — convert to a friendly message
            # instead of exposing raw AWS error codes to the user.
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ThrottlingException":
                friendly = "I'm experiencing high demand right now. Please try again in a moment. 🙏"
            elif code in ("ServiceUnavailableException", "InternalServerError"):
                friendly = "The AI  is temporarily unavailable. Please try again in a few seconds."
            else:
                code_msg = e.response.get("Error", {}).get("Message", str(e))
                friendly = f"AWS Error ({code}): {code_msg}"
            logger.error("[bedrock] fallback message triggered | code=%s | error=%s", code, e)
            queue.put_nowait({"type": "error", "message": friendly})
        except Exception as e:  # noqa: BLE001
            # EventLoopException is a Strands SDK internal race condition with
            # Bedrock streaming (KeyError: 'output'). It is non-fatal — whatever
            # text was already streamed is still in the queue. Log and suppress;
            # the finally block sends None which closes the turn cleanly.
            if "EventLoop" in type(e).__name__:
                logger.warning("[strands] %s suppressed (SDK internal): %s", type(e).__name__, e)
            else:
                queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            queue.put_nowait(None)

    task = asyncio.ensure_future(_run())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            if event is RESPONSE_DONE:
                # Stream is complete. Memory consolidation runs in the background.
                asyncio.ensure_future(_drain_task(task))
                break
            yield event
    except Exception:
        asyncio.ensure_future(_drain_task(task))
        raise

    # Emit chips + done after stream closes (only if not already emitted early
    # by send_chip_response during the turn)
    if not ctx.chips_emitted:
        pres_tool_ids = {
            name: tuid
            for tuid, name in ctx.tool_id_to_name.items()
            if name in _PRESENTATION_TOOLS
        }
        if ctx.chips or ctx.deeplinks:
            yield {
                "type": "chips",
                "chips": ctx.chips,
                "deeplinks": ctx.deeplinks,
                "pres_tool_ids": pres_tool_ids,
            }

    yield {"type": "done", "text": ""}
    logger.info(
        "[turn] DONE  | user=%s session=%s | elapsed=%.3fs",
        user_id, session_id, time.perf_counter() - t_turn_start,
    )


def _sanitize_agent_messages(agent: Agent) -> None:
    """Ensure no message in agent.messages has blank text content blocks."""
    if not hasattr(agent, "messages") or not agent.messages:
        return
    cleaned = []
    for msg in agent.messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        clean_content = []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    if "text" in b:
                        t = (b.get("text") or "").strip()
                        if t:
                            clean_content.append({"text": t})
                    else:
                        clean_content.append(b)
                elif isinstance(b, str) and b.strip():
                    clean_content.append({"text": b.strip()})
        elif isinstance(content, str) and content.strip():
            clean_content.append({"text": content.strip()})

        if clean_content:
            msg["content"] = clean_content
            cleaned.append(msg)
    agent.messages = cleaned


def _invoke_with_lock(agent: Agent, lock: threading.Lock, message: str) -> None:
    """Call agent(message) under the per-session lock with Bedrock retry.

    Retry policy:
    - ThrottlingException → up to 3 retries, exponential backoff (1s → 2s → 4s) + jitter.
    - ServiceUnavailableException / 5xx InternalServerError → 1 retry only.
    - ValidationException / other 4xx → fail immediately, retrying won't help.

    Runs on a thread-pool thread (via run_in_executor).
    The lock prevents two concurrent requests from the same user
    from calling the same Agent simultaneously.
    """
    _BEDROCK_MAX_RETRIES    = 3
    _BEDROCK_BASE_DELAY     = 1.0   # seconds — doubles each attempt
    _BEDROCK_JITTER_MAX     = 0.2   # seconds — random jitter per attempt
    _BEDROCK_5XX_MAX_RETRY  = 1     # only one retry for transient 5xx

    with lock:
        _sanitize_agent_messages(agent)
        t_invoke = time.perf_counter()
        _latency_logger.info("agent invocation started (inside lock)")

        last_exc: Exception | None = None

        for attempt in range(1, _BEDROCK_MAX_RETRIES + 1):
            try:
                agent(message)
                _latency_logger.info(
                    "agent invocation finished | %.3fs inside lock | attempt=%d",
                    time.perf_counter() - t_invoke, attempt,
                )
                return  # success

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)

                if code == "ThrottlingException":
                    # Bedrock quota hit — retry with exponential backoff + jitter
                    last_exc = e
                    wait = _BEDROCK_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, _BEDROCK_JITTER_MAX)
                    logger.warning(
                        "[bedrock] ThrottlingException attempt=%d/%d → retry in %.2fs",
                        attempt, _BEDROCK_MAX_RETRIES, wait,
                    )
                    if attempt < _BEDROCK_MAX_RETRIES:
                        time.sleep(wait)
                        continue
                    raise  # exhausted all retries

                elif code in ("ServiceUnavailableException", "InternalServerError") or status >= 500:
                    # Transient 5xx — retry once only
                    last_exc = e
                    if attempt <= _BEDROCK_5XX_MAX_RETRY:
                        wait = _BEDROCK_BASE_DELAY + random.uniform(0, _BEDROCK_JITTER_MAX)
                        logger.warning(
                            "[bedrock] %s (5xx) attempt=%d → retry in %.2fs",
                            code, attempt, wait,
                        )
                        time.sleep(wait)
                        continue
                    raise  # already retried once

                else:
                    # 4xx ValidationException, ModelNotReadyException, etc.
                    # These won't resolve with retries — fail immediately.
                    logger.error(
                        "[bedrock] %s (non-retryable) attempt=%d — failing immediately",
                        code, attempt,
                    )
                    raise

            except Exception:
                # Non-ClientError exceptions (network, SDK bugs) — don't retry
                raise


async def _drain_task(task: asyncio.Task) -> None:
    """Await the agent executor task in the background.

    Memory consolidation runs inside agent() after output is complete.
    Fire-and-forget so it never delays the SSE stream.
    """
    try:
        await task
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Post-stream memory consolidation error: %s: %s",
            type(e).__name__, e,
        )
