"""AgentCore Memory tracer.

How it works
------------
1. `patch_session_manager(sm, query_label)` wraps two methods on a *single
   session-manager instance*:

   • retrieve_customer_context  — fires on every user message (MessageAddedEvent
     hook registered by the session manager).  We capture:
       - the user query that is sent to AWS as the semantic-search input
       - each namespace searched, its top_k / relevance_score config
       - how many raw hits came back from AWS
       - how many survived the relevance filter
       - the actual text of each kept memory record

   • list_messages               — fires before each turn to restore session
     history.  We cap at MAX_SESSION_MESSAGES (10) to prevent O(N) latency
     growth as conversations get longer.  We log how many messages loaded.

   • create_message             — fires each time a conversation turn is
     persisted to AgentCore short-term memory.  We capture:
       - the role (USER / ASSISTANT / TOOL …)
       - a short preview of the message content
       - the eventId returned by AWS

2. `install_global_logging()` enables DEBUG-level logging on the two library
   loggers that the AWS SDK uses internally, so low-level boto3 logs also
   appear in log streams (optional, controlled by MEMORY_TRACE_BOTO_DEBUG
   env var, default off).

All output is routed to stdout to comply with containerized cloud deployment.
"""
from __future__ import annotations

import json
import logging
import os
import textwrap
from datetime import datetime, timezone
from typing import Any

from bedrock_agentcore.memory.integrations.strands.config import RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from strands.hooks.events import MessageAddedEvent

# ---------------------------------------------------------------------------
# Console stream logger setup
# ---------------------------------------------------------------------------
import sys

def _get_console_logger() -> logging.Logger:
    """Return (and lazily create) the console logger for memory traces."""
    log = logging.getLogger("memory_tracer")
    if log.handlers:
        return log  # already configured
    log.setLevel(logging.DEBUG)
    log.propagate = False  # don't bubble up to root / uvicorn

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(
        logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    log.addHandler(sh)

    # Write a startup marker immediately
    log.info("═" * 72)
    log.info("MEMORY TRACER INITIALISED (stdout stream)")
    log.info("═" * 72)
    sh.flush()
    return log


_tracer = _get_console_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _preview(obj: Any, max_chars: int = 200) -> str:
    """Return a short human-readable preview of any object."""
    if obj is None:
        return "<None>"
    if isinstance(obj, str):
        text = obj
    else:
        try:
            text = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            text = repr(obj)
    return textwrap.shorten(text, width=max_chars, placeholder=" …")


# ---------------------------------------------------------------------------
# Core patch
# ---------------------------------------------------------------------------

def patch_session_manager(
    sm: AgentCoreMemorySessionManager,
    *,
    user_id: str,
    session_id: str,
) -> None:
    """Wrap retrieve_customer_context and create_message on *sm* instance.

    Called once per request turn in _make_session_manager().
    """

    # ── 1. Wrap retrieve_customer_context ───────────────────────────────────
    _original_retrieve = sm.retrieve_customer_context

    def _traced_retrieve(event: MessageAddedEvent) -> None:
        messages = event.agent.messages
        if not messages or messages[-1].get("role") != "user":
            return _original_retrieve(event)

        content = messages[-1].get("content", [])
        user_query = content[0].get("text", "") if content else ""

        # ── Skip retrieval for tool-result messages (empty query) ──────────
        # The SDK fires MessageAddedEvent for every appended message including
        # tool results.  Running 3 namespace searches on an empty query adds
        # ~2 s of latency per tool call with no memory benefit.  Only retrieve
        # when there is an actual user text query.
        if not user_query.strip():
            _tracer.info(
                "MEMORY RETRIEVAL  user=%s  session=%s  SKIPPED (no user text query — tool result message)",
                user_id, session_id,
            )
            return  # do NOT call _original_retrieve — avoids 3 useless AWS searches

        retrieval_cfg: dict[str, RetrievalConfig] = sm.config.retrieval_config or {}

        _tracer.info(_sep("═"))
        _tracer.info(
            "MEMORY RETRIEVAL  user=%s  session=%s  ts=%s",
            user_id, session_id, _ts(),
        )
        _tracer.info("  Query : %s", _preview(user_query, 300))
        _tracer.info("  Namespaces searched (%d):", len(retrieval_cfg))

        for ns, rcfg in retrieval_cfg.items():
            resolved_ns = ns.format(
                actorId=sm.config.actor_id,
                sessionId=sm.config.session_id,
                memoryStrategyId=rcfg.strategy_id or "",
            )
            strategy_label = rcfg.strategy_id or "<default>"
            _tracer.info(
                "    %-50s  top_k=%-3d  relevance_score=%-4s  strategy=%s",
                resolved_ns,
                rcfg.top_k,
                str(rcfg.relevance_score) if rcfg.relevance_score else "none",
                strategy_label,
            )

        # ── Intercept memory_client.retrieve_memories to capture raw hits ──
        _original_retrieve_memories = sm.memory_client.retrieve_memories

        namespace_results: dict[str, dict] = {}

        def _traced_retrieve_memories(*args: Any, **kwargs: Any) -> list:
            ns_val = kwargs.get("namespace", kwargs.get("namespace_path", "?"))
            if len(args) > 1:
                ns_val = args[1]

            raw = _original_retrieve_memories(*args, **kwargs)
            namespace_results[ns_val] = {
                "raw_hits": len(raw),
                "hits": raw,
            }
            return raw

        sm.memory_client.retrieve_memories = _traced_retrieve_memories

        # ── Run the actual retrieval ────────────────────────────────────────
        _original_retrieve(event)

        # ── Move context from assistant prefill msg to user msg to avoid Bedrock error ──
        if event.agent.messages and event.agent.messages[-1].get("role") == "assistant":
            last_content = event.agent.messages[-1].get("content", [])
            if last_content and isinstance(last_content[0], dict) and last_content[0].get("text", "").startswith("<user_context>"):
                injected_msg = event.agent.messages.pop()
                context_text = injected_msg["content"][0]["text"]
                if event.agent.messages and event.agent.messages[-1].get("role") == "user":
                    user_msg = event.agent.messages[-1]
                    original_text = user_msg["content"][0]["text"]
                    user_msg["content"][0]["text"] = f"{context_text}\n\n{original_text}"
                    _tracer.info("  Moved <user_context> from assistant prefill to user message prefix to prevent Bedrock ValidationException.")

        # ── Restore original method ─────────────────────────────────────────
        sm.memory_client.retrieve_memories = _original_retrieve_memories

        # ── Log per-namespace results ───────────────────────────────────────
        total_injected = 0
        for ns_path, data in namespace_results.items():
            raw_hits = data["raw_hits"]
            hits = data["hits"]

            # Re-apply relevance filter to count kept items (mirrors SDK logic)
            rcfg_for_ns = _find_rcfg(sm.config.retrieval_config, ns_path)
            min_score = rcfg_for_ns.relevance_score if rcfg_for_ns else 0.0
            kept = [
                m for m in hits
                if (not min_score) or m.get("score", 0.0) >= min_score
            ]
            total_injected += len(kept)

            _tracer.info(
                "  %s  →  raw=%d  kept=%d (score≥%s)",
                ns_path,
                raw_hits,
                len(kept),
                str(min_score) if min_score else "none",
            )
            for i, mem in enumerate(kept, 1):
                score = mem.get("score", "?")
                text = mem.get("content", {}).get("text", "").strip()
                strategy = mem.get("memoryStrategyId", mem.get("strategyId", "unknown"))
                _tracer.info(
                    "    [%d] score=%-5s  strategy=%-30s  %s",
                    i,
                    f"{score:.3f}" if isinstance(score, float) else str(score),
                    strategy,
                    _preview(text, 160),
                )

        if total_injected == 0:
            _tracer.info("  → No long-term memory injected (no matches above threshold)")
        else:
            _tracer.info("  → %d item(s) injected as <user_context> into prompt", total_injected)
        _tracer.info(_sep())
        # Flush to disk immediately — don't wait for OS buffering
        for h in _tracer.handlers:
            h.flush()

    sm.retrieve_customer_context = _traced_retrieve

    # ── 2. Wrap list_messages to log and cap short-term memory loading ──────
    # The SDK calls list_messages on every turn to reload session history from
    # AWS. But from turn 2 onwards, agent.messages already holds the full
    # conversation in RAM (the Agent object is cached in _agent_cache).
    # Calling AWS again is redundant and costs ~1-2s per turn.
    #
    # Strategy:
    #   - Turn 1 (cold start): load up to MAX_SESSION_MESSAGES from AWS to
    #     restore any prior messages from a previous server restart.
    #   - Turn 2+ (warm cache): return [] immediately — the in-RAM messages
    #     are already correct and complete. Save the AWS round-trip entirely.
    #
    # Long-term context (summaries) is unaffected — that comes through
    # retrieve_customer_context, which still fires every turn.
    MAX_SESSION_MESSAGES = 10

    _list_messages_call_count = [0]   # mutable counter captured by closure
    _original_list_messages = sm.list_messages

    def _traced_list_messages(
        session_id: str,
        agent_id: str,
        limit: int | None = None,
        offset: int = 0,
        **kwargs: Any,
    ) -> list:
        _list_messages_call_count[0] += 1
        call_num = _list_messages_call_count[0]

        # Turn 1: cold start — fetch from AWS to restore session history
        if call_num == 1:
            effective_limit = min(limit, MAX_SESSION_MESSAGES) if limit else MAX_SESSION_MESSAGES
            raw_messages = _original_list_messages(
                session_id,
                agent_id,
                limit=effective_limit,
                offset=offset,
                **kwargs,
            )
            # Filter out any corrupt/blank messages loaded from AWS short-term memory
            messages = []
            for m in (raw_messages or []):
                if not isinstance(m, dict):
                    continue
                content = m.get("content", [])
                clean_content = []
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict):
                            if "text" in b:
                                if (b.get("text") or "").strip():
                                    clean_content.append(b)
                            else:
                                clean_content.append(b)
                        elif isinstance(b, str) and b.strip():
                            clean_content.append({"text": b.strip()})
                elif isinstance(content, str) and content.strip():
                    clean_content.append({"text": content.strip()})

                if clean_content:
                    m["content"] = clean_content
                    messages.append(m)

            _tracer.info(
                "SHORT-TERM LOAD   user=%s  session=%s  turn=1 (cold)  loaded %d valid messages (raw=%d cap=%d)",
                user_id, session_id, len(messages), len(raw_messages or []), effective_limit,
            )
        else:
            # Turn 2+: agent.messages already has everything — skip AWS call
            messages = []
            _tracer.info(
                "SHORT-TERM LOAD   user=%s  session=%s  turn=%d (warm)  skipped AWS call — using in-RAM messages",
                user_id, session_id, call_num,
            )

        for h in _tracer.handlers:
            h.flush()
        return messages

    sm.list_messages = _traced_list_messages

    # ── 3. Wrap create_message ───────────────────────────────────────────────
    _original_create_message = sm.create_message

    def _traced_create_message(
        session_id_arg: str,
        agent_id: str,
        session_message: Any,
        **kwargs: Any,
    ) -> Any:
        result = _original_create_message(session_id_arg, agent_id, session_message, **kwargs)

        role = "?"
        preview_text = ""
        try:
            msg = session_message.message
            role = msg.get("role", "?") if isinstance(msg, dict) else getattr(msg, "role", "?")
            content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
            if content:
                first = content[0] if isinstance(content, list) else content
                if isinstance(first, dict):
                    preview_text = first.get("text", "") or str(first)
                else:
                    preview_text = str(first)
        except Exception:
            pass

        event_id = result.get("eventId", "<buffered>") if isinstance(result, dict) else str(result)

        _tracer.info(
            "MEMORY STORE      user=%s  session=%s  role=%-10s  eventId=%s  │  %s",
            user_id,
            session_id,
            role.upper(),
            event_id,
            _preview(preview_text, 120),
        )
        for h in _tracer.handlers:
            h.flush()
        return result

    sm.create_message = _traced_create_message


# ---------------------------------------------------------------------------
# Utility: match a resolved namespace path back to a RetrievalConfig
# ---------------------------------------------------------------------------

def _find_rcfg(
    retrieval_config: dict[str, RetrievalConfig] | None,
    resolved_ns: str,
) -> RetrievalConfig | None:
    """Best-effort: match the resolved namespace back to its RetrievalConfig."""
    if not retrieval_config:
        return None
    # Exact match first (unlikely but fast)
    for template, rcfg in retrieval_config.items():
        if resolved_ns == template:
            return rcfg
    # Prefix match — resolved_ns starts with a pattern segment
    for template, rcfg in retrieval_config.items():
        # Strip template placeholders to get a static prefix
        static_prefix = template.split("{")[0]
        if resolved_ns.startswith(static_prefix):
            return rcfg
    return None


# ---------------------------------------------------------------------------
# Optional: route library DEBUG logs to trace_log.txt too
# ---------------------------------------------------------------------------

def install_global_logging() -> None:
    """Attach trace_log.txt handler to bedrock_agentcore and strands loggers.

    Only active when env var MEMORY_TRACE_BOTO_DEBUG=1.
    """
    if os.environ.get("MEMORY_TRACE_BOTO_DEBUG", "0") != "1":
        return
    fh = _tracer.handlers[0] if _tracer.handlers else None
    if not fh:
        return
    for name in ("bedrock_agentcore", "strands"):
        lib_log = logging.getLogger(name)
        lib_log.setLevel(logging.DEBUG)
        lib_log.addHandler(fh)
