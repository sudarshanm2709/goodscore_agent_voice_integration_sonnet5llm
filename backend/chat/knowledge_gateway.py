"""Knowledge-retrieval adapter boundary — AgentCore Gateway (production,
not yet implemented) or a local dummy knowledge base (local testing).

Integration gap (see GoodScore_Architecture.drawio vs. the current agent):
the target production flow is

    Strands Agent -> knowledge-retrieval tool -> AgentCore Gateway
    -> approved knowledge search -> S3 Vectors + source documents in S3

but the current codebase has no AgentCore Gateway URL, OAuth/IAM auth
config, S3 bucket, or S3 Vectors index — none of these were supplied to
this integration, and per the project's source-priority rules they must
not be invented. That path (mode="gateway") remains a boundary, not a
working integration, below.

For local development (no AWS), mode="local_dummy" backs the same tool
with dummy_knowledge_base.py's small keyword-search over invented
GoodScore FAQ content — see that module's docstring. Both modes are
opt-in via KNOWLEDGE_BASE_MODE (config.py); the default ("disabled")
reproduces the exact prior behaviour (tool not registered at all).

This is an additive file — it does not modify any of the four existing
customer-data tools in tools.py.
"""
from __future__ import annotations

import logging
import os

import dummy_knowledge_base

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_MODE = os.environ.get("KNOWLEDGE_BASE_MODE", "disabled")

# Only consulted when KNOWLEDGE_BASE_MODE=gateway. Not set by default
# anywhere in this project's config.
AGENTCORE_GATEWAY_URL = os.environ.get("AGENTCORE_GATEWAY_URL")
AGENTCORE_GATEWAY_TOKEN = os.environ.get("AGENTCORE_GATEWAY_TOKEN")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")


class KnowledgeGatewayNotConfigured(RuntimeError):
    """Raised when the knowledge tool is invoked without real Gateway config."""


def is_active() -> bool:
    """Whether agent.py should register the knowledge-retrieval tool at all."""
    return KNOWLEDGE_BASE_MODE in ("local_dummy", "gateway")


def search_knowledge_base(query: str) -> dict:
    """Query the GoodScore knowledge base — dummy locally, Gateway in production.

    mode="local_dummy": searches dummy_knowledge_base.py's invented FAQ
    content with simple keyword overlap. This is a stand-in for local
    testing only — it is not connected to any real GoodScore content.

    mode="gateway": TBD / not implemented. AgentCore Gateway is an
    MCP-compatible endpoint — connecting to it for real requires the
    Gateway URL, its auth scheme (IAM SigV4 or OAuth2 client credentials
    — not confirmed by any project source), and the target/tool schema
    it exposes. None of that was supplied to this integration, so this
    branch deliberately does not attempt a real network call yet and
    returns a structured "not configured" result — the existing prompt's
    "ALWAYS CALL TOOL... I don't have the information needed" ground rule
    already gives the model a safe, truthful fallback line for this case.
    """
    if KNOWLEDGE_BASE_MODE == "local_dummy":
        results = dummy_knowledge_base.search(query)
        if not results:
            return {"status": True, "results": [], "message": "No matching articles found."}
        return {"status": True, "results": results}

    if KNOWLEDGE_BASE_MODE == "gateway":
        if not AGENTCORE_GATEWAY_URL:
            raise KnowledgeGatewayNotConfigured(
                "KNOWLEDGE_BASE_MODE=gateway requires AGENTCORE_GATEWAY_URL."
            )
        logger.warning(
            "[knowledge_gateway] AGENTCORE_GATEWAY_URL is set but the real MCP "
            "connection to AgentCore Gateway is not implemented in this "
            "integration (no confirmed auth scheme / tool schema). Returning "
            "a not-configured result instead of a fabricated one."
        )
        return {
            "status": False,
            "error_code": "KNOWLEDGE_GATEWAY_NOT_CONFIGURED",
            "message": "I don't have the information needed to answer that right now.",
        }

    raise KnowledgeGatewayNotConfigured(
        f"KNOWLEDGE_BASE_MODE={KNOWLEDGE_BASE_MODE!r} does not support knowledge retrieval."
    )
