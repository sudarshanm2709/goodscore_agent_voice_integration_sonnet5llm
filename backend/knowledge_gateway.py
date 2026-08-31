"""AgentCore Gateway knowledge-retrieval adapter boundary.

Integration gap (see GoodScore_Architecture.drawio vs. the current agent):
the target flow is

    Strands Agent -> knowledge-retrieval tool -> AgentCore Gateway
    -> approved knowledge search -> S3 Vectors + source documents in S3

but the current codebase has no AgentCore Gateway URL, OAuth/IAM auth
config, S3 bucket, or S3 Vectors index — none of these were supplied to
this integration, and per the project's source-priority rules they must
not be invented.

This module is therefore a boundary, not a working integration: it
defines the tool the Strands agent WOULD call, and is registered as a
Strands tool by agent.py ONLY when AGENTCORE_GATEWAY_URL is set (see
_make_knowledge_tools() in agent.py) — which it is not, by default, so
chat and voice behaviour is completely unchanged until real Gateway
configuration is supplied.

This is a new, additive file — it does not modify any of the four
existing customer-data tools in tools.py.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Not set by default anywhere in this project's config — presence of this
# variable is the explicit opt-in that activates the tool (see agent.py).
AGENTCORE_GATEWAY_URL = os.environ.get("AGENTCORE_GATEWAY_URL")
AGENTCORE_GATEWAY_TOKEN = os.environ.get("AGENTCORE_GATEWAY_TOKEN")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")


class KnowledgeGatewayNotConfigured(RuntimeError):
    """Raised when the knowledge tool is invoked without real Gateway config."""


def is_configured() -> bool:
    return bool(AGENTCORE_GATEWAY_URL)


def search_knowledge_base(query: str) -> dict:
    """Query the approved GoodScore knowledge base via AgentCore Gateway.

    TBD / not implemented: AgentCore Gateway is an MCP-compatible
    endpoint — connecting to it for real requires the Gateway URL, its
    auth scheme (IAM SigV4 or OAuth2 client credentials — not confirmed
    by any project source), and the target/tool schema it exposes. None
    of that was supplied to this integration, so this function
    deliberately does not attempt a real network call yet.

    When AGENTCORE_GATEWAY_URL is set, callers reach this function (it is
    registered as a Strands tool — see agent.py). Until the real
    connection is implemented, it returns a structured "not configured"
    result rather than raising an unhandled exception into the agent
    loop, and the existing prompt's "ALWAYS CALL TOOL... I don't have
    the information needed" ground rule already gives the model a safe,
    truthful fallback line for this case.
    """
    if not is_configured():
        raise KnowledgeGatewayNotConfigured(
            "AGENTCORE_GATEWAY_URL is not set — knowledge retrieval is not available."
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
