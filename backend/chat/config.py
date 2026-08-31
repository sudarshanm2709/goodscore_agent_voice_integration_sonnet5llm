import os
from pathlib import Path

# Load .env file (looks in backend/ first, then one level up)
# This must happen before any os.environ.get() calls.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if not _env_path.exists():
        _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on system env vars

# --- Bedrock / model -------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "4096"))

# --- Local-development model provider (additive) ----------------------------
# MODEL_PROVIDER defaults to "bedrock" — unset, this reproduces the exact
# prior behaviour (Bedrock is the only provider that ever existed here).
# Set MODEL_PROVIDER=anthropic in a local .env (never committed) to run the
# agent against the direct Anthropic API instead of AWS Bedrock, for local
# testing on a machine with no AWS credentials/account configured. See
# agent.py's _get_active_model().
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "bedrock")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL_ID = os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-5")

# --- Knowledge-retrieval tool (additive, disabled by default) ---------------
# "disabled"    (default) — tool not registered; chat behaviour unchanged.
# "local_dummy" — registers the tool backed by dummy_knowledge_base.py, for
#                 local testing of the knowledge-retrieval path without a
#                 real AgentCore Gateway/S3 Vectors deployment.
# "gateway"     — production path; requires AGENTCORE_GATEWAY_URL (TBD, not
#                 implemented — see knowledge_gateway.py).
KNOWLEDGE_BASE_MODE = os.environ.get("KNOWLEDGE_BASE_MODE", "disabled")

# --- AgentCore Memory --------------------------------------------------------
AGENTCORE_MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "credit_assistant_memory-9XYyucFcfm")
# --- Support ---------------------------------------------------------------
CUSTOMER_SUPPORT_NUMBER = os.environ.get("CUSTOMER_SUPPORT_NUMBER", "+9108065663363")


# --- Server ----------------------------------------------------------------
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

 
