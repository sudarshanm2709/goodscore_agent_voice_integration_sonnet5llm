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

# --- AgentCore Memory --------------------------------------------------------
AGENTCORE_MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "credit_assistant_memory-9XYyucFcfm")
# --- Support ---------------------------------------------------------------
CUSTOMER_SUPPORT_NUMBER = os.environ.get("CUSTOMER_SUPPORT_NUMBER", "+9108065663363")


# --- Server ----------------------------------------------------------------
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

 
