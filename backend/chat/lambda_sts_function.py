# lambda_sts_function.py — Header-Based SigV4 Generator using STS Temporary Credentials (ASIA...)
# Converts permanent AKIA keys to short-lived ASIA credentials so long-term keys are never exposed in browser headers.

import base64
import json
import os
import time
import urllib.parse
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AGENTCORE_URL = os.environ.get(
    "AGENTCORE_URL",
    "https://bedrock-agentcore.ap-south-1.amazonaws.com/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aap-south-1%3A435472313876%3Aruntime%2Fai_assistant_backend-mYW2NvGHmw/invocations?qualifier=DEFAULT"
)

PARSED_URL = urllib.parse.urlparse(AGENTCORE_URL)
REGION = os.environ.get("AWS_REGION", "ap-south-1")
SERVICE = "bedrock-agentcore"
APP_PASSCODE = os.environ.get("APP_PASSCODE", "!@#!23QWE")

# Cache temporary credentials in memory for fast Lambda reuse until close to expiration
_TEMP_CREDS_CACHE = {
    "credentials": None,
    "expires_at": 0
}

def get_sts_credentials():
    """
    Loads AKIA credentials from any environment variables or boto3 session,
    and exchanges them via AWS STS for short-lived temporary ASIA credentials (with Security Token).
    Caches credentials for ~14 minutes to minimize STS API calls.
    """
    now = time.time()
    if _TEMP_CREDS_CACHE["credentials"] and now < _TEMP_CREDS_CACHE["expires_at"]:
        return _TEMP_CREDS_CACHE["credentials"]

    ak = (
        os.environ.get("CUSTOM_AWS_ACCESS_KEY_ID") or 
        os.environ.get("MY_AWS_ACCESS_KEY_ID") or 
        os.environ.get("AWS_ACCESS_KEY_ID") or ""
    ).strip()
    
    sk = (
        os.environ.get("CUSTOM_AWS_SECRET_ACCESS_KEY") or 
        os.environ.get("MY_AWS_SECRET_ACCESS_KEY") or 
        os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
    ).strip()

    st = (
        os.environ.get("CUSTOM_AWS_SESSION_TOKEN") or 
        os.environ.get("AWS_SESSION_TOKEN") or ""
    ).strip()

    if not (ak and sk):
        session = boto3.Session()
        creds = session.get_credentials()
        if creds:
            frozen = creds.get_frozen_credentials()
            ak = frozen.access_key
            sk = frozen.secret_key
            st = frozen.token or ""

    if ak and sk and ak.startswith("AKIA"):
        print(f"[STS] Static AKIA Key detected ({ak[:5]}...). Requesting STS Temporary Token...")
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            region_name=REGION
        )
        
        # Issue 15-minute temporary session token (works for standard IAM users without extra IAM permissions)
        res = sts_client.get_session_token(
            DurationSeconds=900
        )
        c = res["Credentials"]
        temp_creds = Credentials(c["AccessKeyId"], c["SecretAccessKey"], c["SessionToken"])
        
        # Cache for 14 minutes (840 seconds)
        _TEMP_CREDS_CACHE["credentials"] = temp_creds
        _TEMP_CREDS_CACHE["expires_at"] = now + 840
        print(f"[STS] STS Temporary Credentials issued ({c['AccessKeyId'][:5]}...). Expires in 15m.")
        return temp_creds

    if ak and sk:
        return Credentials(ak, sk, st if st else None)

    raise RuntimeError("Unable to load or exchange AWS IAM credentials for STS signing.")


def lambda_handler(event, context=None):
    """
    Official SigV4 Generator using STS Temporary Credentials.
    Validates passcode, fetches short-lived ASIA credentials, signs the prompt payload,
    and returns signed headers containing ASIA key + x-amz-security-token.
    """
    http_method = event.get("requestContext", {}).get("http", {}).get("method", "POST").upper()

    # Fast CORS Preflight
    if http_method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": ""
        }

    try:
        body_raw = event.get("body", "") or ""
        if event.get("isBase64Encoded", False) and body_raw:
            try:
                body_raw = base64.b64decode(body_raw).decode("utf-8", errors="replace")
            except Exception:
                pass

        event_headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}

        # 1. Passcode & Payload Extraction
        auth_header = event_headers.get("authorization", "")
        passcode_input = ""
        prompt = ""
        user_id = "default_user"
        action = ""

        try:
            body_json = json.loads(body_raw) if body_raw.strip().startswith("{") else {}
            passcode_input = str(body_json.get("passcode", "")).strip()
            prompt = str(body_json.get("prompt", "")).strip()
            user_id = str(body_json.get("user_id") or event_headers.get("x-amzn-bedrock-agentcore-runtime-user-id", "default_user")).strip()
            session_id = str(body_json.get("session_id") or event_headers.get("x-amzn-bedrock-agentcore-runtime-session-id", "")).strip()
            action = str(body_json.get("action", "")).strip()  # e.g. "reset"
        except Exception:
            pass

        expected_passcode = APP_PASSCODE.strip()
        passcode_valid = (
            (expected_passcode and expected_passcode in auth_header) or
            (passcode_input.lower() == expected_passcode.lower()) or
            (expected_passcode and expected_passcode in body_raw)
        )

        if APP_PASSCODE and not passcode_valid:
            return {
                "statusCode": 401,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({"authenticated": False, "error": "Unauthorized: Invalid Passcode"})
            }

        if not session_id or len(session_id) < 33:
            session_id = f"sess-{user_id}-{int(time.time())}".ljust(35, "0")

        # 2. Get STS Temporary Credentials (ASIA...)
        creds = get_sts_credentials()

        # 3. Construct payload including user_id and session_id for AgentCore Memory
        payload_dict = {
            "prompt": prompt,
            "user_id": user_id,
            "session_id": session_id,
        }
        if action:
            payload_dict["action"] = action  # pass reset signal through to backend

        payload_json_str = json.dumps(payload_dict, separators=(',', ':'))
        payload_bytes = payload_json_str.encode("utf-8")

        # 4. Calculate Header-Based SigV4 Signature using ASIA Temporary Credentials
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Host": PARSED_URL.netloc,
            "x-amzn-bedrock-agentcore-runtime-user-id": user_id,
            "x-amzn-bedrock-agentcore-runtime-session-id": session_id,
        }
        if creds.token:
            req_headers["x-amz-security-token"] = creds.token

        req = AWSRequest(method="POST", url=AGENTCORE_URL, data=payload_bytes, headers=req_headers)
        signer = SigV4Auth(creds, SERVICE, REGION)
        signer.add_auth(req)

        signed_headers = dict(req.headers)

        # 5. Return signed headers (which now carry ASIA... key and Security Token)
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "authenticated": True,
                "target_url": AGENTCORE_URL,
                "payload": payload_dict,
                "raw_body": payload_json_str,
                "headers": signed_headers
            })
        }

    except Exception as e:
        print(f"[ERROR] STS Signature generator error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"authenticated": False, "error": str(e)})
        }
