"""Environment configuration for the GoodScore voice service.

Every value that identifies a provider, credential, model, or endpoint is
read from the environment (see .env.example) — nothing here fabricates a
default for anything the project sources did not confirm. Values with a
genuine, safe default (timeouts, retry counts, sample rates) keep one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigurationError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _require(name: str) -> str:
    value = _get(name)
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    value = _get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _get_float(name: str, default: float) -> float:
    value = _get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable {name} must be a number, got {value!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    value = _get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class OpenRouterConfig:
    """Configuration for the OpenRouter-hosted STT and TTS providers."""

    base_url: str
    api_key: str
    stt_model: str
    tts_model: str
    tts_voice: str | None
    tts_response_format: str
    request_timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class ChatbotConfig:
    """How to reach the existing GoodScore chatbot's /invocations endpoint.

    mode="local"     — plain HTTP to a locally running server.py (dev/test).
    mode="agentcore" — SigV4-signed HTTPS to the AgentCore Runtime invoke URL,
                        signed with the ECS task's own IAM role (no browser
                        passcode broker needed — this is a trusted backend
                        service, unlike the mobile/web chat client).
    """

    mode: str
    local_base_url: str | None
    agentcore_invoke_url: str | None
    agentcore_region: str
    request_timeout_seconds: float
    connect_timeout_seconds: float


@dataclass(frozen=True)
class GoodScoreApiConfig:
    """Direct access to the existing GoodScore backend staging API.

    Reuses the same host tools.py already calls for the four chat data
    tools — kept as an independent constant here (not imported from
    backend/) so the voice service has no source dependency on the chat
    codebase and can be built/deployed as a separate container.
    """

    base_url: str
    request_timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class AudioConfig:
    input_format: str
    output_format: str
    sample_rate_hz: int


@dataclass(frozen=True)
class FillerConfig:
    """Deterministic filler timing thresholds — see fillers.py."""

    min_wait_before_filler_seconds: float
    repeat_cooldown_seconds: float


@dataclass(frozen=True)
class VoiceServiceConfig:
    host: str
    port: int
    call_ttl_seconds: int
    session_sweep_interval_seconds: int
    max_concurrent_calls: int
    supported_languages: tuple[str, ...]
    default_language: str
    language_selection_mode: str  # "explicit" | "auto_stt" — final product choice is TBD
    openrouter: OpenRouterConfig
    chatbot: ChatbotConfig
    goodscore: GoodScoreApiConfig
    audio: AudioConfig
    fillers: FillerConfig


def load_config() -> VoiceServiceConfig:
    """Build the immutable service configuration from environment variables.

    Raises ConfigurationError early (at startup) if a value required for
    the selected chatbot mode is missing, rather than failing deep inside
    a call.
    """
    chatbot_mode = (_get("CHATBOT_MODE", "local") or "local").strip().lower()
    if chatbot_mode not in ("local", "agentcore"):
        raise ConfigurationError('CHATBOT_MODE must be "local" or "agentcore"')

    local_base_url = _get("CHATBOT_LOCAL_BASE_URL", "http://localhost:8000")
    agentcore_invoke_url = _get("AGENTCORE_INVOKE_URL")
    if chatbot_mode == "agentcore" and not agentcore_invoke_url:
        raise ConfigurationError(
            "CHATBOT_MODE=agentcore requires AGENTCORE_INVOKE_URL "
            "(the existing AgentCore Runtime /invocations endpoint — see README)."
        )

    openrouter = OpenRouterConfig(
        base_url=_get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=_require("OPENROUTER_API_KEY"),
        stt_model=_require("OPENROUTER_STT_MODEL"),
        tts_model=_require("OPENROUTER_TTS_MODEL"),
        # Kokoro 82M on OpenRouter rejects requests with no voice at all
        # ("An explicit voice is required for this TTS provider" — confirmed
        # against the live API) unlike some other OpenRouter TTS providers,
        # so this always resolves to a real voice ID rather than None.
        # af_heart is Kokoro's documented default American-English voice.
        tts_voice=_get("OPENROUTER_TTS_VOICE", "af_heart"),
        tts_response_format=_get("OPENROUTER_TTS_RESPONSE_FORMAT", "pcm"),
        request_timeout_seconds=_get_float("OPENROUTER_TIMEOUT_SECONDS", 20.0),
        max_retries=_get_int("OPENROUTER_MAX_RETRIES", 2),
    )

    chatbot = ChatbotConfig(
        mode=chatbot_mode,
        local_base_url=local_base_url,
        agentcore_invoke_url=agentcore_invoke_url,
        agentcore_region=_get("AWS_REGION", "ap-south-1") or "ap-south-1",
        request_timeout_seconds=_get_float("CHATBOT_TIMEOUT_SECONDS", 45.0),
        connect_timeout_seconds=_get_float("CHATBOT_CONNECT_TIMEOUT_SECONDS", 5.0),
    )

    goodscore = GoodScoreApiConfig(
        base_url=_get("GOODSCORE_API_BASE_URL", "https://subscription.stage.goodscore.io"),
        request_timeout_seconds=_get_float("GOODSCORE_API_TIMEOUT_SECONDS", 15.0),
        max_retries=_get_int("GOODSCORE_API_MAX_RETRIES", 3),
    )

    audio = AudioConfig(
        input_format=_get("VOICE_AUDIO_INPUT_FORMAT", "webm"),
        output_format=_get("VOICE_AUDIO_OUTPUT_FORMAT", "pcm"),
        sample_rate_hz=_get_int("VOICE_AUDIO_SAMPLE_RATE_HZ", 16000),
    )

    fillers = FillerConfig(
        min_wait_before_filler_seconds=_get_float("FILLER_MIN_WAIT_SECONDS", 1.2),
        repeat_cooldown_seconds=_get_float("FILLER_REPEAT_COOLDOWN_SECONDS", 8.0),
    )

    languages = tuple(
        lang.strip()
        for lang in (_get("VOICE_SUPPORTED_LANGUAGES", "en,hi,hi-en") or "").split(",")
        if lang.strip()
    ) or ("en", "hi", "hi-en")

    return VoiceServiceConfig(
        host=_get("HOST", "0.0.0.0") or "0.0.0.0",
        port=_get_int("PORT", 8090),
        call_ttl_seconds=_get_int("CALL_TTL_SECONDS", 60 * 30),
        session_sweep_interval_seconds=_get_int("SESSION_SWEEP_INTERVAL_SECONDS", 30),
        max_concurrent_calls=_get_int("MAX_CONCURRENT_CALLS", 200),
        supported_languages=languages,
        default_language=_get("VOICE_DEFAULT_LANGUAGE", "hi-en") or "hi-en",
        language_selection_mode=_get("VOICE_LANGUAGE_SELECTION_MODE", "auto_stt") or "auto_stt",
        openrouter=openrouter,
        chatbot=chatbot,
        goodscore=goodscore,
        audio=audio,
        fillers=fillers,
    )
