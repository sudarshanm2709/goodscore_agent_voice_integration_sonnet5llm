import pytest

from voice.config import ConfigurationError, load_config

_REQUIRED_OPENROUTER_ENV = {
    "OPENROUTER_API_KEY": "sk-test",
    "OPENROUTER_STT_MODEL": "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
    "OPENROUTER_TTS_MODEL": "hexgrad/kokoro-82m",
}


def _set_env(monkeypatch, **overrides):
    for key, value in _REQUIRED_OPENROUTER_ENV.items():
        monkeypatch.setenv(key, value)
    for key, value in overrides.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_load_config_defaults_to_local_chatbot_mode(monkeypatch):
    _set_env(monkeypatch, CHATBOT_MODE=None)
    config = load_config()
    assert config.chatbot.mode == "local"
    assert config.chatbot.local_base_url == "http://localhost:8000"


def test_load_config_defaults_tts_voice_to_kokoro_default(monkeypatch):
    """Kokoro on OpenRouter rejects requests with no voice at all
    (confirmed against the live API) — this must never resolve to None.
    """
    _set_env(monkeypatch, OPENROUTER_TTS_VOICE=None)
    config = load_config()
    assert config.openrouter.tts_voice == "af_heart"


def test_load_config_missing_openrouter_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_STT_MODEL", "x")
    monkeypatch.setenv("OPENROUTER_TTS_MODEL", "y")
    with pytest.raises(ConfigurationError):
        load_config()


def test_load_config_agentcore_mode_requires_invoke_url(monkeypatch):
    _set_env(monkeypatch, CHATBOT_MODE="agentcore", AGENTCORE_INVOKE_URL=None)
    with pytest.raises(ConfigurationError):
        load_config()


def test_load_config_agentcore_mode_succeeds_with_invoke_url(monkeypatch):
    _set_env(
        monkeypatch,
        CHATBOT_MODE="agentcore",
        AGENTCORE_INVOKE_URL="https://example.amazonaws.com/runtimes/fake/invocations",
    )
    config = load_config()
    assert config.chatbot.mode == "agentcore"
    assert config.chatbot.agentcore_invoke_url is not None


def test_load_config_invalid_chatbot_mode_raises(monkeypatch):
    _set_env(monkeypatch, CHATBOT_MODE="carrier-pigeon")
    with pytest.raises(ConfigurationError):
        load_config()


def test_load_config_default_languages_include_hinglish(monkeypatch):
    _set_env(monkeypatch, VOICE_SUPPORTED_LANGUAGES=None)
    config = load_config()
    assert "hi-en" in config.supported_languages
    assert "en" in config.supported_languages
    assert "hi" in config.supported_languages
