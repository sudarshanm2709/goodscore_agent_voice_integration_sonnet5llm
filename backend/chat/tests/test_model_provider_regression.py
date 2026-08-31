"""Regression tests for the additive MODEL_PROVIDER dispatch in agent.py.

Proves the default (MODEL_PROVIDER unset / "bedrock") always resolves to
_get_bedrock_model() — the only provider that existed before this change
— and that "anthropic" is strictly opt-in and never reached otherwise.
"""
import config as cfg
import agent as agent_module


def test_default_provider_is_bedrock():
    assert cfg.MODEL_PROVIDER == "bedrock"


def test_active_model_dispatches_to_bedrock_by_default(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_PROVIDER", "bedrock")
    sentinel = object()
    monkeypatch.setattr(agent_module, "_get_bedrock_model", lambda: sentinel)

    def _fail_anthropic():
        raise AssertionError("anthropic model must not be constructed when provider=bedrock")

    monkeypatch.setattr(agent_module, "_get_anthropic_model", _fail_anthropic)

    assert agent_module._get_active_model() is sentinel


def test_active_model_dispatches_to_anthropic_only_when_selected(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_PROVIDER", "anthropic")
    sentinel = object()
    monkeypatch.setattr(agent_module, "_get_anthropic_model", lambda: sentinel)

    def _fail_bedrock():
        raise AssertionError("bedrock model must not be constructed when provider=anthropic")

    monkeypatch.setattr(agent_module, "_get_bedrock_model", _fail_bedrock)

    assert agent_module._get_active_model() is sentinel


def test_anthropic_model_requires_api_key(monkeypatch):
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(agent_module, "_ANTHROPIC_MODEL", None)
    try:
        agent_module._get_anthropic_model()
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True
