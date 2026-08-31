"""Regression tests for local testing without AWS (MODEL_PROVIDER=anthropic).

Proves:
- The default (Bedrock) path still builds a real AgentCoreMemorySessionManager
  — unchanged from before this feature existed.
- MODEL_PROVIDER=anthropic skips AgentCore Memory entirely (returns None,
  which Strands' Agent already accepts) rather than making an AWS call.
- A failure while constructing the agent (missing key, expired AWS token,
  etc.) surfaces as a normal `error` SSE event instead of crashing the
  stream with an unhandled exception.
"""
import config as cfg
import agent as agent_module


def test_bedrock_mode_still_builds_agentcore_session_manager(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_PROVIDER", "bedrock")
    captured = {}

    def fake_agentcore(session_id, user_id):
        captured["called"] = (session_id, user_id)
        return "sentinel-agentcore-session-manager"

    monkeypatch.setattr(agent_module, "_make_agentcore_session_manager", fake_agentcore)

    result = agent_module._make_session_manager("sess-1", "user-1")

    assert result == "sentinel-agentcore-session-manager"
    assert captured["called"] == ("sess-1", "user-1")


def test_anthropic_mode_skips_agentcore_memory_entirely(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_PROVIDER", "anthropic")

    def _fail(*a, **kw):
        raise AssertionError("AgentCore Memory must not be touched in local/anthropic mode")

    monkeypatch.setattr(agent_module, "_make_agentcore_session_manager", _fail)

    result = agent_module._make_session_manager("sess-1", "user-1")

    assert result is None


async def test_run_turn_async_surfaces_agent_construction_failure_as_error_event(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("MODEL_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")

    monkeypatch.setattr(agent_module, "_get_or_create_agent", _boom)

    events = [e async for e in agent_module.run_turn_async("user-1", "sess-1", "hello")]

    assert events[0]["type"] == "error"
    assert "ANTHROPIC_API_KEY" in events[0]["message"]
    assert events[1] == {"type": "done", "text": ""}
