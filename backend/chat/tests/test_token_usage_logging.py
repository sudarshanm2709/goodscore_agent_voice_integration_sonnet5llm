"""Regression tests for the additive token_usage field on the done event.

Uses a minimal fake Agent (not a real Strands Agent — that needs a real
model call) that drives agent.callback_handler the way the REAL, installed
Strands + Anthropic-provider combination actually does — verified live
against a real API call, not assumed: the final callback invocation for a
turn carries kwargs["result"] (an AgentResult with .metrics.accumulated_usage),
never kwargs["complete"] (that kwarg is checked elsewhere in agent.py for
the pre-existing RESPONSE_DONE signal, but was observed to never actually
fire for this combination — a pre-existing characteristic, not something
these tests changed).
"""
import threading

import agent as agent_module


class _FakeMetrics:
    def __init__(self, usage: dict):
        self.accumulated_usage = usage


class _FakeAgentResult:
    def __init__(self, usage: dict):
        self.metrics = _FakeMetrics(usage)


class _FakeAgent:
    """Drives agent.callback_handler the way the real Strands+Anthropic
    combination does for one simple turn: one text delta, then the final
    result callback.
    """

    def __init__(self, usage: dict):
        self.callback_handler = None
        self.messages: list = []
        self.hooks = _FakeHooks()
        self._usage = usage

    def __call__(self, message: str):
        self.callback_handler(data="Hello there.")
        self.callback_handler(result=_FakeAgentResult(self._usage))


class _FakeHooks:
    def add_callback(self, *a, **kw):
        pass


def _install_fake_agent(monkeypatch, usage: dict):
    fake_agent = _FakeAgent(usage)
    ctx = agent_module.TurnContext()
    lock = threading.Lock()
    monkeypatch.setattr(agent_module, "_get_or_create_agent", lambda *a, **kw: (fake_agent, ctx, lock))
    return fake_agent, ctx


async def test_voice_channel_done_event_includes_token_usage(monkeypatch):
    _install_fake_agent(monkeypatch, {"inputTokens": 120, "outputTokens": 45, "totalTokens": 165})

    events = [e async for e in agent_module.run_turn_async("user-1", "sess-1", "hi", channel="voice")]

    done = next(e for e in events if e["type"] == "done")
    assert done["token_usage"] == {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}


async def test_chat_channel_done_event_never_includes_token_usage(monkeypatch):
    """Default channel ("chat") must reproduce the exact prior done event
    shape — no token_usage key at all, even though the same usage data
    was captured internally.
    """
    _install_fake_agent(monkeypatch, {"inputTokens": 120, "outputTokens": 45, "totalTokens": 165})

    events = [e async for e in agent_module.run_turn_async("user-1", "sess-1", "hi")]

    done = next(e for e in events if e["type"] == "done")
    assert "token_usage" not in done
    assert done == {"type": "done", "text": ""}


async def test_usage_capture_failure_does_not_break_the_turn(monkeypatch):
    """If reading result.metrics.accumulated_usage ever raises for some
    reason, the turn must still complete normally — usage is a
    nice-to-have, not a load-bearing part of the response.
    """
    class _BrokenResult:
        @property
        def metrics(self):
            raise RuntimeError("metrics unavailable")

    class _BrokenAgent(_FakeAgent):
        def __call__(self, message: str):
            self.callback_handler(data="Hello there.")
            self.callback_handler(result=_BrokenResult())

    fake_agent = _BrokenAgent({"inputTokens": 1, "outputTokens": 1, "totalTokens": 2})
    ctx = agent_module.TurnContext()
    lock = threading.Lock()
    monkeypatch.setattr(agent_module, "_get_or_create_agent", lambda *a, **kw: (fake_agent, ctx, lock))

    events = [e async for e in agent_module.run_turn_async("user-1", "sess-1", "hi", channel="voice")]

    done = next(e for e in events if e["type"] == "done")
    assert "token_usage" not in done
    assert any(e["type"] == "text_delta" for e in events)
