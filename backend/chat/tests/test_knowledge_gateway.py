import pytest

import agent as agent_module
import config as cfg
import dummy_knowledge_base
import knowledge_gateway


def test_default_knowledge_base_mode_is_disabled():
    assert cfg.KNOWLEDGE_BASE_MODE == "disabled"


def test_disabled_mode_registers_no_tool(monkeypatch):
    monkeypatch.setattr(knowledge_gateway, "KNOWLEDGE_BASE_MODE", "disabled")
    assert knowledge_gateway.is_active() is False
    assert agent_module._make_knowledge_tools() == []


def test_disabled_mode_raises_on_direct_call(monkeypatch):
    monkeypatch.setattr(knowledge_gateway, "KNOWLEDGE_BASE_MODE", "disabled")
    with pytest.raises(knowledge_gateway.KnowledgeGatewayNotConfigured):
        knowledge_gateway.search_knowledge_base("what is a credit score")


def test_local_dummy_mode_registers_one_tool(monkeypatch):
    monkeypatch.setattr(knowledge_gateway, "KNOWLEDGE_BASE_MODE", "local_dummy")
    assert knowledge_gateway.is_active() is True
    tools = agent_module._make_knowledge_tools()
    assert len(tools) == 1


def test_local_dummy_mode_returns_relevant_results(monkeypatch):
    monkeypatch.setattr(knowledge_gateway, "KNOWLEDGE_BASE_MODE", "local_dummy")
    result = knowledge_gateway.search_knowledge_base("what is a good credit score")
    assert result["status"] is True
    assert len(result["results"]) > 0
    assert any("credit score" in r["title"].lower() for r in result["results"])


def test_local_dummy_mode_handles_no_match_gracefully(monkeypatch):
    monkeypatch.setattr(knowledge_gateway, "KNOWLEDGE_BASE_MODE", "local_dummy")
    result = knowledge_gateway.search_knowledge_base("xyzzyzzyx unrelated gibberish")
    assert result["status"] is True
    assert result["results"] == []


def test_gateway_mode_without_url_raises_not_configured(monkeypatch):
    monkeypatch.setattr(knowledge_gateway, "KNOWLEDGE_BASE_MODE", "gateway")
    monkeypatch.setattr(knowledge_gateway, "AGENTCORE_GATEWAY_URL", None)
    with pytest.raises(knowledge_gateway.KnowledgeGatewayNotConfigured):
        knowledge_gateway.search_knowledge_base("what is a credit score")


def test_gateway_mode_with_url_returns_structured_not_implemented(monkeypatch):
    monkeypatch.setattr(knowledge_gateway, "KNOWLEDGE_BASE_MODE", "gateway")
    monkeypatch.setattr(knowledge_gateway, "AGENTCORE_GATEWAY_URL", "https://example.com/gateway")
    result = knowledge_gateway.search_knowledge_base("what is a credit score")
    assert result["status"] is False
    assert result["error_code"] == "KNOWLEDGE_GATEWAY_NOT_CONFIGURED"


def test_dummy_knowledge_base_search_ranks_keyword_overlap():
    results = dummy_knowledge_base.search("missed EMI payment effect on score")
    assert results
    assert any("emi" in r["title"].lower() for r in results)


def test_dummy_knowledge_base_search_empty_query_returns_nothing():
    assert dummy_knowledge_base.search("") == []


def test_dummy_knowledge_base_has_no_pii_or_account_data():
    """Sanity check that the dummy content stays educational/FAQ-style and
    never looks like real customer account data (score numbers tied to a
    name, account numbers, etc.) — this file is meant to be safe to read.
    """
    import dummy_knowledge_base as kb

    for doc in kb._DOCUMENTS:  # noqa: SLF001 - test-only introspection
        assert "@" not in doc.content
        assert "account number" not in doc.content.lower()
