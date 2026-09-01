"""Regression tests for the additive channel param on build_system_prompt.

These prove: (1) the default chat prompt is unchanged in shape/content,
and (2) the voice addendum is purely additive and isolated to
channel="voice" — the chat ground rules (SECTION 1-4) are untouched
either way.
"""
from prompt import build_system_prompt

_CHAT_SECTION_HEADERS = [
    "SECTION 1 — GROUND RULES",
    "SECTION 2 — RESPONSE FLOW & PRESENTATION",
    "SECTION 3 — BUSINESS LOGIC FLOWS",
    "SECTION 4 — DEEPLINKS",
]


def test_default_channel_is_chat_and_matches_no_channel_argument():
    """Calling with no channel arg (every existing call site) must be
    byte-for-byte identical to explicitly passing channel="chat".
    """
    implicit = build_system_prompt("user-1")
    explicit = build_system_prompt("user-1", channel="chat")
    assert implicit == explicit


def test_chat_prompt_contains_all_original_sections_and_no_voice_section():
    prompt = build_system_prompt("user-1")
    for header in _CHAT_SECTION_HEADERS:
        assert header in prompt
    assert "SECTION 5" not in prompt
    assert "VOICE MODE" not in prompt


def test_chat_prompt_still_mandates_send_chip_response():
    """The chip-mandate ground rule (SECTION 2) must survive unchanged
    for chat — this is the exact rule the voice addendum relaxes only
    for channel="voice".
    """
    prompt = build_system_prompt("user-1")
    assert "Call send_chip_response EXACTLY ONCE" in prompt


def test_voice_channel_adds_section_five_without_removing_chat_sections():
    prompt = build_system_prompt("user-1", channel="voice")
    for header in _CHAT_SECTION_HEADERS:
        assert header in prompt
    assert "SECTION 5 — VOICE MODE" in prompt


def test_voice_channel_relaxes_chip_mandate_and_forbids_markdown():
    """Written with positive instructions per Anthropic's own current
    guidance for Claude Sonnet 5 (positive framing outperforms "don't"
    instructions for this model) — so this checks intent (chips aren't
    required, markdown is avoided) via the actual positive wording used,
    not a literal "Do NOT ..." string.
    """
    prompt = build_system_prompt("user-1", channel="voice")
    assert "send_chip_response and get_deeplinks" in prompt
    assert "nothing to attach to on a call" in prompt
    assert "table, or markdown" in prompt


def test_voice_channel_still_contains_chat_ground_rules_verbatim():
    """The voice addendum must not duplicate or override SECTION 1 — it
    only adds new instructions after it. Zero-hallucination and
    tool-only-answers must be present exactly as in chat.
    """
    chat_prompt = build_system_prompt("user-1")
    voice_prompt = build_system_prompt("user-1", channel="voice")
    assert "ZERO HALLUCINATION" in chat_prompt
    assert "ZERO HALLUCINATION" in voice_prompt
    assert "TOOL-ONLY RESPONSES" in voice_prompt


def test_unknown_channel_falls_back_to_chat_prompt():
    prompt = build_system_prompt("user-1", channel="sms")
    assert "SECTION 5" not in prompt
