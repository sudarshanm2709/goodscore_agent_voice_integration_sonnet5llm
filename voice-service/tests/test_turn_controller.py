import asyncio

import pytest

from conftest import FakeChatbotClient, FakeSTTAdapter, FakeTTSAdapter
from voice_service.models import TurnState
from voice_service.turn_controller import TurnController


class RecordingStreamer:
    """Test double for OutboundAudioStreamer — same interface, no real socket."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def stop(self) -> None:
        self._stopped = True

    async def send_chunk(self, chunk: bytes) -> bool:
        if self._stopped:
            return False
        self.chunks.append(chunk)
        return True


def _controller(stt=None, tts=None, chatbot=None, filler_config=None):
    return TurnController(
        stt or FakeSTTAdapter(),
        tts or FakeTTSAdapter(),
        chatbot or FakeChatbotClient(),
        filler_config,
    )


async def test_happy_path_transcribes_answers_and_speaks(call_session, turn, filler_config):
    stt = FakeSTTAdapter(text="what is my credit score")
    tts = FakeTTSAdapter()
    chatbot = FakeChatbotClient(events=[
        {"type": "text_delta", "text": "Your score is 742."},
        {"type": "done"},
    ])
    controller = _controller(stt, tts, chatbot, filler_config)
    streamer = RecordingStreamer()

    await controller.run_turn(call_session, turn, b"raw-audio", "webm", streamer)

    assert stt.calls == [b"raw-audio"]
    assert turn.transcript == "what is my credit score"
    assert turn.state == TurnState.DONE
    assert len(streamer.chunks) > 0
    assert tts.synthesized_texts == ["Your score is 742."]
    # chatbot must be called with the voice channel and this call's ids
    assert chatbot.requests[0].call_id == call_session.call_id
    assert chatbot.requests[0].turn_id == turn.turn_id


async def test_stt_failure_speaks_error_message_and_skips_chatbot(call_session, turn, filler_config):
    stt = FakeSTTAdapter(fail=True)
    chatbot = FakeChatbotClient()
    tts = FakeTTSAdapter()
    controller = _controller(stt, tts, chatbot, filler_config)
    streamer = RecordingStreamer()

    await controller.run_turn(call_session, turn, b"raw-audio", "webm", streamer)

    assert chatbot.requests == []  # never reached the chatbot
    assert len(streamer.chunks) > 0  # spoke the terminal error message
    assert any("clearly" in t.lower() for t in tts.synthesized_texts)


async def test_chatbot_failure_speaks_error_message(call_session, turn, filler_config):
    chatbot = FakeChatbotClient(fail=True)
    tts = FakeTTSAdapter()
    controller = _controller(FakeSTTAdapter(), tts, chatbot, filler_config)
    streamer = RecordingStreamer()

    await controller.run_turn(call_session, turn, b"raw-audio", "webm", streamer)

    assert any("trouble" in t.lower() for t in tts.synthesized_texts)


async def test_tts_failure_does_not_crash_the_turn(call_session, turn, filler_config):
    tts = FakeTTSAdapter(fail=True)
    chatbot = FakeChatbotClient(events=[{"type": "text_delta", "text": "Your score is 742."}, {"type": "done"}])
    controller = _controller(FakeSTTAdapter(), tts, chatbot, filler_config)
    streamer = RecordingStreamer()

    # Must complete without raising even though every synthesize() call fails.
    await controller.run_turn(call_session, turn, b"raw-audio", "webm", streamer)
    assert turn.state == TurnState.DONE
    assert streamer.chunks == []


async def test_stale_turn_is_abandoned_without_speaking(call_session, turn, filler_config):
    """Simulates a barge-in that supersedes `turn` with a new current_turn
    before the chatbot has replied — the superseded turn's output must
    never reach the streamer.
    """
    from voice_service.models import Turn, new_turn_id

    chatbot = FakeChatbotClient(
        events=[{"type": "text_delta", "text": "stale answer"}, {"type": "done"}],
        delay_between_events=0.05,
    )
    tts = FakeTTSAdapter()
    controller = _controller(FakeSTTAdapter(), tts, chatbot, filler_config)
    streamer = RecordingStreamer()

    async def run():
        await controller.run_turn(call_session, turn, b"raw-audio", "webm", streamer)

    task = asyncio.ensure_future(run())
    await asyncio.sleep(0.01)  # let STT + first loop iteration start
    # Barge-in: a new turn supersedes the one currently running.
    call_session.current_turn = Turn(turn_id=new_turn_id(), call_id=call_session.call_id)
    turn.mark_cancelled()

    await task

    assert streamer.chunks == []
    assert tts.synthesized_texts == []


async def test_filler_speaks_when_chatbot_is_slow(call_session, turn, filler_config):
    chatbot = FakeChatbotClient(
        events=[{"type": "text_delta", "text": "Your score is 742."}, {"type": "done"}],
        delay_between_events=0.1,  # longer than filler_config.min_wait_before_filler_seconds (0.05)
    )
    tts = FakeTTSAdapter()
    controller = _controller(FakeSTTAdapter(), tts, chatbot, filler_config)
    streamer = RecordingStreamer()

    await controller.run_turn(call_session, turn, b"raw-audio", "webm", streamer)

    # First spoken text should be the deterministic filler, before the real answer.
    assert tts.synthesized_texts[0] == "One moment, please."
    assert "Your score is 742." in tts.synthesized_texts
    assert turn.turn_id in call_session.fillers_spoken_for_turn


async def test_filler_uses_credit_report_operation_when_tool_call_seen_first(call_session, turn, filler_config):
    """The tool_call for get_credit_report arrives before the filler
    threshold, so the filler that eventually fires should be the
    credit-report-specific line, not the generic fallback — and it must
    still only be spoken once for the turn.
    """
    chatbot = FakeChatbotClient(
        events=[
            {"type": "tool_call", "name": "get_credit_report", "input": {}},
            {"type": "text_delta", "text": "Your score is 742."},
            {"type": "done"},
        ],
        delays=[0.01, 0.15, 0.0],  # tool_call before threshold, answer well after it
    )
    tts = FakeTTSAdapter()
    controller = _controller(FakeSTTAdapter(), tts, chatbot, filler_config)
    streamer = RecordingStreamer()

    await controller.run_turn(call_session, turn, b"raw-audio", "webm", streamer)

    filler_count = sum(1 for t in tts.synthesized_texts if "credit report" in t.lower())
    assert filler_count == 1
    assert "Your score is 742." in tts.synthesized_texts
