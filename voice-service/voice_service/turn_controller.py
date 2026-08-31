"""Turn state machine: STT -> chatbot -> TTS, with barge-in cancellation.

One TurnController instance is shared by the app (it is stateless — all
per-turn state lives on the Turn/CallSession objects passed in). A turn
runs as its own asyncio.Task so app.py can cancel it instantly on
barge-in; this module also checks the turn's own `cancelled` flag between
pipeline stages so a stage already past its last await point still stops
before doing further (wasted, possibly stale) work.

Stale-audio guard: every stage re-checks `turn.cancelled` and compares
`turn.turn_id` against the session's current turn before sending anything
to the client — satisfying "cancelled turns cannot send stale audio" even
if a cancellation races a stage that was already mid-flight.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

from .adapters.stt import SpeechToTextAdapter, TranscriptionError
from .adapters.tts import SynthesisError, TextToSpeechAdapter
from .audio import OutboundAudioStreamer
from .clients.chatbot import ChatbotClient, ChatbotClientError, ChatbotTurnRequest
from .fillers import FillerController, FillerOperation
from .models import CallSession, Turn, TurnState
from .observability import StageTimer, log_error, log_event, log_metric

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।\n])\s+")

_TOOL_TO_FILLER_OP = {
    "get_credit_report": FillerOperation.CREDIT_REPORT,
    "get_prefetched_bills": FillerOperation.BILLS,
}

# Deterministic, non-LLM fallback lines spoken when a pipeline stage fails
# outright (provider error/timeout) — see <provider_integration> "handle
# realistic provider failures". Kept tiny and channel-appropriate; these
# are not "filler" (which describes an in-progress wait) but terminal
# error messages for the turn.
_ERROR_MESSAGES = {
    "stt": {
        "en": "Sorry, I couldn't hear that clearly. Could you please repeat?",
        "hi-en": "Sorry, mujhe woh clearly sunayi nahi diya. Kya aap dobara bol sakte hain?",
    },
    "chatbot": {
        "en": "Sorry, I'm having trouble reaching your account right now. Please try again in a moment.",
        "hi-en": "Sorry, abhi aapke account tak pahunchne mein dikkat aa rahi hai. Thodi der mein phir try karein.",
    },
    "tts": {
        "en": "Sorry, something went wrong while responding. Please try again.",
        "hi-en": "Sorry, jawab dete waqt kuch gadbad ho gayi. Please dobara try karein.",
    },
}


def _error_message(stage: str, language: str) -> str:
    table = _ERROR_MESSAGES[stage]
    return table.get(language, table["en"])


class TurnController:
    def __init__(
        self,
        stt: SpeechToTextAdapter,
        tts: TextToSpeechAdapter,
        chatbot: ChatbotClient,
        filler_config,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._chatbot = chatbot
        self._filler_config = filler_config

    async def run_turn(
        self,
        session: CallSession,
        turn: Turn,
        audio_bytes: bytes,
        audio_format: str,
        streamer: OutboundAudioStreamer,
    ) -> None:
        """Run one full turn. Safe to cancel via asyncio.Task.cancel() at
        any point — cleanup happens in the `finally` blocks below.
        """
        t_turn_start = time.monotonic()
        language_hint = None if session.language == "auto" else session.language

        # Step 1: transcribe the buffered utterance.
        turn.state = TurnState.TRANSCRIBING
        try:
            with StageTimer("stt_latency", call_id=session.call_id, turn_id=turn.turn_id):
                result = await self._stt.transcribe(audio_bytes, audio_format, language_hint)
        except TranscriptionError as exc:
            log_error("turn_stt_failed", exc, call_id=session.call_id, turn_id=turn.turn_id)
            await self._speak_terminal_message(
                session, turn, streamer, _error_message("stt", session.language)
            )
            return

        if self._is_stale(session, turn):
            return

        turn.transcript = result.text
        turn.detected_language = result.detected_language
        effective_language = result.detected_language or session.language
        log_event(
            "turn_transcribed",
            call_id=session.call_id,
            turn_id=turn.turn_id,
            transcript_chars=len(result.text),
        )

        # Step 2: send the transcript to the existing chatbot and stream
        # its answer, speaking a deterministic filler if the first token
        # takes longer than the configured threshold.
        turn.state = TurnState.THINKING
        filler = FillerController(
            config=self._filler_config,
            operation=FillerOperation.GENERIC,
            language=effective_language,
        )

        request = ChatbotTurnRequest(
            user_id=session.user_id,
            session_id=session.chat_session_id,
            message=result.text,
            call_id=session.call_id,
            turn_id=turn.turn_id,
            language=effective_language,
            credit_context_ref=(
                "prefetched" if session.credit_prefetch and session.credit_prefetch.ok else None
            ),
        )

        answer_buffer = ""
        spoken_buffer = ""
        first_text_at: Optional[float] = None
        events = self._chatbot.invoke_turn(request)
        # `pending_next` holds the in-flight `events.__anext__()` call so the
        # filler-timeout race (below) can peek at whether it has completed
        # without ever cancelling it — cancelling an async generator's
        # __anext__() mid-flight can leave it unusable for the next call,
        # which would break real SSE streaming (clients/chatbot.py's
        # _iter_sse_events). asyncio.wait(..., timeout=...) never cancels
        # the futures it's waiting on, unlike asyncio.wait_for.
        pending_next: asyncio.Task | None = None
        try:
            while True:
                if self._is_stale(session, turn):
                    break

                if pending_next is None:
                    pending_next = asyncio.ensure_future(events.__anext__())

                if first_text_at is None and not filler.already_spoken:
                    # While no real answer text has arrived yet, race the
                    # pending event against the filler threshold so a slow
                    # backend call (e.g. get_credit_report) gets a filler
                    # spoken proactively — not only when an event happens
                    # to arrive after the threshold has already passed.
                    remaining = filler.config.min_wait_before_filler_seconds - (
                        time.monotonic() - filler.turn_started_at
                    )
                    done, _ = await asyncio.wait({pending_next}, timeout=max(remaining, 0.0))
                    if not done:
                        if not self._is_stale(session, turn) and filler.ready_to_speak():
                            await self._speak_filler(session, turn, streamer, filler)
                        continue  # keep waiting on the same pending_next task

                try:
                    event = pending_next.result() if pending_next.done() else await pending_next
                except StopAsyncIteration:
                    break
                finally:
                    pending_next = None  # consumed — a new one is scheduled next iteration

                if self._is_stale(session, turn):
                    break

                etype = event.get("type")

                if etype == "tool_call":
                    tool_name = event.get("name", "")
                    filler.operation = _TOOL_TO_FILLER_OP.get(tool_name, FillerOperation.GENERIC)

                elif etype == "text_delta":
                    if first_text_at is None:
                        first_text_at = time.monotonic()
                        log_metric(
                            "time_to_first_chatbot_text",
                            (first_text_at - t_turn_start) * 1000,
                            call_id=session.call_id,
                            turn_id=turn.turn_id,
                        )
                        filler.cancel()  # real answer has started — stop any pending filler
                    text = event.get("text", "")
                    answer_buffer += text
                    turn.answer_text_parts.append(text)

                    ready, spoken_buffer = _extract_ready_sentences(spoken_buffer + text)
                    for sentence in ready:
                        if self._is_stale(session, turn):
                            break
                        await self._speak(session, turn, streamer, sentence, t_turn_start, first_audio_logged=[False])

                elif etype == "error":
                    log_error(
                        "turn_chatbot_error",
                        ChatbotClientError(str(event.get("message"))),
                        call_id=session.call_id,
                        turn_id=turn.turn_id,
                    )

        except ChatbotClientError as exc:
            log_error("turn_chatbot_failed", exc, call_id=session.call_id, turn_id=turn.turn_id)
            if not self._is_stale(session, turn):
                await self._speak_terminal_message(
                    session, turn, streamer, _error_message("chatbot", effective_language)
                )
            return
        finally:
            # A pending_next task can still be in flight here (e.g. this
            # turn was cancelled while waiting). It must be cancelled and
            # awaited before aclose() — closing an async generator while
            # one of its __anext__() calls is still running raises
            # "aclose(): asynchronous generator is already running".
            if pending_next is not None and not pending_next.done():
                pending_next.cancel()
                try:
                    await pending_next
                except (asyncio.CancelledError, StopAsyncIteration, Exception):  # noqa: BLE001
                    pass
            await events.aclose()

        if self._is_stale(session, turn):
            return

        # Flush any trailing partial sentence that never hit a boundary.
        if spoken_buffer.strip():
            await self._speak(session, turn, streamer, spoken_buffer.strip(), t_turn_start, first_audio_logged=[False])

        turn.state = TurnState.DONE
        log_metric(
            "turn_total_latency",
            (time.monotonic() - t_turn_start) * 1000,
            call_id=session.call_id,
            turn_id=turn.turn_id,
        )

    async def _speak(
        self,
        session: CallSession,
        turn: Turn,
        streamer: OutboundAudioStreamer,
        text: str,
        t_turn_start: float,
        first_audio_logged: list,
    ) -> None:
        turn.state = TurnState.SPEAKING
        try:
            audio_stream = self._tts.synthesize(text, turn.detected_language or session.language)
            async for chunk in audio_stream:
                if self._is_stale(session, turn) or streamer.stopped:
                    await audio_stream.aclose()
                    return
                if not first_audio_logged[0]:
                    first_audio_logged[0] = True
                    log_metric(
                        "time_to_first_audio",
                        (time.monotonic() - t_turn_start) * 1000,
                        call_id=session.call_id,
                        turn_id=turn.turn_id,
                    )
                sent = await streamer.send_chunk(chunk)
                if not sent:
                    await audio_stream.aclose()
                    return
        except SynthesisError as exc:
            log_error("turn_tts_failed", exc, call_id=session.call_id, turn_id=turn.turn_id)

    async def _speak_filler(
        self, session: CallSession, turn: Turn, streamer: OutboundAudioStreamer, filler: FillerController
    ) -> None:
        if turn.turn_id in session.fillers_spoken_for_turn:
            return
        message = filler.take_message()
        if not message:
            return
        session.fillers_spoken_for_turn.add(turn.turn_id)
        log_event("filler_spoken", call_id=session.call_id, turn_id=turn.turn_id, operation=filler.operation.value)
        await self._speak(session, turn, streamer, message, time.monotonic(), first_audio_logged=[True])

    async def _speak_terminal_message(
        self, session: CallSession, turn: Turn, streamer: OutboundAudioStreamer, message: str
    ) -> None:
        turn.state = TurnState.DONE
        await self._speak(session, turn, streamer, message, time.monotonic(), first_audio_logged=[True])

    @staticmethod
    def _is_stale(session: CallSession, turn: Turn) -> bool:
        """A turn is stale once it's been cancelled or superseded by a
        newer turn on the same call (barge-in started a new turn_id).
        """
        if turn.cancelled:
            return True
        current = session.current_turn
        return current is None or current.turn_id != turn.turn_id


def _extract_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Split accumulated streamed text into complete sentences plus a
    trailing partial fragment, so TTS can start speaking sentence-by-
    sentence instead of waiting for the whole chatbot answer.
    """
    parts = _SENTENCE_BOUNDARY.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    *complete, remainder = parts
    return [p.strip() for p in complete if p.strip()], remainder
