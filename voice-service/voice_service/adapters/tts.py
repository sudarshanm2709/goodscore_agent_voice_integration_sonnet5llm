"""Provider-independent text-to-speech interface."""
from __future__ import annotations

import abc
from typing import AsyncIterator


class SynthesisError(RuntimeError):
    """Raised for any TTS provider failure: timeout, 4xx/5xx, malformed response."""


class TextToSpeechAdapter(abc.ABC):
    """Streams synthesized audio for one piece of text."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        language_hint: str | None,
    ) -> AsyncIterator[bytes]:
        """Yield audio chunks for `text` as they become available.

        Must be cancellable: the caller (turn_controller.py, on barge-in)
        will stop iterating this generator early. Implementations should
        close the underlying HTTP stream promptly when that happens
        (an `async for` loop broken by the caller triggers the
        generator's `finally`/context-manager exit — see
        openrouter_tts.py for how the HTTP stream is closed there).

        Raises:
            SynthesisError: on any provider failure.
        """
        raise NotImplementedError
