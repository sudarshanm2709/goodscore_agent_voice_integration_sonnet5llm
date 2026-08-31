import time

from voice_service.fillers import FillerController, FillerOperation


def test_filler_not_ready_before_min_wait(filler_config):
    f = FillerController(config=filler_config, operation=FillerOperation.GENERIC, language="en")
    assert f.ready_to_speak(now=f.turn_started_at) is False


def test_filler_ready_after_min_wait(filler_config):
    f = FillerController(config=filler_config, operation=FillerOperation.GENERIC, language="en")
    later = f.turn_started_at + filler_config.min_wait_before_filler_seconds + 0.01
    assert f.ready_to_speak(now=later) is True


def test_filler_message_spoken_only_once(filler_config):
    f = FillerController(config=filler_config, operation=FillerOperation.CREDIT_REPORT, language="en")
    first = f.take_message()
    second = f.take_message()
    assert first is not None
    assert "credit report" in first.lower()
    assert second is None
    assert f.already_spoken is True


def test_filler_cancelled_never_speaks(filler_config):
    f = FillerController(config=filler_config, operation=FillerOperation.GENERIC, language="en")
    f.cancel()
    later = f.turn_started_at + 10
    assert f.ready_to_speak(now=later) is False
    assert f.take_message() is None


def test_filler_language_selection_hinglish(filler_config):
    f = FillerController(config=filler_config, operation=FillerOperation.BILLS, language="hi-en")
    message = f.take_message()
    assert message is not None
    assert "bills" in message.lower()


def test_filler_language_fallback_to_english_for_unknown_language(filler_config):
    f = FillerController(config=filler_config, operation=FillerOperation.GENERIC, language="fr")
    message = f.take_message()
    assert message == "One moment, please."
