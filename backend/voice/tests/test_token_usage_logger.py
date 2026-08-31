from pathlib import Path

from voice.models import CallSession, CallState, Turn, new_call_id, new_turn_id
from voice.token_usage_logger import write_token_usage_log


def _session() -> CallSession:
    return CallSession(
        call_id=new_call_id(),
        user_id="user-abc",
        chat_session_id="voice-sess-1",
        language="en",
        state=CallState.ACTIVE,
    )


def test_writes_one_text_file_with_usage_fields(tmp_path):
    session = _session()
    turn = Turn(turn_id=new_turn_id(), call_id=session.call_id)
    turn.token_usage = {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}

    write_token_usage_log(str(tmp_path), session, turn)

    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert session.call_id in content
    assert turn.turn_id in content
    assert "input_tokens:   120" in content
    assert "output_tokens:  45" in content
    assert "total_tokens:   165" in content


def test_does_not_write_a_file_when_no_usage(tmp_path):
    session = _session()
    turn = Turn(turn_id=new_turn_id(), call_id=session.call_id)
    turn.token_usage = None

    write_token_usage_log(str(tmp_path), session, turn)

    assert list(tmp_path.glob("*.txt")) == []


def test_creates_the_log_directory_if_missing(tmp_path):
    nested = tmp_path / "nested" / "logs"
    session = _session()
    turn = Turn(turn_id=new_turn_id(), call_id=session.call_id)
    turn.token_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    write_token_usage_log(str(nested), session, turn)

    assert nested.exists()
    assert len(list(nested.glob("*.txt"))) == 1


def test_never_writes_raw_user_id_only_the_anonymised_reference(tmp_path):
    session = _session()
    turn = Turn(turn_id=new_turn_id(), call_id=session.call_id)
    turn.token_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    write_token_usage_log(str(tmp_path), session, turn)

    content = next(tmp_path.glob("*.txt")).read_text(encoding="utf-8")
    assert "user-abc" not in content


def test_a_write_failure_does_not_raise(monkeypatch, tmp_path):
    """Logging must never break the call it's reporting on."""
    session = _session()
    turn = Turn(turn_id=new_turn_id(), call_id=session.call_id)
    turn.token_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)

    write_token_usage_log(str(tmp_path), session, turn)  # must not raise
