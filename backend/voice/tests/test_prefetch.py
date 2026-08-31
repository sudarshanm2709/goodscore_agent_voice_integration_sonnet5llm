import pytest

from voice.clients.goodscore import GoodScoreApiError
from voice.prefetch import CreditPrefetchController


class _OkClient:
    async def get_credit_report(self, user_id: str) -> dict:
        return {"score": 742, "bureau": "Equifax"}


class _FailingClient:
    async def get_credit_report(self, user_id: str) -> dict:
        raise GoodScoreApiError("staging API unreachable")


class _CrashingClient:
    async def get_credit_report(self, user_id: str) -> dict:
        raise ValueError("unexpected bug")


async def test_prefetch_success_produces_summary_without_raw_report():
    controller = CreditPrefetchController(_OkClient())
    result = await controller.prefetch("call-1", "user-1")

    assert result.ok is True
    assert result.summary == {"available": True}
    # The full report (score, bureau) must never leak into the stored result.
    assert "score" not in result.summary
    assert "bureau" not in result.summary


async def test_prefetch_api_failure_does_not_raise():
    controller = CreditPrefetchController(_FailingClient())
    result = await controller.prefetch("call-1", "user-1")

    assert result.ok is False
    assert result.summary is None
    assert result.error is not None


async def test_prefetch_unexpected_exception_does_not_raise():
    controller = CreditPrefetchController(_CrashingClient())
    result = await controller.prefetch("call-1", "user-1")

    assert result.ok is False
    assert "unexpected bug" in (result.error or "")
