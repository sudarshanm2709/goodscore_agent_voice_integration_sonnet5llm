"""Direct client for the existing GoodScore backend staging API.

Mirrors the retry/backoff pattern already proven in backend/tools.py's
_fetch_stage_api (same host, same {"userId": ...} POST contract) so the
voice service's prefetch behaves identically to the chat tool it parallels.
Deliberately not imported from backend/ — the voice service is a separate
deployable container and must not depend on the chatbot's source tree.

Per <credit_prefetch>: this client talks to the GoodScore backend API only.
It never touches Aurora, S3, or any AWS data store directly.
"""
from __future__ import annotations

import asyncio
import random

import httpx

from ..config import GoodScoreApiConfig
from ..observability import log_error, log_event


class GoodScoreApiError(RuntimeError):
    """Raised when the GoodScore staging API cannot be reached or errors out."""


class GoodScoreClient:
    def __init__(self, config: GoodScoreApiConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url,
            verify=False,  # staging cert is self-signed — matches backend/tools.py precedent
            timeout=httpx.Timeout(config.request_timeout_seconds),
            headers={"Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_credit_report(self, user_id: str) -> dict:
        return await self._post("/subscription/getCreditReportV3", user_id)

    async def _post(self, path: str, user_id: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = await self._client.post(path, json={"userId": user_id})
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise GoodScoreApiError(
                        f"GoodScore API rejected request: {exc.response.status_code}"
                    ) from exc
                last_error = exc

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt < self._config.max_retries:
                wait = min(0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2), 4.0)
                log_event("goodscore_api_retry", path=path, attempt=attempt, wait_seconds=round(wait, 2))
                await asyncio.sleep(wait)

        log_error("goodscore_api_failed", last_error or GoodScoreApiError("unknown"), path=path)
        raise GoodScoreApiError(f"GoodScore API unreachable after retries: {last_error}")
