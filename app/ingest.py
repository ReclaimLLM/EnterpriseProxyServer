from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.backend import GatewayContext
from app.config import settings


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class InsertedGatewayRecord:
    session_id: str
    user_id: str


class SupabaseGatewayIngestor:
    """Compatibility name for the gateway log ingestor.

    The proxy no longer writes session blobs or rows directly. It sends the log
    record to ReclaimLLM-Backend, where encryption policy, S3 writes, and DB
    metadata are owned in one place.
    """

    def __init__(self) -> None:
        self._backend_client = httpx.AsyncClient(
            base_url=settings.backend_base_url.rstrip("/"),
            timeout=settings.backend_timeout_seconds,
        )

    async def close(self) -> None:
        await self._backend_client.aclose()

    async def ingest_gateway_record(
        self,
        *,
        context: GatewayContext,
        record: dict[str, Any],
    ) -> InsertedGatewayRecord:
        response = await self._backend_client.post(
            "/api/enterprise/gateway/ingest",
            headers={"X-Proxy-Secret": settings.proxy_shared_secret},
            json={
                "org_id": context.org_id,
                "org_slug": context.org_slug,
                "team_id": context.team_id,
                "user_id": context.user_id,
                "key_id": context.key_id,
                "provider": context.provider,
                "record": record,
            },
        )
        if response.status_code >= 400:
            raise IngestError(_extract_error(response))
        data = response.json()
        return InsertedGatewayRecord(
            session_id=str(data["session_id"]),
            user_id=str(data["user_id"]),
        )


def _extract_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    detail = body.get("detail") or body.get("error") or body.get("message")
    if isinstance(detail, str):
        return detail
    return response.reason_phrase
