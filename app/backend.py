from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings


class BackendError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class GatewayContext:
    org_id: str
    org_slug: str
    team_id: str
    user_id: str
    key_id: str
    provider: str
    provider_config: dict[str, Any]


class BackendClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.backend_base_url.rstrip("/"),
            timeout=settings.backend_timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def resolve_gateway_key(
        self,
        *,
        org_slug: str,
        bearer_token: str,
        model: str,
    ) -> GatewayContext:
        response = await self._client.post(
            "/api/enterprise/gateway/resolve",
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "X-Proxy-Secret": settings.proxy_shared_secret,
            },
            json={"org_slug": org_slug, "model": model},
        )
        if response.status_code >= 400:
            raise BackendError(response.status_code, _extract_error(response))
        data = response.json()
        return GatewayContext(
            org_id=data["org_id"],
            org_slug=data["org_slug"],
            team_id=data["team_id"],
            user_id=data["user_id"],
            key_id=data["key_id"],
            provider=data["provider"],
            provider_config=data["provider_config"],
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
