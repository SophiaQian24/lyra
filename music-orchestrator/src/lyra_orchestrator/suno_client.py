from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class SunoAPIError(RuntimeError):
    pass


@dataclass
class SunoClient:
    base_url: str
    timeout_seconds: float = 120.0

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        timeout = kwargs.pop("timeout", self.timeout_seconds)
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
                response = await client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise SunoAPIError(
                f"{method} {path} timed out after {timeout}s (base_url={self.base_url})"
            ) from exc
        except httpx.RequestError as exc:
            raise SunoAPIError(f"{method} {path} request error: {repr(exc)}") from exc
        if response.status_code >= 400:
            detail = response.text
            raise SunoAPIError(f"{method} {path} failed: {response.status_code} {detail}")
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.text

    async def get_limit(self) -> dict[str, Any]:
        data = await self._request("GET", "/api/get_limit")
        if not isinstance(data, dict):
            raise SunoAPIError("Unexpected /api/get_limit response")
        return data

    async def custom_generate(
        self,
        *,
        prompt: str,
        tags: str,
        title: str,
        negative_tags: str,
        make_instrumental: bool = True,
        model: str = "chirp-v3-5",
        wait_audio: bool = True,
    ) -> list[dict[str, Any]]:
        payload = {
            "prompt": prompt,
            "tags": tags,
            "title": title,
            "negative_tags": negative_tags,
            "make_instrumental": make_instrumental,
            "model": model,
            "wait_audio": wait_audio,
        }
        data = await self._request("POST", "/api/custom_generate", json=payload)
        if not isinstance(data, list):
            raise SunoAPIError("Unexpected /api/custom_generate response")
        return data

    async def get(self, ids: list[str]) -> list[dict[str, Any]]:
        query = ",".join(ids)
        data = await self._request("GET", f"/api/get?ids={query}")
        if not isinstance(data, list):
            raise SunoAPIError("Unexpected /api/get response")
        return data

    async def download_to_file(self, url: str, dest_path: str) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
