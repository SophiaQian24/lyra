from __future__ import annotations

import pytest
import respx
from httpx import Response

from lyra_orchestrator.suno_client import SunoAPIError, SunoClient


@pytest.mark.asyncio
async def test_custom_generate_happy_path() -> None:
    client = SunoClient("http://localhost:3000", timeout_seconds=10)

    with respx.mock(base_url="http://localhost:3000") as mock:
        mock.post("/api/custom_generate").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "id": "clip1",
                        "audio_url": "https://cdn.example/audio.mp3",
                        "status": "complete",
                    }
                ],
            )
        )

        out = await client.custom_generate(
            prompt="drums only",
            tags="drums",
            title="drums_run",
            negative_tags="no vocals",
            wait_audio=True,
        )

    assert out[0]["id"] == "clip1"


@pytest.mark.asyncio
async def test_custom_generate_402() -> None:
    client = SunoClient("http://localhost:3000", timeout_seconds=10)

    with respx.mock(base_url="http://localhost:3000") as mock:
        mock.post("/api/custom_generate").mock(return_value=Response(402, json={"error": "credits"}))
        with pytest.raises(SunoAPIError):
            await client.custom_generate(
                prompt="bass",
                tags="bass",
                title="bass_run",
                negative_tags="",
            )
