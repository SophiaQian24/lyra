from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lyra_orchestrator.api import create_app
from lyra_orchestrator.manager import RunManager
from lyra_orchestrator.settings import Settings


async def _noop_start(self):
    return None


async def _noop_stop(self):
    return None


def test_api_run_lifecycle_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RunManager, "start", _noop_start)
    monkeypatch.setattr(RunManager, "stop", _noop_stop)

    settings = Settings(
        OPENAI_API_KEY=None,
        ARTIFACTS_ROOT=tmp_path / "artifacts",
        RUNS_DB_PATH=tmp_path / "runs.db",
        SUNO_API_BASE_URL="http://localhost:3000",
    )

    app = create_app(settings)
    client = TestClient(app)

    create_resp = client.post("/v1/runs", json={"prompt": "melodic house 124 bpm", "references": [], "overrides": {}})
    assert create_resp.status_code == 200

    payload = create_resp.json()
    run_id = payload["run_id"]
    assert payload["status"] == "queued"
    assert "music_brief" in payload

    status_resp = client.get(f"/v1/runs/{run_id}")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["run_id"] == run_id
    assert status_payload["status"] in ["queued", "planning", "failed", "done", "generating", "qc", "mixing"]

    retry_resp = client.post(f"/v1/runs/{run_id}/retry", json={"target": "drums", "reason": "test"})
    assert retry_resp.status_code == 200
    assert retry_resp.json()["target"] == "drums"
